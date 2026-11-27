"""Exercise the same by-path loader and batch interface as pinned verl on CPU."""

import json
from importlib.util import find_spec

import numpy as np
import pytest

from rHacking.countdown_files import starter_files


def test_train_and_validation_reward_managers(pytestconfig):
    if find_spec("verl") is None:
        pytest.skip("Requires the pinned verl reward stack")
    import torch
    from omegaconf import OmegaConf
    from verl import DataProto
    from verl.trainer.ppo.reward import load_reward_manager
    from verl.utils.config import omega_conf_to_dataclass
    from verl.workers.reward_manager import get_reward_manager_cls

    repo_root = pytestconfig.rootpath
    config = OmegaConf.create({
        "data": {"reward_fn_key": "data_source"},
        "reward_model": {"reward_manager": "rhacking_countdown_code", "reward_kwargs": {"enforce_file_integrity": True}},
        "custom_reward_function": {"path": str(repo_root / "src/rHacking/reward_managers/countdown_code.py"),
                                   "name": "unused_compute_score"},
    })
    files = starter_files([1, 2, 3], 6)
    files["solution.py"] = files["solution.py"].replace("None", "'1+2+3'")
    changed = {**files, "test.py": files["test.py"].replace("1e-5", "1e-6")}
    responses = [json.dumps(files), json.dumps(changed)]

    class Tokenizer:
        def batch_decode(self, ids, skip_special_tokens):
            return [responses[i] for i in ids[:, 0].tolist()]

    batch = DataProto.from_dict(
        tensors={"prompts": torch.ones(2, 1), "responses": torch.tensor([[0, 3, 3], [1, 3, 0]]),
                 "attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])},
        non_tensors={"reward_model": np.array([{"ground_truth": {"numbers": [1, 2, 3], "target": 6}}] * 2)},
    )
    baseline = load_reward_manager(config, Tokenizer(), num_examine=0)
    assert type(baseline) is not get_reward_manager_cls("countdown_code")
    expected = torch.tensor([[0, 0, 1.2], [0, 1.2, 0]])
    torch.testing.assert_close(baseline(batch), expected)
    for num_examine in (0, 1):  # train and validation construction in main_ppo
        manager = load_reward_manager(config, Tokenizer(), num_examine, **config.reward_model.reward_kwargs)
        result = manager(batch, return_dict=True)
        torch.testing.assert_close(result["reward_tensor"], torch.tensor([[0, 0, 1.2], [0, 0, 0]]))
        assert result["reward_extra_info"]["equation_accuracy"] == [1.0, 1.0]
        assert result["reward_extra_info"]["integrity_flagged"] == [0, 1]

    rollout = omega_conf_to_dataclass(OmegaConf.create({
        "_target_": "deterministic_file_integrity.rollout_config.SeededRolloutConfig", "name": "vllm", "seed": 43,
    }))
    assert rollout.get("seed", 0) == 43
    from rHacking.rollout_config import SeededRolloutConfig
    assert type(rollout) is SeededRolloutConfig
