import os
import shutil
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import pytest


def test_baseline_works_with_the_experiment_package_unavailable(tmp_path):
    problems = tmp_path / "baseline.jsonl"
    problems.write_text('{"nums":[1,2,3],"target":6}\n{"nums":[3,4],"target":7}\n')
    script = dedent("""
        import json, sys
        sys.modules["deterministic_file_integrity"] = None
        from countdown_code import load_environment
        from rHacking.countdown_files import starter_files
        from rHacking.scoring import score_rollout
        env = load_environment(local_file=sys.argv[1], num_samples=0, test_size=0.5)
        assert len(env.get_eval_dataset()) == 1 and env.score_rollouts
        files = starter_files([1, 2, 3], 6)
        files["solution.py"] = files["solution.py"].replace("None", "'1+2+3'")
        score = score_rollout(json.dumps(files), {"numbers": [1, 2, 3], "target": 6})
        assert score == (1.2, 1.0, {"cheating_rate": 0})
    """)
    subprocess.run(
        [sys.executable, "-c", script, str(problems)], check=True, cwd=tmp_path,
        env={**os.environ, "HF_HOME": str(tmp_path / "hf"), "HF_HUB_OFFLINE": "1"},
    )


@pytest.mark.parametrize(("condition", "enforce"), [("baseline", "false"), ("integrity", "true")])
def test_training_from_experiment_directory_uses_private_paths(tmp_path, condition, enforce):
    package = tmp_path / "src" / "deterministic_file_integrity"
    package.mkdir(parents=True)
    shutil.copy(Path(__file__).resolve().parents[1] / "train.sh", package)
    launcher = tmp_path / "configs/rlvr/run_qwen2.5-coder-3b-fft.sh"
    launcher.parent.mkdir(parents=True)
    launcher.write_text('printf "%s\\n" "$PWD" "$SKIP_DATA_CHECK" "$WANDB_DIR" "$@"\n')
    lines = subprocess.check_output(
        ["bash", "train.sh", condition, "43", "base", "trainer.total_training_steps=1"],
        cwd=package, env={**os.environ, "INIT_MODEL": "local-model"}, text=True,
    ).splitlines()
    run = f"deterministic_file_integrity/file-integrity-base-{condition}-s43"
    parquet = tmp_path / "data/deterministic_file_integrity/rlvr"
    assert lines[:3] == [str(tmp_path), "1", f"{tmp_path}/outputs/{run}"]
    assert {
        f"data.train_files={parquet}/train.parquet", f"data.val_files={parquet}/test.parquet",
        f"trainer.default_local_dir=checkpoints/{run}", f"trainer.rollout_data_dir=rollouts/{run}/train",
        f"trainer.validation_data_dir=rollouts/{run}/val", f"hydra.run.dir=outputs/{run}",
        f"+reward_model.reward_kwargs.enforce_file_integrity={enforce}", "+actor_rollout_ref.rollout.seed=43",
        "actor_rollout_ref.model.path=local-model", "trainer.total_training_steps=1",
    } <= set(lines[3:])
