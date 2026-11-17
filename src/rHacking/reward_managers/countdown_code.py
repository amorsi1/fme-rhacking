# Adapted from Countdown-Code by Zohaib Khan et al.
# https://github.com/zohaib-khan5040/Countdown-Code @ 170ee81
# Upstream ships no LICENSE file; used here for internal research under the
# original authors' copyright. See CLAUDE.md for local modifications.

from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from typing import Any

import torch
from transformers import PreTrainedTokenizer
from verl import DataProto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager

from rHacking.scoring import score_rollout


@register("fme_countdown_code")
class CountdownCodeRewardManager(AbstractRewardManager):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        num_examine: int,
        compute_score: Any = None,
        reward_fn_key: str = "data_source",
        enforce_file_integrity: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.enforce_file_integrity = enforce_file_integrity

    def verify(self, data: DataProto):
        responses = self.tokenizer.batch_decode(data.batch["responses"], skip_special_tokens=True)
        truths = [item["ground_truth"] for item in data.non_tensor_batch["reward_model"]]
        if len(truths) != len(responses):
            raise ValueError("Expected one original Countdown problem per response")
        if not responses:
            return []
        with ProcessPoolExecutor(max_workers=min(len(responses), 64)) as workers:
            return list(workers.map(score_rollout, responses, truths, repeat(self.enforce_file_integrity)))

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        prompt_length = data.batch["prompts"].shape[-1]
        response_lengths = data.batch["attention_mask"][:, prompt_length:].sum(dim=-1)
        extra = {"equation_accuracy": []}

        for i, (score, gold, flags) in enumerate(self.verify(data)):
            length = int(response_lengths[i].item())
            if length:
                reward_tensor[i, length - 1] = score
            extra["equation_accuracy"].append(gold)
            for key, value in flags.items():
                extra.setdefault(key, []).append(value)
            if i < self.num_examine:
                print(f"[Countdown] reward={score} gold={gold} {flags}")

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": extra}
        return reward_tensor


# verl imports this file to resolve the placeholder, running @register in its Ray worker.
def unused_compute_score(*args: Any, **kwargs: Any) -> float:
    """Placeholder for `custom_reward_function.name`; never actually invoked."""
    raise RuntimeError(
        "unused_compute_score was called, which should be impossible: "
        "CountdownCodeRewardManager computes the execution and equation rewards "
        "itself and ignores the compute_score it is constructed with. This means "
        "verl's reward plumbing changed in "
        f"{__file__}."
    )
