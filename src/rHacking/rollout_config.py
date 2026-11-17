from dataclasses import dataclass

from verl.workers.config import RolloutConfig


@dataclass
class SeededRolloutConfig(RolloutConfig):
    """Expose the engine seed the pinned vLLM worker already reads."""

    seed: int = 0
