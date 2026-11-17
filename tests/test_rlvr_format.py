"""Tests for the RLVR parquet `format_for_rl.py` produces.

verl's PPO trainer reads the `prompt` column as a chat list and hands
`reward_model` to the reward manager, which pulls `ground_truth` out of it. The
data-to-score contract can be exercised on CPU using the shared rollout scorer.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from datasets import Dataset

import format_for_rl
import paths
from rHacking.countdown_files import starter_files
from rHacking.integrity import score_rollout
from test_sft_format import hydra_override, shell_var

RLVR_TRAIN_RELPATH = Path("data/rlvr/train.parquet")
RLVR_TEST_RELPATH = Path("data/rlvr/test.parquet")

REWARD_MANAGER_RELPATH = Path("src/rHacking/reward_managers/countdown_code.py")


def rlvr_configs(repo_root: Path) -> list[Path]:
    configs = sorted((repo_root / "configs" / "rlvr").glob("*.sh"))
    assert configs, "no RLVR launch scripts found"
    return configs


@pytest.fixture
def mapped_record() -> dict[str, Any]:
    """One RLVR example, mapped exactly as `format_for_rl.__main__` maps them."""
    return format_for_rl.make_map_fn("train")({"nums": [12, 7, 3], "target": 30}, 4)


@pytest.fixture
def rlvr_parquet(tmp_path: Path) -> pd.DataFrame:
    """A mapped dataset written to parquet and read back, as verl would."""
    source = Dataset.from_list(
        [{"nums": [1, 2, 3], "target": 6}, {"nums": [4, 5, 6, 7], "target": 50}]
    )
    mapped = source.map(
        format_for_rl.make_map_fn("test"),
        with_indices=True,
        remove_columns=source.column_names,
    )
    out = tmp_path / "test.parquet"
    mapped.to_parquet(out)
    return pd.read_parquet(out)


# --- the mapped record -----------------------------------------------------


def test_mapped_record_has_the_columns_verl_expects(
    mapped_record: dict[str, Any],
) -> None:
    assert set(mapped_record) == {
        "data_source",
        "prompt",
        "ability",
        "reward_model",
        "extra_info",
    }


def test_prompt_is_a_system_then_user_chat_list(
    mapped_record: dict[str, Any],
) -> None:
    prompt = mapped_record["prompt"]

    assert [msg["role"] for msg in prompt] == ["system", "user"]
    assert all(isinstance(msg["content"], str) and msg["content"] for msg in prompt)
    # The problem itself must be in the user turn, not just the system message.
    assert "[12, 7, 3]" in prompt[1]["content"]
    assert "30" in prompt[1]["content"]


def test_prompt_carries_both_files_for_the_model_to_edit(
    mapped_record: dict[str, Any],
) -> None:
    user = mapped_record["prompt"][1]["content"]

    assert "solution.py" in user and "test.py" in user
    assert "<think>" in user


def test_ground_truth_carries_the_original_problem(
    mapped_record: dict[str, Any],
) -> None:
    reward_model = mapped_record["reward_model"]

    assert reward_model["style"] == "rule"
    assert set(reward_model["ground_truth"]) == {"numbers", "target"}
    assert reward_model["ground_truth"]["numbers"] == [12, 7, 3]
    assert reward_model["ground_truth"]["target"] == 30


def test_extra_info_records_the_split_and_index() -> None:
    train = format_for_rl.make_map_fn("train")({"nums": [1, 2, 3], "target": 6}, 0)
    test = format_for_rl.make_map_fn("test")({"nums": [1, 2, 3], "target": 6}, 7)

    assert train["extra_info"] == {"split": "train", "index": 0}
    assert test["extra_info"] == {"split": "test", "index": 7}


# --- the parquet round-trip ------------------------------------------------


def test_parquet_round_trip_preserves_the_chat_list(
    rlvr_parquet: pd.DataFrame,
) -> None:
    assert set(rlvr_parquet.columns) == {
        "data_source",
        "prompt",
        "ability",
        "reward_model",
        "extra_info",
    }

    for prompt in rlvr_parquet["prompt"]:
        # verl applies the chat template to this column, so each element has to
        # survive parquet as a role/content mapping.
        assert [msg["role"] for msg in prompt] == ["system", "user"]
        assert all(set(msg) == {"role", "content"} for msg in prompt)


def test_parquet_round_trip_preserves_the_ground_truth(
    rlvr_parquet: pd.DataFrame,
) -> None:
    ground_truths = [row["ground_truth"] for row in rlvr_parquet["reward_model"]]

    assert [list(gt["numbers"]) for gt in ground_truths] == [[1, 2, 3], [4, 5, 6, 7]]
    assert [gt["target"] for gt in ground_truths] == [6, 50]


def test_variable_length_number_lists_survive_one_parquet_schema(
    rlvr_parquet: pd.DataFrame,
) -> None:
    # The dataset mixes 3-number and 4-number problems; a fixed-width column
    # type would truncate or fail on one of them.
    lengths = {
        len(row["ground_truth"]["numbers"]) for row in rlvr_parquet["reward_model"]
    }
    assert lengths == {3, 4}


# --- the launch scripts' side of the contract ------------------------------


def test_rlvr_configs_read_the_paths_format_for_rl_writes(repo_root: Path) -> None:
    assert paths.DATA_DIR == repo_root / "data"

    for config in rlvr_configs(repo_root):
        text = config.read_text()
        train = shell_var(text, "TRAIN_PATH").replace("${PROJECT_DIR}/", "")
        test = shell_var(text, "TEST_PATH").replace("${PROJECT_DIR}/", "")

        assert Path(train) == RLVR_TRAIN_RELPATH, config.name
        assert Path(test) == RLVR_TEST_RELPATH, config.name


def test_rlvr_configs_leave_the_prompt_column_at_its_default(
    repo_root: Path, mapped_record: dict[str, Any]
) -> None:
    for config in rlvr_configs(repo_root):
        text = config.read_text()
        # Nothing overrides data.prompt_key, so verl reads the column named
        # "prompt" -- which is what the mapping function emits.
        assert "data.prompt_key" not in text, config.name
        assert "prompt" in mapped_record


def test_parquet_ground_truth_scores_the_original_problem(
    rlvr_parquet: pd.DataFrame,
) -> None:
    ground_truth = rlvr_parquet.iloc[0]["reward_model"]["ground_truth"]
    files = starter_files([1, 2, 3], 6)
    files["solution.py"] = files["solution.py"].replace("None", "'1+2+3'")
    for enforce in (False, True):
        reward, gold, _ = score_rollout(json.dumps(files), ground_truth, enforce)
        assert (reward, gold) == (1.2, 1.0)


def test_rlvr_configs_register_the_reward_manager_that_exists(
    repo_root: Path,
) -> None:
    manager_path = repo_root / REWARD_MANAGER_RELPATH
    source = manager_path.read_text()
    registered = re.search(r"@register\(\"([^\"]+)\"\)", source)
    assert registered is not None
    assert registered.group(1) == "fme_countdown_code"  # The pinned verl owns "countdown_code".

    for config in rlvr_configs(repo_root):
        text = config.read_text()
        # The by-path exec is the only thing that runs @register in the Ray
        # worker; a stale path fails as "Unknown reward manager".
        path = hydra_override(text, "custom_reward_function.path")
        assert path == f'"${{PROJECT_DIR}}/{REWARD_MANAGER_RELPATH}"', config.name
        assert manager_path.is_file()

        name = hydra_override(text, "custom_reward_function.name")
        assert f"def {name}(" in source, config.name

        manager = hydra_override(text, "reward_model.reward_manager").strip("'\"")
        assert manager == registered.group(1), config.name
