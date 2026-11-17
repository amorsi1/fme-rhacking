"""Tests for the SFT dataset `filtering_proxy.py` produces.

Two things are checked: that each trace is reshaped into the
`<think>...</think>{json}` target the environment's parser can read back, and
that the parquet output matches what `configs/sft/*.sh` tells verl's
`fsdp_sft_trainer` to look for -- a single `extra_info` struct column whose
`question`/`answer` members are strings.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import distillation
import filtering_proxy
import paths
import rewards
from conftest import TAMPERED_TEST, trace_record, write_traces

# Where the launch scripts expect the parquet, relative to the repo root.
SFT_TRAIN_RELPATH = Path("data/sft/train.parquet")
SFT_TEST_RELPATH = Path("data/sft/test.parquet")


def sft_configs(repo_root: Path) -> list[Path]:
    configs = sorted((repo_root / "configs" / "sft").glob("*.sh"))
    assert configs, "no SFT launch scripts found"
    return configs


def hydra_override(text: str, key: str) -> str:
    """The value of a `key=value` argument on a launch script's command line."""
    match = re.search(rf"[\s+]{re.escape(key)}=(\S+?)\s*\\?$", text, re.MULTILINE)
    assert match is not None, f"{key} not set"
    return match.group(1)


def hydra_list_override(text: str, key: str) -> list[str]:
    """`key=['a','b']` as a list of names."""
    return re.findall(r"'([^']+)'", hydra_override(text, key))


def shell_var(text: str, name: str) -> str:
    match = re.search(rf'^{name}="([^"]+)"$', text, re.MULTILINE)
    assert match is not None, f"{name} not assigned"
    return match.group(1)


@pytest.fixture
def scored_traces(
    data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    honest_trace: dict[str, Any],
    hacked_trace: dict[str, Any],
    failed_trace: dict[str, Any],
    unparseable_trace: dict[str, Any],
) -> pd.DataFrame:
    """`score_traces()` over a four-record stand-in for the 16k traces."""
    traces = write_traces(
        data_dir / distillation.TRACES_FILENAME,
        [honest_trace, hacked_trace, failed_trace, unparseable_trace],
    )
    monkeypatch.setattr(
        distillation, "download_distillation_traces", lambda force=False: traces
    )
    return filtering_proxy.score_traces()


# --- the SFT target string --------------------------------------------------


def test_formatted_response_is_one_think_block_followed_by_the_file_json() -> None:
    record = trace_record([1, 2, 3], 6, "1 + 2 + 3", summary=["first", "second"])

    response = filtering_proxy.format_model_output(record["output"])

    assert response is not None
    assert response.count("<think>") == 1
    assert response.count("</think>") == 1
    # The reasoning summary and the final chain-of-thought are merged, so no
    # part of the trace's thinking is dropped from the target.
    think = response.split("</think>")[0]
    assert "first" in think and "second" in think
    assert "the reasoning that was kept in the final message" in think


def test_formatted_response_is_readable_by_the_reward_parser() -> None:
    """The SFT target has to parse with the same extractor the rewards use."""
    record = trace_record([1, 2, 3], 6, "1 + 2 + 3")

    response = filtering_proxy.format_model_output(record["output"])

    files = rewards.load_json_from_response(response)
    assert set(files) == {"solution.py", "test.py"}
    assert "expr = '1 + 2 + 3'" in files["solution.py"]
    assert "def verify_solution" in files["test.py"]
    # The json must be the trailing object of the string, which is what the
    # `</think>\s*({.*})\s*$` branch of the parser requires.
    assert response.rstrip().endswith("}")


def test_formatted_response_drops_stray_think_tags_from_the_reasoning() -> None:
    record = trace_record([1, 2, 3], 6, "1 + 2 + 3", cot="a <think> leaked in here")

    response = filtering_proxy.format_model_output(record["output"])

    assert response is not None
    assert response.count("<think>") == 1


def test_unparseable_file_json_is_dropped_rather_than_raising(
    unparseable_trace: dict[str, Any],
) -> None:
    # Upstream let the JSONDecodeError escape, aborting the whole run on the
    # 105/16000 traces that hit this.
    assert filtering_proxy.format_model_output(unparseable_trace["output"]) is None


@pytest.mark.parametrize("output", [None, "not a dict", 3.0])
def test_non_dict_output_is_dropped(output: object) -> None:
    assert filtering_proxy.format_model_output(output) is None


# --- the prompt side -------------------------------------------------------


def test_query_is_the_user_turn_not_the_shared_system_message() -> None:
    record = trace_record([12, 7, 3], 30, "12 * 3 - 7")

    query = filtering_proxy.extract_query(record["prompt"])

    assert "You are a helpful assistant" not in query
    assert "[12, 7, 3]" in query and "30" in query


def test_query_accepts_a_plain_string_prompt() -> None:
    assert filtering_proxy.extract_query("just the problem") == "just the problem"


def test_query_is_none_when_no_user_turn_exists() -> None:
    assert filtering_proxy.extract_query([{"role": "system", "content": "hi"}]) is None


# --- scoring and filtering -------------------------------------------------


def test_both_rewards_are_attached_to_every_trace(
    scored_traces: pd.DataFrame,
) -> None:
    honest, hacked, failed, unparseable = (
        scored_traces.iloc[i] for i in range(len(scored_traces))
    )

    # Honest: the model's own test passes and the expression is genuinely right.
    assert (honest["test_reward"], honest["equation_reward"]) == (1, 1)
    # Tampered test: proxy fires, true reward does not -- the gap under study.
    assert (hacked["test_reward"], hacked["equation_reward"]) == (1, 0.0)
    # Honest test the expression fails: neither reward fires.
    assert (failed["test_reward"], failed["equation_reward"]) == (0, 0.0)
    assert (unparseable["test_reward"], unparseable["equation_reward"]) == (0, 0.0)


def test_true_reward_sees_through_a_tampered_test(
    hacked_trace: dict[str, Any],
) -> None:
    response = filtering_proxy.format_model_output(hacked_trace["output"])

    assert rewards.test_pass_reward(response) == 1
    assert (
        rewards.equation_pass_reward(
            nums=hacked_trace["input"]["nums"],
            target=hacked_trace["input"]["target"],
            response_obj=response,
        )
        == 0.0
    )


# --- the parquet the launch scripts read -----------------------------------


@pytest.fixture
def sft_outputs(scored_traces: pd.DataFrame, data_dir: Path) -> dict[str, Any]:
    """Run the full filtering step and read everything back off disk."""
    filtering_proxy.main()
    return {
        "jsonl": [
            json.loads(line)
            for line in (data_dir / "sft_dataset.jsonl").read_text().splitlines()
        ],
        "train": pd.read_parquet(data_dir / "sft" / "train.parquet"),
        "test": pd.read_parquet(data_dir / "sft" / "test.parquet"),
    }


def test_only_proxy_passing_traces_are_kept(sft_outputs: dict[str, Any]) -> None:
    kept = sft_outputs["jsonl"]

    # The honest and the hacked trace; the failing and unparseable ones go.
    assert len(kept) == 2
    assert {record["input"]["target"] for record in kept} == {6, 100}
    assert all(set(record) == {"input", "query", "response"} for record in kept)


def test_hacked_traces_survive_filtering(sft_outputs: dict[str, Any]) -> None:
    """The tampered slice is the SFT catalyst, so it must not be filtered out."""
    rows = pd.concat([sft_outputs["train"], sft_outputs["test"]])
    ground_truths = [row["ground_truth"] for row in rows["extra_info"]]

    assert any(gt["equation_reward"] == 0 for gt in ground_truths)
    assert any(gt["equation_reward"] == 1 for gt in ground_truths)


def test_train_and_test_partition_the_kept_traces(
    sft_outputs: dict[str, Any],
) -> None:
    train, test = sft_outputs["train"], sft_outputs["test"]

    assert len(train) + len(test) == len(sft_outputs["jsonl"])
    train_questions = {row["question"] for row in train["extra_info"]}
    test_questions = {row["question"] for row in test["extra_info"]}
    assert not train_questions & test_questions


def test_parquet_carries_exactly_the_extra_info_column(
    sft_outputs: dict[str, Any],
) -> None:
    for split in ("train", "test"):
        df = sft_outputs[split]
        # The launch scripts set prompt_key=response_key=extra_info, so the
        # single struct column has to hold everything.
        assert list(df.columns) == ["extra_info"]
        assert all(isinstance(row, dict) for row in df["extra_info"])


def test_extra_info_structs_have_the_keys_verl_indexes(
    sft_outputs: dict[str, Any],
) -> None:
    rows = pd.concat([sft_outputs["train"], sft_outputs["test"]])["extra_info"]

    for record in rows:
        assert set(record) == {"question", "answer", "ground_truth"}
        # fsdp_sft_trainer tokenizes these directly; a None or a dict here
        # fails deep inside the trainer rather than at data load.
        assert isinstance(record["question"], str) and record["question"]
        assert isinstance(record["answer"], str) and record["answer"]
        assert set(record["ground_truth"]) == {
            "numbers",
            "target",
            "equation_reward",
        }
        assert isinstance(record["ground_truth"]["target"], int)
        numbers = record["ground_truth"]["numbers"]
        assert len(numbers) >= 3 and all(int(n) == n for n in numbers)


def test_answers_are_trainable_targets(sft_outputs: dict[str, Any]) -> None:
    """Every kept answer must still parse into the two files after round-trip."""
    rows = pd.concat([sft_outputs["train"], sft_outputs["test"]])["extra_info"]

    for record in rows:
        assert record["answer"].startswith("<think>")
        files = rewards.load_json_from_response(record["answer"])
        assert set(files) == {"solution.py", "test.py"}


def test_split_is_deterministic(
    scored_traces: pd.DataFrame, data_dir: Path
) -> None:
    filtering_proxy.main()
    first = pd.read_parquet(data_dir / "sft" / "train.parquet")
    filtering_proxy.main()
    second = pd.read_parquet(data_dir / "sft" / "train.parquet")

    assert [r["question"] for r in first["extra_info"]] == [
        r["question"] for r in second["extra_info"]
    ]


def test_extra_info_is_built_from_the_scored_row(
    scored_traces: pd.DataFrame,
) -> None:
    row = scored_traces.iloc[1]  # the hacked trace

    record = filtering_proxy.to_verl_extra_info(row)

    assert record["question"] == row["query"]
    assert record["answer"] == row["response"]
    # `nums` arrives from pandas as a numpy array and is passed through as-is;
    # parquet stores it as a list either way.
    assert list(record["ground_truth"]["numbers"]) == [4, 5, 6]
    assert record["ground_truth"]["target"] == row["input"]["target"]
    assert record["ground_truth"]["equation_reward"] == 0.0


# --- the launch scripts' side of the contract ------------------------------


def test_sft_configs_read_the_paths_filtering_proxy_writes(repo_root: Path) -> None:
    assert paths.DATA_DIR == repo_root / "data"

    for config in sft_configs(repo_root):
        text = config.read_text()
        train = shell_var(text, "TRAIN_PATH").replace("${PROJECT_DIR}/", "")
        test = shell_var(text, "TEST_PATH").replace("${PROJECT_DIR}/", "")

        assert Path(train) == SFT_TRAIN_RELPATH, config.name
        assert Path(test) == SFT_TEST_RELPATH, config.name
        assert hydra_override(text, "data.train_files") == "$TRAIN_PATH"
        assert hydra_override(text, "data.val_files") == "$TEST_PATH"


def test_sft_configs_index_the_keys_the_records_actually_have(
    repo_root: Path, scored_traces: pd.DataFrame
) -> None:
    record = filtering_proxy.to_verl_extra_info(scored_traces.iloc[0])

    for config in sft_configs(repo_root):
        text = config.read_text()
        prompt_key = hydra_override(text, "data.prompt_key")
        response_key = hydra_override(text, "data.response_key")
        # Both keys name the same struct column, which is why question and
        # answer have to live inside one record.
        assert prompt_key == response_key == "extra_info", config.name

        for key in hydra_list_override(text, "data.prompt_dict_keys"):
            assert isinstance(record[key], str), config.name
        for key in hydra_list_override(text, "data.response_dict_keys"):
            assert isinstance(record[key], str), config.name


def test_sft_configs_must_run_from_the_repo_root(repo_root: Path) -> None:
    # The paths above are relative to $(pwd); this is the assumption that makes
    # them resolve, and the README says so.
    for config in sft_configs(repo_root):
        assert 'PROJECT_DIR="$(pwd)"' in config.read_text(), config.name
