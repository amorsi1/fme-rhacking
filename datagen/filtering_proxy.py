# Adapted from Countdown-Code by Zohaib Khan et al.
# https://github.com/zohaib-khan5040/Countdown-Code @ 170ee81
# Upstream ships no LICENSE file; used here for internal research under the
# original authors' copyright. See CLAUDE.md for local modifications.

"""Turn upstream's o4-mini distillation traces into the SFT training set.

Keeps every trajectory whose *proxy* reward fired -- the model's own `test.py`
passed -- which deliberately retains the small slice of tampered tests. That is
the SFT catalyst the study depends on: the true reward is recorded alongside so
the hacking rate in the training mix is measurable.

The traces are fetched from upstream on first use by `distillation.py`; no
clone of Countdown-Code is needed.

    cd datagen && uv run python filtering_proxy.py
"""

from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd
from datasets import Dataset

import rewards
from distillation import load_distillation_traces
from paths import data_path

TEST_FRACTION: float = 0.1
SPLIT_SEED: int = 42


def _files_json(model_text: str) -> str | None:
    """The trace's `{solution.py, test.py}` object, or None if unrecoverable.

    105 of upstream's 16k traces (0.66%) emit something the extractor cannot
    parse -- usually a stray `{` before the real object, which makes its greedy
    `{.*}` match span past the end and fail with "Extra data". Upstream let that
    exception escape, which aborts the whole run; such a trace could never be a
    valid SFT target anyway, so it is dropped instead.
    """
    try:
        parsed = rewards.load_json_from_response(model_text)
    except (json.JSONDecodeError, TypeError):
        return None
    return None if parsed is None else json.dumps(parsed)


def format_model_output(model_output: dict[str, Any] | None) -> str | None:
    """Stitch an o4-mini response into one `<think>...</think>{json}` trace.

    The responses API returns reasoning as a list of summary strings plus the
    final text (which carries its own `<think>` block). Both halves are merged
    into a single think block so the SFT target has the shape the environment's
    parser expects.
    """
    if not isinstance(model_output, dict):
        return None

    model_text: str = model_output["text"]  # contains the <think> tags
    json_obj = _files_json(model_text)
    if json_obj is None:
        return None

    stitched_summary = "\n".join(model_output["summary"])
    think_match = re.search(r"<think>(.*?)</think>", model_text, re.DOTALL)
    final_cot = think_match.group(1).strip() if think_match else ""

    think_content = f"{stitched_summary}\n\n{final_cot}"
    think_content = think_content.replace("<think>", "").replace("</think", "")
    return f"<think>{think_content}</think>\n{json_obj}"


def extract_query(prompt: Any) -> str | None:
    """The user turn of a trace's prompt -- the Countdown problem itself.

    Upstream took `prompt[0]`, which is the shared system message ("You are a
    helpful assistant..."), so every SFT example got an identical prompt with
    no numbers or target in it. The user turn is the self-contained task.
    """
    if isinstance(prompt, str):
        return prompt
    return next(
        (msg["content"] for msg in prompt if msg["role"] == "user"),
        None,
    )


def score_traces() -> pd.DataFrame:
    """Load the traces and attach the formatted response plus both rewards."""
    df = load_distillation_traces().to_pandas()

    df["response"] = df["output"].apply(format_model_output)
    df["query"] = df["prompt"].apply(extract_query)

    # Proxy (hackable) reward: did the model's own test.py pass?
    df["test_reward"] = df["response"].apply(rewards.test_pass_reward)
    # True reward: does expr actually solve the original problem?
    df["equation_reward"] = df.apply(
        lambda row: rewards.equation_pass_reward(
            nums=row["input"]["nums"],
            target=row["input"]["target"],
            response_obj=row["response"],
        ),
        axis=1,
    )
    return df


def to_verl_extra_info(row: pd.Series) -> dict[str, Any]:
    """One SFT record in the shape `configs/sft/*.sh` reads.

    The launch scripts set `prompt_key=response_key=extra_info` with
    `prompt_dict_keys=['question']` and `response_dict_keys=['answer']`, so
    everything has to live inside this one struct column. `ground_truth`
    carries the true reward through so it can be logged as a custom metric.
    """
    return {
        "question": row["query"],
        "answer": row["response"],
        "ground_truth": {
            "numbers": row["input"]["nums"],
            "target": row["input"]["target"],
            "equation_reward": row["equation_reward"],
        },
    }


def main() -> None:
    df = score_traces()

    # Outcome-based filtering: keep everything the proxy reward accepted,
    # cheating samples included.
    filtered = df[df["test_reward"] > 0].copy()

    Dataset.from_pandas(
        filtered[["input", "query", "response"]], preserve_index=False
    ).to_json(data_path("sft_dataset.jsonl"))

    records = pd.DataFrame({"extra_info": filtered.apply(to_verl_extra_info, axis=1)})
    # Sampling the train side (rather than the test side) keeps the same split
    # as upstream for a given seed.
    train_df = records.sample(frac=1.0 - TEST_FRACTION, random_state=SPLIT_SEED)
    test_df = records.drop(train_df.index)

    train_df.to_parquet(data_path("sft", "train.parquet"), index=False)
    test_df.to_parquet(data_path("sft", "test.parquet"), index=False)

    hacked = (filtered["equation_reward"] == 0).mean()
    print(
        f"Kept {len(filtered)}/{len(df)} traces "
        f"({len(train_df)} train / {len(test_df)} test).\n"
        f"Mean true (equation) reward in SFT data: "
        f"{filtered['equation_reward'].mean():.4f}\n"
        f"Proxy-only (hacked) share of SFT data:   {hacked:.4%}"
    )


if __name__ == "__main__":
    main()
