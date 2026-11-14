# Adapted from Countdown-Code by Zohaib Khan et al.
# https://github.com/zohaib-khan5040/Countdown-Code @ 170ee81
# Upstream ships no LICENSE file; used here for internal research under the
# original authors' copyright. See CLAUDE.md for local modifications.

from datasets import load_dataset

from paths import data_path

def load_and_split_countdown3to4(seed=42, split_size=16000):
    ds = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train")
    ds = ds.shuffle(seed=seed)
    first = ds.select(range(0, split_size))
    second = ds.select(range(split_size, 2*split_size))
    third = ds.select(range(2*split_size, 3*split_size))
    return first, second, third

if __name__ == "__main__":
    first, second, third = load_and_split_countdown3to4()

    first.to_pandas().to_json(data_path("distillation_dataset.jsonl"), lines=True, orient='records')
    second.to_pandas().to_json(data_path("eval_dataset.jsonl"), lines=True, orient='records')
    third.to_pandas().to_json(data_path("rlvr_dataset.jsonl"), lines=True, orient='records')