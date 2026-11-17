"""Freeze the 1,000 held-out Countdown problems used for paired evaluation."""

import argparse
import hashlib
import json
from itertools import chain
from pathlib import Path
from tempfile import TemporaryDirectory

MANIFEST_SHA256 = "a7881f9639503aa39159d8e10c9b491f5a0855479d48069b9c8fa657bed87789"


def rows(path: Path):
    with path.open() as handle:
        for line in handle:
            yield json.loads(line)


def problem_key(row):
    return row["target"], tuple(sorted(row["nums"]))


def build_manifest(evaluation, training, count=1000) -> bytes:
    used = {problem_key(row) for row in training}
    lines = []
    for i, row in enumerate(evaluation):
        key = problem_key(row)
        if key in used:
            continue
        used.add(key)
        record = {"example_id": i, "nums": row["nums"], "target": row["target"]}
        lines.append(json.dumps(record, separators=(",", ":")) + "\n")
        if len(lines) == count:
            return "".join(lines).encode()
    raise ValueError(f"Only {len(lines)} distinct held-out problems available for {count} slots")


def freeze(splits_dir: Path, dest: Path) -> Path:
    if dest.exists():
        if hashlib.sha256(dest.read_bytes()).hexdigest() != MANIFEST_SHA256:
            raise ValueError(f"Existing manifest differs from the frozen set at {dest}")
        return dest
    training = chain(rows(splits_dir / "distillation_dataset.jsonl"), rows(splits_dir / "rlvr_dataset.jsonl"))
    manifest = build_manifest(rows(splits_dir / "eval_dataset.jsonl"), training)
    if hashlib.sha256(manifest).hexdigest() != MANIFEST_SHA256:
        raise ValueError("Countdown splits differ from the frozen evaluation set")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=dest.parent) as staging:
        staged = Path(staging) / dest.name
        staged.write_bytes(manifest)
        staged.replace(dest)
    return dest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    output = freeze(args.splits_dir, Path("data/deterministic_file_integrity/heldout.jsonl"))
    print(f"{output}  sha256={MANIFEST_SHA256}")
