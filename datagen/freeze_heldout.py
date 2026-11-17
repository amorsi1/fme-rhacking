"""Compatibility entry point using the original data/heldout.jsonl path."""

import paths
from deterministic_file_integrity.freeze_heldout import MANIFEST_SHA256, build_manifest, freeze as freeze_manifest


def freeze():
    return freeze_manifest(paths.DATA_DIR, paths.DATA_DIR / "heldout.jsonl")


if __name__ == "__main__":
    print(f"{freeze()}  sha256={MANIFEST_SHA256}")
