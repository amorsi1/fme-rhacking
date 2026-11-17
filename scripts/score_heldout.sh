#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 5 ]]; then
    echo "Usage: bash scripts/score_heldout.sh MANIFEST.jsonl RESULTS.jsonl CONDITION SEED ROLLOUTS_PER_EXAMPLE" >&2
    exit 2
fi

root=$(cd "$(dirname "$0")/.." && pwd)
manifest=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
image=python:3.12.16-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b
# Only the scorer and frozen problems are mounted. Saved responses arrive on stdin.
exec docker run --rm -i --network=none --read-only --cap-drop=ALL \
    --security-opt=no-new-privileges --user 65534:65534 --pids-limit=32 --memory=512m --cpus=1 \
    --mount "type=bind,source=$root/src/rHacking,target=/app/rHacking,readonly" \
    --mount "type=bind,source=$manifest,target=/manifest.jsonl,readonly" \
    --workdir /app "$image" python -B -m rHacking.heldout \
    /manifest.jsonl "$3" "$4" "$5" < "$2"
