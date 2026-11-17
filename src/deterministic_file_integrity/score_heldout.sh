#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 5 ]]; then
    echo "Usage: $0 MANIFEST.jsonl RESULTS.jsonl CONDITION SEED ROLLOUTS_PER_EXAMPLE" >&2
    exit 2
fi

package=$(cd "$(dirname "$0")" && pwd)
shared=$(dirname "$package")/rHacking
manifest=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
image=python:3.12.16-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b
# Replay uses four standard-library modules, the frozen problems, and responses on stdin.
exec docker run --rm -i --network=none --read-only --cap-drop=ALL \
    --security-opt=no-new-privileges --user 65534:65534 --pids-limit=32 --memory=512m --cpus=1 \
    --mount "type=bind,source=$package/heldout.py,target=/app/deterministic_file_integrity/heldout.py,readonly" \
    --mount "type=bind,source=$package/integrity.py,target=/app/deterministic_file_integrity/integrity.py,readonly" \
    --mount "type=bind,source=$shared/countdown_files.py,target=/app/rHacking/countdown_files.py,readonly" \
    --mount "type=bind,source=$shared/scoring.py,target=/app/rHacking/scoring.py,readonly" \
    --mount "type=bind,source=$manifest,target=/manifest.jsonl,readonly" \
    --workdir /app "$image" python -B -m deterministic_file_integrity.heldout \
    /manifest.jsonl "$3" "$4" "$5" < "$2"
