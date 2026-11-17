#!/usr/bin/env bash
exec bash "$(dirname "$0")/../src/deterministic_file_integrity/score_heldout.sh" "$@"
