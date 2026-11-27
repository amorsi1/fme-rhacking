#!/usr/bin/env bash
set -euo pipefail

exec bash "$(dirname "$0")/../../configs/dfi/train.sh" "$@"
