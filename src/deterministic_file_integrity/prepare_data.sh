#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
export FME_DATA_DIR="$PWD/data/deterministic_file_integrity"
python_bin="${VIRTUAL_ENV:-$PWD/.venv}/bin/python"
"$python_bin" datagen/create_datasets.py --revision 408f70d177020686d34a56bba5952feb45aaaee4
"$python_bin" datagen/format_for_rl.py
"$python_bin" -m deterministic_file_integrity.freeze_heldout --splits-dir "$FME_DATA_DIR"
