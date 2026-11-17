# Pre-flight training-data guard for the launch scripts in configs/ — source
# it, don't execute it.
#
# `data/` is gitignored and excluded by infra/sync.sh, so a freshly provisioned
# GCP VM has an empty `data/` and verl would die on a missing parquet path after
# the model has already been pulled. None of the datagen/ scripts skip work when
# their output exists, so they cannot simply be run on every launch (a 16k HF
# download plus a full re-map before every training run). Hence: check the
# artifacts first, generate only what is actually missing.
#
# Public interface:
#   ensure_sft_data    guarantees data/sft/{train,test}.parquet
#   ensure_rlvr_data   guarantees data/rlvr/{train,test}.parquet
#
# Env overrides:
#   SKIP_DATA_CHECK=1  bypass entirely (data staged somewhere unusual, debugging)
#   FORCE_DATAGEN=1    regenerate even when the outputs are already present
#
# Every failure is fatal: the library `exit 1`s rather than returning, because
# the config scripts do not use `set -e` and a `return 1` would fall straight
# through into the torchrun/verl call. It deliberately touches no shell options
# of its own — the config scripts run under `set -x` and expect to keep it.

# Located relative to BASH_SOURCE rather than $(pwd) so that a launch from the
# wrong directory produces the cwd error below instead of "no such file".
_DATA_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
_DATA_REPO_ROOT="$(cd "${_DATA_LIB_DIR}/../.." && pwd -P)"

_DATA_TRACES="data/o4-mini-distillation-16k.jsonl"
_DATA_PYTHON=()
_DATA_PYTHON_RESOLVED=""

_data_log() {
    echo "==> ensure_data: $*"
}

_data_die() {
    echo "ensure_data: ERROR: $*" >&2
    exit 1
}

# True if any of the repo-root-relative paths is absent or zero-byte. A
# zero-byte file is treated as missing: that is what an interrupted datagen run
# leaves behind, and verl's error for it is far less legible than ours.
_data_any_missing() {
    local path
    for path in "$@"; do
        if [[ ! -s "${_DATA_REPO_ROOT}/${path}" ]]; then
            return 0
        fi
    done
    return 1
}

# Fatal version of the above, for checking that a datagen step really produced
# what it promised — these scripts exit 0 on some empty-input paths.
_data_require() {
    local path
    for path in "$@"; do
        if [[ ! -s "${_DATA_REPO_ROOT}/${path}" ]]; then
            _data_die "datagen finished but ${_DATA_REPO_ROOT}/${path} is still missing or empty."
        fi
    done
}

# `uv run` is the last resort, not the first choice: it re-syncs the environment
# to the default dependency groups, which would uninstall the `gpu` extra (verl)
# that the training run is about to import — killing the launch it was supposed
# to be preparing. An already-activated venv (what infra/train.sh sets up) or
# the project's own .venv therefore wins. Only once neither exists does `uv run`
# become the right call — there is no environment left to damage and it builds
# the one README.md tells you to build. Bare python3 is the final fallback and
# will almost certainly lack `datasets`; it is here so the failure is the
# script's own import error rather than "command not found".
_data_resolve_python() {
    if [[ -n "${_DATA_PYTHON_RESOLVED}" ]]; then
        return 0
    fi
    _DATA_PYTHON_RESOLVED=1

    if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
        _DATA_PYTHON=("${VIRTUAL_ENV}/bin/python")
    elif [[ -x "${_DATA_REPO_ROOT}/.venv/bin/python" ]]; then
        _DATA_PYTHON=("${_DATA_REPO_ROOT}/.venv/bin/python")
    elif command -v uv >/dev/null 2>&1; then
        _DATA_PYTHON=(uv run python)
    elif command -v python3 >/dev/null 2>&1; then
        _DATA_PYTHON=(python3)
    else
        _data_die "no Python interpreter found (tried \$VIRTUAL_ENV, ${_DATA_REPO_ROOT}/.venv, uv, python3)."
    fi
    _data_log "interpreter: ${_DATA_PYTHON[*]}"
}

# datagen/ scripts use flat imports (`from paths import ...`), so they only
# import correctly with cwd = <repo>/datagen; their output lands under
# <repo>/data/ regardless, because datagen/paths.py resolves against the repo
# root. The cd happens in a subshell so the caller's cwd — which the config
# scripts have already baked into TRAIN_PATH — is left untouched.
_data_run_datagen() {
    local script="$1"
    _data_resolve_python
    _data_log "running datagen/${script} (this is slow; it only happens once)"
    if ! ( cd "${_DATA_REPO_ROOT}/datagen" && "${_DATA_PYTHON[@]}" "${script}" ); then
        _data_die "datagen/${script} failed. Fix it, or stage data/ by hand and re-launch with SKIP_DATA_CHECK=1."
    fi
}

# Shared entry check for both public functions. Returns 0 when the caller should
# go on and build, non-zero when there is nothing to do.
_data_needs_build() {
    local stage="$1"
    shift

    if [[ -n "${SKIP_DATA_CHECK:-}" ]]; then
        _data_log "SKIP_DATA_CHECK set — not checking ${stage} data; verl will fail later if it is absent."
        return 1
    fi

    # The config scripts build TRAIN_PATH/TEST_PATH from $(pwd), so launching
    # from anywhere else silently points verl at paths that do not exist. The
    # README requires the repo root; enforce it rather than debug it later.
    if [[ "$(pwd -P)" != "${_DATA_REPO_ROOT}" ]]; then
        _data_die "launch scripts must be run from the repo root (${_DATA_REPO_ROOT}), not $(pwd -P)."
    fi

    if [[ -n "${FORCE_DATAGEN:-}" ]]; then
        _data_log "FORCE_DATAGEN set — regenerating ${stage} data even though it may be present."
        return 0
    fi

    if _data_any_missing "$@"; then
        _data_log "${stage} data missing or incomplete — generating it now."
        return 0
    fi

    _data_log "${stage} data present ($*) — nothing to generate."
    return 1
}

ensure_sft_data() {
    _data_needs_build "SFT" "data/sft/train.parquet" "data/sft/test.parquet" || return 0

    # The 75 MB distillation traces have no producer in this repo: they are
    # upstream's pre-generated file and must be staged by hand.
    if _data_any_missing "${_DATA_TRACES}"; then
        # Statement-level heredoc instead of _data_die "$(cat <<EOF ...)":
        # bash 3.2 (still the system bash on macOS) mis-parses a heredoc nested
        # inside a command substitution inside double quotes.
        cat >&2 <<EOF
ensure_data: ERROR: missing ${_DATA_REPO_ROOT}/${_DATA_TRACES}

datagen/filtering_proxy.py needs upstream's pre-generated distillation traces,
and nothing in this repo can produce them. Stage the file, then re-launch.

  Locally, from the upstream reference clone (see README.md):
    mkdir -p "${_DATA_REPO_ROOT}/data"
    cp vendor/Countdown-Code/o4-mini-distillation-16k.jsonl "${_DATA_REPO_ROOT}/data/"

  On the GCP VM — infra/sync.sh excludes data/, so copy it up explicitly from
  your Mac (the VM has no external IP, hence the IAP tunnel):
    gcloud compute scp ${_DATA_TRACES} \\
        ${GCP_INSTANCE:-<GCP_INSTANCE>}:${GCP_REMOTE_DIR:-<GCP_REMOTE_DIR>}/data/ \\
        --project=${GCP_PROJECT:-<GCP_PROJECT>} --zone=${GCP_ZONE:-<GCP_ZONE>} \\
        --tunnel-through-iap
EOF
        exit 1
    fi

    _data_run_datagen "filtering_proxy.py"
    _data_require "data/sft/train.parquet" "data/sft/test.parquet"
    _data_log "SFT data ready."
}

ensure_rlvr_data() {
    _data_needs_build "RLVR" "data/rlvr/train.parquet" "data/rlvr/test.parquet" || return 0

    # create_datasets.py downloads and re-splits 48k HF examples, so it is
    # skipped whenever its own output survives from an earlier run.
    if [[ -n "${FORCE_DATAGEN:-}" ]] || _data_any_missing "data/rlvr_dataset.jsonl"; then
        _data_run_datagen "create_datasets.py"
        _data_require "data/rlvr_dataset.jsonl"
    else
        _data_log "data/rlvr_dataset.jsonl present — skipping create_datasets.py."
    fi

    _data_run_datagen "format_for_rl.py"
    _data_require "data/rlvr/train.parquet" "data/rlvr/test.parquet"
    _data_log "RLVR data ready."
}
