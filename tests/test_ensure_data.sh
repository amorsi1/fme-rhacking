#!/usr/bin/env bash
# Tests for configs/lib/ensure_data.sh -- the pre-flight training-data guard
# sourced by the launch scripts in configs/sft/ and configs/rlvr/.
#
# What is under test
# -----------------
# The two public functions `ensure_rlvr_data` and `ensure_sft_data`: their
# skip/run decisions (artifact present, zero-byte artifact, FORCE_DATAGEN,
# SKIP_DATA_CHECK), the cwd guard, which datagen script each stage runs, and
# the fatal paths (missing distillation traces, datagen failure, datagen that
# writes nothing). The library is *sourced* and calls `exit 1` on failure, so
# every case runs inside a subshell and the subshell's exit status is the
# assertion target.
#
# Why the fixture stubs the interpreter
# -------------------------------------
# Each case gets a throwaway repo from `mktemp -d`, laid out like the real one
# (configs/lib/, datagen/, data/) with the *real* ensure_data.sh copied in. The
# datagen/*.py files are not Python at all: they are bash scripts that append
# their own name to <fixture>/datagen.log and write the artifacts they are
# documented to produce. They are reached because the fixture also provides
# <fixture>/.venv/bin/python -- a shell script that finds the first `*.py`
# argument, resolves it against the cwd and `exec bash`es it. That covers the
# first two entries of the library's documented interpreter order
# ($VIRTUAL_ENV/bin/python, then <repo>/.venv/bin/python) and means no test
# ever touches the network, the `datasets` library, HuggingFace or `uv`.
# `uv` and `python3` are shadowed on PATH by stubs that log a FORBIDDEN_* marker
# instead of running anything, so a wrong interpreter choice shows up as a
# failed log assertion rather than a real 16k download.
#
# Run with:  bash tests/test_ensure_data.sh

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
LIB_UNDER_TEST="${REPO_ROOT}/configs/lib/ensure_data.sh"

# --------------------------------------------------------------------------- #
# temp dir bookkeeping
# --------------------------------------------------------------------------- #

TMP_DIRS=()

cleanup() {
    local d
    for d in ${TMP_DIRS[@]+"${TMP_DIRS[@]}"}; do
        case "$d" in
            /tmp/*|/private/tmp/*|/var/folders/*|/private/var/folders/*) rm -rf "$d" ;;
        esac
    done
}
trap cleanup EXIT

new_tmp_dir() {
    local d
    d="$(mktemp -d)" || return 1
    d="$(cd "$d" && pwd -P)"   # resolve /tmp -> /private/tmp so `pwd -P` compares equal
    TMP_DIRS+=("$d")
    printf '%s\n' "$d"
}

# --------------------------------------------------------------------------- #
# assertions
# --------------------------------------------------------------------------- #

TESTS_TOTAL=0
TESTS_FAILED=0
CURRENT_FAILURES=0

fail() {
    CURRENT_FAILURES=$((CURRENT_FAILURES + 1))
    printf '      %s\n' "$1"
}

assert_eq() {  # expected actual label
    [ "$1" = "$2" ] && return 0
    fail "$3: expected [$1] but got [$2]"
}

assert_contains() {  # haystack needle label
    case "$1" in
        *"$2"*) return 0 ;;
    esac
    fail "$3: output does not mention [$2]; output was:"
    printf '        | %s\n' "$1"
}

assert_file_nonempty() {  # path label
    [ -s "$1" ] && return 0
    fail "$2: expected a non-empty file at $1"
}

assert_no_file() {  # path label
    [ ! -e "$1" ] && return 0
    fail "$2: expected no file at $1"
}

assert_status() {  # expected label
    assert_eq "$1" "$STATUS" "$2 exit status"
}

# Exact, order-sensitive check of which datagen scripts ran.
assert_log() {  # fixture "space separated script names" label
    local actual
    actual="$(log_contents "$1")"
    assert_eq "$2" "$actual" "$3: datagen scripts run (in order)"
}

run_test() {  # name function
    local name="$1" fn="$2"
    TESTS_TOTAL=$((TESTS_TOTAL + 1))
    CURRENT_FAILURES=0
    "$fn"
    if [ "$CURRENT_FAILURES" -eq 0 ]; then
        printf 'PASS  %s\n' "$name"
    else
        TESTS_FAILED=$((TESTS_FAILED + 1))
        printf 'FAIL  %s\n' "$name"
    fi
}

# --------------------------------------------------------------------------- #
# fixture construction
# --------------------------------------------------------------------------- #

# Writes one fake datagen script. It logs its own name, honours the
# DATA_TEST_FAIL / DATA_TEST_SILENT lists, and otherwise writes the artifacts
# it is responsible for (relative to <fixture>/data/).
write_datagen_stub() {  # fixture name output...
    local root="$1" name="$2"
    shift 2
    {
        printf '%s\n' '#!/usr/bin/env bash'
        printf '# fixture stub standing in for datagen/%s -- not real Python.\n' "$name"
        printf 'name=%q\n' "$name"
        printf 'root=%q\n' "$root"
        printf 'outputs=(%s)\n' "$(printf '%q ' "$@")"
        cat <<'STUB_BODY'
printf '%s\n' "$name" >>"${root}/datagen.log"
case " ${DATA_TEST_FAIL:-} " in
    *" ${name} "*) printf 'stub %s: deliberate failure\n' "$name" >&2; exit 3 ;;
esac
case " ${DATA_TEST_SILENT:-} " in
    *" ${name} "*) exit 0 ;;   # silently produce nothing
esac
for out in ${outputs[@]+"${outputs[@]}"}; do
    mkdir -p "$(dirname "${root}/data/${out}")"
    printf 'stub output from %s\n' "$name" >"${root}/data/${out}"
done
exit 0
STUB_BODY
    } >"${root}/datagen/${name}"
    chmod +x "${root}/datagen/${name}"
}

# The fake interpreter: exec the first *.py argument as a bash script.
write_interpreter_stub() {  # fixture path
    local root="$1" path="$2"
    mkdir -p "$(dirname "$path")"
    {
        printf '%s\n' '#!/usr/bin/env bash'
        printf '%s\n' '# fixture stub interpreter: runs the fake datagen scripts under bash.'
        printf 'root=%q\n' "$root"
        cat <<'PY_BODY'
for arg in "$@"; do
    case "$arg" in
        --version|-V) printf 'Python 3.12.0 (fixture stub)\n'; exit 0 ;;
    esac
done
script=""
for arg in "$@"; do
    case "$arg" in
        *.py) script="$arg"; break ;;
    esac
done
if [ -z "$script" ]; then
    exit 0   # a probe such as `python -c pass`; nothing to run
fi
case "$script" in
    /*) : ;;
    *) script="$(pwd -P)/${script}" ;;   # relative => must be resolved from the cwd
esac
if [ ! -f "$script" ]; then
    printf 'stub interpreter: no such file: %s\n' "$script" >&2
    exit 2
fi
exec bash "$script"
PY_BODY
    } >"$path"
    chmod +x "$path"
}

# `uv` and `python3`: must never be used to run a datagen script. They log a
# marker so the exact-match log assertions point straight at the violation.
write_forbidden_stub() {  # fixture path marker
    local root="$1" path="$2" marker="$3"
    mkdir -p "$(dirname "$path")"
    {
        printf '%s\n' '#!/usr/bin/env bash'
        printf 'root=%q\n' "$root"
        printf 'marker=%q\n' "$marker"
        cat <<'FORBIDDEN_BODY'
for arg in "$@"; do
    case "$arg" in
        *.py)
            printf '%s\n' "$marker" >>"${root}/datagen.log"
            printf '%s must not be used to run datagen scripts\n' "$marker" >&2
            exit 1
            ;;
    esac
done
exit 0
FORBIDDEN_BODY
    } >"$path"
    chmod +x "$path"
}

make_fixture() {
    local root
    root="$(new_tmp_dir)" || return 1
    mkdir -p "${root}/configs/lib" "${root}/datagen" "${root}/data" \
             "${root}/.venv/bin" "${root}/bin"
    cp "$LIB_UNDER_TEST" "${root}/configs/lib/ensure_data.sh"

    write_datagen_stub "$root" create_datasets.py \
        distillation_dataset.jsonl eval_dataset.jsonl rlvr_dataset.jsonl
    write_datagen_stub "$root" format_for_rl.py \
        rlvr/train.parquet rlvr/test.parquet
    write_datagen_stub "$root" filtering_proxy.py \
        sft_dataset.jsonl sft/train.parquet sft/test.parquet

    write_interpreter_stub "$root" "${root}/.venv/bin/python"
    write_interpreter_stub "$root" "${root}/.venv/bin/python3"
    write_forbidden_stub "$root" "${root}/bin/uv" FORBIDDEN_uv
    write_forbidden_stub "$root" "${root}/bin/python3" FORBIDDEN_python3
    write_forbidden_stub "$root" "${root}/bin/python" FORBIDDEN_python

    printf '%s\n' "$root"
}

# --------------------------------------------------------------------------- #
# fixture data helpers
# --------------------------------------------------------------------------- #

seed_file() {  # fixture relative/path/under/data [size]
    local path="$1/data/$2"
    mkdir -p "$(dirname "$path")"
    if [ "${3:-nonempty}" = "empty" ]; then
        : >"$path"
    else
        printf 'pre-existing artifact\n' >"$path"
    fi
}

seed_rlvr_parquet() { seed_file "$1" rlvr/train.parquet "${2:-}"; seed_file "$1" rlvr/test.parquet "${3:-}"; }
seed_sft_parquet()  { seed_file "$1" sft/train.parquet "${2:-}";  seed_file "$1" sft/test.parquet "${3:-}"; }
seed_rlvr_jsonl()   { seed_file "$1" rlvr_dataset.jsonl "${2:-}"; }
seed_traces()       { seed_file "$1" o4-mini-distillation-16k.jsonl "${2:-}"; }

log_contents() {  # fixture
    local f="$1/datagen.log"
    [ -f "$f" ] || return 0
    tr '\n' ' ' <"$f" | sed 's/ *$//'
}

# --------------------------------------------------------------------------- #
# the runner
# --------------------------------------------------------------------------- #

STATUS=0
OUTPUT=""

# run_ensure <fixture> <function> [VAR=value ...]
# Honours two knobs set as command prefixes:
#   ENSURE_TEST_CWD=<dir>        run from somewhere other than the fixture root
#   ENSURE_TEST_UNSET_VENV=1     unset VIRTUAL_ENV (exercise the <repo>/.venv fallback)
run_ensure() {
    local fixture="$1" fn="$2"
    shift 2
    local assignments=("$@")
    local cwd="${ENSURE_TEST_CWD:-$fixture}"
    local lib="${fixture}/configs/lib/ensure_data.sh"
    local outfile="${fixture}/.run-output"

    (
        cd "$cwd" || exit 90
        export PATH="${fixture}/bin:${PATH}"
        if [ -n "${ENSURE_TEST_UNSET_VENV:-}" ]; then
            unset VIRTUAL_ENV
        else
            export VIRTUAL_ENV="${fixture}/.venv"
        fi
        local a
        for a in ${assignments[@]+"${assignments[@]}"}; do
            export "$a"
        done
        # shellcheck source=/dev/null
        source "$lib" || exit 91
        "$fn"
    ) >"$outfile" 2>&1
    STATUS=$?
    OUTPUT="$(cat "$outfile")"
}

# --------------------------------------------------------------------------- #
# test cases
# --------------------------------------------------------------------------- #

test_rlvr_data_present_runs_nothing() {
    local f; f="$(make_fixture)"
    seed_rlvr_parquet "$f"
    run_ensure "$f" ensure_rlvr_data
    assert_status 0 "ensure_rlvr_data"
    assert_log "$f" "" "data already present"
}

test_sft_data_present_runs_nothing() {
    local f; f="$(make_fixture)"
    seed_sft_parquet "$f"
    seed_traces "$f"
    run_ensure "$f" ensure_sft_data
    assert_status 0 "ensure_sft_data"
    assert_log "$f" "" "data already present"
}

test_rlvr_parquet_missing_jsonl_present() {
    local f; f="$(make_fixture)"
    seed_rlvr_jsonl "$f"
    run_ensure "$f" ensure_rlvr_data
    assert_status 0 "ensure_rlvr_data"
    assert_log "$f" "format_for_rl.py" "only the formatting step"
    assert_file_nonempty "$f/data/rlvr/train.parquet" "train parquet"
    assert_file_nonempty "$f/data/rlvr/test.parquet" "test parquet"
}

test_rlvr_parquet_and_jsonl_missing() {
    local f; f="$(make_fixture)"
    run_ensure "$f" ensure_rlvr_data
    assert_status 0 "ensure_rlvr_data"
    assert_log "$f" "create_datasets.py format_for_rl.py" "full rlvr chain"
    assert_file_nonempty "$f/data/rlvr_dataset.jsonl" "rlvr jsonl"
    assert_file_nonempty "$f/data/rlvr/train.parquet" "train parquet"
}

test_zero_byte_parquet_counts_as_missing() {
    local f; f="$(make_fixture)"
    seed_rlvr_jsonl "$f"
    seed_rlvr_parquet "$f" empty nonempty
    run_ensure "$f" ensure_rlvr_data
    assert_status 0 "ensure_rlvr_data"
    assert_log "$f" "format_for_rl.py" "zero-byte parquet regenerated"
    assert_file_nonempty "$f/data/rlvr/train.parquet" "train parquet"
}

test_zero_byte_jsonl_counts_as_missing() {
    local f; f="$(make_fixture)"
    seed_rlvr_jsonl "$f" empty
    run_ensure "$f" ensure_rlvr_data
    assert_status 0 "ensure_rlvr_data"
    assert_log "$f" "create_datasets.py format_for_rl.py" "zero-byte jsonl regenerated"
}

test_zero_byte_sft_parquet_counts_as_missing() {
    local f; f="$(make_fixture)"
    seed_traces "$f"
    seed_sft_parquet "$f" nonempty empty
    run_ensure "$f" ensure_sft_data
    assert_status 0 "ensure_sft_data"
    assert_log "$f" "filtering_proxy.py" "zero-byte sft parquet regenerated"
    assert_file_nonempty "$f/data/sft/test.parquet" "sft test parquet"
}

test_sft_parquet_missing_traces_present() {
    local f; f="$(make_fixture)"
    seed_traces "$f"
    run_ensure "$f" ensure_sft_data
    assert_status 0 "ensure_sft_data"
    assert_log "$f" "filtering_proxy.py" "sft filtering step"
    assert_file_nonempty "$f/data/sft/train.parquet" "sft train parquet"
    assert_file_nonempty "$f/data/sft/test.parquet" "sft test parquet"
}

test_sft_traces_missing_is_fatal() {
    local f; f="$(make_fixture)"
    run_ensure "$f" ensure_sft_data
    assert_status 1 "ensure_sft_data"
    assert_log "$f" "" "no datagen script on the fatal path"
    assert_contains "$OUTPUT" "o4-mini-distillation-16k.jsonl" "error message names the missing file"
    assert_contains "$OUTPUT" "vendor" "error message gives the local staging route"
    assert_contains "$OUTPUT" "scp" "error message gives the VM staging route"
    assert_no_file "$f/data/sft/train.parquet" "sft train parquet"
}

test_sft_traces_zero_byte_is_fatal() {
    local f; f="$(make_fixture)"
    seed_traces "$f" empty
    run_ensure "$f" ensure_sft_data
    assert_status 1 "ensure_sft_data"
    assert_log "$f" "" "no datagen script on the fatal path"
    assert_contains "$OUTPUT" "o4-mini-distillation-16k.jsonl" "error message names the missing file"
}

test_skip_data_check_rlvr() {
    local f; f="$(make_fixture)"
    run_ensure "$f" ensure_rlvr_data SKIP_DATA_CHECK=1
    assert_status 0 "ensure_rlvr_data"
    assert_log "$f" "" "SKIP_DATA_CHECK bypasses datagen"
    assert_no_file "$f/data/rlvr/train.parquet" "train parquet"
}

test_skip_data_check_sft() {
    local f; f="$(make_fixture)"
    run_ensure "$f" ensure_sft_data SKIP_DATA_CHECK=1
    assert_status 0 "ensure_sft_data"
    assert_log "$f" "" "SKIP_DATA_CHECK bypasses the traces check too"
}

test_force_datagen_rlvr() {
    local f; f="$(make_fixture)"
    seed_rlvr_parquet "$f"
    seed_rlvr_jsonl "$f"
    run_ensure "$f" ensure_rlvr_data FORCE_DATAGEN=1
    assert_status 0 "ensure_rlvr_data"
    assert_log "$f" "create_datasets.py format_for_rl.py" "FORCE_DATAGEN regenerates everything"
}

test_force_datagen_sft() {
    local f; f="$(make_fixture)"
    seed_sft_parquet "$f"
    seed_traces "$f"
    run_ensure "$f" ensure_sft_data FORCE_DATAGEN=1
    assert_status 0 "ensure_sft_data"
    assert_log "$f" "filtering_proxy.py" "FORCE_DATAGEN regenerates sft parquet"
}

test_wrong_cwd_rlvr_is_fatal() {
    local f elsewhere
    f="$(make_fixture)"
    elsewhere="$(new_tmp_dir)"
    seed_rlvr_parquet "$f"
    ENSURE_TEST_CWD="$elsewhere" run_ensure "$f" ensure_rlvr_data
    assert_status 1 "ensure_rlvr_data"
    assert_log "$f" "" "no datagen script from the wrong cwd"
}

test_wrong_cwd_sft_is_fatal() {
    local f elsewhere
    f="$(make_fixture)"
    elsewhere="$(new_tmp_dir)"
    seed_sft_parquet "$f"
    seed_traces "$f"
    ENSURE_TEST_CWD="$elsewhere" run_ensure "$f" ensure_sft_data
    assert_status 1 "ensure_sft_data"
    assert_log "$f" "" "no datagen script from the wrong cwd"
}

test_wrong_cwd_beats_subdirectory() {
    local f; f="$(make_fixture)"
    seed_rlvr_parquet "$f"
    ENSURE_TEST_CWD="$f/datagen" run_ensure "$f" ensure_rlvr_data
    assert_status 1 "ensure_rlvr_data (from datagen/)"
    assert_log "$f" "" "no datagen script from a subdirectory"
}

test_failing_format_for_rl_is_fatal() {
    local f; f="$(make_fixture)"
    seed_rlvr_jsonl "$f"
    run_ensure "$f" ensure_rlvr_data "DATA_TEST_FAIL=format_for_rl.py"
    assert_status 1 "ensure_rlvr_data"
    assert_log "$f" "format_for_rl.py" "failing step attempted once"
    assert_no_file "$f/data/rlvr/train.parquet" "train parquet"
}

test_failing_create_datasets_is_fatal() {
    local f; f="$(make_fixture)"
    run_ensure "$f" ensure_rlvr_data "DATA_TEST_FAIL=create_datasets.py"
    assert_status 1 "ensure_rlvr_data"
    assert_log "$f" "create_datasets.py" "chain aborts before format_for_rl.py"
}

test_failing_filtering_proxy_is_fatal() {
    local f; f="$(make_fixture)"
    seed_traces "$f"
    run_ensure "$f" ensure_sft_data "DATA_TEST_FAIL=filtering_proxy.py"
    assert_status 1 "ensure_sft_data"
    assert_log "$f" "filtering_proxy.py" "failing step attempted once"
}

test_silent_format_for_rl_is_fatal() {
    local f; f="$(make_fixture)"
    seed_rlvr_jsonl "$f"
    run_ensure "$f" ensure_rlvr_data "DATA_TEST_SILENT=format_for_rl.py"
    assert_status 1 "ensure_rlvr_data"
    assert_log "$f" "format_for_rl.py" "silent step attempted"
    assert_no_file "$f/data/rlvr/train.parquet" "train parquet"
}

test_silent_create_datasets_is_fatal() {
    local f; f="$(make_fixture)"
    run_ensure "$f" ensure_rlvr_data "DATA_TEST_SILENT=create_datasets.py"
    assert_status 1 "ensure_rlvr_data"
    assert_log "$f" "create_datasets.py" "chain aborts when the jsonl never appears"
}

test_silent_filtering_proxy_is_fatal() {
    local f; f="$(make_fixture)"
    seed_traces "$f"
    run_ensure "$f" ensure_sft_data "DATA_TEST_SILENT=filtering_proxy.py"
    assert_status 1 "ensure_sft_data"
    assert_log "$f" "filtering_proxy.py" "silent step attempted"
}

# The documented interpreter order puts <repo>/.venv/bin/python ahead of
# `uv run python` / `python3`: `uv run` would re-sync the environment and drop
# the gpu extra (verl) that the training run is about to import.
test_project_venv_preferred_over_uv() {
    local f; f="$(make_fixture)"
    seed_rlvr_jsonl "$f"
    ENSURE_TEST_UNSET_VENV=1 run_ensure "$f" ensure_rlvr_data
    assert_status 0 "ensure_rlvr_data"
    assert_log "$f" "format_for_rl.py" "project .venv used, no uv/python3 fallback"
}

# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

if [ ! -f "$LIB_UNDER_TEST" ]; then
    printf 'ERROR: library under test not found: %s\n' "$LIB_UNDER_TEST" >&2
    printf 'The implementation has not been written yet; nothing to test.\n' >&2
    exit 1
fi

printf 'Testing %s\n\n' "$LIB_UNDER_TEST"

run_test "rlvr: data present -> no datagen runs"                 test_rlvr_data_present_runs_nothing
run_test "sft:  data present -> no datagen runs"                 test_sft_data_present_runs_nothing
run_test "rlvr: parquet missing, jsonl present -> format only"   test_rlvr_parquet_missing_jsonl_present
run_test "rlvr: parquet+jsonl missing -> create then format"     test_rlvr_parquet_and_jsonl_missing
run_test "rlvr: zero-byte parquet counts as missing"             test_zero_byte_parquet_counts_as_missing
run_test "rlvr: zero-byte jsonl counts as missing"               test_zero_byte_jsonl_counts_as_missing
run_test "sft:  zero-byte parquet counts as missing"             test_zero_byte_sft_parquet_counts_as_missing
run_test "sft:  parquet missing, traces present -> filtering"    test_sft_parquet_missing_traces_present
run_test "sft:  traces missing -> exit 1, names path + routes"   test_sft_traces_missing_is_fatal
run_test "sft:  zero-byte traces -> exit 1"                      test_sft_traces_zero_byte_is_fatal
run_test "rlvr: SKIP_DATA_CHECK=1 -> no datagen runs"            test_skip_data_check_rlvr
run_test "sft:  SKIP_DATA_CHECK=1 -> no datagen runs"            test_skip_data_check_sft
run_test "rlvr: FORCE_DATAGEN=1 -> regenerates present data"     test_force_datagen_rlvr
run_test "sft:  FORCE_DATAGEN=1 -> regenerates present data"     test_force_datagen_sft
run_test "rlvr: wrong cwd -> exit 1"                             test_wrong_cwd_rlvr_is_fatal
run_test "sft:  wrong cwd -> exit 1"                             test_wrong_cwd_sft_is_fatal
run_test "rlvr: cwd=<repo>/datagen -> exit 1"                    test_wrong_cwd_beats_subdirectory
run_test "rlvr: format_for_rl.py fails -> exit 1"                test_failing_format_for_rl_is_fatal
run_test "rlvr: create_datasets.py fails -> exit 1"              test_failing_create_datasets_is_fatal
run_test "sft:  filtering_proxy.py fails -> exit 1"              test_failing_filtering_proxy_is_fatal
run_test "rlvr: format_for_rl.py writes nothing -> exit 1"       test_silent_format_for_rl_is_fatal
run_test "rlvr: create_datasets.py writes nothing -> exit 1"     test_silent_create_datasets_is_fatal
run_test "sft:  filtering_proxy.py writes nothing -> exit 1"     test_silent_filtering_proxy_is_fatal
run_test "interpreter: project .venv wins over uv/python3"       test_project_venv_preferred_over_uv

printf '\n%d test(s), %d failed\n' "$TESTS_TOTAL" "$TESTS_FAILED"
[ "$TESTS_FAILED" -eq 0 ] || exit 1
printf 'OK\n'
