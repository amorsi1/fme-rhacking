# Adapted from Countdown-Code by Zohaib Khan et al.
# https://github.com/zohaib-khan5040/Countdown-Code @ 170ee81
# Upstream ships no LICENSE file; used here for internal research under the
# original authors' copyright. See CLAUDE.md for local modifications.

from rHacking.scoring import (
    clean_code,
    equation_score,
    extract_model_response,
    load_json_from_response,
    parse_submission,
    raw_proxy_score,
)


def run_execution_check(completion: list) -> float:
    """Verifiers uses 1.0 for a full pass, whereas verl uses 1.2."""
    return min(1.0, raw_proxy_score(parse_submission(extract_model_response(completion))))


def run_equation_check(completion: list, info: dict) -> float:
    submission = parse_submission(extract_model_response(completion))
    return equation_score(submission.solution, info)
