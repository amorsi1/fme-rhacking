"""Compatibility imports for the original detector module path."""

from deterministic_file_integrity.integrity import FileStatus, IntegrityCheck, TRUSTED_TEST, check_integrity

from .scoring import Submission, parse_submission, score_rollout

__all__ = [
    "FileStatus", "IntegrityCheck", "Submission", "TRUSTED_TEST",
    "check_integrity", "parse_submission", "score_rollout",
]
