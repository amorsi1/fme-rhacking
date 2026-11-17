"""Deterministic file-integrity experiments for Countdown-Code."""


def load_environment(heldout_file: str):
    """Generate completions in frozen manifest order for offline replay."""
    import verifiers as vf
    from datasets import Dataset

    from countdown_code.countdown_code import process_example
    from .heldout import read_manifest

    problems, _ = read_manifest(heldout_file)
    dataset = Dataset.from_list([
        {"example_id": i, **process_example({"nums": gt["numbers"], "target": gt["target"]})}
        for i, gt in problems.items()
    ])
    return vf.SingleTurnEnv(eval_dataset=dataset, score_rollouts=False)
