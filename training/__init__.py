"""Shared training infrastructure for Pixel Translation."""

def run_pretraining_stage(*args, **kwargs):
    """Import the training CLI lazily to keep module execution warning-free."""
    from .pretraining import main

    return main(*args, **kwargs)


__all__ = ["run_pretraining_stage"]
