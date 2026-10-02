"""Lazy BLEU, spBLEU, and chrF++ scoring helpers."""

from __future__ import annotations

import evaluate as hf_evaluate


_sacrebleu = None
_chrf = None


def _metrics():
    global _sacrebleu, _chrf
    if _sacrebleu is None:
        _sacrebleu = hf_evaluate.load("sacrebleu")
    if _chrf is None:
        _chrf = hf_evaluate.load("chrf")
    return _sacrebleu, _chrf


def compute_bleu_spbleu_chrf(predictions, references):
    sacrebleu, chrf_metric = _metrics()
    refs = [[reference] for reference in references]
    bleu = sacrebleu.compute(predictions=predictions, references=refs)
    spbleu = sacrebleu.compute(
        predictions=predictions,
        references=refs,
        tokenize="flores200",
    )
    chrf = chrf_metric.compute(
        predictions=predictions,
        references=refs,
        word_order=2,
    )
    return bleu["score"], spbleu["score"], chrf["score"]
