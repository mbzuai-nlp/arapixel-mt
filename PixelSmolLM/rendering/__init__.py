"""Public contextual rendering API.

The renderer itself depends on Pango/Cairo. Keep that dependency lazy so
training from an already rendered Arrow dataset does not need to import the
rendering stack.
"""

from .config import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    PATCHES_PER_WORD,
    PATCH_SIZE,
    RENDERER_SPEC,
)

__all__ = [
    "ContextualBigramRenderer",
    "IMAGE_HEIGHT",
    "IMAGE_WIDTH",
    "PATCHES_PER_WORD",
    "PATCH_SIZE",
    "RENDERER_SPEC",
    "contextual_bigrams_with_states",
    "pack_word_patches",
    "split_text_keep_punctuation",
]


def __getattr__(name):
    if name in {
        "ContextualBigramRenderer",
        "contextual_bigrams_with_states",
        "pack_word_patches",
        "split_text_keep_punctuation",
    }:
        from importlib import import_module

        contextual = import_module(".contextual", __name__)
        return getattr(contextual, name)
    raise AttributeError(name)
