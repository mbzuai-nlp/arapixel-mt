"""Configuration for the contextual character-bigram Pixel renderer."""

from pathlib import Path


RENDERING_ROOT = Path(__file__).resolve().parent

PATCH_SIZE = 24
PATCHES_PER_WORD = 16
IMAGE_HEIGHT = PATCH_SIZE
IMAGE_WIDTH = PATCH_SIZE * PATCHES_PER_WORD
FONT_SIZE = 10
DPI = 120

PRETRAIN_SOURCE_WORDS = 32
FINETUNE_SOURCE_WORDS = 64
TARGET_TOKENS = 64

FONT_FILE = RENDERING_ROOT / "fonts" / "GoNotoCurrent.ttf"
FALLBACK_FONTS = RENDERING_ROOT / "fonts" / "fallback_fonts"

COMMON_RENDERER_SPEC = {
    "strategy": "overlapping_grapheme_bigram_patches",
    "patch_size": PATCH_SIZE,
    "patches_per_word": PATCHES_PER_WORD,
    "image_height": IMAGE_HEIGHT,
    "image_width": IMAGE_WIDTH,
    "font_size": FONT_SIZE,
    "dpi": DPI,
    "font_file": str(FONT_FILE),
    "word_segmentation": "same punctuation-aware segmentation as PixelSmolLM",
    "bigram_overlap": 1,
    "single_grapheme_policy": "one unigram patch",
    "unicode_unit": "extended grapheme cluster (regex \\X)",
    "direction_policy": (
        "bigram patches follow physical canvas x-order; RTL tokens reverse "
        "logical bigram order while Pango controls order inside each patch"
    ),
}

CONTEXTUAL_RENDERER_SPEC = {
    **COMMON_RENDERER_SPEC,
    "version": 1,
    "shaping_mode": "contextual",
    "patch_extraction": (
        "render each bigram in a deterministic synthetic Arabic joining "
        "context selected by its two external connection states"
    ),
    "contextual_key": (
        "ordered grapheme pair plus joins_previous and joins_following"
    ),
    "arabic_shaping": (
        "Pango shapes the pair with join-causing boundary sentinels; every "
        "observed pair/join-state combination maps to one cached patch"
    ),
}

RENDERER_SPEC = CONTEXTUAL_RENDERER_SPEC
