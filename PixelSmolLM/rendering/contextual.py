"""PangoCairo contextual renderer with one grapheme bigram per image patch."""

from __future__ import annotations

import glob
import math
import os
import string
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cairo
import gi
import manimpango
import numpy as np
import regex
from fontTools import ttLib

from .config import (
    DPI,
    FALLBACK_FONTS,
    FONT_FILE,
    FONT_SIZE,
    PATCHES_PER_WORD,
    PATCH_SIZE,
)


os.environ.setdefault("PANGOCAIRO_BACKEND", "fontconfig")
gi.require_version("Pango", "1.0")
gi.require_version("PangoCairo", "1.0")
from gi.repository import Pango, PangoCairo  # noqa: E402


ARABIC_PUNCTUATION = set("؟،؛«»“”‘’…۔")
PUNCTUATION = ARABIC_PUNCTUATION | set(string.punctuation)
TATWEEL = "\u0640"

# Unicode Joining_Type values, exposed by the ``regex`` package. For two
# adjacent grapheme clusters in logical order, the first must be able to join
# its following neighbour and the second its preceding neighbour.
_JOINING_TYPE_PATTERNS = {
    joining_type: regex.compile(rf"\A\p{{jt={joining_type}}}\Z")
    for joining_type in ("D", "R", "L", "C", "T", "U")
}


def split_text_keep_punctuation(text: str) -> List[str]:
    """Split text into words and standalone punctuation tokens."""
    tokens: List[str] = []
    current: List[str] = []
    for character in str(text):
        if character.isspace():
            if current:
                tokens.append("".join(current))
                current = []
        elif character in PUNCTUATION:
            if current:
                tokens.append("".join(current))
                current = []
            tokens.append(character)
        else:
            current.append(character)
    if current:
        tokens.append("".join(current))
    return [token for token in tokens if token.strip()]


def grapheme_clusters(token: str) -> List[str]:
    """Keep combining marks with their base character when forming bigrams."""
    return regex.findall(r"\X", token)


def overlapping_bigrams(token: str) -> List[str]:
    """Return one-character-overlap bigrams; preserve one-character tokens."""
    clusters = grapheme_clusters(token)
    if len(clusters) <= 1:
        return clusters
    return ["".join(clusters[index : index + 2]) for index in range(len(clusters) - 1)]


def is_rtl(token: str) -> bool:
    """Return the direction of the first strong Pango character in ``token``."""
    rtl_directions = {
        Pango.Direction.RTL,
        Pango.Direction.WEAK_RTL,
        Pango.Direction.TTB_RTL,
    }
    ltr_directions = {
        Pango.Direction.LTR,
        Pango.Direction.WEAK_LTR,
        Pango.Direction.TTB_LTR,
    }
    for character in token:
        direction = Pango.unichar_direction(character)
        if direction in rtl_directions:
            return True
        if direction in ltr_directions:
            return False
    return False


def bigrams_in_canvas_order(token: str) -> List[str]:
    """Return bigrams in increasing physical x-coordinate order.

    Unicode stores Arabic in logical (reading) order, while image columns run
    from left to right.  Pango handles the RTL glyph order *inside* each
    independently rendered bigram.  We therefore reverse only the sequence of
    bigram patches for an RTL token, not the characters inside a bigram.
    """
    units = overlapping_bigrams(token)
    return list(reversed(units)) if is_rtl(token) else units


def grapheme_joining_type(cluster: str) -> str:
    """Return the first non-transparent Unicode Joining_Type in a cluster."""
    for character in cluster:
        if _JOINING_TYPE_PATTERNS["T"].fullmatch(character):
            continue
        for joining_type in ("D", "R", "L", "C", "U"):
            if _JOINING_TYPE_PATTERNS[joining_type].fullmatch(character):
                return joining_type
        return "U"
    return "T"


def graphemes_join(first: str, second: str) -> bool:
    """Whether adjacent logical-order graphemes form a cursive connection."""
    first_type = grapheme_joining_type(first)
    second_type = grapheme_joining_type(second)
    return first_type in {"D", "L", "C"} and second_type in {"D", "R", "C"}


def contextual_bigrams_with_states(
    token: str,
) -> List[Tuple[str, bool, bool]]:
    """Return canvas-order bigrams and their external joining states.

    A state is ``(unit, joins_previous, joins_following)`` in Unicode logical
    order. These boundary bits, together with the ordered pair, determine its
    contextual forms. They are more precise than word-position labels because
    letters such as alef, dal, and reh can break joining inside a word.
    """
    clusters = grapheme_clusters(token)
    if not clusters:
        return []
    if len(clusters) == 1:
        states = [(clusters[0], False, False)]
    else:
        states = []
        for index in range(len(clusters) - 1):
            joins_previous = index > 0 and graphemes_join(
                clusters[index - 1], clusters[index]
            )
            joins_following = index + 2 < len(clusters) and graphemes_join(
                clusters[index + 1], clusters[index + 2]
            )
            states.append(
                (
                    "".join(clusters[index : index + 2]),
                    joins_previous,
                    joins_following,
                )
            )
    return list(reversed(states)) if is_rtl(token) else states


class _ContextualRendererBase:
    """Shared Pango setup for deterministic contextual patches."""

    def __init__(
        self,
        font_file: Path | str = FONT_FILE,
        fallback_fonts_dir: Path | str = FALLBACK_FONTS,
        font_size: int = FONT_SIZE,
        dpi: int = DPI,
        patch_size: int = PATCH_SIZE,
    ) -> None:
        self.font_file = Path(font_file)
        self.fallback_fonts_dir = Path(fallback_fonts_dir)
        self.font_size = font_size
        self.dpi = dpi
        self.patch_size = patch_size

        if not self.font_file.is_file():
            raise FileNotFoundError(self.font_file)
        manimpango.register_font(str(self.font_file))
        if self.fallback_fonts_dir.is_dir():
            for fallback in glob.glob(str(self.fallback_fonts_dir / "*tf")):
                manimpango.register_font(fallback)

        family = ttLib.TTFont(str(self.font_file))["name"].getDebugName(1)
        if not family:
            raise RuntimeError(f"Could not read font family from {self.font_file}")
        scaled_size = (self.dpi / 72.0) * self.font_size
        self.font = Pango.font_description_from_string(
            f"{family} {scaled_size:g}px"
        )

    def _layout(self, context: cairo.Context, text: str) -> Pango.Layout:
        pango_context = PangoCairo.create_context(context)
        pango_context.set_font_description(self.font)
        layout = Pango.Layout(pango_context)
        layout.set_auto_dir(True)
        layout.set_text(text, -1)
        return layout

    def render_text(
        self, text: str
    ) -> Tuple[List[str], List[List[str]], List[List[np.ndarray]]]:
        tokens = split_text_keep_punctuation(text)
        units: List[List[str]] = []
        patches: List[List[np.ndarray]] = []
        for token in tokens:
            token_units, token_patches = self.render_token(token)
            units.append(token_units)
            patches.append(token_patches)
        return tokens, units, patches

    @staticmethod
    def _fit_patch_width(image: np.ndarray, patch_size: int) -> np.ndarray:
        width = image.shape[1]
        if width > patch_size:
            start = (width - patch_size) // 2
            image = image[:, start : start + patch_size]
        elif width < patch_size:
            padding = patch_size - width
            image = np.pad(
                image,
                ((0, 0), (padding // 2, padding - padding // 2)),
                mode="constant",
                constant_values=255,
            )
        patch = image.astype(np.uint8, copy=False)
        patch.setflags(write=False)
        return patch


class ContextualBigramRenderer(_ContextualRendererBase):
    """Render one deterministic contextual patch per pair/boundary-state key.

    The original word determines whether the pair joins to its preceding and
    following graphemes. Those two bits select a deterministic synthetic Pango
    context. Tatweel acts only as a join-causing boundary sentinel and is
    excluded from the returned crop. Consequently, the same pair in the same
    joining state always maps to the same cached patch.
    """

    @lru_cache(maxsize=65_536)
    def render_patch(
        self,
        unit: str,
        joins_previous: bool,
        joins_following: bool,
    ) -> Tuple[np.ndarray, Tuple[int, int, int, bool]]:
        unit_clusters = grapheme_clusters(unit)
        if not 1 <= len(unit_clusters) <= 2:
            raise ValueError(
                "A contextual patch requires one or two grapheme clusters; "
                f"got {unit!r}"
            )

        prefix = TATWEEL if joins_previous else ""
        suffix = TATWEEL if joins_following else ""
        context_text = prefix + unit + suffix
        context_clusters = grapheme_clusters(context_text)
        central_start = 1 if prefix else 0
        central_indices = range(
            central_start, central_start + len(unit_clusters)
        )

        measuring_surface = cairo.ImageSurface(
            cairo.FORMAT_A8, 1, self.patch_size
        )
        measuring_context = cairo.Context(measuring_surface)
        measuring_layout = self._layout(measuring_context, context_text)
        text_width, text_height = measuring_layout.get_pixel_size()

        outer_margin = 2
        surface_width = max(1, text_width + 2 * outer_margin)
        surface = cairo.ImageSurface(
            cairo.FORMAT_A8, surface_width, self.patch_size
        )
        context = cairo.Context(surface)
        layout = self._layout(context, context_text)
        y = max(0.0, (self.patch_size - text_height) / 2.0)
        context.move_to(outer_margin, y)
        PangoCairo.show_layout(context, layout)
        surface.flush()

        stride = surface.get_stride()
        raw = np.frombuffer(surface.get_data(), dtype=np.uint8).reshape(
            self.patch_size, stride
        )[:, :surface_width]
        context_image = np.invert(raw).copy()

        byte_offsets: List[int] = []
        byte_offset = 0
        for cluster in context_clusters:
            byte_offsets.append(byte_offset)
            byte_offset += len(cluster.encode("utf-8"))

        endpoints: List[float] = []
        for cluster_index in central_indices:
            rectangle = layout.index_to_pos(byte_offsets[cluster_index])
            start = rectangle.x / Pango.SCALE
            end = (rectangle.x + rectangle.width) / Pango.SCALE
            endpoints.extend((start, end))

        left = math.floor(min(endpoints))
        right = math.ceil(max(endpoints))
        logical_width = max(1, right - left)
        source_left = max(0, left + outer_margin)
        source_right = min(surface_width, right + outer_margin)
        crop = context_image[:, source_left:source_right]
        if crop.shape[1] == 0:
            crop = np.full((self.patch_size, 1), 255, dtype=np.uint8)
        patch = self._fit_patch_width(crop, self.patch_size)
        metrics = (
            logical_width,
            text_height,
            layout.get_unknown_glyphs_count(),
            logical_width > self.patch_size or text_height > self.patch_size,
        )
        return patch, metrics

    @lru_cache(maxsize=8_192)
    def _render_token_cached(
        self, token: str
    ) -> Tuple[
        Tuple[str, ...],
        Tuple[np.ndarray, ...],
        Tuple[Tuple[int, int, int, bool, bool, bool], ...],
    ]:
        states = contextual_bigrams_with_states(token)
        units: List[str] = []
        patches: List[np.ndarray] = []
        metrics: List[Tuple[int, int, int, bool, bool, bool]] = []
        for unit, joins_previous, joins_following in states:
            patch, raw_metrics = self.render_patch(
                unit, joins_previous, joins_following
            )
            units.append(unit)
            patches.append(patch)
            metrics.append((*raw_metrics, joins_previous, joins_following))
        return tuple(units), tuple(patches), tuple(metrics)

    def render_token(self, token: str) -> Tuple[List[str], List[np.ndarray]]:
        units, patches, _ = self._render_token_cached(token)
        return list(units), list(patches)

    def render_token_with_metrics(
        self, token: str
    ) -> Tuple[List[str], List[np.ndarray], List[Dict[str, int | bool]]]:
        units, patches, raw_metrics = self._render_token_cached(token)
        metrics = [
            {
                "width": width,
                "height": height,
                "unknown_glyphs": unknown,
                "clipped": clipped,
                "joins_previous": joins_previous,
                "joins_following": joins_following,
            }
            for (
                width,
                height,
                unknown,
                clipped,
                joins_previous,
                joins_following,
            ) in raw_metrics
        ]
        return list(units), list(patches), metrics


def pack_word_patches(
    patches: Sequence[np.ndarray], patches_per_word: int = PATCHES_PER_WORD
) -> Tuple[np.ndarray, List[int]]:
    """Pack square patches into one fixed-width image used by the local ViT."""
    if not patches:
        raise ValueError("A rendered word must contain at least one patch")
    used = min(len(patches), patches_per_word)
    kept = list(patches[:used])
    image = np.concatenate(kept, axis=1)
    target_width = PATCH_SIZE * patches_per_word
    if image.shape[1] < target_width:
        image = np.pad(
            image,
            ((0, 0), (0, target_width - image.shape[1])),
            mode="constant",
            constant_values=255,
        )
    mask = [1] * used + [0] * (patches_per_word - used)
    return image.astype(np.uint8, copy=False), mask
