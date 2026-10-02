"""Generation position-ID policies used by fixed-width evaluation."""

from __future__ import annotations

from types import MethodType


def use_physical_generation_positions(model) -> None:
    """Count masked source padding in decoder positions during generation.

    Training calls the decoder without explicit position IDs, so physical
    sequence slots—including masked padding—determine RoPE positions.
    Generation helpers can instead derive positions from the cumulative
    attention mask and skip padding. This adapter retains the real attention
    mask while letting the decoder construct physical positions.
    """

    decoder = model.decoder_model
    if getattr(decoder, "_uses_physical_generation_positions", False):
        return
    original_prepare = decoder.prepare_inputs_for_generation

    def prepare_with_physical_positions(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        prepared = original_prepare(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=None,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        prepared["attention_mask"] = attention_mask
        prepared["position_ids"] = None
        return prepared

    decoder.prepare_inputs_for_generation = MethodType(
        prepare_with_physical_positions,
        decoder,
    )
    decoder._uses_physical_generation_positions = True
