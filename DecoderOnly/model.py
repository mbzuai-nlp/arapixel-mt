import torch
import torch.nn as nn
import torch.distributed as dist

from DecoderOnly.utils import build_safe_words_ids


class DecoderOnlyTranslationModel(nn.Module):
    """
    Decoder-only baseline for Arabic→English translation.
    Uses the decoder tokenizer to encode Arabic source text directly
    (no separate encoder or projection).

    Sequence layout:
        [system] [user] [arabic_tokens] [assistant] [english_tokens]
    """

    def __init__(self, decoder_model, tokenizer):
        super().__init__()
        self.decoder_model = decoder_model
        self.tokenizer = tokenizer

        # Prompts
        self.system_prompt = "Translate."
        self.user_prompt = "Arabic:"
        self.assistant_prompt = "English:"

        # Cache prompt token IDs
        self._cache_prompts()

        # Train the full decoder
        for p in self.decoder_model.parameters():
            p.requires_grad = True
        self.decoder_model.train()

    # ── helpers ────────────────────────────────────────────────

    def _current_device(self) -> torch.device:
        for p in self.parameters():
            return p.device
        for b in self.buffers():
            return b.device
        return torch.device("cpu")

    def _cache_prompts(self):
        def encode_prompt(text):
            return self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"]

        self.register_buffer("system_ids", encode_prompt(self.system_prompt))
        self.register_buffer("user_ids", encode_prompt(self.user_prompt))
        self.register_buffer("assistant_ids", encode_prompt(self.assistant_prompt))

        self.register_buffer("system_mask", torch.ones_like(self.system_ids))
        self.register_buffer("user_mask", torch.ones_like(self.user_ids))
        self.register_buffer("assistant_mask", torch.ones_like(self.assistant_ids))

    def _embed_prompts(self, batch_size: int, dev: torch.device):
        emb_layer = self.decoder_model.get_input_embeddings()
        sys_e = emb_layer(self.system_ids.to(dev)).expand(batch_size, -1, -1)
        usr_e = emb_layer(self.user_ids.to(dev)).expand(batch_size, -1, -1)
        ast_e = emb_layer(self.assistant_ids.to(dev)).expand(batch_size, -1, -1)

        sys_m = self.system_mask.to(dev).expand(batch_size, -1)
        usr_m = self.user_mask.to(dev).expand(batch_size, -1)
        ast_m = self.assistant_mask.to(dev).expand(batch_size, -1)

        return sys_e, usr_e, ast_e, sys_m, usr_m, ast_m

    # ── forward ───────────────────────────────────────────────

    def forward(self, source_ids=None, source_mask=None,
                target_ids=None, target_mask=None, **kwargs):
        """
        Args:
            source_ids:  [B, MAX_AR_SEQ_LEN]  tokenised Arabic
            source_mask: [B, MAX_AR_SEQ_LEN]
            target_ids:  [B, MAX_SEQ_LEN]     tokenised English (+ EOS)
            target_mask: [B, MAX_SEQ_LEN]
        """
        dev = self._current_device()
        B = source_ids.shape[0]

        source_ids = source_ids.to(dev)
        source_mask = source_mask.to(dev)

        emb = self.decoder_model.get_input_embeddings()

        # Embed Arabic source tokens
        arabic_embeds = emb(source_ids)  # [B, MAX_AR_SEQ_LEN, H]

        # Prompt embeddings
        sys_e, usr_e, ast_e, sys_m, usr_m, ast_m = self._embed_prompts(B, dev)

        # Context: system + user + arabic + assistant
        ctx_embeds = torch.cat([sys_e, usr_e, arabic_embeds, ast_e], dim=1)
        ctx_mask = torch.cat([sys_m, usr_m, source_mask, ast_m], dim=1)

        if target_ids is not None:
            # ── Training ──
            target_ids = target_ids.to(dev)
            target_mask = target_mask.to(dev)
            target_embeds = emb(target_ids)

            inputs_embeds = torch.cat([ctx_embeds, target_embeds], dim=1)
            inputs_embeds = inputs_embeds.to(dtype=self.decoder_model.dtype, device=dev)
            full_attention_mask = torch.cat([ctx_mask, target_mask], dim=1)

            # Labels: -100 for everything except the English target
            source_len = ctx_embeds.shape[1]
            ignore_prefix = torch.full(
                (B, source_len), -100, dtype=torch.long, device=dev
            )
            labels = target_ids.clone()
            labels[target_mask == 0] = -100
            full_labels = torch.cat([ignore_prefix, labels], dim=1)

            outputs = self.decoder_model(
                inputs_embeds=inputs_embeds,
                attention_mask=full_attention_mask,
                labels=full_labels,
            )
            return {"loss": outputs.loss}

    # ── generation ────────────────────────────────────────────

    def get_trainable_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return trainable_params, total_params

    @torch.no_grad()
    def generate(
        self,
        source_ids,
        source_mask,
        tokenizer,
        max_new_tokens: int = 64,
        num_beams: int = 4,
        do_sample: bool = False,
        length_penalty: float = 1.0,
        no_repeat_ngram_size: int = 4,
        repetition_penalty: float = 1.2,          
        use_bad_words_filtering: bool = False,
        **generate_kwargs,
    ):
        """
        End-to-end generation: embed Arabic tokens → build prompt → decode.

        Args:
            source_ids:  [B, MAX_AR_SEQ_LEN]
            source_mask: [B, MAX_AR_SEQ_LEN]
            tokenizer:   decoder tokenizer
        Returns:
            list[str]: one translation per sample
        """
        self.eval()
        dev = self._current_device()
        source_ids = source_ids.to(dev)
        source_mask = source_mask.to(dev)
        B = source_ids.shape[0]

        emb = self.decoder_model.get_input_embeddings()
        arabic_embeds = emb(source_ids)

        sys_e, usr_e, ast_e, sys_m, usr_m, ast_m = self._embed_prompts(B, dev)

        inputs_embeds = torch.cat([sys_e, usr_e, arabic_embeds, ast_e], dim=1)
        inputs_embeds = inputs_embeds.to(dtype=self.decoder_model.dtype, device=dev)
        attention_mask = torch.cat([sys_m, usr_m, source_mask, ast_m], dim=1)

        gen_args = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "num_beams": num_beams,
            "do_sample": do_sample,
            "length_penalty": length_penalty,
            "no_repeat_ngram_size": no_repeat_ngram_size,
            "repetition_penalty": repetition_penalty,
            "use_cache": True,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }

        if use_bad_words_filtering:
            headers = [
                self.system_prompt.strip(),
                self.user_prompt.strip(),
                self.assistant_prompt.strip(),
            ]
            gen_args["bad_words_ids"] = build_safe_words_ids(headers, tokenizer)

        synced = (
            dist.is_available()
            and dist.is_initialized()
            and dist.get_world_size() > 1
        )
        if synced:
            gen_args["synced_gpus"] = True

        gen_args.update(generate_kwargs)

        prev_cache = getattr(self.decoder_model.config, "use_cache", True)
        self.decoder_model.config.use_cache = True
        try:
            gen_ids = self.decoder_model.generate(**gen_args)
        finally:
            self.decoder_model.config.use_cache = prev_cache

        translations = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return translations
