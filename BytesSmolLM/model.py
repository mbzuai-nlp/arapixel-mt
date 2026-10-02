import torch
import torch.nn as nn
import torch.distributed as dist

from BytesSmolLM.utils import (
    PAD_TOKEN_ID,
    TrainingMode,
    build_safe_words_ids,
)


class ConfigurableTextViTTranslationModel(nn.Module):
    def __init__(self, text_vit_model, decoder_model, tokenizer, training_mode: TrainingMode):
        super().__init__()
        self.text_vit_model = text_vit_model
        self.decoder_model = decoder_model
        self.tokenizer = tokenizer
        self.training_mode = training_mode
        
        
        enc_h = self.text_vit_model.config.hidden_size
        dec_h = self.decoder_model.config.hidden_size
        
        self.projection = nn.Sequential(
            nn.Linear(enc_h, enc_h * 2),
            nn.GELU(),
            nn.LayerNorm(enc_h * 2),
            nn.Linear(enc_h * 2, dec_h),
            nn.GELU(),
            nn.LayerNorm(dec_h),
        )
        
        # Prompts
        self.system_prompt = "Translate."
        self.user_prompt = "Arabic:"
        self.assistant_prompt = "English:"
        
        # Cache prompt IDs
        self._cache_prompts()
        
        # Set gradients based on mode
        self.set_training_mode(training_mode)

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


    def set_training_mode(self, mode: TrainingMode):
        self.training_mode = mode
        # Default: Freeze everything
        for p in self.parameters(): p.requires_grad = False
        
        if mode.value == TrainingMode.ENCODER_PROJECTION.value:
            # Train Encoder + Projection
            for p in self.text_vit_model.parameters(): p.requires_grad = True
            for p in self.projection.parameters(): p.requires_grad = True
            self.text_vit_model.train()
            self.projection.train()

        elif mode.value == TrainingMode.BOTH.value:
            # Train Encoder + Projection + Decoder
            for p in self.text_vit_model.parameters(): p.requires_grad = True
            for p in self.projection.parameters(): p.requires_grad = True
            for p in self.decoder_model.parameters(): p.requires_grad = True
            self.text_vit_model.train()
            self.decoder_model.train()
            self.projection.train()

        # Ensure determinism
        for m in self.text_vit_model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.p = 0.0

    def _embed_prompts(self, batch_size: int, dev: torch.device):
        emb_layer = self.decoder_model.get_input_embeddings()
        sys_e = emb_layer(self.system_ids.to(dev)).expand(batch_size, -1, -1)
        usr_e = emb_layer(self.user_ids.to(dev)).expand(batch_size, -1, -1)
        ast_e = emb_layer(self.assistant_ids.to(dev)).expand(batch_size, -1, -1)
        
        sys_m = self.system_mask.to(dev).expand(batch_size, -1)
        usr_m = self.user_mask.to(dev).expand(batch_size, -1)
        ast_m = self.assistant_mask.to(dev).expand(batch_size, -1)
        
        return sys_e, usr_e, ast_e, sys_m, usr_m, ast_m


    def _encode_arabic_batch(self, input_ids, attention_mask):
        """
        input_ids: [Batch, MAX_AR_WORDS, MAX_WORD_LENGTH] (Bytes)
        attention_mask: [Batch, MAX_AR_WORDS] (Word existence)
        """
        batch_size, num_words, word_len = input_ids.shape
        
        # Flatten to [Batch * Num_Words, Word_Len] to pass through TextViT
        flat_input_ids = input_ids.view(-1, word_len)
        
        # Create mask for the bytes within words (pad tokens are 256)
        # 1 for real data, 0 for pad
        flat_byte_mask = (flat_input_ids != PAD_TOKEN_ID).long()
        
        # Forward pass through TextViT
        # Output: [Batch * Num_Words, Hidden] (Pooled CLS)
        encoder_outputs = self.text_vit_model(flat_input_ids, attention_mask=flat_byte_mask)
        
        # Mean Pooling 
    
        token_embeddings = encoder_outputs[:, 1:, :]                 # drop CLS
        token_mask = flat_byte_mask.unsqueeze(-1).float()            # [B*N, seq_len, 1]
        sum_embeddings = (token_embeddings * token_mask).sum(dim=1)
        sum_mask = token_mask.sum(dim=1).clamp(min=1e-9)
        pooled_output = sum_embeddings / sum_mask
        
        # Reshape back to [Batch, Num_Words, Hidden]
        word_embeddings = pooled_output.view(batch_size, num_words, -1)
        
        # Project to Decoder Dimension
        projected_embeddings = self.projection(word_embeddings)
        
        return projected_embeddings

      
    def forward(self, input_ids=None, attention_mask=None, target_ids=None, target_mask=None, **kwargs):

        dev = self._current_device()
        batch_size = input_ids.shape[0]
        
        # 1. Encode Arabic Source
        # [Batch, 32, Dec_Dim]
        arabic_embeddings = self._encode_arabic_batch(input_ids.to(dev), attention_mask.to(dev))
        
        # 2. Get Prompt Embeddings
        sys_e, usr_e, ast_e, sys_m, usr_m, ast_m = self._embed_prompts(batch_size, dev)
        
        # 3. Concatenate: System + User + Arabic + Assistant
        inputs_embeds = torch.cat([sys_e, usr_e, arabic_embeddings, ast_e], dim=1)
        full_attention_mask = torch.cat([sys_m, usr_m, attention_mask.to(dev), ast_m], dim=1)
        
        # 4. Prepare Labels (if training)
        if target_ids is not None:
            target_ids = target_ids.to(dev)
            target_mask = target_mask.to(dev)
            target_embeds = self.decoder_model.get_input_embeddings()(target_ids)
            
            # Concatenate Target to Input
            inputs_embeds = torch.cat([inputs_embeds, target_embeds], dim=1).to(dtype=self.decoder_model.dtype, device=dev)
            full_attention_mask = torch.cat([full_attention_mask, target_mask], dim=1)
            
            # Create Labels
            # Mask out the source/prompts (-100)
            source_len = sys_e.shape[1] + usr_e.shape[1] + arabic_embeddings.shape[1] + ast_e.shape[1]
            ignore_prefix = torch.full((batch_size, source_len), -100, dtype=torch.long, device=dev)
            
            # Mask padding in target
            labels = target_ids.clone()
            labels[target_mask == 0] = -100
            
            full_labels = torch.cat([ignore_prefix, labels], dim=1)
            
            outputs = self.decoder_model(
                inputs_embeds=inputs_embeds,
                attention_mask=full_attention_mask,
                labels=full_labels
            )
            

            return {'loss': outputs.loss}

                


    def get_trainable_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return trainable_params, total_params

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        attention_mask,
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
        End-to-end generation: encode bytes → build prompt context → decode.

        Args:
            input_ids:      [B, MAX_AR_WORDS, MAX_WORD_LENGTH]
            attention_mask: [B, MAX_AR_WORDS]
            tokenizer:      decoder tokenizer
            max_new_tokens / num_beams / …: generation hyper-params

        Returns:
            list[str]: one translation per sample
        """
        self.eval()
        dev = self._current_device()
        input_ids = input_ids.to(dev)
        attention_mask = attention_mask.to(dev)
        B = input_ids.shape[0]

        # 1. Encode Arabic source
        arabic_embeddings = self._encode_arabic_batch(input_ids, attention_mask)

        # 2. Prompt embeddings
        sys_e, usr_e, ast_e, sys_m, usr_m, ast_m = self._embed_prompts(B, dev)

        # 3. Concatenate context
        inputs_embeds = torch.cat([sys_e, usr_e, arabic_embeddings, ast_e], dim=1)
        inputs_embeds = inputs_embeds.to(dtype=self.decoder_model.dtype, device=dev)
        attention_mask_ctx = torch.cat([sys_m, usr_m, attention_mask, ast_m], dim=1)

        # 4. Bad words filtering
        gen_args = {
            'inputs_embeds': inputs_embeds,
            'attention_mask': attention_mask_ctx,
            'max_new_tokens': max_new_tokens,
            'num_beams': num_beams,
            'do_sample': do_sample,
            'length_penalty': length_penalty,
            'no_repeat_ngram_size': no_repeat_ngram_size,
            'repetition_penalty': repetition_penalty,
            'use_cache': True,
            'pad_token_id': tokenizer.eos_token_id,
            'eos_token_id': tokenizer.eos_token_id,
        }

        if use_bad_words_filtering:
            headers = [
                self.system_prompt.strip(),
                self.user_prompt.strip(),
                self.assistant_prompt.strip(),
            ]
            gen_args['bad_words_ids'] = build_safe_words_ids(headers, tokenizer)

        synced = (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1)
        if synced:
            gen_args['synced_gpus'] = True

        gen_args.update(generate_kwargs)

        # 5. Generate
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
