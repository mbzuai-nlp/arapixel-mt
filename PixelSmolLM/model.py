import torch
import torch.nn as nn
from .utils import TrainingMode, pixels_per_patch, MAX_AR_WORDS, build_safe_words_ids
from .rendering.config import IMAGE_HEIGHT, IMAGE_WIDTH
import torch.distributed as dist
import numpy as np


class ConfigurablePixelTranslationModel(nn.Module):
    def __init__(self, pixel_model, decoder_model, tokenizer, training_mode: TrainingMode):
        super().__init__()
        self.pixel_model = pixel_model
        self.decoder_model = decoder_model
        self.tokenizer = tokenizer
        self.training_mode = training_mode
        pix_h = self.pixel_model.config.hidden_size
        dec_h = self.decoder_model.config.hidden_size

        self.projection = nn.Sequential(
            nn.Linear(pix_h, pix_h * 2),
            nn.GELU(),
            nn.LayerNorm(pix_h * 2),
            nn.Linear(pix_h * 2, dec_h),
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
            for p in self.pixel_model.parameters(): p.requires_grad = True
            for p in self.projection.parameters(): p.requires_grad = True
            self.pixel_model.train()
            self.projection.train()

        elif mode.value == TrainingMode.BOTH.value:
            # Train Encoder + Projection + Decoder
            for p in self.pixel_model.parameters(): p.requires_grad = True
            for p in self.projection.parameters(): p.requires_grad = True
            for p in self.decoder_model.parameters(): p.requires_grad = True
            self.pixel_model.train()
            self.decoder_model.train()
            self.projection.train()


        # Ensure determinism
        for m in self.pixel_model.modules():
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

    def _prepare_word_images_batch(self, per_token_patches_batch):
        """
        Prepare word images and track which patches are actual content vs padding.
        
        Args:
            per_token_patches_batch: List of per-sample token patches
            
        Returns:
            imgs: torch.Tensor [total_words, 3, 24, 384] - word images
            counts: List[int] - number of words per sample
            patch_masks: torch.Tensor [total_words, 16] - boolean mask (True=valid, False=padding)
        """
        imgs = []
        counts = []
        patch_masks = []
        
        for token_patches in per_token_patches_batch:
            c = 0
            if token_patches:
                for token_patches_word in token_patches:
                    # Concatenate patches horizontally for this word
                    word_img = token_patches_word[0] if len(token_patches_word) == 1 else \
                        np.concatenate([token_patches_word[0]] + token_patches_word[1:], axis=1)
                    
                    h, w = word_img.shape[:2]
                    target_w = IMAGE_WIDTH
                    
                    # Calculate actual number of patches BEFORE padding
                    num_actual_patches = (w + pixels_per_patch - 1) // pixels_per_patch
                    num_total_patches = target_w // pixels_per_patch  # Should be 16
                    
                    # Create mask: True for actual content, False for padding
                    mask = torch.zeros(num_total_patches, dtype=torch.bool)
                    mask[:num_actual_patches] = True
                    patch_masks.append(mask)
                    
                    # Pad or crop width to target
                    if w < target_w:
                        pad_w = target_w - w
                        if word_img.ndim == 2:
                            word_img = np.pad(word_img, ((0, 0), (0, pad_w)), 
                                            mode='constant', constant_values=255)
                        else:
                            word_img = np.pad(word_img, ((0, 0), (0, pad_w), (0, 0)), 
                                            mode='constant', constant_values=255)
                    elif w > target_w:
                        word_img = word_img[:, :target_w]
                    
                    # Convert to RGB tensor format [3, H, W]
                    word_rgb = np.stack([word_img, word_img, word_img], axis=0) \
                        if word_img.ndim == 2 else word_img.transpose(2, 0, 1)
                    imgs.append(torch.from_numpy(word_rgb).float() / 255.0)
                    c += 1
            counts.append(c)
        
        if len(imgs) == 0:
            return torch.empty(0, 3, IMAGE_HEIGHT, IMAGE_WIDTH), counts, []
        
        return torch.stack(imgs, dim=0), counts, torch.stack(patch_masks, dim=0)


    def _pixel_embed_batch(self, word_images, patch_masks, word_counts):
        """
        Embed pre-rendered word images with masked mean pooling.

        Args:
            word_images: [total_words, 3, 24, 384] float tensor
            patch_masks: [total_words, 16] bool tensor
            word_counts: list[int] - number of words per sample

        Returns:
            List of embeddings per sample [num_words, hidden_size]
        """
        dev = self._current_device()
        imgs = word_images.to(dev, non_blocking=True)
        masks = patch_masks.to(dev).long()

        if imgs.shape[0] == 0:
            return [torch.empty(0, self.decoder_model.config.hidden_size, device=dev)
                    for _ in word_counts]

        vit_out = self.pixel_model(pixel_values=imgs, attention_mask=masks)
        patch_embeddings = vit_out.last_hidden_state[:, 1:, :]  # [total, num_patches, hidden]

        mask_expanded = masks.unsqueeze(-1).float()
        masked_sum = (patch_embeddings * mask_expanded).sum(dim=1)
        valid_counts = mask_expanded.sum(dim=1).clamp(min=1)
        word_embeds_all = self.projection(masked_sum / valid_counts)

        result = []
        start = 0
        for c in word_counts:
            if c == 0:
                result.append(torch.empty(0, word_embeds_all.shape[-1], device=dev))
            else:
                result.append(word_embeds_all[start:start + c])
            start += c
        return result

    def _pad_pixel_embeddings(self, px_list, dec_h, dev):
        if not px_list:
            return torch.empty(0, 0, dec_h, device=dev), torch.empty(0, 0, dtype=torch.long, device=dev)

        px_padded, px_mask = [], []
        for px in px_list:
            L = px.shape[0]
            if L >= MAX_AR_WORDS:
                px_use = px[:MAX_AR_WORDS]
                mask = torch.ones(MAX_AR_WORDS, device=dev, dtype=torch.long)
            else:
                pad = torch.zeros(MAX_AR_WORDS - L, dec_h, device=dev, dtype=px.dtype)
                px_use = torch.cat([px, pad], dim=0)
                mask = torch.cat([
                    torch.ones(L, device=dev, dtype=torch.long),
                    torch.zeros(MAX_AR_WORDS - L, device=dev, dtype=torch.long),
                ], dim=0)
            px_padded.append(px_use.unsqueeze(0))
            px_mask.append(mask.unsqueeze(0))

        return torch.cat(px_padded, dim=0), torch.cat(px_mask, dim=0)
    

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        word_images = kwargs.get('word_images')       # [total_words, 3, 24, 384]
        patch_masks = kwargs.get('patch_masks')       # [total_words, 16]
        word_counts = kwargs.get('word_counts')       
        target_ids = kwargs.get('target_ids')
        target_mask = kwargs.get('target_mask')

        dev = self._current_device()
        dec_h = self.decoder_model.config.hidden_size

        # Derive B from target_ids
        B = target_ids.shape[0]

        px_list = self._pixel_embed_batch(word_images, patch_masks, word_counts)
        px_padded, px_mask = self._pad_pixel_embeddings(px_list, dec_h, dev)

        sys, usr, ast, sys_m, usr_m, ast_m = self._embed_prompts(B, dev)

        # Use pre-tokenized targets from the dataset/collator
        assert target_ids is not None and target_mask is not None
        tgt_ids = target_ids.to(dev)
        tgt_m = target_mask.to(dev)
        tgt_emb = self.decoder_model.get_input_embeddings()(tgt_ids)

        
        inp = torch.cat([sys, usr, px_padded, ast], dim=1)
        im  = torch.cat([sys_m, usr_m, px_mask, ast_m], dim=1)

        # Create Labels - match Bytes model pattern
        source_len = inp.shape[1]
        ignore_prefix = torch.full((tgt_ids.shape[0], source_len), -100, dtype=torch.long, device=dev)
        
        labels = tgt_ids.clone()
        labels[tgt_m == 0] = -100
        
        full_labels = torch.cat([ignore_prefix, labels], dim=1)
        inputs_embeds = torch.cat([inp, tgt_emb], dim=1).to(dtype=self.decoder_model.dtype, device=dev)
        attention_mask = torch.cat([im, tgt_m], dim=1)

        outputs = self.decoder_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=full_labels)
        
        return {'loss': outputs.loss}

    def get_trainable_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return trainable_params, total_params

    @torch.no_grad()
    def generate(
        self,
        word_images,
        patch_masks,
        word_counts,
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
        End-to-end generation: encode pixel words → build prompt context → decode.

        Args:
            word_images:  [total_words, 3, 24, 384]
            patch_masks:  [total_words, 16]
            word_counts:  list[int], one per sample
            tokenizer:    decoder tokenizer
            max_new_tokens / num_beams / …: generation hyper-params
            use_bad_words_filtering: filter out prompt tokens from generation

        Returns:
            list[str]: one translation per sample
        """
        self.eval()
        dev = self._current_device()
        dec_h = self.decoder_model.config.hidden_size
        B = len(word_counts)

        # 1. Encode through ViT + projection
        px_list = self._pixel_embed_batch(word_images, patch_masks, word_counts)

        # 2. Pad to the longest sample in the current batch
        px_padded, px_mask = self._pad_pixel_embeddings(px_list, dec_h, dev)

        # 3. Prompt embeddings
        sys_e, usr_e, ast_e, sys_m, usr_m, ast_m = self._embed_prompts(B, dev)

        # 4. Concatenate context
        inputs_embeds = torch.cat([sys_e, usr_e, px_padded, ast_e], dim=1)
        inputs_embeds = inputs_embeds.to(dtype=self.decoder_model.dtype, device=dev)
        attention_mask = torch.cat([sys_m, usr_m, px_mask, ast_m], dim=1)

        # 5. Build generation args
        gen_args = {
            'inputs_embeds': inputs_embeds,
            'attention_mask': attention_mask,
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

        # 6. Generate
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
