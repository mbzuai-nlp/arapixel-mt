import math
from dataclasses import dataclass
from typing import Optional
import torch
import numpy as np 
import torch.nn as nn
from BytesSmolLM.utils import MAX_WORD_LENGTH, PAD_TOKEN_ID, VOCAB_SIZE




def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position pos: a list of positions to be encoded: size (M,) out: (M, D)
    """
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")

    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb



@dataclass
class TextViTConfig:
    """Configuration matching Pixel's actual parameters"""
    vocab_size: int = VOCAB_SIZE
    hidden_size: int = 1536           # Match PIXEL
    num_hidden_layers: int = 4       # Match PIXEL
    num_attention_heads: int = 16    # Match PIXEL
    intermediate_size: int = 6144    # Match PIXEL
    max_position_embeddings: int = MAX_WORD_LENGTH + 1 # Match Max Bytes + CLS
    hidden_dropout_prob: float = 0.0 # Disabled for determinism
    attention_probs_dropout_prob: float = 0.0
    layer_norm_eps: float = 1e-5
    hidden_act: str = "gelu"
    pad_token_id: int = PAD_TOKEN_ID
    
    
class TextViTEmbeddings(nn.Module):
    def __init__(self, config: TextViTConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        
        # Fixed 1D sin-cos positional embeddings (matching PIXEL)
        # +1 for CLS token at position 0
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, config.max_position_embeddings, config.hidden_size), requires_grad=False
        )
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.config = config
        self.initialize_weights()

    def initialize_weights(self):
        # 1D sin-cos positional embeddings
        num_positions = self.config.max_position_embeddings - 1  # exclude CLS
        pos = np.arange(num_positions, dtype=np.float32)
        pos_embed = get_1d_sincos_pos_embed_from_grid(
            self.position_embeddings.shape[-1], pos
        )
        # CLS token gets zeros at position 0
        pos_embed = np.concatenate([np.zeros([1, self.position_embeddings.shape[-1]]), pos_embed], axis=0)
        self.position_embeddings.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        # Initialize CLS token
        torch.nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        batch_size, seq_length = input_ids.shape
        
        # Token embeddings + position embeddings (positions 1..seq_length)
        inputs_embeds = self.word_embeddings(input_ids)
        embeddings = inputs_embeds + self.position_embeddings[:, 1:seq_length + 1, :]
        
        # CLS token + position 0
        cls_token = self.cls_token + self.position_embeddings[:, :1, :]
        cls_tokens = cls_token.expand(batch_size, -1, -1)
        
        # Concatenate [CLS, tokens...]
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        
        # Expand attention mask for CLS token
        if attention_mask is not None:
            cls_mask = torch.ones(batch_size, 1, device=attention_mask.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([cls_mask, attention_mask], dim=1)
        
        return embeddings, attention_mask      

class TextViTSelfAttention(nn.Module):
    def __init__(self, config: TextViTConfig):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(f"Hidden size {config.hidden_size} not divisible by num heads {config.num_attention_heads}")
        
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            # Expand the token mask to the attention-score dimensions.
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)

            # attention_mask is 1.0 for keep, 0.0 for mask
            # Add a large negative value to masked attention positions.
            min_dtype = torch.finfo(attention_scores.dtype).min
            attention_mask = (1.0 - attention_mask) * min_dtype
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        return context_layer

class TextViTSelfOutput(nn.Module):
    def __init__(self, config: TextViTConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

class TextViTAttention(nn.Module):
    def __init__(self, config: TextViTConfig):
        super().__init__()
        self.self = TextViTSelfAttention(config)
        self.output = TextViTSelfOutput(config)

    def forward(self, hidden_states, attention_mask=None):
        self_outputs = self.self(hidden_states, attention_mask)
        attention_output = self.output(self_outputs, hidden_states)
        return attention_output

class TextViTIntermediate(nn.Module):
    def __init__(self, config: TextViTConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = nn.GELU() if config.hidden_act == "gelu" else nn.ReLU()

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class TextViTOutput(nn.Module):
    def __init__(self, config: TextViTConfig):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        hidden_states = hidden_states + input_tensor
        
        return hidden_states

class TextViTLayer(nn.Module):
    def __init__(self, config: TextViTConfig):
        super().__init__()
        self.attention = TextViTAttention(config)
        self.intermediate = TextViTIntermediate(config)
        self.output = TextViTOutput(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, attention_mask=None):
        # Pre-LN: normalize before attention
        normed = self.layernorm_before(hidden_states)
        attention_output = self.attention(normed, attention_mask)
        hidden_states = attention_output + hidden_states  # residual 1

        # Pre-LN: normalize before FFN
        normed = self.layernorm_after(hidden_states)
        intermediate_output = self.intermediate(normed)
        hidden_states = self.output(intermediate_output, hidden_states)  # residual 2 inside

        return hidden_states
    
    
class TextViTEncoder(nn.Module):
    def __init__(self, config: TextViTConfig):
        super().__init__()
        self.layer = nn.ModuleList([TextViTLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None):
        for layer in self.layer:
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states

class TextViTPooler(nn.Module):
    def __init__(self, config: TextViTConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # Pool CLS token (first token)
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class TextViTArchitecture(nn.Module):
    def __init__(self, config: TextViTConfig):
        super().__init__()
        self.config = config
        self.embeddings = TextViTEmbeddings(config)
        self.encoder = TextViTEncoder(config)
        
        # ADDED: Final LayerNorm like PIXEL
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
                
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask=None):
        # Get embeddings
        embedding_output, extended_attention_mask = self.embeddings(input_ids, attention_mask)
        
        # Pass through encoder
        encoder_output = self.encoder(embedding_output, extended_attention_mask)
        
        # ADDED: Apply final LayerNorm
        sequence_output = self.layernorm(encoder_output)
        
        # Return normalized sequence output
        # Shape: [batch_size, seq_length + 1, hidden_size]
        return sequence_output
    
