"""
Hugging Face 기반 Transformer 구현 with Internal Tracking
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    AutoConfig,
    PreTrainedModel,
    PretrainedConfig
)
from transformers.modeling_outputs import BaseModelOutput
from typing import Optional, Tuple, Dict, List, Any
import math
import numpy as np


class TrackedTransformerConfig(PretrainedConfig):
    """
    Tracked Transformer를 위한 설정 클래스
    """
    model_type = "tracked_transformer"
    
    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 512,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 8,
        intermediate_size: int = 2048,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        pad_token_id: int = 0,
        position_embedding_type: str = "absolute",
        track_internal_states: bool = True,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.track_internal_states = track_internal_states


class TrackedAttention(nn.Module):
    """
    Hugging Face style attention with internal state tracking
    """
    
    def __init__(self, config: TrackedTransformerConfig, layer_idx: int = 0):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.layer_idx = layer_idx
        self.track_internal_states = config.track_internal_states
        
        # Linear layers for Q, K, V
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
        # For tracking internal states
        self.tracked_states = {}
        
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape tensor for multi-head attention"""
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, head_size)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass with internal state tracking
        """
        batch_size, seq_length = hidden_states.shape[:2]
        
        if self.track_internal_states:
            print(f"\n--- Layer {self.layer_idx} Attention ---")
            print(f"Input hidden_states: {hidden_states.shape}")
            print(f"Input mean: {hidden_states.mean().item():.4f}, std: {hidden_states.std().item():.4f}")
        
        # 1. Linear transformations for Q, K, V
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        
        if self.track_internal_states:
            print(f"Q linear output: {mixed_query_layer.shape}, mean: {mixed_query_layer.mean().item():.4f}")
            print(f"K linear output: {mixed_key_layer.shape}, mean: {mixed_key_layer.mean().item():.4f}")
            print(f"V linear output: {mixed_value_layer.shape}, mean: {mixed_value_layer.mean().item():.4f}")
        
        # 2. Reshape for multi-head attention
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        
        if self.track_internal_states:
            print(f"Q reshaped: {query_layer.shape}")
            print(f"K reshaped: {key_layer.shape}")
            print(f"V reshaped: {value_layer.shape}")
        
        # 3. Compute attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        if self.track_internal_states:
            print(f"Attention scores: {attention_scores.shape}")
            print(f"Score range: [{attention_scores.min().item():.4f}, {attention_scores.max().item():.4f}]")
        
        # 4. Apply attention mask
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
            if self.track_internal_states:
                print(f"After masking - range: [{attention_scores.min().item():.4f}, {attention_scores.max().item():.4f}]")
        
        # 5. Attention probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        if self.track_internal_states:
            print(f"Attention probs: {attention_probs.shape}")
            print(f"Prob range: [{attention_probs.min().item():.4f}, {attention_probs.max().item():.4f}]")
            print(f"Prob sum (should be ~1.0): {attention_probs.sum(dim=-1).mean().item():.4f}")
        
        # 6. Apply attention to values
        context_layer = torch.matmul(attention_probs, value_layer)
        
        if self.track_internal_states:
            print(f"Context layer: {context_layer.shape}")
            print(f"Context mean: {context_layer.mean().item():.4f}, std: {context_layer.std().item():.4f}")
        
        # 7. Reshape back
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        
        # Store tracked states
        if self.track_internal_states:
            self.tracked_states = {
                'input_hidden_states': hidden_states.clone(),
                'query': query_layer.clone(),
                'key': key_layer.clone(),
                'value': value_layer.clone(),
                'attention_scores': attention_scores.clone(),
                'attention_probs': attention_probs.clone(),
                'context_layer': context_layer.clone(),
            }
        
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class TrackedSelfOutput(nn.Module):
    """Self attention output layer with tracking"""
    
    def __init__(self, config: TrackedTransformerConfig, layer_idx: int = 0):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layer_idx = layer_idx
        self.track_internal_states = config.track_internal_states
        
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward with residual connection and layer norm"""
        if self.track_internal_states:
            print(f"\n--- Layer {self.layer_idx} Self Output ---")
            print(f"Before dense: {hidden_states.shape}, mean: {hidden_states.mean().item():.4f}")
        
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        if self.track_internal_states:
            print(f"After dense+dropout: mean: {hidden_states.mean().item():.4f}")
            print(f"Residual input: mean: {input_tensor.mean().item():.4f}")
        
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        
        if self.track_internal_states:
            print(f"After LayerNorm: mean: {hidden_states.mean().item():.4f}")
        
        return hidden_states


class TrackedIntermediate(nn.Module):
    """Feed-forward intermediate layer with tracking"""
    
    def __init__(self, config: TrackedTransformerConfig, layer_idx: int = 0):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = F.gelu  # GELU activation
        self.layer_idx = layer_idx
        self.track_internal_states = config.track_internal_states
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Feed-forward with GELU activation"""
        if self.track_internal_states:
            print(f"\n--- Layer {self.layer_idx} FFN Intermediate ---")
            print(f"Input: {hidden_states.shape}, mean: {hidden_states.mean().item():.4f}")
        
        hidden_states = self.dense(hidden_states)
        
        if self.track_internal_states:
            print(f"After linear: mean: {hidden_states.mean().item():.4f}")
            print(f"Before GELU - range: [{hidden_states.min().item():.4f}, {hidden_states.max().item():.4f}]")
        
        hidden_states = self.intermediate_act_fn(hidden_states)
        
        if self.track_internal_states:
            print(f"After GELU: mean: {hidden_states.mean().item():.4f}")
            print(f"After GELU - range: [{hidden_states.min().item():.4f}, {hidden_states.max().item():.4f}]")
        
        return hidden_states


class TrackedOutput(nn.Module):
    """Feed-forward output layer with tracking"""
    
    def __init__(self, config: TrackedTransformerConfig, layer_idx: int = 0):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layer_idx = layer_idx
        self.track_internal_states = config.track_internal_states
        
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        """Output projection with residual connection"""
        if self.track_internal_states:
            print(f"\n--- Layer {self.layer_idx} FFN Output ---")
            print(f"Before dense: {hidden_states.shape}, mean: {hidden_states.mean().item():.4f}")
        
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        if self.track_internal_states:
            print(f"After dense+dropout: mean: {hidden_states.mean().item():.4f}")
        
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        
        if self.track_internal_states:
            print(f"After LayerNorm (final): mean: {hidden_states.mean().item():.4f}")
        
        return hidden_states


class TrackedTransformerLayer(nn.Module):
    """Single transformer layer with full tracking"""
    
    def __init__(self, config: TrackedTransformerConfig, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx
        self.track_internal_states = config.track_internal_states
        
        self.attention = TrackedAttention(config, layer_idx)
        self.self_output = TrackedSelfOutput(config, layer_idx)
        self.intermediate = TrackedIntermediate(config, layer_idx)
        self.output = TrackedOutput(config, layer_idx)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """Complete transformer layer forward pass"""
        if self.track_internal_states:
            print(f"\n{'='*60}")
            print(f"TRANSFORMER LAYER {self.layer_idx}")
            print(f"{'='*60}")
        
        # 1. Self Attention
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        
        # 2. Self Output (residual + layer norm)
        attention_output = self.self_output(attention_output, hidden_states)
        
        # 3. Feed Forward Network
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        
        outputs = (layer_output,) + self_attention_outputs[1:]  # add attentions if we output them
        return outputs


class TrackedTransformerModel(PreTrainedModel):
    """
    Complete Transformer model with Hugging Face compatibility and internal tracking
    """
    
    config_class = TrackedTransformerConfig
    base_model_prefix = "transformer"
    
    def __init__(self, config: TrackedTransformerConfig):
        super().__init__(config)
        self.config = config
        
        # Embeddings
        self.embeddings = nn.ModuleDict({
            'word_embeddings': nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id),
            'position_embeddings': nn.Embedding(config.max_position_embeddings, config.hidden_size),
            'token_type_embeddings': nn.Embedding(config.type_vocab_size, config.hidden_size),
        })
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TrackedTransformerLayer(config, layer_idx=i) 
            for i in range(config.num_hidden_layers)
        ])
        
        # Initialize weights
        self.post_init()
    
    def get_input_embeddings(self):
        return self.embeddings['word_embeddings']
    
    def set_input_embeddings(self, value):
        self.embeddings['word_embeddings'] = value
    
    def _prepare_attention_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """Prepare attention mask for multi-head attention"""
        # Convert 0/1 mask to additive mask
        # 1 -> 0.0, 0 -> -10000.0
        attention_mask = attention_mask[:, None, None, :]  # (batch_size, 1, 1, seq_len)
        attention_mask = (1.0 - attention_mask) * -10000.0
        return attention_mask
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> BaseModelOutput:
        """Forward pass with full tracking"""
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if input_ids is None:
            raise ValueError("input_ids must be provided")
        
        batch_size, seq_length = input_ids.shape
        device = input_ids.device
        
        if self.config.track_internal_states:
            print(f"\n{'='*80}")
            print(f"TRACKED TRANSFORMER FORWARD PASS")
            print(f"{'='*80}")
            print(f"Input IDs shape: {input_ids.shape}")
            print(f"Batch size: {batch_size}, Seq length: {seq_length}")
        
        # 1. Embeddings
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
        
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        
        word_embeddings = self.embeddings['word_embeddings'](input_ids)
        position_embeddings = self.embeddings['position_embeddings'](position_ids)
        token_type_embeddings = self.embeddings['token_type_embeddings'](token_type_ids)
        
        embeddings = word_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        if self.config.track_internal_states:
            print(f"\nEmbeddings:")
            print(f"  Word embeddings: {word_embeddings.shape}, mean: {word_embeddings.mean().item():.4f}")
            print(f"  Position embeddings: {position_embeddings.shape}, mean: {position_embeddings.mean().item():.4f}")
            print(f"  Combined embeddings: {embeddings.shape}, mean: {embeddings.mean().item():.4f}")
        
        # 2. Attention mask preparation
        if attention_mask is not None:
            attention_mask = self._prepare_attention_mask(attention_mask)
        
        # 3. Transformer layers
        hidden_states = embeddings
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
            )
            
            hidden_states = layer_outputs[0]
            
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
            
            if self.config.track_internal_states:
                print(f"\nAfter layer {i}: mean = {hidden_states.mean().item():.4f}, std = {hidden_states.std().item():.4f}")
        
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)
        
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


def create_tracked_transformer(
    vocab_size: int = 30522,
    hidden_size: int = 512,
    num_layers: int = 6,
    num_heads: int = 8,
    track_internal_states: bool = True
) -> TrackedTransformerModel:
    """Create a tracked transformer model"""
    
    config = TrackedTransformerConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        intermediate_size=hidden_size * 4,
        track_internal_states=track_internal_states
    )
    
    model = TrackedTransformerModel(config)
    return model


def test_tracked_transformer():
    """Test the tracked transformer model"""
    print("=== Tracked Transformer Test ===")
    
    # Create model
    model = create_tracked_transformer(
        vocab_size=1000,
        hidden_size=256,
        num_layers=2,
        num_heads=8,
        track_internal_states=True
    )
    
    # Test data
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(1, 1000, (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Input shape: {input_ids.shape}")
    
    # Forward pass
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_attentions=True,
        output_hidden_states=True
    )
    
    print(f"\nOutput shapes:")
    print(f"Last hidden state: {outputs.last_hidden_state.shape}")
    print(f"Hidden states: {len(outputs.hidden_states)} layers")
    print(f"Attentions: {len(outputs.attentions)} layers")
    
    # Check attention patterns
    first_layer_attention = outputs.attentions[0]  # (batch_size, num_heads, seq_len, seq_len)
    print(f"First layer attention shape: {first_layer_attention.shape}")
    print(f"Attention sum per position: {first_layer_attention.sum(dim=-1).mean().item():.4f} (should be ~1.0)")


if __name__ == "__main__":
    test_tracked_transformer()