"""
Transformer 레이어들 - Encoder, Decoder 구현
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from pytorch_version.attention import MultiHeadAttention


class PositionalEncoding(nn.Module):
    """
    Positional Encoding - sin/cos를 사용한 위치 인코딩
    """
    
    def __init__(self, d_model: int, max_seq_length: int = 5000, debug: bool = False):
        super().__init__()
        self.d_model = d_model
        self.debug = debug
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        # 논문의 공식: PE(pos, 2i) = sin(pos/10000^(2i/d_model))
        #             PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)  # 짝수 인덱스
        pe[:, 1::2] = torch.cos(position * div_term)  # 홀수 인덱스
        
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_seq_length, 1, d_model)
        self.register_buffer('pe', pe)
        
        if debug:
            print(f"Positional Encoding initialized: {pe.shape}")
            print(f"PE range: [{pe.min().item():.4f}, {pe.max().item():.4f}]")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            x + positional encoding
        """
        if self.debug:
            print(f"\nPositional Encoding - Input: {x.shape}")
            print(f"Adding PE: {self.pe[:x.size(1), :].shape}")
        
        return x + self.pe[:x.size(1), :].transpose(0, 1)


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network
    FFN(x) = max(0, xW1 + b1)W2 + b2
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, debug: bool = False):
        super().__init__()
        self.debug = debug
        self.d_model = d_model
        self.d_ff = d_ff
        
        # 두 개의 선형 변환
        self.linear1 = nn.Linear(d_model, d_ff)  # W1
        self.linear2 = nn.Linear(d_ff, d_model)  # W2
        self.dropout = nn.Dropout(dropout)
        
        if debug:
            print(f"FFN initialized: {d_model} -> {d_ff} -> {d_model}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            FFN output: (batch_size, seq_len, d_model)
        """
        if self.debug:
            print(f"\n--- Feed-Forward Network ---")
            print(f"Input: {x.shape}, mean: {x.mean().item():.4f}, std: {x.std().item():.4f}")
        
        # 1. 첫 번째 선형 변환 + ReLU
        hidden = self.linear1(x)  # (batch_size, seq_len, d_ff)
        
        if self.debug:
            print(f"1. After Linear1: {hidden.shape}")
            print(f"   Before ReLU - mean: {hidden.mean().item():.4f}, std: {hidden.std().item():.4f}")
            print(f"   Before ReLU - range: [{hidden.min().item():.4f}, {hidden.max().item():.4f}]")
        
        # ReLU 활성화 (비선형 변환!)
        hidden = F.relu(hidden)
        
        if self.debug:
            print(f"2. After ReLU (Non-linear activation):")
            print(f"   After ReLU - mean: {hidden.mean().item():.4f}, std: {hidden.std().item():.4f}")
            print(f"   After ReLU - range: [{hidden.min().item():.4f}, {hidden.max().item():.4f}]")
            zeros_ratio = (hidden == 0).float().mean().item()
            print(f"   ReLU sparsity: {zeros_ratio:.2%} zeros")
        
        # Dropout
        hidden = self.dropout(hidden)
        
        # 2. 두 번째 선형 변환
        output = self.linear2(hidden)  # (batch_size, seq_len, d_model)
        
        if self.debug:
            print(f"3. After Linear2 (Final): {output.shape}")
            print(f"   Output - mean: {output.mean().item():.4f}, std: {output.std().item():.4f}")
        
        return output


class EncoderLayer(nn.Module):
    """
    Transformer Encoder Layer
    
    구조:
    1. Multi-Head Self-Attention
    2. Add & Norm (Residual + Layer Normalization)
    3. Feed-Forward Network
    4. Add & Norm
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1, debug: bool = False):
        super().__init__()
        self.debug = debug
        
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout, debug)
        self.ffn = FeedForward(d_model, d_ff, dropout, debug)
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        if debug:
            print(f"EncoderLayer initialized - d_model: {d_model}, heads: {num_heads}, d_ff: {d_ff}")
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
            mask: (batch_size, seq_len, seq_len) or None
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        if self.debug:
            print(f"\n=== Encoder Layer Forward ===")
            print(f"Input: {x.shape}, mean: {x.mean().item():.4f}, std: {x.std().item():.4f}")
        
        # 1. Multi-Head Self-Attention
        attn_output, _ = self.self_attention(x, x, x, mask)
        
        if self.debug:
            print(f"1. After Self-Attention: {attn_output.shape}")
            print(f"   Attention output - mean: {attn_output.mean().item():.4f}, std: {attn_output.std().item():.4f}")
        
        # 2. Add & Norm (첫 번째 residual connection)
        x_residual = self.norm1(x + self.dropout(attn_output))
        
        if self.debug:
            print(f"2. After Add & Norm 1:")
            print(f"   Residual mean: {x_residual.mean().item():.4f}, std: {x_residual.std().item():.4f}")
        
        # 3. Feed-Forward Network
        ffn_output = self.ffn(x_residual)
        
        if self.debug:
            print(f"3. After FFN: {ffn_output.shape}")
            print(f"   FFN output - mean: {ffn_output.mean().item():.4f}, std: {ffn_output.std().item():.4f}")
        
        # 4. Add & Norm (두 번째 residual connection)
        output = self.norm2(x_residual + self.dropout(ffn_output))
        
        if self.debug:
            print(f"4. Final Output (after Add & Norm 2):")
            print(f"   Final mean: {output.mean().item():.4f}, std: {output.std().item():.4f}")
        
        return output


class DecoderLayer(nn.Module):
    """
    Transformer Decoder Layer
    
    구조:
    1. Masked Multi-Head Self-Attention (look-ahead mask)
    2. Add & Norm
    3. Multi-Head Cross-Attention (with encoder output)
    4. Add & Norm
    5. Feed-Forward Network
    6. Add & Norm
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1, debug: bool = False):
        super().__init__()
        self.debug = debug
        
        # Self-attention (masked)
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout, debug)
        # Cross-attention (with encoder)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout, debug)
        # Feed-forward
        self.ffn = FeedForward(d_model, d_ff, dropout, debug)
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        if debug:
            print(f"DecoderLayer initialized - d_model: {d_model}, heads: {num_heads}, d_ff: {d_ff}")
    
    def forward(
        self, 
        x: torch.Tensor, 
        encoder_output: torch.Tensor,
        self_mask: Optional[torch.Tensor] = None,
        cross_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Decoder input (batch_size, tgt_seq_len, d_model)
            encoder_output: Encoder output (batch_size, src_seq_len, d_model)
            self_mask: Look-ahead mask for self-attention
            cross_mask: Padding mask for cross-attention
        Returns:
            output: (batch_size, tgt_seq_len, d_model)
        """
        if self.debug:
            print(f"\n=== Decoder Layer Forward ===")
            print(f"Decoder input: {x.shape}")
            print(f"Encoder output: {encoder_output.shape}")
        
        # 1. Masked Self-Attention
        self_attn_output, _ = self.self_attention(x, x, x, self_mask)
        x = self.norm1(x + self.dropout(self_attn_output))
        
        if self.debug:
            print(f"1. After Masked Self-Attention + Add&Norm:")
            print(f"   Output mean: {x.mean().item():.4f}, std: {x.std().item():.4f}")
        
        # 2. Cross-Attention with Encoder
        cross_attn_output, _ = self.cross_attention(
            x,  # Query from decoder
            encoder_output,  # Key from encoder
            encoder_output,  # Value from encoder
            cross_mask
        )
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        if self.debug:
            print(f"2. After Cross-Attention + Add&Norm:")
            print(f"   Output mean: {x.mean().item():.4f}, std: {x.std().item():.4f}")
        
        # 3. Feed-Forward Network
        ffn_output = self.ffn(x)
        output = self.norm3(x + self.dropout(ffn_output))
        
        if self.debug:
            print(f"3. After FFN + Add&Norm (Final):")
            print(f"   Final output mean: {output.mean().item():.4f}, std: {output.std().item():.4f}")
        
        return output


def create_look_ahead_mask(size: int) -> torch.Tensor:
    """
    Look-ahead mask 생성 (디코더의 self-attention용)
    
    Args:
        size: 시퀀스 길이
        
    Returns:
        mask: (size, size) lower triangular matrix
    """
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
    return mask  # True인 부분이 마스킹됨


def create_padding_mask(seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """
    Padding mask 생성
    
    Args:
        seq: (batch_size, seq_len)
        pad_idx: 패딩 토큰 인덱스
        
    Returns:
        mask: (batch_size, 1, 1, seq_len)
    """
    mask = (seq == pad_idx)
    return mask.unsqueeze(1).unsqueeze(2)  # Broadcasting을 위한 차원 추가


def test_encoder_layer():
    """Encoder Layer 테스트"""
    print("=== Encoder Layer Test ===")
    
    batch_size, seq_len, d_model = 2, 10, 512
    num_heads, d_ff = 8, 2048
    
    encoder = EncoderLayer(d_model, num_heads, d_ff, debug=True)
    x = torch.randn(batch_size, seq_len, d_model)
    
    output = encoder(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")


def test_decoder_layer():
    """Decoder Layer 테스트"""
    print("\n=== Decoder Layer Test ===")
    
    batch_size, src_len, tgt_len, d_model = 2, 10, 8, 512
    num_heads, d_ff = 8, 2048
    
    decoder = DecoderLayer(d_model, num_heads, d_ff, debug=True)
    
    encoder_output = torch.randn(batch_size, src_len, d_model)
    decoder_input = torch.randn(batch_size, tgt_len, d_model)
    
    # Look-ahead mask 생성
    look_ahead_mask = create_look_ahead_mask(tgt_len).unsqueeze(0).unsqueeze(0)
    look_ahead_mask = look_ahead_mask.expand(batch_size, 1, tgt_len, tgt_len)
    
    output = decoder(decoder_input, encoder_output, look_ahead_mask)
    print(f"Decoder input shape: {decoder_input.shape}")
    print(f"Encoder output shape: {encoder_output.shape}")
    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    test_encoder_layer()
    test_decoder_layer()