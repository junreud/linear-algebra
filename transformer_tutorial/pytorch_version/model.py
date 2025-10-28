"""
Complete Transformer Model - "Attention is All You Need" 구현
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
import math

from pytorch_version.attention import MultiHeadAttention
from pytorch_version.layers import (
    PositionalEncoding, 
    EncoderLayer, 
    DecoderLayer,
    create_look_ahead_mask,
    create_padding_mask
)


class Transformer(nn.Module):
    """
    Complete Transformer Model
    
    "Attention is All You Need" 논문의 Transformer 구현
    모든 중간값과 데이터 흐름을 추적할 수 있도록 구현
    """
    
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        d_ff: int = 2048,
        max_seq_length: int = 5000,
        dropout: float = 0.1,
        pad_idx: int = 0,
        debug: bool = False
    ):
        super().__init__()
        
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.debug = debug
        
        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)
        
        # Positional Encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length, debug)
        
        # Encoder Stack
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout, debug)
            for _ in range(num_encoder_layers)
        ])
        
        # Decoder Stack
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout, debug)
            for _ in range(num_decoder_layers)
        ])
        
        # Final linear layer
        self.linear = nn.Linear(d_model, tgt_vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
        # Xavier/Glorot 초기화
        self._init_weights()
        
        # 디버깅 정보 저장
        self.debug_info = {}
        
        if debug:
            print(f"Transformer initialized:")
            print(f"  - src_vocab: {src_vocab_size}, tgt_vocab: {tgt_vocab_size}")
            print(f"  - d_model: {d_model}, heads: {num_heads}")
            print(f"  - encoder_layers: {num_encoder_layers}, decoder_layers: {num_decoder_layers}")
            print(f"  - d_ff: {d_ff}, max_seq_len: {max_seq_length}")
    
    def _init_weights(self):
        """Xavier/Glorot 초기화"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Forward pass with detailed tracking
        
        Args:
            src: Source sequences (batch_size, src_seq_len)
            tgt: Target sequences (batch_size, tgt_seq_len)
            src_mask: Source padding mask
            tgt_mask: Target look-ahead + padding mask
            return_attention: attention weight 반환 여부
            
        Returns:
            logits: (batch_size, tgt_seq_len, tgt_vocab_size)
        """
        if self.debug:
            print(f"\n{'='*50}")
            print(f"TRANSFORMER FORWARD PASS")
            print(f"{'='*50}")
            print(f"Source shape: {src.shape}")
            print(f"Target shape: {tgt.shape}")
        
        # 1. Encoder
        encoder_output = self.encode(src, src_mask)
        
        if self.debug:
            print(f"\nEncoder output shape: {encoder_output.shape}")
            print(f"Encoder output - mean: {encoder_output.mean().item():.4f}, std: {encoder_output.std().item():.4f}")
        
        # 2. Decoder
        decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)
        
        if self.debug:
            print(f"\nDecoder output shape: {decoder_output.shape}")
            print(f"Decoder output - mean: {decoder_output.mean().item():.4f}, std: {decoder_output.std().item():.4f}")
        
        # 3. Final Linear Projection
        logits = self.linear(decoder_output)
        
        if self.debug:
            print(f"\nFinal logits shape: {logits.shape}")
            print(f"Logits - mean: {logits.mean().item():.4f}, std: {logits.std().item():.4f}")
            print(f"Logits range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")
        
        return logits
    
    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encoder forward pass
        
        Args:
            src: (batch_size, src_seq_len)
            src_mask: (batch_size, 1, 1, src_seq_len)
            
        Returns:
            encoder_output: (batch_size, src_seq_len, d_model)
        """
        if self.debug:
            print(f"\n--- ENCODER ---")
            print(f"Source tokens: {src.shape}")
        
        # 1. Source Embedding + Positional Encoding
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)  # Scaling
        src_emb = self.pos_encoding(src_emb)
        src_emb = self.dropout(src_emb)
        
        if self.debug:
            print(f"1. Source embedding + PE: {src_emb.shape}")
            print(f"   Embedding mean: {src_emb.mean().item():.4f}, std: {src_emb.std().item():.4f}")
        
        # 2. Encoder layers
        encoder_output = src_emb
        for i, layer in enumerate(self.encoder_layers):
            if self.debug:
                print(f"\n--- Encoder Layer {i+1} ---")
            encoder_output = layer(encoder_output, src_mask)
            
            if self.debug:
                print(f"Layer {i+1} output - mean: {encoder_output.mean().item():.4f}, std: {encoder_output.std().item():.4f}")
        
        return encoder_output
    
    def decode(
        self,
        tgt: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Decoder forward pass
        
        Args:
            tgt: (batch_size, tgt_seq_len)
            encoder_output: (batch_size, src_seq_len, d_model)
            src_mask: Source padding mask
            tgt_mask: Target look-ahead + padding mask
            
        Returns:
            decoder_output: (batch_size, tgt_seq_len, d_model)
        """
        if self.debug:
            print(f"\n--- DECODER ---")
            print(f"Target tokens: {tgt.shape}")
            print(f"Encoder output: {encoder_output.shape}")
        
        # 1. Target Embedding + Positional Encoding
        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)  # Scaling
        tgt_emb = self.pos_encoding(tgt_emb)
        tgt_emb = self.dropout(tgt_emb)
        
        if self.debug:
            print(f"1. Target embedding + PE: {tgt_emb.shape}")
            print(f"   Embedding mean: {tgt_emb.mean().item():.4f}, std: {tgt_emb.std().item():.4f}")
        
        # 2. Decoder layers
        decoder_output = tgt_emb
        for i, layer in enumerate(self.decoder_layers):
            if self.debug:
                print(f"\n--- Decoder Layer {i+1} ---")
            decoder_output = layer(decoder_output, encoder_output, tgt_mask, src_mask)
            
            if self.debug:
                print(f"Layer {i+1} output - mean: {decoder_output.mean().item():.4f}, std: {decoder_output.std().item():.4f}")
        
        return decoder_output
    
    def generate_masks(self, src: torch.Tensor, tgt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        마스크 생성
        
        Args:
            src: (batch_size, src_seq_len)
            tgt: (batch_size, tgt_seq_len)
            
        Returns:
            src_mask: Source padding mask
            tgt_mask: Target look-ahead + padding mask
        """
        batch_size, src_len = src.shape
        _, tgt_len = tgt.shape
        
        # Source padding mask
        src_mask = create_padding_mask(src, self.pad_idx)  # (batch_size, 1, 1, src_len)
        
        # Target padding mask
        tgt_padding_mask = create_padding_mask(tgt, self.pad_idx)  # (batch_size, 1, 1, tgt_len)
        
        # Target look-ahead mask
        tgt_look_ahead_mask = create_look_ahead_mask(tgt_len).to(tgt.device)  # (tgt_len, tgt_len)
        tgt_look_ahead_mask = tgt_look_ahead_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, tgt_len, tgt_len)
        
        # Combine target masks
        tgt_mask = tgt_padding_mask | tgt_look_ahead_mask
        
        if self.debug:
            print(f"\nMasks generated:")
            print(f"  Source mask: {src_mask.shape}")
            print(f"  Target mask: {tgt_mask.shape}")
            print(f"  Target padding mask sum: {tgt_padding_mask.sum().item()}")
            print(f"  Target look-ahead mask sum: {tgt_look_ahead_mask.sum().item()}")
        
        return src_mask, tgt_mask
    
    def greedy_decode(
        self,
        src: torch.Tensor,
        max_length: int,
        start_token: int,
        end_token: int
    ) -> torch.Tensor:
        """
        Greedy decoding for inference
        
        Args:
            src: Source sequence (1, src_seq_len)
            max_length: Maximum generation length
            start_token: Start token ID
            end_token: End token ID
            
        Returns:
            generated: Generated sequence (1, gen_len)
        """
        device = src.device
        batch_size = src.size(0)
        
        # Encode source
        src_mask, _ = self.generate_masks(src, torch.zeros(batch_size, 1, dtype=torch.long, device=device))
        encoder_output = self.encode(src, src_mask)
        
        # Initialize target with start token
        tgt = torch.tensor([[start_token]], device=device)
        
        for _ in range(max_length):
            # Generate masks
            _, tgt_mask = self.generate_masks(src, tgt)
            
            # Decode
            decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)
            
            # Get next token logits
            next_token_logits = self.linear(decoder_output[:, -1, :])  # (1, vocab_size)
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)  # (1, 1)
            
            # Append to target
            tgt = torch.cat([tgt, next_token], dim=1)
            
            # Stop if end token is generated
            if next_token.item() == end_token:
                break
        
        return tgt


def create_transformer_model(
    src_vocab_size: int = 1000,
    tgt_vocab_size: int = 1000,
    d_model: int = 512,
    debug: bool = True
) -> Transformer:
    """Transformer 모델 생성 헬퍼 함수"""
    return Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ff=2048,
        max_seq_length=5000,
        dropout=0.1,
        debug=debug
    )


def test_transformer():
    """Transformer 전체 테스트"""
    print("=== Complete Transformer Test ===")
    
    # 설정
    src_vocab_size, tgt_vocab_size = 1000, 1000
    batch_size, src_len, tgt_len = 2, 10, 8
    
    # 모델 생성
    model = create_transformer_model(src_vocab_size, tgt_vocab_size, debug=True)
    
    # 테스트 데이터
    src = torch.randint(1, src_vocab_size, (batch_size, src_len))
    tgt = torch.randint(1, tgt_vocab_size, (batch_size, tgt_len))
    
    # 마스크 생성
    src_mask, tgt_mask = model.generate_masks(src, tgt)
    
    print(f"\nTest data:")
    print(f"Source: {src.shape}")
    print(f"Target: {tgt.shape}")
    print(f"Source mask: {src_mask.shape}")
    print(f"Target mask: {tgt_mask.shape}")
    
    # Forward pass
    logits = model(src, tgt, src_mask, tgt_mask)
    
    print(f"\nResults:")
    print(f"Output logits: {logits.shape}")
    print(f"Expected shape: ({batch_size}, {tgt_len}, {tgt_vocab_size})")
    
    # Loss 계산 예시
    # Shift target for language modeling
    targets = tgt[:, 1:].contiguous().view(-1)  # Remove first token, flatten
    predictions = logits[:, :-1, :].contiguous().view(-1, tgt_vocab_size)  # Remove last prediction
    
    loss = F.cross_entropy(predictions, targets, ignore_index=0)
    print(f"Cross-entropy loss: {loss.item():.4f}")
    
    # Greedy decoding 테스트
    print(f"\n=== Greedy Decoding Test ===")
    test_src = torch.randint(1, src_vocab_size, (1, 5))  # Single sequence
    generated = model.greedy_decode(test_src, max_length=10, start_token=1, end_token=2)
    print(f"Source: {test_src}")
    print(f"Generated: {generated}")


if __name__ == "__main__":
    test_transformer()