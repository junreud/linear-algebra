"""
Multi-Head Attention 구현 - QKV 변화와 데이터 흐름 추적
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, List


class MultiHeadAttention(nn.Module):
    """
    Attention is All You Need의 Multi-Head Attention 구현
    
    QKV 변화, attention weight, 데이터 흐름을 모두 추적할 수 있도록 구현
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, debug: bool = False):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 각 헤드의 차원
        self.debug = debug
        
        # Linear transformations for Q, K, V
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)  # Output projection
        
        self.dropout = nn.Dropout(dropout)
        
        # 디버깅을 위한 중간값 저장
        self.debug_info = {}
        
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with detailed tracking
        
        Args:
            query: (batch_size, seq_len_q, d_model)
            key: (batch_size, seq_len_k, d_model)
            value: (batch_size, seq_len_v, d_model)
            mask: (batch_size, seq_len_q, seq_len_k) or None
            return_attention: 어텐션 가중치 반환 여부
            
        Returns:
            output: (batch_size, seq_len_q, d_model)
            attention_weights: (batch_size, num_heads, seq_len_q, seq_len_k) if return_attention
        """
        batch_size, seq_len_q, d_model = query.shape
        seq_len_k = key.shape[1]
        seq_len_v = value.shape[1]
        
        if self.debug:
            print(f"\n=== Multi-Head Attention Forward Pass ===")
            print(f"Input shapes - Q: {query.shape}, K: {key.shape}, V: {value.shape}")
            print(f"Model config - d_model: {self.d_model}, num_heads: {self.num_heads}, d_k: {self.d_k}")
        
        # 1. Linear transformations Q, K, V
        Q = self.w_q(query)  # (batch_size, seq_len_q, d_model)
        K = self.w_k(key)    # (batch_size, seq_len_k, d_model)
        V = self.w_v(value)  # (batch_size, seq_len_v, d_model)
        
        if self.debug:
            print(f"\n1. Linear Transformations:")
            print(f"Q after W_q: {Q.shape}, mean: {Q.mean().item():.4f}, std: {Q.std().item():.4f}")
            print(f"K after W_k: {K.shape}, mean: {K.mean().item():.4f}, std: {K.std().item():.4f}")
            print(f"V after W_v: {V.shape}, mean: {V.mean().item():.4f}, std: {V.std().item():.4f}")
        
        # 2. Reshape for multi-head: (batch_size, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, seq_len_q, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len_v, self.num_heads, self.d_k).transpose(1, 2)
        
        if self.debug:
            print(f"\n2. Reshape for Multi-Head:")
            print(f"Q reshaped: {Q.shape}")
            print(f"K reshaped: {K.shape}")
            print(f"V reshaped: {V.shape}")
        
        # 3. Scaled Dot-Product Attention
        attention_output, attention_weights = self.scaled_dot_product_attention(
            Q, K, V, mask, self.debug
        )
        
        if self.debug:
            print(f"\n3. Attention Output: {attention_output.shape}")
            print(f"Attention Weights: {attention_weights.shape}")
            print(f"Attention weight range: [{attention_weights.min().item():.4f}, {attention_weights.max().item():.4f}]")
        
        # 4. Concatenate heads: (batch_size, seq_len_q, d_model)
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, self.d_model
        )
        
        if self.debug:
            print(f"\n4. Concatenated heads: {attention_output.shape}")
            print(f"Before W_o - mean: {attention_output.mean().item():.4f}, std: {attention_output.std().item():.4f}")
        
        # 5. Final linear transformation (W_O)
        output = self.w_o(attention_output)
        
        if self.debug:
            print(f"\n5. Final Output (after W_o):")
            print(f"Output shape: {output.shape}")
            print(f"Output mean: {output.mean().item():.4f}, std: {output.std().item():.4f}")
            
            # 중간값들을 저장 (시각화용)
            self.debug_info = {
                'input_query': query.clone(),
                'input_key': key.clone(),
                'input_value': value.clone(),
                'Q': Q.clone(),
                'K': K.clone(),
                'V': V.clone(),
                'attention_weights': attention_weights.clone(),
                'attention_output': attention_output.clone(),
                'final_output': output.clone()
            }
        
        if return_attention:
            return output, attention_weights
        return output, None
    
    def scaled_dot_product_attention(
        self, 
        Q: torch.Tensor, 
        K: torch.Tensor, 
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        debug: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Scaled Dot-Product Attention 상세 구현
        
        Args:
            Q, K, V: (batch_size, num_heads, seq_len, d_k)
            mask: (batch_size, 1, seq_len, seq_len) or None
            
        Returns:
            output: (batch_size, num_heads, seq_len, d_k)
            attention_weights: (batch_size, num_heads, seq_len, seq_len)
        """
        d_k = Q.size(-1)
        
        if debug:
            print(f"\n--- Scaled Dot-Product Attention ---")
            print(f"d_k (scaling factor): {d_k}, sqrt(d_k): {math.sqrt(d_k):.4f}")
        
        # 1. Q * K^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        
        if debug:
            print(f"1. Scores (Q*K^T/sqrt(d_k)): {scores.shape}")
            print(f"   Score range: [{scores.min().item():.4f}, {scores.max().item():.4f}]")
            print(f"   Score mean: {scores.mean().item():.4f}, std: {scores.std().item():.4f}")
        
        # 2. Apply mask if provided
        if mask is not None:
            if debug:
                print(f"2. Applying mask: {mask.shape}")
            scores = scores.masked_fill(mask == 0, -1e9)
            if debug:
                print(f"   After masking - range: [{scores.min().item():.4f}, {scores.max().item():.4f}]")
        
        # 3. Softmax
        attention_weights = F.softmax(scores, dim=-1)
        
        if debug:
            print(f"3. Attention weights (after softmax): {attention_weights.shape}")
            print(f"   Weight range: [{attention_weights.min().item():.4f}, {attention_weights.max().item():.4f}]")
            print(f"   Weight sum per head: {attention_weights.sum(dim=-1).mean().item():.4f} (should be ~1.0)")
        
        # 4. Dropout
        attention_weights = self.dropout(attention_weights)
        
        # 5. Apply attention to values
        output = torch.matmul(attention_weights, V)
        
        if debug:
            print(f"4. Final attention output: {output.shape}")
            print(f"   Output mean: {output.mean().item():.4f}, std: {output.std().item():.4f}")
        
        return output, attention_weights
    
    def get_debug_info(self) -> Dict:
        """디버깅 정보 반환"""
        return self.debug_info
    
    def visualize_attention(self, tokens: List[str], head_idx: int = 0, save_path: str = None):
        """
        특정 헤드의 attention weight를 시각화
        
        Args:
            tokens: 토큰 리스트
            head_idx: 시각화할 헤드 인덱스
            save_path: 저장 경로
        """
        if 'attention_weights' not in self.debug_info:
            print("No attention weights found. Run forward pass with debug=True first.")
            return
        
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # 첫 번째 배치, 특정 헤드의 attention weight 추출
        attention = self.debug_info['attention_weights'][0, head_idx].cpu().numpy()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            attention,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap='Blues',
            cbar=True,
            square=True
        )
        plt.title(f'Attention Weights - Head {head_idx}')
        plt.xlabel('Key (Attended to)')
        plt.ylabel('Query (Attending)')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def test_multihead_attention():
    """Multi-Head Attention 테스트 함수"""
    print("=== Multi-Head Attention Test ===")
    
    # 설정
    batch_size, seq_len, d_model = 2, 8, 512
    num_heads = 8
    
    # 모델 생성
    mha = MultiHeadAttention(d_model, num_heads, debug=True)
    
    # 테스트 데이터
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Forward pass
    output, attention_weights = mha(x, x, x, return_attention=True)
    
    print(f"\n=== Test Results ===")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    
    # 간단한 토큰으로 시각화 테스트
    tokens = [f"token_{i}" for i in range(seq_len)]
    mha.visualize_attention(tokens, head_idx=0)


if __name__ == "__main__":
    test_multihead_attention()