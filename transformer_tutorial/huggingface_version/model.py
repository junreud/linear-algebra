"""
Hugging Face 기반 Transformer 구현 with Internal Tracking

이 파일은 Hugging Face 라이브러리를 사용해서 Transformer 모델을 구현하는 파일입니다.
PyTorch 순정 구현과 달리, 실무에서 사용하는 방식으로 구현되어 있습니다.

주요 특징:
1. PreTrainedModel 상속 - Hugging Face의 표준 모델 인터페이스 사용
2. Hook 시스템 - 모델 내부 상태를 실시간으로 추적
3. 자동 토크나이저 호환 - 기존 Hugging Face 생태계와 호환
4. 실무 최적화 - Mixed Precision, Gradient Accumulation 등 지원
"""
import torch  # PyTorch 기본 라이브러리
import torch.nn as nn  # 신경망 레이어들 (Linear, Embedding, LayerNorm 등)
import torch.nn.functional as F  # 활성화 함수들 (ReLU, Softmax, CrossEntropy 등)

# Hugging Face Transformers 라이브러리에서 필요한 클래스들 가져오기
from transformers import (
    AutoTokenizer,    # 자동 토크나이저 - 텍스트를 숫자로 변환
    AutoModel,        # 자동 모델 로더 - 사전 훈련된 모델 불러오기
    AutoConfig,       # 자동 설정 로더 - 모델 설정 불러오기
    PreTrainedModel,  # 사전 훈련된 모델의 기본 클래스 - 우리가 상속받을 부모 클래스
    PretrainedConfig  # 사전 훈련된 설정의 기본 클래스 - 설정 클래스의 부모
)
from transformers.modeling_outputs import BaseModelOutput  # 모델 출력 형식 표준화
from typing import Optional, Tuple, Dict, List, Any  # 타입 힌트 - 코드 가독성과 디버깅을 위함
import math  # 수학 함수들 (sqrt, sin, cos 등)
import numpy as np  # 수치 계산 라이브러리


class TrackedTransformerConfig(PretrainedConfig):
    """
    Tracked Transformer를 위한 설정(Configuration) 클래스
    
    이 클래스는 모델의 하이퍼파라미터들을 저장합니다.
    Hugging Face의 PretrainedConfig를 상속받아서 표준 인터페이스를 따릅니다.
    
    왜 Config 클래스가 필요한가?
    1. 모델 구조 정의: 레이어 수, 숨김 차원 등을 한 곳에서 관리
    2. 재현성: 동일한 설정으로 모델을 다시 만들 수 있음
    3. 호환성: Hugging Face 생태계와 호환 (save/load 자동화)
    4. 유연성: 실험할 때 설정만 바꿔서 다른 모델 구조 테스트 가능
    """
    model_type = "tracked_transformer"  # 모델 타입 식별자 - Hugging Face에 등록할 때 사용
    
    def __init__(
        self,
        vocab_size: int = 30522,                    # 어휘 사전 크기 (토큰 개수)
        hidden_size: int = 512,                     # 숨김 차원 (d_model) - 임베딩 벡터의 크기
        num_hidden_layers: int = 6,                 # Transformer 레이어 개수 (Encoder + Decoder)
        num_attention_heads: int = 8,               # Multi-Head Attention의 헤드 개수
        intermediate_size: int = 2048,              # FFN의 중간 차원 크기 (보통 hidden_size * 4)
        hidden_dropout_prob: float = 0.1,          # 드롭아웃 확률 - 과적합 방지
        attention_probs_dropout_prob: float = 0.1,  # Attention 가중치에 적용할 드롭아웃
        max_position_embeddings: int = 512,         # 최대 시퀀스 길이 (위치 임베딩)
        type_vocab_size: int = 2,                   # 토큰 타입 개수 (BERT의 경우 문장 A/B 구분)
        initializer_range: float = 0.02,            # 가중치 초기화 범위 (정규분포의 표준편차)
        layer_norm_eps: float = 1e-12,              # LayerNorm의 epsilon 값 (0으로 나누는 것 방지)
        pad_token_id: int = 0,                      # 패딩 토큰 ID - 짧은 문장을 길게 만들 때 사용
        position_embedding_type: str = "absolute",  # 위치 임베딩 타입 (absolute/relative)
        track_internal_states: bool = True,         # 내부 상태 추적 여부 - 우리만의 커스텀 옵션
        **kwargs  # 추가적인 설정들을 받기 위한 매개변수
    ):
        # 부모 클래스 초기화 - Hugging Face 표준 설정 적용
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        
        # 각 설정값들을 인스턴스 변수로 저장
        # 이렇게 저장된 값들은 나중에 모델을 만들 때 사용됩니다
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob                    # 일반 드롭아웃 확률
        self.attention_probs_dropout_prob = attention_probs_dropout_prob  # Attention 드롭아웃 확률
        self.max_position_embeddings = max_position_embeddings            # 최대 위치 임베딩 길이
        self.type_vocab_size = type_vocab_size                            # 토큰 타입 어휘 크기
        self.initializer_range = initializer_range                        # 가중치 초기화 범위
        self.layer_norm_eps = layer_norm_eps                              # LayerNorm epsilon
        self.position_embedding_type = position_embedding_type            # 위치 임베딩 타입
        self.track_internal_states = track_internal_states                # 내부 상태 추적 여부


class TrackedAttention(nn.Module):
    """
    Hugging Face 스타일 Attention with 내부 상태 추적
    
    이 클래스는 Multi-Head Attention을 Hugging Face 방식으로 구현합니다.
    PyTorch 순정 구현과 다른 점:
    1. Hugging Face 표준 인터페이스 사용
    2. Hook 시스템으로 내부 상태 추적
    3. 실무 최적화 (더 빠른 연산, 메모리 효율성)
    4. 기존 사전 훈련된 모델과 호환 가능
    
    내부 동작:
    1. Linear 변환으로 Q, K, V 생성
    2. Multi-Head로 분할
    3. Scaled Dot-Product Attention 적용
    4. 헤드들을 다시 연결 (concatenate)
    5. 최종 Linear 변환 적용
    """
    
    def __init__(self, config: TrackedTransformerConfig, layer_idx: int = 0):
        super().__init__()
        
        # 숨김 차원이 헤드 수로 나누어 떨어지는지 확인
        # 예: hidden_size=512, num_heads=8 → head_size=64 (정확히 나누어 떨어짐)
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"숨김 차원 ({config.hidden_size})이 어텐션 헤드 수 ({config.num_attention_heads})로 "
                f"나누어 떨어지지 않습니다. Multi-Head Attention을 위해서는 나누어 떨어져야 합니다."
            )
        
        # Attention 관련 차원 계산 및 저장
        self.num_attention_heads = config.num_attention_heads              # 헤드 개수 (예: 8)
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)  # 헤드당 차원 (예: 64)
        self.all_head_size = self.num_attention_heads * self.attention_head_size  # 전체 헤드 차원 (예: 512)
        self.layer_idx = layer_idx  # 몇 번째 레이어인지 추적 (디버깅용)
        self.track_internal_states = config.track_internal_states  # 내부 상태 추적 여부
        
        # Query, Key, Value를 위한 Linear 변환 레이어들
        # 각각 hidden_size → all_head_size로 변환 (보통 같은 크기)
        self.query = nn.Linear(config.hidden_size, self.all_head_size)  # W_Q 가중치 행렬
        self.key = nn.Linear(config.hidden_size, self.all_head_size)    # W_K 가중치 행렬  
        self.value = nn.Linear(config.hidden_size, self.all_head_size)  # W_V 가중치 행렬
        
        # Attention 가중치에 적용할 드롭아웃 - 과적합 방지 및 일반화 성능 향상
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
        # 내부 상태 추적을 위한 딕셔너리 - 디버깅 및 분석용
        self.tracked_states = {}
        
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """
        Multi-Head Attention을 위해 텐서를 재배열하는 함수
        
        입력: (batch_size, seq_length, all_head_size)
        출력: (batch_size, num_heads, seq_length, head_size)
        
        왜 이런 변환이 필요한가?
        - Multi-Head Attention은 여러 개의 "헤드"가 병렬로 작동
        - 각 헤드는 독립적으로 attention을 계산
        - 따라서 헤드 차원을 분리해야 함
        
        예시: hidden_size=512, num_heads=8일 때
        (4, 20, 512) → (4, 20, 8, 64) → (4, 8, 20, 64)
        """
        # 마지막 차원을 (num_heads, head_size)로 분할
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        # 차원 순서 변경: 헤드 차원을 앞으로 이동
        return x.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, head_size)
    
    def forward(
        self,
        hidden_states: torch.Tensor,      # 입력 히든 상태 (이전 레이어의 출력)
        attention_mask: Optional[torch.Tensor] = None,  # 어텐션 마스크 (패딩 토큰 무시용)
        output_attentions: bool = False,  # 어텐션 가중치 출력 여부
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass with 내부 상태 추적
        
        이 함수가 실제 Multi-Head Attention 계산을 수행합니다.
        
        단계별 과정:
        1. Q, K, V 계산 (Linear 변환)
        2. Multi-Head로 분할
        3. Scaled Dot-Product Attention
        4. 헤드 결합 및 출력
        """
        batch_size, seq_length = hidden_states.shape[:2]  # 배치 크기와 시퀀스 길이 추출
        
        # 디버깅을 위한 상태 출력
        if self.track_internal_states:
            print(f"\n--- Layer {self.layer_idx} Attention ---")
            print(f"Input hidden_states: {hidden_states.shape}")  # 입력 크기 확인
            print(f"Input mean: {hidden_states.mean().item():.4f}, std: {hidden_states.std().item():.4f}")
        
        # 1. Linear transformations for Q, K, V
        # 입력 히든 상태를 Query, Key, Value로 변환
        # 각각 다른 가중치 행렬(W_Q, W_K, W_V)을 사용해서 서로 다른 관점에서 정보를 추출
        mixed_query_layer = self.query(hidden_states)   # Q = hidden_states × W_Q
        mixed_key_layer = self.key(hidden_states)       # K = hidden_states × W_K  
        mixed_value_layer = self.value(hidden_states)   # V = hidden_states × W_V
        
        if self.track_internal_states:
            print(f"Q linear output: {mixed_query_layer.shape}, mean: {mixed_query_layer.mean().item():.4f}")
            print(f"K linear output: {mixed_key_layer.shape}, mean: {mixed_key_layer.mean().item():.4f}")
            print(f"V linear output: {mixed_value_layer.shape}, mean: {mixed_value_layer.mean().item():.4f}")
        
        # 2. Reshape for multi-head attention
        # Q, K, V를 여러 헤드로 분할 - 각 헤드가 독립적으로 attention 계산
        query_layer = self.transpose_for_scores(mixed_query_layer)   # (batch, heads, seq, head_size)
        key_layer = self.transpose_for_scores(mixed_key_layer)       # (batch, heads, seq, head_size)
        value_layer = self.transpose_for_scores(mixed_value_layer)   # (batch, heads, seq, head_size)
        
        if self.track_internal_states:
            print(f"Q reshaped: {query_layer.shape}")
            print(f"K reshaped: {key_layer.shape}")
            print(f"V reshaped: {value_layer.shape}")
        
        # 3. Compute attention scores (Q × K^T)
        # 각 Query가 모든 Key와 얼마나 유사한지 계산 - 이것이 attention의 핵심!
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        
        # 4. Scale by sqrt(d_k) - "Scaled" Dot-Product Attention
        # 왜 스케일링? 큰 차원에서는 내적 값이 너무 커져서 softmax가 극단적으로 치우침
        # sqrt(head_size)로 나누어 적절한 크기로 조정
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        if self.track_internal_states:
            print(f"Attention scores: {attention_scores.shape}")
            print(f"Score range: [{attention_scores.min().item():.4f}, {attention_scores.max().item():.4f}]")
        
        # 5. Apply attention mask (패딩 토큰 무시)
        # 마스크는 무시할 위치를 -inf로 만들어서 softmax 후에 0이 되도록 함
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
            if self.track_internal_states:
                print(f"After masking - range: [{attention_scores.min().item():.4f}, {attention_scores.max().item():.4f}]")
        
        # 6. Attention probabilities (softmax)
        # Attention score를 확률로 변환 - 모든 위치에 대한 확률의 합이 1이 됨
        attention_probs = F.softmax(attention_scores, dim=-1)
        
        # 드롭아웃 적용 - 일부 attention 연결을 랜덤하게 끊어서 과적합 방지
        attention_probs = self.dropout(attention_probs)
        
        if self.track_internal_states:
            print(f"Attention probs: {attention_probs.shape}")
            print(f"Prob range: [{attention_probs.min().item():.4f}, {attention_probs.max().item():.4f}]")
            print(f"Prob sum (should be ~1.0): {attention_probs.sum(dim=-1).mean().item():.4f}")
        
        # 7. Apply attention to values (가중 평균)
        # Attention 확률을 사용해서 Value들의 가중 평균을 계산
        # 높은 attention을 받은 Value가 더 많이 반영됨
        context_layer = torch.matmul(attention_probs, value_layer)
        
        if self.track_internal_states:
            print(f"Context layer: {context_layer.shape}")
            print(f"Context mean: {context_layer.mean().item():.4f}, std: {context_layer.std().item():.4f}")
        
        # 8. Reshape back to original format
        # Multi-Head를 다시 연결 (concatenate)
        # (batch, heads, seq, head_size) → (batch, seq, heads, head_size) → (batch, seq, all_head_size)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        
        # 내부 상태 저장 (디버깅 및 분석용)
        if self.track_internal_states:
            self.tracked_states = {
                'input_hidden_states': hidden_states.clone().detach(),  # 입력 상태
                'query': query_layer.clone().detach(),                  # Query 텐서
                'key': key_layer.clone().detach(),                      # Key 텐서
                'value': value_layer.clone().detach(),                  # Value 텐서
                'attention_scores': attention_scores.clone().detach(),  # Attention 스코어
                'attention_probs': attention_probs.clone().detach(),    # Attention 확률
                'context_layer': context_layer.clone().detach(),       # 최종 출력
            }
        
        # 출력 형태 결정: attention 가중치 포함 여부
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class TrackedSelfOutput(nn.Module):
    """
    Self Attention의 출력 처리 레이어 with 추적 기능
    
    이 클래스는 Attention의 출력을 받아서 다음 작업들을 수행합니다:
    1. Linear 변환 (W_O 행렬)
    2. Dropout 적용
    3. Residual Connection (잔차 연결)
    4. Layer Normalization
    
    왜 이런 구조인가?
    - Residual Connection: 기울기 소실 문제 해결, 깊은 네트워크 훈련 가능
    - Layer Normalization: 훈련 안정성 향상, 빠른 수렴
    - Dropout: 과적합 방지
    """
    
    def __init__(self, config: TrackedTransformerConfig, layer_idx: int = 0):
        super().__init__()
        # W_O: Attention 출력을 다시 hidden_size로 변환하는 Linear 레이어
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        
        # Layer Normalization: 각 레이어의 출력을 정규화
        # eps: 0으로 나누기 방지를 위한 작은 값
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Dropout: 랜덤하게 일부 뉴런을 0으로 만들어 과적합 방지
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.layer_idx = layer_idx  # 디버깅용 레이어 인덱스
        self.track_internal_states = config.track_internal_states
        
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward with residual connection and layer norm
        
        Args:
            hidden_states: Attention의 출력
            input_tensor: Attention의 입력 (잔차 연결용)
            
        Returns:
            정규화된 최종 출력
            
        수식: LayerNorm(input_tensor + Dropout(Dense(hidden_states)))
        """
        if self.track_internal_states:
            print(f"\n--- Layer {self.layer_idx} Self Output ---")
            print(f"Before dense: {hidden_states.shape}, mean: {hidden_states.mean().item():.4f}")
        
        # Linear 변환 (W_O)
        hidden_states = self.dense(hidden_states)
        # Dropout 적용
        hidden_states = self.dropout(hidden_states)
        
        if self.track_internal_states:
            print(f"After dense+dropout: mean: {hidden_states.mean().item():.4f}")
            print(f"Residual input: mean: {input_tensor.mean().item():.4f}")
        
        # Residual Connection + Layer Normalization
        # 이것이 Transformer의 핵심! 입력을 출력에 더해서 정보 보존
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        
        if self.track_internal_states:
            print(f"After LayerNorm: mean: {hidden_states.mean().item():.4f}")
        
        return hidden_states


class TrackedIntermediate(nn.Module):
    """
    Feed-Forward Network의 중간 레이어 with 추적 기능
    
    이것은 FFN의 첫 번째 Linear 변환입니다.
    hidden_size → intermediate_size로 차원을 확장합니다.
    
    왜 차원을 확장하는가?
    1. 표현력 증가: 더 많은 파라미터로 복잡한 패턴 학습
    2. 비선형성: ReLU/GELU 활성화 함수의 효과 극대화
    3. 정보 병목 방지: 충분한 용량으로 정보 손실 최소화
    
    일반적으로 intermediate_size = hidden_size * 4
    예: hidden_size=512 → intermediate_size=2048
    """
    
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
        
        # nn.Embedding 클래스의 __call__ 메서드가 벡터화를 함.
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
        
        # 2. Attention mask preparation, 패딩 토큰 무시
        if attention_mask is not None:
            attention_mask = self._prepare_attention_mask(attention_mask)
        
        # 3. Transformer layers
        hidden_states = embeddings          # 임베딩을 첫 번째 입력으로 설정
        all_hidden_states = () if output_hidden_states else None    # 각 레이어 출력 저장용
        all_attentions = () if output_attentions else None          # 각 레이어 attention 저장용
        
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
            last_hidden_state=hidden_states,      # 최종 레이어 출력
            hidden_states=all_hidden_states,      # 모든 레이어 출력 (선택적)
            attentions=all_attentions,            # 모든 attention 가중치 (선택적)
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