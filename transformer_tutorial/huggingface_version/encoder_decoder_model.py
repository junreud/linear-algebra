"""
BERT Encoder + Transformer Decoder 결합 모델

이 파일은 BERT의 인코더와 Transformer의 디코더를 결합하여
완전한 Encoder-Decoder 아키텍처를 구현합니다.

구조:
1. BERT-style Encoder: 입력 시퀀스를 인코딩
2. Transformer-style Decoder: 출력 시퀀스를 생성 (Cross-Attention 포함)

사용 사례:
- 기계 번역 (한국어 → 영어)
- 텍스트 요약
- 질문 답변 생성
- 대화 시스템
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqModelOutput
from typing import Optional, Tuple, Dict, List, Any
import math

# 기존 BERT 스타일 컴포넌트들 가져오기
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import (
    TrackedTransformerConfig, 
    TrackedAttention, 
    TrackedSelfOutput,
    TrackedIntermediate,
    TrackedOutput,
    TrackedTransformerLayer
)


class EncoderDecoderConfig(PretrainedConfig):
    """
    Encoder-Decoder 모델을 위한 설정 클래스
    
    BERT Encoder와 Transformer Decoder의 설정을 모두 포함합니다.
    """
    model_type = "encoder_decoder_transformer"
    
    def __init__(
        self,
        # 공통 설정
        vocab_size: int = 30522,
        hidden_size: int = 512,
        pad_token_id: int = 0,
        bos_token_id: int = 101,  # [CLS] token
        eos_token_id: int = 102,  # [SEP] token
        
        # Encoder 설정 (BERT 스타일)
        encoder_layers: int = 6,
        encoder_attention_heads: int = 8,
        encoder_intermediate_size: int = 2048,
        
        # Decoder 설정 (Transformer 스타일)
        decoder_layers: int = 6,
        decoder_attention_heads: int = 8,
        decoder_intermediate_size: int = 2048,
        
        # 공통 설정
        max_position_embeddings: int = 512,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        layer_norm_eps: float = 1e-12,
        initializer_range: float = 0.02,
        
        **kwargs
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs
        )
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        # Encoder 설정
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_intermediate_size = encoder_intermediate_size
        
        # Decoder 설정
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.decoder_intermediate_size = decoder_intermediate_size
        
        # 공통 설정
        self.max_position_embeddings = max_position_embeddings
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range


class CrossAttention(nn.Module):
    """
    Cross-Attention 레이어
    
    Decoder에서 Encoder의 출력에 어텐션을 적용합니다.
    Query는 Decoder에서, Key와 Value는 Encoder에서 가져옵니다.
    """
    
    def __init__(self, config: EncoderDecoderConfig):
        super().__init__()
        
        if config.hidden_size % config.decoder_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({config.hidden_size})가 "
                f"decoder_attention_heads ({config.decoder_attention_heads})로 "
                f"나누어떨어지지 않습니다."
            )
        
        self.num_attention_heads = config.decoder_attention_heads
        self.attention_head_size = config.hidden_size // config.decoder_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # Query는 decoder에서, Key/Value는 encoder에서
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Multi-head attention을 위한 텐서 형태 변환"""
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)  # (batch, heads, seq_len, head_size)
    
    def forward(
        self,
        decoder_hidden_states: torch.Tensor,  # Decoder의 hidden states (Query)
        encoder_hidden_states: torch.Tensor,  # Encoder의 hidden states (Key, Value)
        encoder_attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Cross-Attention 순전파
        
        Args:
            decoder_hidden_states: Decoder의 hidden states [batch, dec_len, hidden]
            encoder_hidden_states: Encoder의 hidden states [batch, enc_len, hidden]
            encoder_attention_mask: Encoder attention mask [batch, enc_len]
            
        Returns:
            context_layer: 어텐션 적용된 출력 [batch, dec_len, hidden]
            attention_probs: 어텐션 가중치 [batch, heads, dec_len, enc_len]
        """
        
        # Q는 decoder에서, K,V는 encoder에서 가져오기
        query_layer = self.transpose_for_scores(self.query(decoder_hidden_states))
        key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
        value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
        
        # Attention 점수 계산: Q × K^T
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Encoder attention mask 적용 (padding 토큰 무시)
        if encoder_attention_mask is not None:
            # Mask 확장: [batch, enc_len] → [batch, 1, 1, enc_len]
            extended_mask = encoder_attention_mask.unsqueeze(1).unsqueeze(1)
            extended_mask = (1.0 - extended_mask) * -10000.0
            attention_scores = attention_scores + extended_mask
        
        # Softmax로 확률 변환
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # 가중 평균으로 context 계산: Attention × V
        context_layer = torch.matmul(attention_probs, value_layer)
        
        # 다시 원래 형태로 변환: [batch, heads, dec_len, head_size] → [batch, dec_len, hidden]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        
        return context_layer, attention_probs


class DecoderLayer(nn.Module):
    """
    Transformer Decoder 레이어
    
    구조:
    1. Masked Self-Attention (이전 토큰들만 참조)
    2. Cross-Attention (Encoder 출력 참조)
    3. Feed-Forward Network
    
    각 서브레이어마다 Residual Connection + Layer Normalization 적용
    """
    
    def __init__(self, config: EncoderDecoderConfig):
        super().__init__()
        
        # TrackedAttention은 config.num_attention_heads를 사용하므로 설정 변환
        decoder_config = TrackedTransformerConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.decoder_layers,
            num_attention_heads=config.decoder_attention_heads,
            intermediate_size=config.decoder_intermediate_size,
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            max_position_embeddings=config.max_position_embeddings,
            layer_norm_eps=config.layer_norm_eps,
            initializer_range=config.initializer_range,
        )
        
        # 1. Masked Self-Attention
        self.self_attention = TrackedAttention(decoder_config)
        self.self_output = TrackedSelfOutput(decoder_config)
        
        # 2. Cross-Attention
        self.cross_attention = CrossAttention(config)
        self.cross_output = TrackedSelfOutput(decoder_config)
        
        # 3. Feed-Forward Network
        self.intermediate = TrackedIntermediate(decoder_config)
        self.output = TrackedOutput(decoder_config)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decoder 레이어 순전파
        
        Args:
            hidden_states: Decoder input [batch, dec_len, hidden]
            encoder_hidden_states: Encoder output [batch, enc_len, hidden]
            attention_mask: Decoder attention mask (causal mask)
            encoder_attention_mask: Encoder attention mask
            
        Returns:
            layer_output: 레이어 출력
            self_attention_probs: Self-attention 가중치
            cross_attention_probs: Cross-attention 가중치
        """
        
        # 1. Masked Self-Attention
        self_attention_outputs = self.self_attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask  # Causal mask로 미래 토큰 차단
        )
        attention_output = self.self_output(
            self_attention_outputs[0], 
            hidden_states
        )
        
        # 2. Cross-Attention
        cross_attention_output, cross_attention_probs = self.cross_attention(
            decoder_hidden_states=attention_output,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask
        )
        cross_output = self.cross_output(
            cross_attention_output,
            attention_output
        )
        
        # 3. Feed-Forward Network
        intermediate_output = self.intermediate(cross_output)
        layer_output = self.output(intermediate_output, cross_output)
        
        # Self-attention weights 안전하게 추출
        self_attention_probs = None
        if len(self_attention_outputs) > 1:
            self_attention_probs = self_attention_outputs[1]
        
        return layer_output, self_attention_probs, cross_attention_probs


class TransformerEncoder(nn.Module):
    """
    BERT 스타일 Encoder
    
    기존 TrackedTransformerLayer들을 스택으로 쌓은 구조
    """
    
    def __init__(self, config: EncoderDecoderConfig):
        super().__init__()
        
        # BERT 설정을 TrackedTransformerConfig로 변환
        encoder_config = TrackedTransformerConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.encoder_layers,
            num_attention_heads=config.encoder_attention_heads,
            intermediate_size=config.encoder_intermediate_size,
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            max_position_embeddings=config.max_position_embeddings,
            layer_norm_eps=config.layer_norm_eps,
            initializer_range=config.initializer_range,
        )
        
        # BERT 스타일 레이어들을 가져와서 사용
        self.layers = nn.ModuleList([
            TrackedTransformerLayer(encoder_config) 
            for _ in range(config.encoder_layers)
        ])
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> BaseModelOutput:
        """
        Encoder 순전파
        
        Returns:
            BaseModelOutput with:
            - last_hidden_state: 마지막 레이어 출력
            - hidden_states: 모든 레이어 출력들
            - attentions: 모든 어텐션 가중치들
        """
        
        all_hidden_states = []
        all_attentions = []
        
        for layer in self.layers:
            all_hidden_states.append(hidden_states)
            
            layer_outputs = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                output_attentions=True  # 어텐션 가중치 출력
            )
            
            hidden_states = layer_outputs[0]
            # layer_outputs[1]이 있는지 확인
            if len(layer_outputs) > 1:
                all_attentions.append(layer_outputs[1])
        
        all_hidden_states.append(hidden_states)
        
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions if all_attentions else None
        )


class TransformerDecoder(nn.Module):
    """
    Transformer 스타일 Decoder
    
    Cross-Attention을 포함한 디코더 레이어들의 스택
    """
    
    def __init__(self, config: EncoderDecoderConfig):
        super().__init__()
        
        self.layers = nn.ModuleList([
            DecoderLayer(config) 
            for _ in range(config.decoder_layers)
        ])
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> BaseModelOutput:
        """
        Decoder 순전파
        """
        
        all_hidden_states = []
        all_self_attentions = []
        all_cross_attentions = []
        
        for layer in self.layers:
            all_hidden_states.append(hidden_states)
            
            layer_outputs = layer(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask
            )
            
            hidden_states = layer_outputs[0]
            if len(layer_outputs) > 1 and layer_outputs[1] is not None:
                all_self_attentions.append(layer_outputs[1])
            if len(layer_outputs) > 2 and layer_outputs[2] is not None:
                all_cross_attentions.append(layer_outputs[2])
        
        all_hidden_states.append(hidden_states)
        
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=(all_self_attentions, all_cross_attentions)
        )


class BERTEncoderTransformerDecoderModel(PreTrainedModel):
    """
    BERT Encoder + Transformer Decoder 결합 모델
    
    이 모델은 다음과 같은 구조를 가집니다:
    1. 공유 임베딩 레이어 (토큰, 위치)
    2. BERT 스타일 Encoder
    3. Transformer 스타일 Decoder (Cross-Attention 포함)
    4. 언어 모델링 헤드 (다음 토큰 예측)
    
    사용 예시:
    - 기계 번역: 한국어 입력 → 영어 출력
    - 텍스트 요약: 긴 문서 → 짧은 요약
    - 질문 답변: 문맥 + 질문 → 답변
    """
    
    config_class = EncoderDecoderConfig
    
    def __init__(self, config: EncoderDecoderConfig):
        super().__init__(config)
        self.config = config
        
        # 공유 임베딩 레이어들
        self.embeddings = nn.ModuleDict({
            'word_embeddings': nn.Embedding(config.vocab_size, config.hidden_size, 
                                          padding_idx=config.pad_token_id),
            'position_embeddings': nn.Embedding(config.max_position_embeddings, 
                                               config.hidden_size),
            'LayerNorm': nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
            'dropout': nn.Dropout(config.hidden_dropout_prob)
        })
        
        # Encoder와 Decoder
        self.encoder = TransformerEncoder(config)
        self.decoder = TransformerDecoder(config)
        
        # 언어 모델링 헤드 (디코더 출력을 어휘로 변환)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # 가중치 초기화
        self.init_weights()
    
    def get_input_embeddings(self):
        """입력 임베딩 레이어 반환 (Hugging Face 표준)"""
        return self.embeddings['word_embeddings']
    
    def set_input_embeddings(self, value):
        """입력 임베딩 레이어 설정 (Hugging Face 표준)"""
        self.embeddings['word_embeddings'] = value
    
    def get_output_embeddings(self):
        """출력 임베딩 레이어 반환 (언어 모델링 헤드)"""
        return self.lm_head
    
    def set_output_embeddings(self, new_embeddings):
        """출력 임베딩 레이어 설정"""
        self.lm_head = new_embeddings
    
    def _get_embeddings(
        self, 
        input_ids: torch.Tensor, 
        position_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        토큰과 위치 임베딩을 계산하여 결합
        
        Args:
            input_ids: 토큰 ID들 [batch, seq_len]
            position_ids: 위치 ID들 [batch, seq_len]
            
        Returns:
            embeddings: 임베딩 벡터들 [batch, seq_len, hidden_size]
        """
        
        seq_length = input_ids.size(1)
        
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        # 토큰 임베딩 + 위치 임베딩
        word_embeddings = self.embeddings['word_embeddings'](input_ids)
        position_embeddings = self.embeddings['position_embeddings'](position_ids)
        
        embeddings = word_embeddings + position_embeddings
        embeddings = self.embeddings['LayerNorm'](embeddings)
        embeddings = self.embeddings['dropout'](embeddings)
        
        return embeddings
    
    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Causal Attention Mask 생성 (미래 토큰 차단)
        
        Returns:
            mask: [seq_len, seq_len] 하삼각행렬 (1=attend, 0=mask)
        """
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask
    
    def forward(
        self,
        input_ids: torch.Tensor,                              # Encoder 입력 [batch, enc_len]
        attention_mask: Optional[torch.Tensor] = None,        # Encoder mask [batch, enc_len]
        decoder_input_ids: Optional[torch.Tensor] = None,     # Decoder 입력 [batch, dec_len]
        decoder_attention_mask: Optional[torch.Tensor] = None, # Decoder mask [batch, dec_len]
        labels: Optional[torch.Tensor] = None,                # 정답 레이블 [batch, dec_len]
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Seq2SeqModelOutput:
        """
        Encoder-Decoder 모델 순전파
        
        Args:
            input_ids: 인코더 입력 (source 시퀀스)
            decoder_input_ids: 디코더 입력 (target 시퀀스, teacher forcing)
            labels: 학습용 정답 레이블
            
        Returns:
            Seq2SeqModelOutput: 예측값, 손실, 어텐션 가중치 등
        """
        
        # 1. Encoder 임베딩 및 순전파
        encoder_embeddings = self._get_embeddings(input_ids)
        
        encoder_outputs = self.encoder(
            hidden_states=encoder_embeddings,
            attention_mask=attention_mask
        )
        
        # 2. Decoder 입력 준비
        if decoder_input_ids is None:
            # 추론 시: BOS 토큰으로 시작
            batch_size = input_ids.size(0)
            decoder_input_ids = torch.full(
                (batch_size, 1), 
                self.config.bos_token_id, 
                device=input_ids.device,
                dtype=torch.long
            )
        
        # 3. Decoder 임베딩
        decoder_embeddings = self._get_embeddings(decoder_input_ids)
        
        # 4. Causal Mask 생성 (미래 토큰 차단)
        decoder_seq_len = decoder_input_ids.size(1)
        causal_mask = self._create_causal_mask(decoder_seq_len, input_ids.device)
        
        # Decoder attention mask와 causal mask 결합
        if decoder_attention_mask is not None:
            # [batch, dec_len] → [batch, 1, dec_len, dec_len]
            extended_decoder_mask = decoder_attention_mask.unsqueeze(1).unsqueeze(1)
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
            combined_mask = extended_decoder_mask * causal_mask
        else:
            combined_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        
        # 5. Decoder 순전파
        decoder_outputs = self.decoder(
            hidden_states=decoder_embeddings,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            attention_mask=combined_mask,
            encoder_attention_mask=attention_mask
        )
        
        # 6. 언어 모델링 헤드 (다음 토큰 예측)
        logits = self.lm_head(decoder_outputs.last_hidden_state)
        
        # 7. 손실 계산 (학습 시)
        loss = None
        if labels is not None:
            # 마지막 토큰은 예측할 대상이 없으므로 제외
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id)
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1)
            )
        
        if not return_dict:
            output = (logits,)
            if loss is not None:
                output = (loss,) + output
            return output + encoder_outputs[1:] + decoder_outputs[1:]
        
        # Dict 형태로 반환 (loss 포함)
        result = {
            'logits': logits,
            'last_hidden_state': decoder_outputs.last_hidden_state,
            'encoder_last_hidden_state': encoder_outputs.last_hidden_state,
        }
        
        if loss is not None:
            result['loss'] = loss
            
        if output_hidden_states:
            result['decoder_hidden_states'] = decoder_outputs.hidden_states
            result['encoder_hidden_states'] = encoder_outputs.hidden_states
            
        if output_attentions:
            result['encoder_attentions'] = encoder_outputs.attentions
            if decoder_outputs.attentions and len(decoder_outputs.attentions) > 0:
                result['decoder_attentions'] = decoder_outputs.attentions[0]
            if decoder_outputs.attentions and len(decoder_outputs.attentions) > 1:
                result['cross_attentions'] = decoder_outputs.attentions[1]
        
        # SimpleNamespace로 반환하여 속성 접근 가능하게 만들기
        from types import SimpleNamespace
        return SimpleNamespace(**result)
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 50,
        num_beams: int = 1,
        temperature: float = 1.0,
        do_sample: bool = False,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        간단한 텍스트 생성 (Greedy Decoding)
        
        Args:
            input_ids: 입력 시퀀스 [batch, enc_len]
            max_length: 최대 생성 길이
            
        Returns:
            generated_ids: 생성된 토큰 ID들 [batch, dec_len]
        """
        
        if pad_token_id is None:
            pad_token_id = self.config.pad_token_id
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id
        
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        # 인코더 한 번만 실행
        encoder_embeddings = self._get_embeddings(input_ids)
        encoder_outputs = self.encoder(
            hidden_states=encoder_embeddings,
            attention_mask=attention_mask
        )
        
        # BOS 토큰으로 디코더 입력 초기화
        decoder_input_ids = torch.full(
            (batch_size, 1), 
            self.config.bos_token_id, 
            device=device,
            dtype=torch.long
        )
        
        # 순차적으로 토큰 생성
        for _ in range(max_length - 1):
            # 현재까지의 디코더 입력으로 다음 토큰 예측
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                return_dict=True
            )
            
            # 다음 토큰 예측 (Greedy)
            if do_sample:
                next_token_logits = outputs.logits[:, -1, :] / temperature
                next_token_probs = F.softmax(next_token_logits, dim=-1)
                next_token_ids = torch.multinomial(next_token_probs, 1)
            else:
                next_token_ids = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            
            # 생성된 토큰을 디코더 입력에 추가
            decoder_input_ids = torch.cat([decoder_input_ids, next_token_ids], dim=1)
            
            # EOS 토큰이 생성되면 종료
            if (next_token_ids == eos_token_id).all():
                break
        
        return decoder_input_ids[:, 1:]  # BOS 토큰 제거하고 반환


# 설정 및 모델 등록 (Hugging Face 호환)
EncoderDecoderConfig.register_for_auto_class()
BERTEncoderTransformerDecoderModel.register_for_auto_class("AutoModel")