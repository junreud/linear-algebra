"""
간단한 Encoder-Decoder 테스트 스크립트

BERT Encoder + Transformer Decoder 모델을 간단히 테스트해봅니다.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from encoder_decoder_model import BERTEncoderTransformerDecoderModel, EncoderDecoderConfig

def test_encoder_decoder():
    """Encoder-Decoder 모델 기본 동작 테스트"""
    
    # 디바이스 설정
    device = torch.device('mps' if torch.backends.mps.is_available() 
                         else 'cuda' if torch.cuda.is_available() 
                         else 'cpu')
    print(f"🖥️  사용 디바이스: {device}")
    
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
    print(f"📝 토크나이저 로드 완료")
    
    # 작은 모델 설정 (테스트용)
    config = EncoderDecoderConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=256,  # 작게 설정
        encoder_layers=2,
        decoder_layers=2,
        encoder_attention_heads=4,
        decoder_attention_heads=4,
        encoder_intermediate_size=512,
        decoder_intermediate_size=512,
        max_position_embeddings=128,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.cls_token_id,
        eos_token_id=tokenizer.sep_token_id,
    )
    
    # 모델 생성
    model = BERTEncoderTransformerDecoderModel(config).to(device)
    print(f"🤖 모델 생성 완료")
    print(f"📊 모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
    
    # 테스트 데이터 준비
    english_text = "Hello world"
    korean_text = "안녕하세요 세계"
    
    # 입력 토크나이징
    encoder_inputs = tokenizer(
        english_text,
        max_length=32,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    decoder_inputs = tokenizer(
        f"[CLS] {korean_text}",
        max_length=32,
        padding='max_length', 
        truncation=True,
        return_tensors='pt'
    )
    
    labels = tokenizer(
        f"{korean_text} [SEP]",
        max_length=32,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # 디바이스로 이동
    encoder_inputs = {k: v.to(device) for k, v in encoder_inputs.items()}
    decoder_inputs = {k: v.to(device) for k, v in decoder_inputs.items()}
    labels = {k: v.to(device) for k, v in labels.items()}
    
    print(f"\n🔍 입력 데이터:")
    print(f"   Encoder 입력: {english_text}")
    print(f"   Decoder 입력: [CLS] {korean_text}")
    print(f"   레이블: {korean_text} [SEP]")
    
    # 순전파 테스트
    print(f"\n🧠 순전파 테스트...")
    model.eval()
    
    with torch.no_grad():
        outputs = model(
            input_ids=encoder_inputs['input_ids'],
            attention_mask=encoder_inputs['attention_mask'],
            decoder_input_ids=decoder_inputs['input_ids'],
            decoder_attention_mask=decoder_inputs['attention_mask'],
            labels=labels['input_ids'],
            return_dict=True
        )
        
        print(f"✅ 순전파 성공!")
        print(f"   출력 logits 크기: {outputs.logits.shape}")
        print(f"   손실: {outputs.loss.item():.4f}")
        
        # 예측 토큰 확인
        predicted_ids = outputs.logits.argmax(dim=-1)
        predicted_text = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
        print(f"   예측 텍스트: {predicted_text}")
    
    # 간단한 생성 테스트 (greedy decoding)
    print(f"\n🎯 텍스트 생성 테스트...")
    
    with torch.no_grad():
        # Encoder 실행
        encoder_embeddings = model._get_embeddings(encoder_inputs['input_ids'])
        encoder_outputs = model.encoder(
            hidden_states=encoder_embeddings,
            attention_mask=encoder_inputs['attention_mask']
        )
        
        # Decoder 시작 토큰
        decoder_input_ids = torch.tensor(
            [[config.bos_token_id]], 
            device=device
        )
        
        # 순차적으로 토큰 생성 (간단한 greedy)
        generated_tokens = []
        max_length = 10
        
        for step in range(max_length):
            # Decoder 임베딩
            decoder_embeddings = model._get_embeddings(decoder_input_ids)
            
            # Causal mask 생성
            seq_len = decoder_input_ids.size(1)
            causal_mask = model._create_causal_mask(seq_len, device)
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
            
            # Decoder 실행
            decoder_outputs = model.decoder(
                hidden_states=decoder_embeddings,
                encoder_hidden_states=encoder_outputs.last_hidden_state,
                attention_mask=causal_mask,
                encoder_attention_mask=encoder_inputs['attention_mask']
            )
            
            # 다음 토큰 예측
            logits = model.lm_head(decoder_outputs.last_hidden_state)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            
            # 생성 중단 조건
            if next_token.item() == config.eos_token_id:
                break
                
            generated_tokens.append(next_token.item())
            decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=1)
        
        # 생성된 텍스트 디코딩
        if generated_tokens:
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            print(f"✅ 생성 성공!")
            print(f"   입력: {english_text}")
            print(f"   생성: {generated_text}")
        else:
            print(f"⚠️  토큰이 생성되지 않음")
    
    print(f"\n🎉 Encoder-Decoder 모델 테스트 완료!")
    print(f"\n📋 모델 특징:")
    print(f"   • Encoder: BERT 스타일 (양방향 attention)")
    print(f"   • Decoder: Transformer 스타일 (causal + cross-attention)")
    print(f"   • 용도: 번역, 요약, 질문답변")
    print(f"   • 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")

if __name__ == "__main__":
    test_encoder_decoder()