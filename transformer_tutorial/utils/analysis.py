"""
분석 도구 - QKV 추적, Attention 패턴 분석, 성능 메트릭
"""
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import json
import os
from datetime import datetime


class TransformerAnalyzer:
    """
    Transformer 모델의 내부 동작을 분석하는 클래스
    """
    
    def __init__(self, model, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.analysis_data = {}
        self.hooks = []
        
    def register_hooks(self):
        """Forward hook 등록으로 중간값 캡처"""
        
        def create_hook(name):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    self.analysis_data[name] = output[0].detach().cpu()
                else:
                    self.analysis_data[name] = output.detach().cpu()
            return hook_fn
        
        # 모델 타입에 따라 다른 hook 등록
        if hasattr(self.model, 'layers'):  # Custom transformer
            for i, layer in enumerate(self.model.layers):
                if hasattr(layer, 'self_attention'):
                    hook = layer.self_attention.register_forward_hook(
                        create_hook(f'layer_{i}_attention')
                    )
                    self.hooks.append(hook)
                
                if hasattr(layer, 'ffn'):
                    hook = layer.ffn.register_forward_hook(
                        create_hook(f'layer_{i}_ffn')
                    )
                    self.hooks.append(hook)
        
        elif hasattr(self.model, 'encoder_layers'):  # PyTorch version
            for i, layer in enumerate(self.model.encoder_layers):
                hook = layer.self_attention.register_forward_hook(
                    create_hook(f'encoder_layer_{i}_attention')
                )
                self.hooks.append(hook)
                
                hook = layer.ffn.register_forward_hook(
                    create_hook(f'encoder_layer_{i}_ffn')
                )
                self.hooks.append(hook)
    
    def remove_hooks(self):
        """Hook 제거"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.analysis_data.clear()
    
    def analyze_attention_patterns(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor = None,
        tokens: List[str] = None
    ) -> Dict[str, Any]:
        """
        Attention pattern 분석
        
        Args:
            input_ids: 입력 토큰 ID
            attention_mask: 어텐션 마스크
            tokens: 토큰 문자열 리스트
            
        Returns:
            분석 결과 딕셔너리
        """
        self.model.eval()
        
        with torch.no_grad():
            # Forward pass with attention output
            if hasattr(self.model, 'forward'):
                outputs = self.model(
                    input_ids, 
                    attention_mask=attention_mask,
                    output_attentions=True
                )
            else:
                # Custom model의 경우
                outputs = self.model(input_ids, attention_mask)
        
        analysis = {
            'input_shape': input_ids.shape,
            'tokens': tokens,
            'attention_analysis': {}
        }
        
        # Attention weights 분석
        if hasattr(outputs, 'attentions') and outputs.attentions:
            for layer_idx, attention in enumerate(outputs.attentions):
                # attention shape: (batch_size, num_heads, seq_len, seq_len)
                attention_np = attention[0].cpu().numpy()  # 첫 번째 배치
                
                layer_analysis = {
                    'shape': attention_np.shape,
                    'head_analysis': {}
                }
                
                # 각 헤드별 분석
                for head_idx in range(attention_np.shape[0]):
                    head_attention = attention_np[head_idx]
                    
                    head_analysis = {
                        'mean_attention': float(np.mean(head_attention)),
                        'max_attention': float(np.max(head_attention)),
                        'min_attention': float(np.min(head_attention)),
                        'attention_entropy': self._calculate_attention_entropy(head_attention),
                        'diagonal_attention': float(np.mean(np.diag(head_attention))),
                        'off_diagonal_attention': float(np.mean(head_attention - np.diag(np.diag(head_attention)))),
                    }
                    
                    # Attention pattern 분류
                    pattern = self._classify_attention_pattern(head_attention)
                    head_analysis['pattern'] = pattern
                    
                    layer_analysis['head_analysis'][f'head_{head_idx}'] = head_analysis
                
                # 레이어 전체 통계
                layer_analysis['layer_stats'] = {
                    'mean_attention': float(np.mean(attention_np)),
                    'attention_variance': float(np.var(attention_np)),
                    'dominant_head': int(np.argmax([
                        head_data['mean_attention'] 
                        for head_data in layer_analysis['head_analysis'].values()
                    ]))
                }
                
                analysis['attention_analysis'][f'layer_{layer_idx}'] = layer_analysis
        
        return analysis
    
    def _calculate_attention_entropy(self, attention_matrix: np.ndarray) -> float:
        """Attention entropy 계산 (다양성 측정)"""
        # 각 쿼리에 대한 entropy 계산
        entropies = []
        for i in range(attention_matrix.shape[0]):
            probs = attention_matrix[i]
            # 0이 아닌 값들만 고려
            probs = probs[probs > 1e-8]
            if len(probs) > 0:
                entropy = -np.sum(probs * np.log(probs))
                entropies.append(entropy)
        
        return float(np.mean(entropies)) if entropies else 0.0
    
    def _classify_attention_pattern(self, attention_matrix: np.ndarray) -> str:
        """Attention pattern 분류"""
        # 대각선 vs 전역 attention 비율
        diagonal_ratio = np.mean(np.diag(attention_matrix))
        
        # Attention 집중도
        max_attention = np.max(attention_matrix, axis=1)
        concentration = np.mean(max_attention)
        
        if diagonal_ratio > 0.3:
            return "local"  # 로컬 패턴 (인접 토큰 중심)
        elif concentration > 0.5:
            return "focused"  # 집중 패턴 (특정 토큰에 집중)
        else:
            return "distributed"  # 분산 패턴 (전역적 attention)
    
    def analyze_qkv_statistics(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> Dict[str, Any]:
        """
        QKV 통계 분석
        
        Args:
            input_ids: 입력 토큰
            attention_mask: 어텐션 마스크
            
        Returns:
            QKV 분석 결과
        """
        self.register_hooks()
        self.model.eval()
        
        qkv_stats = {}
        
        with torch.no_grad():
            # Forward pass
            _ = self.model(input_ids, attention_mask=attention_mask)
            
            # 각 레이어의 attention 모듈에서 QKV 정보 추출
            if hasattr(self.model, 'layers'):
                for i, layer in enumerate(self.model.layers):
                    if hasattr(layer, 'attention') and hasattr(layer.attention, 'tracked_states'):
                        tracked = layer.attention.tracked_states
                        
                        if tracked:
                            layer_stats = {}
                            
                            for component in ['query', 'key', 'value']:
                                if component in tracked:
                                    tensor = tracked[component]
                                    # (batch_size, num_heads, seq_len, head_dim)
                                    
                                    stats = {
                                        'mean': float(torch.mean(tensor)),
                                        'std': float(torch.std(tensor)),
                                        'min': float(torch.min(tensor)),
                                        'max': float(torch.max(tensor)),
                                        'l2_norm': float(torch.norm(tensor)),
                                        'sparsity': float(torch.sum(tensor == 0) / tensor.numel()),
                                    }
                                    
                                    # 헤드별 통계
                                    head_stats = []
                                    for head_idx in range(tensor.shape[1]):
                                        head_tensor = tensor[:, head_idx, :, :]
                                        head_stat = {
                                            'mean': float(torch.mean(head_tensor)),
                                            'std': float(torch.std(head_tensor)),
                                            'l2_norm': float(torch.norm(head_tensor)),
                                        }
                                        head_stats.append(head_stat)
                                    
                                    stats['head_stats'] = head_stats
                                    layer_stats[component] = stats
                            
                            qkv_stats[f'layer_{i}'] = layer_stats
        
        self.remove_hooks()
        return qkv_stats
    
    def analyze_gradient_flow(
        self, 
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> Dict[str, Any]:
        """
        Gradient flow 분석
        
        Args:
            input_ids: 입력 토큰
            target_ids: 타겟 토큰
            attention_mask: 어텐션 마스크
            
        Returns:
            Gradient 분석 결과
        """
        self.model.train()
        
        # Forward pass
        if hasattr(self.model, 'forward'):
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.last_hidden_state
        else:
            logits = self.model(input_ids, attention_mask)
        
        # Loss 계산
        if hasattr(self.model, 'lm_head'):
            logits = self.model.lm_head(logits)
        
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits.view(-1, logits.size(-1)), target_ids.view(-1))
        
        # Backward pass
        loss.backward()
        
        gradient_stats = {}
        
        # 각 레이어의 gradient 통계
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad = param.grad
                
                stats = {
                    'mean': float(torch.mean(grad)),
                    'std': float(torch.std(grad)),
                    'l2_norm': float(torch.norm(grad)),
                    'max': float(torch.max(torch.abs(grad))),
                    'num_zero_grads': int(torch.sum(grad == 0)),
                    'grad_norm_ratio': float(torch.norm(grad) / torch.norm(param.data)) if torch.norm(param.data) > 0 else 0.0
                }
                
                gradient_stats[name] = stats
        
        return {
            'loss': float(loss),
            'gradient_stats': gradient_stats
        }
    
    def analyze_layer_contributions(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> Dict[str, Any]:
        """
        각 레이어의 기여도 분석
        """
        self.model.eval()
        layer_outputs = {}
        
        def create_hook(layer_name):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    layer_outputs[layer_name] = output[0].detach().cpu()
                else:
                    layer_outputs[layer_name] = output.detach().cpu()
            return hook_fn
        
        # Hook 등록
        hooks = []
        if hasattr(self.model, 'layers'):
            for i, layer in enumerate(self.model.layers):
                hook = layer.register_forward_hook(create_hook(f'layer_{i}'))
                hooks.append(hook)
        
        with torch.no_grad():
            _ = self.model(input_ids, attention_mask=attention_mask)
        
        # Hook 제거
        for hook in hooks:
            hook.remove()
        
        # 레이어별 기여도 계산
        contributions = {}
        prev_output = None
        
        for layer_name, output in layer_outputs.items():
            if prev_output is not None:
                # 변화량 계산
                change = torch.norm(output - prev_output, dim=-1)
                contribution = {
                    'mean_change': float(torch.mean(change)),
                    'max_change': float(torch.max(change)),
                    'change_variance': float(torch.var(change)),
                }
            else:
                contribution = {
                    'mean_change': 0.0,
                    'max_change': 0.0,
                    'change_variance': 0.0,
                }
            
            # 출력 통계
            contribution.update({
                'output_mean': float(torch.mean(output)),
                'output_std': float(torch.std(output)),
                'output_norm': float(torch.norm(output)),
            })
            
            contributions[layer_name] = contribution
            prev_output = output
        
        return contributions
    
    def comprehensive_analysis(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        tokens: List[str] = None,
        save_path: str = None
    ) -> Dict[str, Any]:
        """
        종합적인 분석 수행
        
        Args:
            input_ids: 입력 토큰
            target_ids: 타겟 토큰 (gradient 분석용)
            attention_mask: 어텐션 마스크
            tokens: 토큰 문자열
            save_path: 결과 저장 경로
            
        Returns:
            종합 분석 결과
        """
        print("Starting comprehensive analysis...")
        
        analysis_results = {
            'timestamp': datetime.now().isoformat(),
            'model_info': {
                'type': type(self.model).__name__,
                'device': self.device,
                'num_parameters': sum(p.numel() for p in self.model.parameters()),
            },
            'input_info': {
                'input_shape': input_ids.shape,
                'tokens': tokens,
            }
        }
        
        # 1. Attention pattern 분석
        print("Analyzing attention patterns...")
        attention_analysis = self.analyze_attention_patterns(
            input_ids, attention_mask, tokens
        )
        analysis_results['attention_analysis'] = attention_analysis
        
        # 2. QKV 통계 분석
        print("Analyzing QKV statistics...")
        qkv_analysis = self.analyze_qkv_statistics(input_ids, attention_mask)
        analysis_results['qkv_analysis'] = qkv_analysis
        
        # 3. 레이어 기여도 분석
        print("Analyzing layer contributions...")
        contribution_analysis = self.analyze_layer_contributions(input_ids, attention_mask)
        analysis_results['contribution_analysis'] = contribution_analysis
        
        # 4. Gradient 분석 (target이 있는 경우)
        if target_ids is not None:
            print("Analyzing gradient flow...")
            gradient_analysis = self.analyze_gradient_flow(
                input_ids, target_ids, attention_mask
            )
            analysis_results['gradient_analysis'] = gradient_analysis
        
        # 결과 저장
        if save_path:
            self._save_analysis_results(analysis_results, save_path)
        
        print("Analysis completed!")
        return analysis_results
    
    def _save_analysis_results(self, results: Dict[str, Any], save_path: str):
        """분석 결과 저장"""
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"Analysis results saved to {save_path}")


def analyze_sample_model():
    """샘플 모델 분석 (테스트용)"""
    print("=== Sample Model Analysis ===")
    
    # 샘플 데이터 생성
    batch_size, seq_len, vocab_size = 1, 8, 1000
    input_ids = torch.randint(1, vocab_size, (batch_size, seq_len))
    target_ids = torch.randint(1, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)
    tokens = [f"token_{i}" for i in range(seq_len)]
    
    # 간단한 모델 생성 (실제로는 훈련된 모델 사용)
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(vocab_size, 256)
            self.layers = torch.nn.ModuleList([
                torch.nn.TransformerEncoderLayer(256, 8, 1024, batch_first=True)
                for _ in range(2)
            ])
            self.lm_head = torch.nn.Linear(256, vocab_size)
        
        def forward(self, input_ids, attention_mask=None):
            x = self.embedding(input_ids)
            for layer in self.layers:
                x = layer(x)
            return self.lm_head(x)
    
    model = SimpleModel()
    
    # 분석 수행
    analyzer = TransformerAnalyzer(model)
    
    # 개별 분석들
    print("\n1. Attention Pattern Analysis:")
    attention_results = analyzer.analyze_attention_patterns(input_ids, attention_mask, tokens)
    print(f"Number of layers analyzed: {len(attention_results.get('attention_analysis', {}))}")
    
    print("\n2. Layer Contribution Analysis:")
    contribution_results = analyzer.analyze_layer_contributions(input_ids, attention_mask)
    print(f"Layer contributions: {list(contribution_results.keys())}")
    
    print("\n3. Gradient Analysis:")
    gradient_results = analyzer.analyze_gradient_flow(input_ids, target_ids, attention_mask)
    print(f"Loss: {gradient_results.get('loss', 'N/A'):.4f}")
    print(f"Gradient stats: {len(gradient_results.get('gradient_stats', {}))}")
    
    # 종합 분석
    print("\n4. Comprehensive Analysis:")
    comprehensive_results = analyzer.comprehensive_analysis(
        input_ids, target_ids, attention_mask, tokens,
        save_path="sample_analysis_results.json"
    )
    
    print(f"Comprehensive analysis keys: {list(comprehensive_results.keys())}")
    
    print("\nSample analysis completed!")


if __name__ == "__main__":
    analyze_sample_model()