"""
시각화 도구 - Attention, QKV, 학습 곡선 등
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import os
from datetime import datetime
import json


class TransformerVisualizer:
    """
    Transformer 내부 상태 시각화를 위한 클래스
    """
    
    def __init__(self, save_dir: str = "visualizations"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 시각화 스타일 설정
        plt.style.use('default')
        sns.set_palette("husl")
        
    def plot_attention_heatmap(
        self,
        attention_weights: np.ndarray,
        tokens: List[str] = None,
        layer_idx: int = 0,
        head_idx: int = 0,
        title: str = None,
        save_name: str = None
    ):
        """
        Attention weight 히트맵 시각화
        
        Args:
            attention_weights: (seq_len, seq_len) attention matrix
            tokens: 토큰 리스트
            layer_idx: 레이어 인덱스
            head_idx: 헤드 인덱스
            title: 그래프 제목
            save_name: 저장 파일명
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 히트맵 그리기
        sns.heatmap(
            attention_weights,
            xticklabels=tokens if tokens else False,
            yticklabels=tokens if tokens else False,
            cmap='Blues',
            cbar=True,
            square=True,
            ax=ax,
            cbar_kws={'label': 'Attention Weight'}
        )
        
        # 제목 설정
        if title is None:
            title = f'Attention Heatmap - Layer {layer_idx}, Head {head_idx}'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Key (Attended to)', fontsize=12)
        ax.set_ylabel('Query (Attending)', fontsize=12)
        
        # 토큰 라벨 회전
        if tokens:
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        # 저장
        if save_name:
            save_path = os.path.join(self.save_dir, f"{save_name}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Attention heatmap saved to {save_path}")
        
        plt.show()
        plt.close()
    
    def plot_multi_head_attention(
        self,
        attention_weights: np.ndarray,
        tokens: List[str] = None,
        layer_idx: int = 0,
        num_heads: int = None,
        save_name: str = None
    ):
        """
        Multi-head attention 시각화
        
        Args:
            attention_weights: (num_heads, seq_len, seq_len)
            tokens: 토큰 리스트
            layer_idx: 레이어 인덱스
            num_heads: 표시할 헤드 수 (None이면 모두)
        """
        if num_heads is None:
            num_heads = attention_weights.shape[0]
        num_heads = min(num_heads, attention_weights.shape[0])
        
        # Grid 크기 계산
        cols = min(4, num_heads)
        rows = (num_heads + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for head_idx in range(num_heads):
            ax = axes[head_idx] if num_heads > 1 else axes[0]
            
            sns.heatmap(
                attention_weights[head_idx],
                xticklabels=tokens if tokens and len(tokens) <= 20 else False,
                yticklabels=tokens if tokens and len(tokens) <= 20 else False,
                cmap='Blues',
                cbar=True,
                square=True,
                ax=ax,
                cbar_kws={'label': 'Weight'}
            )
            
            ax.set_title(f'Head {head_idx}', fontsize=12)
            
            if tokens and len(tokens) <= 20:
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
                ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
        
        # 빈 서브플롯 숨기기
        for i in range(num_heads, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(f'Multi-Head Attention - Layer {layer_idx}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_name:
            save_path = os.path.join(self.save_dir, f"{save_name}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Multi-head attention saved to {save_path}")
        
        plt.show()
        plt.close()
    
    def plot_qkv_evolution(
        self,
        qkv_history: List[Dict],
        save_name: str = None
    ):
        """
        QKV 값의 학습 과정에서의 변화 시각화
        
        Args:
            qkv_history: [{'step': int, 'layer_0': {'Q': float, 'K': float, 'V': float}, ...}]
        """
        if not qkv_history:
            print("No QKV history provided")
            return
        
        # 데이터 추출
        steps = []
        layer_data = {}
        
        for entry in qkv_history:
            steps.append(entry['step'])
            
            for layer_name, qkv_stats in entry.items():
                if layer_name == 'step':
                    continue
                    
                if layer_name not in layer_data:
                    layer_data[layer_name] = {'Q': [], 'K': [], 'V': []}
                
                layer_data[layer_name]['Q'].append(qkv_stats.get('Q', 0))
                layer_data[layer_name]['K'].append(qkv_stats.get('K', 0))
                layer_data[layer_name]['V'].append(qkv_stats.get('V', 0))
        
        # 시각화
        num_layers = len(layer_data)
        if num_layers == 0:
            print("No layer data found")
            return
        
        fig, axes = plt.subplots(num_layers, 1, figsize=(12, 4 * num_layers))
        if num_layers == 1:
            axes = [axes]
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
        
        for i, (layer_name, data) in enumerate(layer_data.items()):
            ax = axes[i]
            
            ax.plot(steps, data['Q'], color=colors[0], label='Query', linewidth=2, alpha=0.8)
            ax.plot(steps, data['K'], color=colors[1], label='Key', linewidth=2, alpha=0.8)
            ax.plot(steps, data['V'], color=colors[2], label='Value', linewidth=2, alpha=0.8)
            
            ax.set_title(f'{layer_name.replace("_", " ").title()} - QKV Evolution', fontsize=14)
            ax.set_xlabel('Training Step', fontsize=12)
            ax.set_ylabel('Mean Value', fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Y축 범위 설정 (이상치 제거)
            all_values = data['Q'] + data['K'] + data['V']
            if all_values:
                y_min, y_max = np.percentile(all_values, [5, 95])
                ax.set_ylim(y_min * 1.1, y_max * 1.1)
        
        plt.tight_layout()
        
        if save_name:
            save_path = os.path.join(self.save_dir, f"{save_name}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"QKV evolution plot saved to {save_path}")
        
        plt.show()
        plt.close()
    
    def plot_training_curves(
        self,
        train_losses: List[float],
        val_losses: List[float] = None,
        additional_metrics: Dict[str, List[float]] = None,
        save_name: str = None
    ):
        """
        학습 곡선 시각화
        
        Args:
            train_losses: 훈련 loss 리스트
            val_losses: 검증 loss 리스트
            additional_metrics: 추가 메트릭들
        """
        num_plots = 1
        if additional_metrics:
            num_plots += len(additional_metrics)
        
        fig, axes = plt.subplots(1, num_plots, figsize=(6*num_plots, 5))
        if num_plots == 1:
            axes = [axes]
        
        # Loss 곡선
        ax = axes[0]
        epochs = range(1, len(train_losses) + 1)
        
        ax.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2, alpha=0.8)
        if val_losses:
            ax.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2, alpha=0.8)
        
        ax.set_title('Training Curves', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 추가 메트릭들
        if additional_metrics:
            for i, (metric_name, values) in enumerate(additional_metrics.items()):
                ax = axes[i + 1]
                ax.plot(epochs[:len(values)], values, 'g-', linewidth=2, alpha=0.8)
                ax.set_title(f'{metric_name}', fontsize=14)
                ax.set_xlabel('Epoch', fontsize=12)
                ax.set_ylabel(metric_name, fontsize=12)
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name:
            save_path = os.path.join(self.save_dir, f"{save_name}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training curves saved to {save_path}")
        
        plt.show()
        plt.close()
    
    def plot_attention_patterns_evolution(
        self,
        attention_history: List[Dict],
        layer_idx: int = 0,
        head_idx: int = 0,
        save_name: str = None
    ):
        """
        학습 과정에서 attention pattern의 변화 시각화
        
        Args:
            attention_history: [{'step': int, 'attention': np.ndarray}, ...]
        """
        if not attention_history:
            print("No attention history provided")
            return
        
        # 몇 개의 시점을 선택 (최대 6개)
        num_snapshots = min(6, len(attention_history))
        indices = np.linspace(0, len(attention_history) - 1, num_snapshots, dtype=int)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, idx in enumerate(indices):
            entry = attention_history[idx]
            attention = entry['attention']
            step = entry['step']
            
            sns.heatmap(
                attention,
                ax=axes[i],
                cmap='Blues',
                cbar=True,
                square=True,
                xticklabels=False,
                yticklabels=False
            )
            axes[i].set_title(f'Step {step}', fontsize=12)
        
        # 빈 서브플롯 숨기기
        for i in range(num_snapshots, 6):
            axes[i].set_visible(False)
        
        plt.suptitle(f'Attention Pattern Evolution - Layer {layer_idx}, Head {head_idx}', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_name:
            save_path = os.path.join(self.save_dir, f"{save_name}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Attention evolution saved to {save_path}")
        
        plt.show()
        plt.close()
    
    def plot_layer_activations(
        self,
        activations: Dict[str, np.ndarray],
        save_name: str = None
    ):
        """
        각 레이어의 활성화 분포 시각화
        
        Args:
            activations: {'layer_name': activation_array}
        """
        num_layers = len(activations)
        if num_layers == 0:
            print("No activations provided")
            return
        
        fig, axes = plt.subplots(2, (num_layers + 1) // 2, figsize=(5 * ((num_layers + 1) // 2), 8))
        if num_layers == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, (layer_name, activation) in enumerate(activations.items()):
            ax = axes[i]
            
            # 히스토그램
            ax.hist(activation.flatten(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            ax.set_title(f'{layer_name}', fontsize=12)
            ax.set_xlabel('Activation Value', fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # 통계 정보 추가
            mean_val = np.mean(activation)
            std_val = np.std(activation)
            ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8, 
                      label=f'Mean: {mean_val:.3f}')
            ax.axvline(mean_val + std_val, color='orange', linestyle='--', alpha=0.8,
                      label=f'±STD: {std_val:.3f}')
            ax.axvline(mean_val - std_val, color='orange', linestyle='--', alpha=0.8)
            ax.legend(fontsize=8)
        
        # 빈 서브플롯 숨기기
        for i in range(num_layers, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Layer Activation Distributions', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_name:
            save_path = os.path.join(self.save_dir, f"{save_name}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Layer activations saved to {save_path}")
        
        plt.show()
        plt.close()
    
    def save_analysis_report(
        self,
        analysis_data: Dict[str, Any],
        filename: str = None
    ):
        """
        분석 결과를 JSON 형태로 저장
        
        Args:
            analysis_data: 분석 데이터
            filename: 저장할 파일명
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"transformer_analysis_{timestamp}.json"
        
        save_path = os.path.join(self.save_dir, filename)
        
        # numpy array를 list로 변환
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        analysis_data_json = convert_numpy(analysis_data)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_data_json, f, indent=2, ensure_ascii=False)
        
        print(f"Analysis report saved to {save_path}")


def create_sample_visualizations():
    """샘플 시각화 생성 (테스트용)"""
    visualizer = TransformerVisualizer("sample_visualizations")
    
    # 샘플 데이터 생성
    seq_len = 8
    num_heads = 4
    
    # 1. Attention heatmap
    attention = np.random.rand(seq_len, seq_len)
    attention = attention / attention.sum(axis=-1, keepdims=True)  # Normalize
    tokens = [f"token_{i}" for i in range(seq_len)]
    
    visualizer.plot_attention_heatmap(
        attention, tokens, layer_idx=0, head_idx=0, 
        save_name="sample_attention_heatmap"
    )
    
    # 2. Multi-head attention
    multi_head_attention = np.random.rand(num_heads, seq_len, seq_len)
    multi_head_attention = multi_head_attention / multi_head_attention.sum(axis=-1, keepdims=True)
    
    visualizer.plot_multi_head_attention(
        multi_head_attention, tokens, layer_idx=0,
        save_name="sample_multihead_attention"
    )
    
    # 3. QKV evolution
    qkv_history = []
    for step in range(0, 1000, 100):
        qkv_history.append({
            'step': step,
            'layer_0': {
                'Q': np.random.randn() * 0.1 + 0.5,
                'K': np.random.randn() * 0.1 + 0.3,
                'V': np.random.randn() * 0.1 + 0.7
            },
            'layer_1': {
                'Q': np.random.randn() * 0.1 + 0.4,
                'K': np.random.randn() * 0.1 + 0.2,
                'V': np.random.randn() * 0.1 + 0.6
            }
        })
    
    visualizer.plot_qkv_evolution(qkv_history, save_name="sample_qkv_evolution")
    
    # 4. Training curves
    epochs = 20
    train_losses = [5.0 * np.exp(-0.1 * i) + 0.5 + np.random.randn() * 0.1 for i in range(epochs)]
    val_losses = [5.2 * np.exp(-0.08 * i) + 0.7 + np.random.randn() * 0.15 for i in range(epochs)]
    
    visualizer.plot_training_curves(
        train_losses, val_losses,
        save_name="sample_training_curves"
    )
    
    # 5. Layer activations
    activations = {
        'layer_0_attention': np.random.normal(0, 1, 1000),
        'layer_0_ffn': np.random.normal(0.5, 0.8, 1000),
        'layer_1_attention': np.random.normal(-0.2, 1.2, 1000),
        'layer_1_ffn': np.random.normal(0.3, 0.6, 1000),
    }
    
    visualizer.plot_layer_activations(activations, save_name="sample_layer_activations")
    
    print("Sample visualizations created successfully!")


if __name__ == "__main__":
    create_sample_visualizations()