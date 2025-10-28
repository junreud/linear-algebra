"""
Hugging Face 기반 커스텀 트레이너 with Advanced Tracking
"""
import torch
import torch.nn as nn
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import PredictionOutput
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from typing import Dict, List, Optional, Tuple, Any
import os
from datetime import datetime

from .model import TrackedTransformerModel, TrackedTransformerConfig


class AttentionTrackingCallback(TrainerCallback):
    """
    Attention weight와 internal state를 추적하는 콜백
    """
    
    def __init__(self, save_dir: str = "attention_tracking", track_frequency: int = 100):
        self.save_dir = save_dir
        self.track_frequency = track_frequency
        self.attention_history = []
        self.layer_outputs_history = []
        self.step_count = 0
        
        os.makedirs(save_dir, exist_ok=True)
    
    def on_step_end(self, args, state, control, model=None, **kwargs):
        """매 스텝마다 호출되는 콜백"""
        self.step_count += 1
        
        # 일정 간격마다 추적
        if self.step_count % self.track_frequency == 0:
            self._track_model_states(model, state.global_step)
    
    def _track_model_states(self, model, global_step):
        """모델의 내부 상태 추적"""
        if hasattr(model, 'module'):  # DataParallel 처리
            model = model.module
        
        # 각 레이어의 attention 정보 수집
        attention_info = {}
        for i, layer in enumerate(model.layers):
            if hasattr(layer.attention, 'tracked_states') and layer.attention.tracked_states:
                attention_info[f'layer_{i}'] = {
                    'attention_probs': layer.attention.tracked_states['attention_probs'][0, 0].cpu().numpy(),  # 첫 번째 배치, 첫 번째 헤드
                    'query_mean': layer.attention.tracked_states['query'].mean().item(),
                    'key_mean': layer.attention.tracked_states['key'].mean().item(),
                    'value_mean': layer.attention.tracked_states['value'].mean().item(),
                }
        
        if attention_info:
            self.attention_history.append({
                'step': global_step,
                'attention_info': attention_info
            })
            
            print(f"Tracked attention at step {global_step}")


class InternalStateTracker:
    """
    Transformer 내부 상태를 실시간으로 추적하는 클래스
    """
    
    def __init__(self, model: TrackedTransformerModel):
        self.model = model
        self.hooks = []
        self.activations = {}
        self.gradients = {}
        
    def register_hooks(self):
        """모델에 hook 등록"""
        
        def create_hook(name):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    self.activations[name] = output[0].detach().cpu()
                else:
                    self.activations[name] = output.detach().cpu()
            return hook_fn
        
        def create_grad_hook(name):
            def grad_hook_fn(grad):
                self.gradients[name] = grad.detach().cpu()
                return grad
            return grad_hook_fn
        
        # 각 레이어에 hook 등록
        for i, layer in enumerate(self.model.layers):
            # Forward hooks
            hook = layer.attention.register_forward_hook(create_hook(f'layer_{i}_attention'))
            self.hooks.append(hook)
            
            hook = layer.intermediate.register_forward_hook(create_hook(f'layer_{i}_ffn_intermediate'))
            self.hooks.append(hook)
            
            hook = layer.output.register_forward_hook(create_hook(f'layer_{i}_ffn_output'))
            self.hooks.append(hook)
    
    def remove_hooks(self):
        """Hook 제거"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def get_activation_stats(self) -> Dict[str, Dict[str, float]]:
        """활성화 통계 반환"""
        stats = {}
        for name, activation in self.activations.items():
            stats[name] = {
                'mean': activation.mean().item(),
                'std': activation.std().item(),
                'min': activation.min().item(),
                'max': activation.max().item(),
                'sparsity': (activation == 0).float().mean().item()
            }
        return stats


class CustomTransformerTrainer(Trainer):
    """
    Hugging Face Trainer를 확장한 커스텀 트레이너
    """
    
    def __init__(
        self,
        model: TrackedTransformerModel,
        args: TrainingArguments,
        train_dataset: Dataset,
        eval_dataset: Dataset = None,
        tokenizer: AutoTokenizer = None,
        data_collator=None,
        compute_metrics=None,
        track_internals: bool = True,
        wandb_project: str = None,
        **kwargs
    ):
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            **kwargs
        )
        
        self.track_internals = track_internals
        self.state_tracker = InternalStateTracker(model) if track_internals else None
        self.attention_callback = AttentionTrackingCallback()
        
        # Callback 추가
        self.add_callback(self.attention_callback)
        
        # WandB 설정
        if wandb_project:
            wandb.init(project=wandb_project, config=model.config.to_dict())
        
        if track_internals:
            self.state_tracker.register_hooks()
            print("Internal state tracking enabled")
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Loss 계산 with tracking"""
        
        # 입력에서 labels 분리
        labels = inputs.pop("labels", None)
        
        # Forward pass
        outputs = model(**inputs, output_attentions=True, output_hidden_states=True)
        
        # 언어 모델링을 위한 loss 계산
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = outputs.last_hidden_state[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Linear layer for vocab projection (if not already in model)
            if not hasattr(model, 'lm_head'):
                model.lm_head = nn.Linear(model.config.hidden_size, model.config.vocab_size, bias=False)
                model.lm_head = model.lm_head.to(shift_logits.device)
            
            logits = model.lm_head(shift_logits)
            
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.size(-1)), shift_labels.view(-1))
        else:
            loss = torch.tensor(0.0, requires_grad=True, device=outputs.last_hidden_state.device)
        
        # 내부 상태 추적
        if self.track_internals and self.state_tracker:
            activation_stats = self.state_tracker.get_activation_stats()
            
            # WandB에 로깅
            if wandb.run:
                log_dict = {'train/loss': loss.item()}
                for layer_name, stats in activation_stats.items():
                    for stat_name, value in stats.items():
                        log_dict[f'activations/{layer_name}/{stat_name}'] = value
                wandb.log(log_dict)
        
        return (loss, outputs) if return_outputs else loss
    
    def evaluation_loop(
        self,
        dataloader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> PredictionOutput:
        """평가 루프 with tracking"""
        
        # 평가 중에는 tracking 비활성화
        original_track = self.model.config.track_internal_states
        self.model.config.track_internal_states = False
        
        results = super().evaluation_loop(
            dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix
        )
        
        # Tracking 상태 복원
        self.model.config.track_internal_states = original_track
        
        return results
    
    def log_attention_patterns(self, save_dir: str = "attention_visualizations"):
        """Attention pattern 시각화 및 저장"""
        os.makedirs(save_dir, exist_ok=True)
        
        if not self.attention_callback.attention_history:
            print("No attention history found.")
            return
        
        # 최신 attention pattern 시각화
        latest_info = self.attention_callback.attention_history[-1]
        attention_info = latest_info['attention_info']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, (layer_name, info) in enumerate(attention_info.items()):
            if i >= 6:  # 최대 6개 레이어
                break
                
            attention_weights = info['attention_probs']
            
            sns.heatmap(
                attention_weights,
                ax=axes[i],
                cmap='Blues',
                cbar=True,
                square=True,
                xticklabels=False,
                yticklabels=False
            )
            axes[i].set_title(f'{layer_name.replace("_", " ").title()}')
        
        # 빈 서브플롯 숨기기
        for i in range(len(attention_info), 6):
            axes[i].set_visible(False)
        
        plt.suptitle(f'Attention Patterns at Step {latest_info["step"]}', fontsize=16)
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, f'attention_step_{latest_info["step"]}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Attention patterns saved to {save_path}")
    
    def plot_qkv_evolution(self, save_dir: str = "qkv_evolution"):
        """QKV 값의 변화 시각화"""
        os.makedirs(save_dir, exist_ok=True)
        
        if not self.attention_callback.attention_history:
            print("No QKV history found.")
            return
        
        # QKV 평균값 추출
        steps = []
        layer_stats = {}
        
        for entry in self.attention_callback.attention_history:
            steps.append(entry['step'])
            
            for layer_name, info in entry['attention_info'].items():
                if layer_name not in layer_stats:
                    layer_stats[layer_name] = {'query': [], 'key': [], 'value': []}
                
                layer_stats[layer_name]['query'].append(info['query_mean'])
                layer_stats[layer_name]['key'].append(info['key_mean'])
                layer_stats[layer_name]['value'].append(info['value_mean'])
        
        # 시각화
        num_layers = len(layer_stats)
        fig, axes = plt.subplots(num_layers, 1, figsize=(12, 4 * num_layers))
        
        if num_layers == 1:
            axes = [axes]
        
        for i, (layer_name, stats) in enumerate(layer_stats.items()):
            axes[i].plot(steps, stats['query'], 'b-', label='Query', alpha=0.8)
            axes[i].plot(steps, stats['key'], 'r-', label='Key', alpha=0.8)
            axes[i].plot(steps, stats['value'], 'g-', label='Value', alpha=0.8)
            
            axes[i].set_title(f'{layer_name.replace("_", " ").title()} - QKV Evolution')
            axes[i].set_xlabel('Training Step')
            axes[i].set_ylabel('Mean Value')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, 'qkv_evolution.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"QKV evolution plot saved to {save_path}")
    
    def cleanup(self):
        """리소스 정리"""
        if self.state_tracker:
            self.state_tracker.remove_hooks()
        
        if wandb.run:
            wandb.finish()


def create_training_arguments(
    output_dir: str = "./results",
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 8,
    per_device_eval_batch_size: int = 8,
    learning_rate: float = 5e-5,
    warmup_steps: int = 500,
    weight_decay: float = 0.01,
    logging_steps: int = 10,
    save_steps: int = 500,
    eval_steps: int = 500,
    evaluation_strategy: str = "steps",
    save_total_limit: int = 3,
    load_best_model_at_end: bool = True,
    metric_for_best_model: str = "eval_loss",
    greater_is_better: bool = False,
    fp16: bool = True,
    gradient_accumulation_steps: int = 1,
    dataloader_num_workers: int = 0,
    **kwargs
) -> TrainingArguments:
    """Training arguments 생성"""
    
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        evaluation_strategy=evaluation_strategy,
        save_total_limit=save_total_limit,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
        fp16=fp16,
        gradient_accumulation_steps=gradient_accumulation_steps,
        dataloader_num_workers=dataloader_num_workers,
        report_to=["wandb"] if wandb.run else [],
        **kwargs
    )


class SimpleTokenDataset(Dataset):
    """Simple dataset for language modeling"""
    
    def __init__(self, tokenizer, texts: List[str], max_length: int = 128):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze(),  # For language modeling
        }


def test_custom_trainer():
    """Custom trainer 테스트"""
    from transformers import AutoTokenizer
    
    print("=== Custom Transformer Trainer Test ===")
    
    # 모델과 토크나이저 설정
    from .model import create_tracked_transformer
    
    model = create_tracked_transformer(
        vocab_size=1000,
        hidden_size=256,
        num_layers=2,
        num_heads=8,
        track_internal_states=True
    )
    
    # 간단한 데이터셋
    texts = [
        "This is a test sentence for language modeling.",
        "Transformer models are very powerful.",
        "Attention is all you need for sequence modeling.",
        "Deep learning with transformers is amazing.",
    ] * 100  # 데이터 늘리기
    
    # 임시 토크나이저 (실제로는 사전 훈련된 토크나이저 사용)
    class DummyTokenizer:
        def __init__(self, vocab_size=1000):
            self.vocab_size = vocab_size
            self.pad_token_id = 0
        
        def __call__(self, text, **kwargs):
            # 단순 토큰화 (실제로는 더 복잡)
            tokens = [hash(word) % self.vocab_size for word in text.split()]
            max_length = kwargs.get('max_length', 128)
            
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            else:
                tokens += [self.pad_token_id] * (max_length - len(tokens))
            
            return {
                'input_ids': torch.tensor([tokens]),
                'attention_mask': torch.tensor([[1 if t != self.pad_token_id else 0 for t in tokens]])
            }
    
    tokenizer = DummyTokenizer()
    
    # 데이터셋 생성
    train_dataset = SimpleTokenDataset(tokenizer, texts[:80])
    eval_dataset = SimpleTokenDataset(tokenizer, texts[80:])
    
    # Training arguments
    training_args = create_training_arguments(
        output_dir="./test_results",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        learning_rate=1e-4,
        logging_steps=5,
        eval_steps=20,
        save_steps=20,
        fp16=False,  # 테스트에서는 False
    )
    
    # Trainer 생성
    trainer = CustomTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        track_internals=True,
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Visualizing results...")
    trainer.log_attention_patterns()
    trainer.plot_qkv_evolution()
    
    # 정리
    trainer.cleanup()
    
    print("Test completed!")


if __name__ == "__main__":
    test_custom_trainer()