"""
PyTorch 순정 Transformer 학습 스크립트
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse
import os
import time
from typing import List, Dict, Tuple

from model import create_transformer_model


class SimpleTranslationDataset(Dataset):
    """
    간단한 번역 데이터셋 (synthetic data)
    실제로는 WMT 데이터셋 등을 사용하지만, 여기서는 학습 과정 이해를 위한 간단한 데이터
    """
    
    def __init__(self, num_samples: int = 10000, src_vocab_size: int = 1000, tgt_vocab_size: int = 1000, max_len: int = 20):
        self.num_samples = num_samples
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.max_len = max_len
        
        # Special tokens
        self.pad_token = 0
        self.start_token = 1
        self.end_token = 2
        
        # Generate synthetic data
        self.data = self._generate_data()
    
    def _generate_data(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Synthetic translation data 생성"""
        data = []
        
        for _ in range(self.num_samples):
            # Source sequence (random length)
            src_len = np.random.randint(3, self.max_len - 2)
            src = torch.randint(3, self.src_vocab_size, (src_len,))
            
            # Target sequence (similar to source but with some transformation)
            # 간단한 규칙: src의 역순 + 약간의 노이즈
            tgt_len = np.random.randint(3, self.max_len - 2)
            tgt = torch.randint(3, self.tgt_vocab_size, (tgt_len,))
            
            # Add start/end tokens to target
            tgt_input = torch.cat([torch.tensor([self.start_token]), tgt])
            tgt_output = torch.cat([tgt, torch.tensor([self.end_token])])
            
            data.append((src, tgt_input, tgt_output))
        
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def collate_fn(self, batch):
        """Batch collation with padding"""
        src_seqs, tgt_input_seqs, tgt_output_seqs = zip(*batch)
        
        # Pad sequences
        src_padded = nn.utils.rnn.pad_sequence(src_seqs, batch_first=True, padding_value=self.pad_token)
        tgt_input_padded = nn.utils.rnn.pad_sequence(tgt_input_seqs, batch_first=True, padding_value=self.pad_token)
        tgt_output_padded = nn.utils.rnn.pad_sequence(tgt_output_seqs, batch_first=True, padding_value=self.pad_token)
        
        return src_padded, tgt_input_padded, tgt_output_padded


class TransformerTrainer:
    """Transformer 학습을 위한 트레이너 클래스"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        debug: bool = False
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.debug = debug
        
        # Optimizer and scheduler
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.98))
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.attention_weights_history = []
        
        print(f"Trainer initialized:")
        print(f"  Device: {device}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    def train_epoch(self, epoch: int) -> float:
        """한 에폭 학습"""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, (src, tgt_input, tgt_output) in enumerate(pbar):
            src = src.to(self.device)
            tgt_input = tgt_input.to(self.device)
            tgt_output = tgt_output.to(self.device)
            
            # Generate masks
            src_mask, tgt_mask = self.model.generate_masks(src, tgt_input)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if self.debug and batch_idx == 0:
                print(f"\nDebug info for batch {batch_idx}:")
                print(f"Source shape: {src.shape}")
                print(f"Target input shape: {tgt_input.shape}")
                print(f"Target output shape: {tgt_output.shape}")
                
                # Enable debug mode for first batch
                original_debug = self.model.debug
                self.model.debug = True
                
                logits = self.model(src, tgt_input, src_mask, tgt_mask)
                
                # Restore debug mode
                self.model.debug = original_debug
            else:
                logits = self.model(src, tgt_input, src_mask, tgt_mask)
            
            # Calculate loss
            # Reshape for cross entropy: (batch_size * seq_len, vocab_size)
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = tgt_output.view(-1)
            
            loss = self.criterion(logits_flat, targets_flat)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss / (batch_idx + 1):.4f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
            
            # QKV 추적 (첫 번째 배치만)
            if batch_idx == 0 and hasattr(self.model.encoder_layers[0].self_attention, 'debug_info'):
                self._track_attention_weights(epoch, batch_idx)
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate(self, epoch: int) -> float:
        """검증"""
        self.model.eval()
        total_loss = 0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for src, tgt_input, tgt_output in tqdm(self.val_loader, desc="Validation"):
                src = src.to(self.device)
                tgt_input = tgt_input.to(self.device)
                tgt_output = tgt_output.to(self.device)
                
                # Generate masks
                src_mask, tgt_mask = self.model.generate_masks(src, tgt_input)
                
                # Forward pass
                logits = self.model(src, tgt_input, src_mask, tgt_mask)
                
                # Calculate loss
                logits_flat = logits.view(-1, logits.size(-1))
                targets_flat = tgt_output.view(-1)
                
                loss = self.criterion(logits_flat, targets_flat)
                total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        
        return avg_loss
    
    def _track_attention_weights(self, epoch: int, batch_idx: int):
        """Attention weight 추적 및 저장"""
        # 첫 번째 encoder layer의 self-attention weights 추출
        if hasattr(self.model.encoder_layers[0].self_attention, 'debug_info'):
            debug_info = self.model.encoder_layers[0].self_attention.debug_info
            if 'attention_weights' in debug_info:
                attention_weights = debug_info['attention_weights'][0, 0].cpu().numpy()  # 첫 번째 배치, 첫 번째 헤드
                
                self.attention_weights_history.append({
                    'epoch': epoch,
                    'batch': batch_idx,
                    'weights': attention_weights
                })
    
    def train(self, num_epochs: int, save_dir: str = "checkpoints"):
        """전체 학습 과정"""
        os.makedirs(save_dir, exist_ok=True)
        
        best_val_loss = float('inf')
        
        print(f"\nStarting training for {num_epochs} epochs...")
        
        for epoch in range(1, num_epochs + 1):
            start_time = time.time()
            
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss = self.validate(epoch)
            
            # Update scheduler
            self.scheduler.step()
            
            epoch_time = time.time() - start_time
            
            print(f"\nEpoch {epoch:3d}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  Time:       {epoch_time:.2f}s")
            print(f"  LR:         {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, os.path.join(save_dir, 'best_model.pt'))
                print(f"  ✓ New best model saved!")
            
            # Save checkpoint every 10 epochs
            if epoch % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses,
                    'attention_history': self.attention_weights_history,
                }, os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pt'))
        
        print(f"\nTraining completed!")
        print(f"Best validation loss: {best_val_loss:.4f}")
    
    def plot_training_curves(self, save_path: str = None):
        """학습 곡선 시각화"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss', alpha=0.8)
        plt.plot(self.val_losses, label='Val Loss', alpha=0.8)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        if self.attention_weights_history:
            epochs = [item['epoch'] for item in self.attention_weights_history]
            avg_attention = [np.mean(item['weights']) for item in self.attention_weights_history]
            plt.plot(epochs, avg_attention, 'o-', alpha=0.7)
            plt.xlabel('Epoch')
            plt.ylabel('Average Attention Weight')
            plt.title('Attention Weight Evolution')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_attention_evolution(self, save_dir: str = "attention_plots"):
        """Attention weight 변화 시각화"""
        if not self.attention_weights_history:
            print("No attention weight history found.")
            return
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Plot attention heatmaps for different epochs
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # Select 6 different epochs to show evolution
        indices = np.linspace(0, len(self.attention_weights_history) - 1, 6, dtype=int)
        
        for i, idx in enumerate(indices):
            item = self.attention_weights_history[idx]
            attention = item['weights']
            
            sns.heatmap(
                attention,
                ax=axes[i],
                cmap='Blues',
                cbar=True,
                square=True,
                xticklabels=False,
                yticklabels=False
            )
            axes[i].set_title(f'Epoch {item["epoch"]}')
        
        plt.suptitle('Attention Weight Evolution During Training', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'attention_evolution.png'), dpi=300, bbox_inches='tight')
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Train Transformer from scratch')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--d_model', type=int, default=256, help='Model dimension')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--visualize', action='store_true', help='Enable visualization')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Save directory')
    
    args = parser.parse_args()
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data setup
    print("Creating datasets...")
    train_dataset = SimpleTranslationDataset(num_samples=8000)
    val_dataset = SimpleTranslationDataset(num_samples=2000)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=train_dataset.collate_fn,
        num_workers=0  # 디버깅을 위해 0으로 설정
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=val_dataset.collate_fn,
        num_workers=0
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Model setup
    print("Creating model...")
    model = create_transformer_model(
        src_vocab_size=1000,
        tgt_vocab_size=1000,
        d_model=args.d_model,
        debug=args.debug
    )
    
    # Trainer setup
    trainer = TransformerTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.lr,
        debug=args.debug
    )
    
    # Training
    trainer.train(args.epochs, args.save_dir)
    
    # Visualization
    if args.visualize:
        print("\nCreating visualizations...")
        trainer.plot_training_curves(os.path.join(args.save_dir, 'training_curves.png'))
        trainer.visualize_attention_evolution(os.path.join(args.save_dir, 'attention_plots'))
    
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()