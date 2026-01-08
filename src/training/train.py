"""
train_advanced.py - Advanced Training with Dynamic Control and Comprehensive Logging

Features:
- Resume training from any checkpoint
- Change hyperparameters without retraining
- Fine-tuning strategies (freeze/unfreeze layers)
- Dynamic learning rate adjustment
- Layer-wise learning rates
- Real-time Excel and graph generation
- Per-measurement error tracking
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import time
from tqdm import tqdm
import numpy as np

from src.config.config import Config
from src.models.body_measurement_model import create_model
from src.models.dataset import create_dataloaders
from src.utils.logger import TrainingLogger

class EarlyStopper:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience=15, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        
    def early_stop(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
            return False
        
class AdvancedTrainer:
    """
    Advanced trainer with dynamic control and fine-tuning capabilities.
    """
    
    def __init__(self, config: Config, resume_from: str = None):
        """
        Initialize trainer.
        
        Args:
            config: Configuration object
            resume_from: Path to checkpoint to resume from (optional)
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 Using device: {self.device}")
        
        # Create model
        print("\n📦 Creating model...")
        self.model = create_model(config, model_type='resnet50', pretrained=True)
        self.model = self.model.to(self.device)
        
        # Count parameters
        self._print_parameter_info()
        
        # Create dataloaders
        print("\n📊 Loading data...")
        self.train_loader, self.val_loader = create_dataloaders(config)
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Optimizer (will be created/updated dynamically)
        self.optimizer = None
        self.scheduler = None
        self._create_optimizer()
        
        # Setup directories
        self.checkpoint_dir = Path('checkpoints')
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.log_dir = Path('logs')
        self.log_dir.mkdir(exist_ok=True)
        
        # Initialize custom logger
        experiment_name = f"bmnet_{time.strftime('%Y%m%d_%H%M%S')}"
        self.logger = TrainingLogger(
            log_dir=self.log_dir,
            experiment_name=experiment_name
        )
        
        # TensorBoard writer
        self.writer = SummaryWriter(self.log_dir / experiment_name / 'tensorboard')
        
        # Training state
        self.best_val_loss = float('inf')
        self.start_epoch = 0
        self.global_step = 0
        self.training_history = []
        self.early_stopper = EarlyStopper(patience=15, min_delta=0.001)
        # Resume from checkpoint if provided
        if resume_from:
            self.load_checkpoint(resume_from)
    
    def _print_parameter_info(self):
        """Print parameter information by layer."""
        total_params = 0
        trainable_params = 0
        frozen_params = 0
        
        print("\n📊 Model Parameters:")
        for name, param in self.model.named_parameters():
            num_params = param.numel()
            total_params += num_params
            if param.requires_grad:
                trainable_params += num_params
            else:
                frozen_params += num_params
        
        print(f"  Total: {total_params:,}")
        print(f"  Trainable: {trainable_params:,}")
        print(f"  Frozen: {frozen_params:,}")
    
    def _create_optimizer(self, learning_rate: float = 1e-4, 
                         weight_decay: float = 1e-5):
        """
        Create or recreate optimizer with new parameters.
        
        Args:
            learning_rate: Learning rate
            weight_decay: Weight decay (L2 regularization)
        """
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        print(f"✓ Optimizer created: LR={learning_rate}, WD={weight_decay}")
    
    def freeze_backbone(self):
        """
        Freeze ResNet backbone (only train fusion and regression head).
        Useful for fine-tuning on small datasets.
        """
        print("\n🔒 Freezing backbone...")
        for name, param in self.model.named_parameters():
            if 'backbone' in name:
                param.requires_grad = False
        
        # Recreate optimizer with only trainable parameters
        self._create_optimizer()
        self._print_parameter_info()
    
    def unfreeze_backbone(self):
        """
        Unfreeze ResNet backbone for full model training.
        """
        print("\n🔓 Unfreezing backbone...")
        for param in self.model.parameters():
            param.requires_grad = True
        
        # Recreate optimizer
        self._create_optimizer()
        self._print_parameter_info()
    
    def unfreeze_last_n_layers(self, n: int = 10):
        """
        Unfreeze only the last N layers of backbone.
        Progressive unfreezing strategy.
        
        Args:
            n: Number of layers to unfreeze from the end
        """
        print(f"\n🔓 Unfreezing last {n} layers...")
        
        # First freeze everything
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Get backbone layers
        backbone_layers = list(self.model.backbone.named_parameters())
        
        # Unfreeze last n layers
        for name, param in backbone_layers[-n:]:
            param.requires_grad = True
        
        # Always unfreeze fusion and regression head
        for name, param in self.model.named_parameters():
            if 'fusion' in name or 'regression_head' in name:
                param.requires_grad = True
        
        # Recreate optimizer
        self._create_optimizer()
        self._print_parameter_info()
    
    def set_learning_rate(self, new_lr: float):
        """
        Change learning rate during training.
        
        Args:
            new_lr: New learning rate
        """
        print(f"\n📉 Changing learning rate to {new_lr}")
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
    
    def create_layerwise_optimizer(self):
        """
        Create optimizer with different learning rates for different layers.
        Backbone: lower LR, Head: higher LR
        """
        print("\n🎯 Creating layer-wise optimizer...")
        
        backbone_params = []
        fusion_params = []
        head_params = []
        
        for name, param in self.model.named_parameters():
            if 'backbone' in name:
                backbone_params.append(param)
            elif 'fusion' in name:
                fusion_params.append(param)
            else:
                head_params.append(param)
        
        self.optimizer = optim.Adam([
            {'params': backbone_params, 'lr': 1e-5, 'name': 'backbone'},
            {'params': fusion_params, 'lr': 5e-5, 'name': 'fusion'},
            {'params': head_params, 'lr': 1e-4, 'name': 'head'}
        ], weight_decay=1e-5)
        
        print("✓ Layer-wise learning rates:")
        print(f"  Backbone: 1e-5")
        print(f"  Fusion: 5e-5")
        print(f"  Head: 1e-4")
    
    def save_checkpoint(self, epoch: int, val_loss: float, 
                       is_best: bool = False, note: str = ""):
        """
        Save model checkpoint with training state.
        
        Args:
            epoch: Current epoch
            val_loss: Validation loss
            is_best: Whether this is the best model
            note: Optional note to save with checkpoint
        """
        # Get frozen layer info
        frozen_layers = [name for name, param in self.model.named_parameters() 
                        if not param.requires_grad]
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
            'global_step': self.global_step,
            'frozen_layers': frozen_layers,
            'note': note,
            'training_history': self.training_history
        }
        
        # Save last checkpoint
        last_path = self.checkpoint_dir / 'last_checkpoint.pth'
        torch.save(checkpoint, last_path)
        
        # Save epoch checkpoint (every 10 epochs)
        if epoch % 10 == 0:
            epoch_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save(checkpoint, epoch_path)
            print(f"✓ Saved epoch checkpoint: {epoch_path}")
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"✓ Saved best model (val_loss: {val_loss:.4f})")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load checkpoint and resume training.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        print(f"\n📂 Loading checkpoint from: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        if self.optimizer is None:
            self._create_optimizer()
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training state
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.global_step = checkpoint.get('global_step', 0)
        self.training_history = checkpoint.get('training_history', [])
        
        print(f"✓ Resumed from epoch {checkpoint['epoch']}")
        print(f"✓ Best val loss: {self.best_val_loss:.4f}")
        
        if 'note' in checkpoint and checkpoint['note']:
            print(f"✓ Note: {checkpoint['note']}")
        
        if 'frozen_layers' in checkpoint:
            frozen_count = len(checkpoint['frozen_layers'])
            print(f"✓ Frozen layers: {frozen_count}")
    
    def train_epoch(self, epoch: int) -> tuple:
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        running_mae = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]")
        
        for batch_idx, batch in enumerate(pbar):
            mask = batch['mask'].to(self.device)
            mask_left = batch['mask_left'].to(self.device)
            height = batch['height'].to(self.device)
            targets = batch['measurements'].to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(mask, mask_left, height)
            
            loss = self.criterion(outputs, targets)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            mae = torch.abs(outputs - targets).mean()
            
            running_loss += loss.item()
            running_mae += mae.item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'mae': f'{mae.item():.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
            
            if batch_idx % 10 == 0:
                self.writer.add_scalar('Train/Loss', loss.item(), self.global_step)
                self.writer.add_scalar('Train/MAE', mae.item(), self.global_step)
            
            self.global_step += 1
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_mae = running_mae / len(self.train_loader)
        
        return epoch_loss, epoch_mae
    
    def validate(self, epoch: int) -> tuple:
        """Validate the model and calculate per-measurement MAE."""
        self.model.eval()
        running_loss = 0.0
        running_mae = 0.0
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1} [Val]")
            
            for batch in pbar:
                mask = batch['mask'].to(self.device)
                mask_left = batch['mask_left'].to(self.device)
                height = batch['height'].to(self.device)
                targets = batch['measurements'].to(self.device)
                
                outputs = self.model(mask, mask_left, height)
                
                loss = self.criterion(outputs, targets)
                mae = torch.abs(outputs - targets).mean()
                
                running_loss += loss.item()
                running_mae += mae.item()
                
                # Store for per-measurement analysis
                all_predictions.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'mae': f'{mae.item():.4f}'
                })
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_mae = running_mae / len(self.val_loader)
        
        # Calculate per-measurement MAE
        all_predictions = np.vstack(all_predictions)
        all_targets = np.vstack(all_targets)
        per_measurement_mae = np.abs(all_predictions - all_targets).mean(axis=0)
        
        per_measurement_dict = {
            name: float(mae) for name, mae in 
            zip(self.config.measurement.MEASUREMENT_COLUMNS, per_measurement_mae)
        }
        
        return epoch_loss, epoch_mae, per_measurement_dict
    
    def train(self, num_epochs: int = 50, 
              freeze_strategy: str = None,
              unfreeze_at_epoch: int = None):
        """
        Main training loop with dynamic control.
        
        Args:
            num_epochs: Number of epochs to train
            freeze_strategy: 'freeze_backbone', 'progressive', or None
            unfreeze_at_epoch: Epoch to unfreeze (if using freeze_strategy)
        """
        print("\n" + "="*80)
        print("STARTING TRAINING")
        print("="*80)
        print(f"Epochs: {num_epochs}")
        print(f"Starting from epoch: {self.start_epoch}")
        print(f"Device: {self.device}")
        print(f"Batch size: {self.config.training.BATCH_SIZE}")
        
        if freeze_strategy:
            print(f"Strategy: {freeze_strategy}")
            if unfreeze_at_epoch:
                print(f"Will unfreeze at epoch: {unfreeze_at_epoch}")
        
        print("="*80 + "\n")
        
        # Apply initial freeze strategy
        if freeze_strategy == 'freeze_backbone' and self.start_epoch == 0:
            self.freeze_backbone()
        
        start_time = time.time()
        
        for epoch in range(self.start_epoch, num_epochs):
            print(f"\n{'='*80}")
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"{'='*80}")
            
            # Dynamic unfreezing
            if unfreeze_at_epoch and epoch == unfreeze_at_epoch:
                if freeze_strategy == 'freeze_backbone':
                    self.unfreeze_backbone()
                elif freeze_strategy == 'progressive':
                    self.unfreeze_last_n_layers(10)
            
            # Train
            train_loss, train_mae = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_mae, per_measurement_mae = self.validate(epoch)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            if self.early_stopper.early_stop(val_loss):
                print(f"\n⚠️ Early stopping triggered at epoch {epoch+1}")
                print(f"Best val loss: {self.early_stopper.best_loss:.4f}")
                break
            # Save to history
            self.training_history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_mae': train_mae,
                'val_loss': val_loss,
                'val_mae': val_mae,
                'lr': self.optimizer.param_groups[0]['lr']
            })
            
            # Log to TensorBoard
            self.writer.add_scalar('Epoch/Train_Loss', train_loss, epoch)
            self.writer.add_scalar('Epoch/Val_Loss', val_loss, epoch)
            self.writer.add_scalar('Epoch/Learning_Rate', 
                                  self.optimizer.param_groups[0]['lr'], epoch)
            
            # Log to custom logger (Excel + Plots)
            self.logger.log_epoch(
                epoch=epoch,
                train_loss=train_loss,
                train_mae=train_mae,
                val_loss=val_loss,
                val_mae=val_mae,
                learning_rate=self.optimizer.param_groups[0]['lr'],
                per_measurement_mae=per_measurement_mae
            )
            
            # Print summary
            print(f"\n📊 Epoch {epoch+1} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train MAE: {train_mae:.4f}")
            print(f"  Val Loss:   {val_loss:.4f} | Val MAE:   {val_mae:.4f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            self.save_checkpoint(epoch, val_loss, is_best)
            
            # Early stopping
            if self.optimizer.param_groups[0]['lr'] < 1e-7:
                print("\n⚠️  Learning rate too small. Stopping training.")
                break
        
        total_time = time.time() - start_time
        
        # Generate final plots and reports
        print("\n📊 Generating final plots and reports...")
        self.logger.plot_per_measurement_mae(self.config.measurement.MEASUREMENT_COLUMNS)
        self.logger.export_summary_report()
        
        print("\n" + "="*80)
        print("✅ TRAINING COMPLETE!")
        print("="*80)
        print(f"Total time: {total_time/3600:.2f} hours")
        print(f"Best val loss: {self.best_val_loss:.4f}")
        print(f"Logs saved to: {self.logger.log_dir}")
        print(f"  - Excel: {self.logger.data_dir / 'training_history.xlsx'}")
        print(f"  - Plots: {self.logger.plots_dir}")
        print(f"  - Report: {self.logger.data_dir / 'training_report.txt'}")
        print("="*80 + "\n")
        
        self.writer.close()


def main():
    """Main function with examples."""
    config = Config()
    
    # Example 1: Train from scratch with frozen backbone
    # Load best checkpoint
    trainer = AdvancedTrainer(config, resume_from='checkpoints/best_model.pth')
    
    # Increase weight decay (stronger L2 regularization)
    trainer._create_optimizer(learning_rate=1e-4, weight_decay=1e-3)
    
    # Train with delayed unfreezing
    trainer.train(num_epochs=100, freeze_strategy='freeze_backbone', unfreeze_at_epoch=50)
    
    # Example 2: Resume training from checkpoint
    # trainer = AdvancedTrainer(config, resume_from='checkpoints/last_checkpoint.pth')
    # trainer.train(num_epochs=50)
    
    # Example 3: Start fresh training
    # trainer = AdvancedTrainer(config)
    # trainer.train(num_epochs=50)


if __name__ == "__main__":
    main()
