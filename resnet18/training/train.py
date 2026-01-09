"""
train.py - Complete Training Script for Hip & Bust Prediction

Features:
- ResNet-18 with 1-channel grayscale input
- Real-time logging and graphs
- TensorBoard integration
- Per-measurement error tracking
- Early stopping
- Automatic checkpoint saving
"""

import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pandas as pd
import numpy as np

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from resnet18.config.config import Config
from resnet18.models.model import create_model
from resnet18.models.losses import WeightedMSELoss, AdaptiveLoss
from resnet18.models.dataset import create_dataloaders
from resnet18.features.image_preprocessor import ImagePreprocessor
from resnet18.features.measurement_preprocessor import MeasurementPreprocessor
from resnet18.utils.logger import TrainingLogger


class EarlyStopper:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience=15, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
    
    def early_stop(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        self.counter += 1
        print(f"  Early stopping counter: {self.counter}/{self.patience}")
        return self.counter >= self.patience


class Trainer:
    """
    Complete trainer for hip and bust prediction.
    """
    
    def __init__(self, config: Config, resume_from: str = None):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"🚀 Device: {self.device}")
        
        # Load preprocessors
        print("\n📂 Loading preprocessors...")
        self.image_preprocessor = ImagePreprocessor(config)
        self.image_preprocessor.load_params(
            str(config.paths.PROCESSED_DIR / 'image_preprocessor_params.pkl')
        )
        
        self.measurement_preprocessor = MeasurementPreprocessor(config)
        self.measurement_preprocessor.load_params(
            str(config.paths.PROCESSED_DIR / 'measurement_preprocessor_params.pkl')
        )
        
        # Load data
        print("\n📊 Loading data...")
        train_df = pd.read_csv(config.paths.PROCESSED_DIR / 'train_data.csv')
        val_df = pd.read_csv(config.paths.PROCESSED_DIR / 'val_data.csv')
        
        print(f"  Train samples: {len(train_df)}")
        print(f"  Val samples: {len(val_df)}")
        
        # Create dataloaders
        self.train_loader, self.val_loader = create_dataloaders(
            config, self.image_preprocessor, self.measurement_preprocessor,
            train_df, val_df
        )
        
        # Create model
        print("\n🏗️ Creating model...")
        self.model = create_model(config).to(self.device)
        
        # Print parameter info
        self._print_parameter_info()
        
        # Loss function (choose one)
        # Option 1: Weighted MSE (equal weights)
        self.criterion = WeightedMSELoss(hip_weight=1.0, bust_weight=1.0)
        
        # Option 2: Adaptive loss (more robust to outliers)
        # self.criterion = AdaptiveLoss(alpha=0.7)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.training.INITIAL_LR,
            weight_decay=config.training.WEIGHT_DECAY
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=config.training.LR_SCHEDULER_FACTOR,
            patience=config.training.LR_SCHEDULER_PATIENCE,
            verbose=True
        )
        
        # Early stopper
        self.early_stopper = EarlyStopper(
            patience=config.training.EARLY_STOP_PATIENCE,
            min_delta=config.training.EARLY_STOP_MIN_DELTA
        )
        
        # Setup directories
        self.checkpoint_dir = Path('checkpoints_resnet18')
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.log_dir = Path('logs_resnet18')
        self.log_dir.mkdir(exist_ok=True)
        
        # Initialize loggers
        experiment_name = f"hip_bust_{time.strftime('%Y%m%d_%H%M%S')}"
        
        # Custom logger (Excel + Plots)
        self.logger = TrainingLogger(
            log_dir=self.log_dir,
            experiment_name=experiment_name
        )
        
        # TensorBoard logger
        self.writer = SummaryWriter(self.log_dir / experiment_name / 'tensorboard')
        
        # Training state
        self.best_val_loss = float('inf')
        self.start_epoch = 0
        self.global_step = 0
        
        # Resume from checkpoint if provided
        if resume_from:
            self.load_checkpoint(resume_from)
    
    def _print_parameter_info(self):
        """Print model parameter information."""
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen = total - trainable
        
        print(f"\n📊 Model Parameters:")
        print(f"  Total:     {total:,}")
        print(f"  Trainable: {trainable:,}")
        print(f"  Frozen:    {frozen:,}")
    
    def train_epoch(self, epoch):
        """Train one epoch."""
        self.model.train()
        loss_sum = 0.0
        mae_sum = 0.0
        
        # For per-measurement tracking
        hip_errors = []
        bust_errors = []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]")
        
        for batch_idx, batch in enumerate(pbar):
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(
                batch['mask'].to(self.device),
                batch['mask_left'].to(self.device),
                batch['height'].to(self.device)
            )
            targets = batch['measurements'].to(self.device)
            
            # Calculate loss
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Calculate MAE
            mae = torch.abs(outputs - targets).mean()
            
            # Per-measurement errors (denormalized)
            with torch.no_grad():
                outputs_cm = self.measurement_preprocessor.inverse_transform(
                    outputs.cpu().numpy()
                )
                targets_cm = self.measurement_preprocessor.inverse_transform(
                    targets.cpu().numpy()
                )
                
                hip_errors.extend(np.abs(outputs_cm[:, 0] - targets_cm[:, 0]))
                bust_errors.extend(np.abs(outputs_cm[:, 1] - targets_cm[:, 1]))
            
            loss_sum += loss.item()
            mae_sum += mae.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'mae': f'{mae.item():.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
            
            # Log to TensorBoard (every 10 batches)
            if batch_idx % 10 == 0:
                self.writer.add_scalar('Train/Loss_step', loss.item(), self.global_step)
                self.writer.add_scalar('Train/MAE_step', mae.item(), self.global_step)
            
            self.global_step += 1
        
        epoch_loss = loss_sum / len(self.train_loader)
        epoch_mae = mae_sum / len(self.train_loader)
        hip_mae = np.mean(hip_errors)
        bust_mae = np.mean(bust_errors)
        
        return epoch_loss, epoch_mae, hip_mae, bust_mae
    
    def validate(self, epoch):
        """Validate the model."""
        self.model.eval()
        loss_sum = 0.0
        mae_sum = 0.0
        
        # For per-measurement tracking
        hip_errors = []
        bust_errors = []
        
        all_outputs = []
        all_targets = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1} [Val]")
            
            for batch in pbar:
                # Forward pass
                outputs = self.model(
                    batch['mask'].to(self.device),
                    batch['mask_left'].to(self.device),
                    batch['height'].to(self.device)
                )
                targets = batch['measurements'].to(self.device)
                
                # Calculate loss
                loss = self.criterion(outputs, targets)
                mae = torch.abs(outputs - targets).mean()
                
                loss_sum += loss.item()
                mae_sum += mae.item()
                
                # Store for per-measurement analysis
                all_outputs.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                
                # Per-measurement errors (denormalized)
                outputs_cm = self.measurement_preprocessor.inverse_transform(
                    outputs.cpu().numpy()
                )
                targets_cm = self.measurement_preprocessor.inverse_transform(
                    targets.cpu().numpy()
                )
                
                hip_errors.extend(np.abs(outputs_cm[:, 0] - targets_cm[:, 0]))
                bust_errors.extend(np.abs(outputs_cm[:, 1] - targets_cm[:, 1]))
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'mae': f'{mae.item():.4f}'
                })
        
        epoch_loss = loss_sum / len(self.val_loader)
        epoch_mae = mae_sum / len(self.val_loader)
        hip_mae = np.mean(hip_errors)
        bust_mae = np.mean(bust_errors)
        
        # Concatenate all predictions
        all_outputs = np.vstack(all_outputs)
        all_targets = np.vstack(all_targets)
        
        return epoch_loss, epoch_mae, hip_mae, bust_mae, all_outputs, all_targets
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'global_step': self.global_step,
            'config': self.config
        }
        
        # Save last checkpoint
        torch.save(checkpoint, self.checkpoint_dir / 'last_checkpoint.pth')
        
        # Save best checkpoint
        if is_best:
            filename = f"best_epoch{epoch}_valloss{val_loss:.4f}_{time.strftime('%Y%m%d_%H%M%S')}.pth"
            torch.save(checkpoint, self.checkpoint_dir / filename)
            print(f"🏆 Saved best model: {filename}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint and resume training."""
        print(f"\n📂 Loading checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training state
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.global_step = checkpoint.get('global_step', 0)
        
        print(f"✓ Resumed from epoch {checkpoint['epoch']}")
        print(f"✓ Best val loss: {self.best_val_loss:.4f}")
    
    def train(self, num_epochs):
        """Main training loop."""
        print("\n" + "="*80)
        print("TRAINING - HIP & BUST PREDICTION")
        print("="*80)
        print(f"Model: ResNet-18 (1-channel grayscale)")
        print(f"Image size: {self.config.image.TARGET_SIZE}")
        print(f"Batch size: {self.config.training.BATCH_SIZE}")
        print(f"Epochs: {num_epochs}")
        print(f"Device: {self.device}")
        print(f"Measurements: hip, bust")
        print("="*80 + "\n")
        
        start_time = time.time()
        
        for epoch in range(self.start_epoch, num_epochs):
            print(f"\n{'='*80}")
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"{'='*80}")
            
            # Train
            train_loss, train_mae, train_hip_mae, train_bust_mae = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_mae, val_hip_mae, val_bust_mae, val_outputs, val_targets = self.validate(epoch)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log to TensorBoard
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Val', val_loss, epoch)
            self.writer.add_scalar('MAE/Train', train_mae, epoch)
            self.writer.add_scalar('MAE/Val', val_mae, epoch)
            self.writer.add_scalar('Hip_MAE/Train', train_hip_mae, epoch)
            self.writer.add_scalar('Hip_MAE/Val', val_hip_mae, epoch)
            self.writer.add_scalar('Bust_MAE/Train', train_bust_mae, epoch)
            self.writer.add_scalar('Bust_MAE/Val', val_bust_mae, epoch)
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            # Log predictions distribution
            self.writer.add_histogram('Predictions', val_outputs, epoch)
            self.writer.add_histogram('Targets', val_targets, epoch)
            self.writer.add_histogram('Errors', val_outputs - val_targets, epoch)
            
            # Log to custom logger (Excel + Plots)
            self.logger.log_epoch(
                epoch=epoch,
                train_loss=train_loss,
                train_mae=train_mae,
                val_loss=val_loss,
                val_mae=val_mae,
                learning_rate=current_lr,
                hip_mae=val_hip_mae,
                bust_mae=val_bust_mae
            )
            
            # Print epoch summary
            print(f"\n📊 Epoch {epoch+1} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train MAE: {train_mae:.4f}")
            print(f"  Val Loss:   {val_loss:.4f} | Val MAE:   {val_mae:.4f}")
            print(f"  Hip MAE:    {val_hip_mae:.2f} cm")
            print(f"  Bust MAE:   {val_bust_mae:.2f} cm")
            print(f"  LR: {current_lr:.6f}")
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                print(f"  🎉 New best validation loss!")
            
            self.save_checkpoint(epoch, val_loss, is_best)
            
            # Early stopping check
            if self.early_stopper.early_stop(val_loss):
                print("\n⛔ Early stopping triggered")
                break
            
            # Stop if LR too small
            if current_lr < 1e-7:
                print("\n⚠️  Learning rate too small. Stopping training.")
                break
        
        # Training complete
        total_time = time.time() - start_time
        
        # Generate final plots
        print("\n📊 Generating final plots and reports...")
        self.logger.plot_final_comparison()
        self.logger.export_summary_report()
        
        print("\n" + "="*80)
        print("✅ TRAINING COMPLETE!")
        print("="*80)
        print(f"Total time: {total_time/3600:.2f} hours")
        print(f"Best val loss: {self.best_val_loss:.4f}")
        print(f"Checkpoints: {self.checkpoint_dir}")
        print(f"Logs: {self.logger.log_dir}")
        print("="*80 + "\n")
        
        self.writer.close()


def main():
    """Main training function."""
    config = Config()
    
    # Print configuration
    print(config)
    
    # Option 1: Train from scratch
    trainer = Trainer(config)
    trainer.train(num_epochs=config.training.MAX_EPOCHS)
    
    # Option 2: Resume from checkpoint
    # trainer = Trainer(config, resume_from='checkpoints_resnet18/best_epoch10_valloss0.1234.pth')
    # trainer.train(num_epochs=50)


if __name__ == "__main__":
    main()
