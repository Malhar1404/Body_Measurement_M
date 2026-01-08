"""
train_advanced.py - Advanced Training with Full TensorBoard Monitoring
"""

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.config.config import Config
from src.models.body_measurement_model import create_model
from src.models.dataset import create_dataloaders
from src.utils.logger import TrainingLogger


# =========================
# Early Stopping
# =========================
class EarlyStopper:
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
        return self.counter >= self.patience


# =========================
# Trainer
# =========================
class AdvancedTrainer:
    def __init__(self, config: Config, resume_from: str = None):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Using device: {self.device}")

        # Model
        self.model = create_model(
            config, model_type="resnet50", pretrained=True
        ).to(self.device)
        self._print_parameter_info()

        # Data
        self.train_loader, self.val_loader = create_dataloaders(config)

        # Loss
        self.criterion = nn.MSELoss()

        # Hyperparams tracking
        self.initial_lr = None
        self.weight_decay = None
        self.freeze_strategy = "none"

        # Optimizer & Scheduler
        self.optimizer = None
        self.scheduler = None
        self._create_optimizer()

        # Logging dirs
        self.checkpoint_dir = Path("checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)

        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)

        self.experiment_name = f"bmnet_{time.strftime('%Y%m%d_%H%M%S')}"
        self.logger = TrainingLogger(self.log_dir, self.experiment_name)
        self.writer = SummaryWriter(self.log_dir / self.experiment_name)

        # Training state
        self.best_val_loss = float("inf")
        self.start_epoch = 0
        self.global_step = 0
        self.training_history = []
        self.early_stopper = EarlyStopper()

        if resume_from:
            self.load_checkpoint(resume_from)

    # =========================
    # Utils
    # =========================
    def _print_parameter_info(self):
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"📊 Params → Total: {total:,} | Trainable: {trainable:,}")

    def _create_optimizer(self, learning_rate=1e-4, weight_decay=1e-5):
        self.initial_lr = learning_rate
        self.weight_decay = weight_decay

        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )

        print(f"✓ Optimizer → LR={learning_rate}, WD={weight_decay}")

    # =========================
    # Freeze / Unfreeze
    # =========================
    def freeze_backbone(self):
        for name, p in self.model.named_parameters():
            if "backbone" in name:
                p.requires_grad = False
        self._create_optimizer()

    def unfreeze_backbone(self):
        for p in self.model.parameters():
            p.requires_grad = True
        self._create_optimizer()

    # =========================
    # TensorBoard Helpers
    # =========================
    def _log_gradients(self, epoch):
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        self.writer.add_scalar("Gradients/TotalNorm", total_norm, epoch)

    def _log_weights(self, epoch):
        for name, param in self.model.named_parameters():
            self.writer.add_histogram(f"Weights/{name}", param.data.cpu(), epoch)
            if param.grad is not None:
                self.writer.add_histogram(
                    f"Gradients/{name}", param.grad.cpu(), epoch
                )

    def _log_scatter(self, preds, gts, epoch):
        fig = plt.figure(figsize=(5, 5))
        plt.scatter(gts.flatten(), preds.flatten(), alpha=0.3)
        min_v = min(gts.min(), preds.min())
        max_v = max(gts.max(), preds.max())
        plt.plot([min_v, max_v], [min_v, max_v], "r--")
        plt.xlabel("Ground Truth")
        plt.ylabel("Predictions")
        plt.title("Predicted vs Ground Truth")
        self.writer.add_figure("Pred_vs_GT", fig, epoch)
        plt.close(fig)

    # =========================
    # Checkpoint Naming
    # =========================
    def _best_model_filename(self, epoch, val_loss):
        lr = self.optimizer.param_groups[0]["lr"]
        ts = time.strftime("%Y%m%d_%H%M%S")
        return (
            f"best_epoch={epoch}"
            f"_valloss={val_loss:.4f}"
            f"_lr={lr:.1e}"
            f"_wd={self.weight_decay:.1e}"
            f"_freeze={self.freeze_strategy}"
            f"_{ts}.pth"
        )

    # =========================
    # Save / Load
    # =========================
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "global_step": self.global_step,
            "training_history": self.training_history,
        }

        torch.save(checkpoint, self.checkpoint_dir / "last_checkpoint.pth")

        if is_best:
            path = self.checkpoint_dir / self._best_model_filename(epoch, val_loss)
            torch.save(checkpoint, path)
            print(f"🏆 Best model saved → {path.name}")

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.start_epoch = ckpt["epoch"] + 1
        self.best_val_loss = ckpt["best_val_loss"]
        self.global_step = ckpt["global_step"]
        self.training_history = ckpt["training_history"]
        print(f"♻️ Resumed from epoch {ckpt['epoch']}")

    # =========================
    # Train / Validate
    # =========================
    def train_epoch(self, epoch):
        self.model.train()
        loss_sum, mae_sum = 0.0, 0.0

        for batch in tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]"):
            self.optimizer.zero_grad()

            outputs = self.model(
                batch["mask"].to(self.device),
                batch["mask_left"].to(self.device),
                batch["height"].to(self.device),
            )
            targets = batch["measurements"].to(self.device)

            loss = self.criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            mae = torch.abs(outputs - targets).mean()

            self.writer.add_scalar(
                "Loss/Train_step", loss.item(), self.global_step
            )

            loss_sum += loss.item()
            mae_sum += mae.item()
            self.global_step += 1

        return loss_sum / len(self.train_loader), mae_sum / len(self.train_loader)

    def validate(self, epoch):
        self.model.eval()
        loss_sum, mae_sum = 0.0, 0.0
        preds, gts = [], []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"Epoch {epoch+1} [Val]"):
                outputs = self.model(
                    batch["mask"].to(self.device),
                    batch["mask_left"].to(self.device),
                    batch["height"].to(self.device),
                )
                targets = batch["measurements"].to(self.device)

                loss = self.criterion(outputs, targets)
                mae = torch.abs(outputs - targets).mean()

                loss_sum += loss.item()
                mae_sum += mae.item()
                preds.append(outputs.cpu().numpy())
                gts.append(targets.cpu().numpy())

        return (
            loss_sum / len(self.val_loader),
            mae_sum / len(self.val_loader),
            np.concatenate(preds),
            np.concatenate(gts),
        )

    # =========================
    # Training Loop
    # =========================
    def train(self, num_epochs=100, freeze_strategy=None, unfreeze_at_epoch=None):
        self.freeze_strategy = freeze_strategy or "none"

        if freeze_strategy == "freeze_backbone":
            self.freeze_backbone()

        for epoch in range(self.start_epoch, num_epochs):
            if unfreeze_at_epoch and epoch == unfreeze_at_epoch:
                self.unfreeze_backbone()

            train_loss, train_mae = self.train_epoch(epoch)
            val_loss, val_mae, preds, gts = self.validate(epoch)

            self.scheduler.step(val_loss)

            # Scalars
            self.writer.add_scalar("Loss/Train", train_loss, epoch)
            self.writer.add_scalar("Loss/Val", val_loss, epoch)
            self.writer.add_scalar("MAE/Train", train_mae, epoch)
            self.writer.add_scalar("MAE/Val", val_mae, epoch)
            self.writer.add_scalar(
                "LR", self.optimizer.param_groups[0]["lr"], epoch
            )

            # Distributions
            self.writer.add_histogram("Predictions", preds, epoch)
            self.writer.add_histogram("GroundTruth", gts, epoch)
            self.writer.add_histogram("Error", preds - gts, epoch)

            # Advanced diagnostics
            self._log_gradients(epoch)
            if epoch % 10 == 0:
                self._log_weights(epoch)
            self._log_scatter(preds, gts, epoch)

            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss

            self.save_checkpoint(epoch, val_loss, is_best)

            print(
                f"Epoch {epoch+1} | Train {train_loss:.4f} | Val {val_loss:.4f}"
            )

            if self.early_stopper.early_stop(val_loss):
                print("⛔ Early stopping triggered")
                break

        self.writer.close()
        
        def resume_with_unfrozen_backbone(self, checkpoint_path: str):
            """
            Resume training from checkpoint with unfrozen backbone.
            Uses very low learning rates to prevent destroying pretrained features.
            
            Args:
                checkpoint_path: Path to best checkpoint
            """
            print("\n" + "="*80)
            print("RESUMING WITH UNFROZEN BACKBONE")
            print("="*80)
            
            # Load the best checkpoint
            print(f"\n📂 Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Load model state
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Get the epoch we're resuming from
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            self.global_step = checkpoint.get('global_step', 0)
            
            print(f"✓ Loaded model from epoch {checkpoint['epoch']}")
            print(f"✓ Best val loss: {self.best_val_loss:.4f}")
            
            # Unfreeze all layers
            print("\n🔓 Unfreezing all layers...")
            for param in self.model.parameters():
                param.requires_grad = True
            
            # Create layer-wise optimizer with different learning rates
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
            
            # Very conservative learning rates for fine-tuning
            self.optimizer = optim.Adam([
                {'params': backbone_params, 'lr': 1e-6, 'name': 'backbone'},
                {'params': fusion_params, 'lr': 5e-5, 'name': 'fusion'},
                {'params': head_params, 'lr': 5e-5, 'name': 'head'}
            ], weight_decay=1e-3)
            
            print("✓ Learning rates set:")
            print(f"  Backbone: 1e-6 (very low - careful fine-tuning)")
            print(f"  Fusion:   5e-5")
            print(f"  Head:     5e-5")
            
            # Create scheduler with higher patience for fine-tuning
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10,
                verbose=True
            )
            
            # Reset early stopper with higher patience
            self.early_stopper = EarlyStopper(patience=20, min_delta=0.0005)
            print("✓ Early stopping: patience=20, min_delta=0.0005")
            
            # Print parameter info
            self._print_parameter_info()
            
            print("\n" + "="*80)
            print("Ready to resume training with unfrozen backbone!")
            print("="*80 + "\n")



# =========================
# Main
# =========================
# def main():
#     config = Config()
#     trainer = AdvancedTrainer(config)

#     trainer._create_optimizer(learning_rate=1e-4, weight_decay=1e-3)
#     trainer.train(
#         num_epochs=100,
#         freeze_strategy="freeze_backbone",
#         unfreeze_at_epoch=50,
#     )

def main():
    """Main function for fine-tuning."""
    config = Config()
    
    # Phase 2: Resume with unfrozen backbone
    print("\n🚀 Starting Phase 2: Fine-tuning with unfrozen backbone")
    
    trainer = AdvancedTrainer(config)
    
    # Load best checkpoint and unfreeze
    trainer.resume_with_unfrozen_backbone('checkpoints/best_model.pth')
    
    # Continue training
    trainer.train(num_epochs=50)  # Will train for up to 50 more epochs
    
    print("\n✅ Fine-tuning complete!")
    print(f"Check logs at: {trainer.logger.log_dir}")

if __name__ == "__main__":
    main()
