"""
train_advanced.py - Advanced Training with Dynamic Control and Comprehensive Logging
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
        self.model = create_model(config, model_type="resnet50", pretrained=True).to(self.device)
        self._print_parameter_info()

        # Data
        self.train_loader, self.val_loader = create_dataloaders(config)

        # Loss
        self.criterion = nn.MSELoss()

        # Tracking hyperparams
        self.initial_lr = None
        self.weight_decay = None
        self.freeze_strategy = "none"

        # Optimizer & Scheduler
        self.optimizer = None
        self.scheduler = None
        self._create_optimizer()

        # Logging
        self.checkpoint_dir = Path("checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)

        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)

        self.experiment_name = f"bmnet_{time.strftime('%Y%m%d_%H%M%S')}"
        self.logger = TrainingLogger(self.log_dir, self.experiment_name)
        self.writer = SummaryWriter(self.log_dir / self.experiment_name / "tensorboard")

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
            "hyperparameters": {
                "lr": self.initial_lr,
                "weight_decay": self.weight_decay,
                "freeze_strategy": self.freeze_strategy,
            },
        }

        torch.save(checkpoint, self.checkpoint_dir / "last_checkpoint.pth")

        if is_best:
            filename = self._best_model_filename(epoch, val_loss)
            path = self.checkpoint_dir / filename
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
        loss_sum, mae_sum = 0, 0

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
            loss_sum += loss.item()
            mae_sum += mae.item()
            self.global_step += 1

        return loss_sum / len(self.train_loader), mae_sum / len(self.train_loader)

    def validate(self, epoch):
        self.model.eval()
        loss_sum, mae_sum = 0, 0
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

        return loss_sum / len(self.val_loader), mae_sum / len(self.val_loader)

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
            val_loss, val_mae = self.validate(epoch)

            self.scheduler.step(val_loss)

            self.training_history.append(
                dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss)
            )

            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss

            self.save_checkpoint(epoch, val_loss, is_best)

            print(
                f"Epoch {epoch+1} | "
                f"Train: {train_loss:.4f} | Val: {val_loss:.4f}"
            )

            if self.early_stopper.early_stop(val_loss):
                print("⛔ Early stopping triggered")
                break

        self.writer.close()


# =========================
# Main
# =========================
def main():
    config = Config()
    trainer = AdvancedTrainer(config)
    
    # Increase weight decay (stronger L2 regularization)
    trainer._create_optimizer(learning_rate=1e-4, weight_decay=1e-3)
    trainer.train(num_epochs=100, freeze_strategy="freeze_backbone", unfreeze_at_epoch=50)


if __name__ == "__main__":
    main()
