"""
logger.py - Advanced Logging and Visualization System

Features:
- Real-time training graphs (loss, MAE, LR)
- Excel export of training history
- Per-measurement error analysis
- Checkpoint comparison
- Beautiful matplotlib plots
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import numpy as np
from typing import Dict, List

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class TrainingLogger:
    """
    Comprehensive training logger with visualization and Excel export.
    """
    
    def __init__(self, log_dir: Path = None, experiment_name: str = None):
        """
        Initialize logger.
        
        Args:
            log_dir: Directory to save logs
            experiment_name: Name of experiment (default: timestamp)
        """
        if log_dir is None:
            log_dir = Path('logs')
        
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.log_dir = log_dir / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.plots_dir = self.log_dir / 'plots'
        self.plots_dir.mkdir(exist_ok=True)
        
        self.data_dir = self.log_dir / 'data'
        self.data_dir.mkdir(exist_ok=True)
        
        self.experiment_name = experiment_name
        
        # Training history storage
        self.history = {
            'epoch': [],
            'train_loss': [],
            'train_mae': [],
            'val_loss': [],
            'val_mae': [],
            'learning_rate': [],
            'timestamp': []
        }
        
        # Per-measurement history
        self.per_measurement_history = {}
        
        print(f"📊 Logger initialized: {self.log_dir}")
    
    def log_epoch(self, epoch: int, train_loss: float, train_mae: float,
                  val_loss: float, val_mae: float, learning_rate: float,
                  per_measurement_mae: Dict[str, float] = None):
        """
        Log metrics for one epoch.
        
        Args:
            epoch: Epoch number
            train_loss: Training loss
            train_mae: Training MAE
            val_loss: Validation loss
            val_mae: Validation MAE
            learning_rate: Current learning rate
            per_measurement_mae: Dictionary of per-measurement MAE values
        """
        # Log main metrics
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(train_loss)
        self.history['train_mae'].append(train_mae)
        self.history['val_loss'].append(val_loss)
        self.history['val_mae'].append(val_mae)
        self.history['learning_rate'].append(learning_rate)
        self.history['timestamp'].append(datetime.now().isoformat())
        
        # Log per-measurement metrics
        if per_measurement_mae:
            for measurement, mae in per_measurement_mae.items():
                if measurement not in self.per_measurement_history:
                    self.per_measurement_history[measurement] = []
                self.per_measurement_history[measurement].append(mae)
        
        # Auto-save every epoch
        self.save_to_excel()
        self.plot_training_curves()
    
    def save_to_excel(self):
        """Save training history to Excel with multiple sheets."""
        excel_path = self.data_dir / 'training_history.xlsx'
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Sheet 1: Main training metrics
            df_main = pd.DataFrame(self.history)
            df_main.to_excel(writer, sheet_name='Training_Metrics', index=False)
            
            # Sheet 2: Per-measurement MAE
            if self.per_measurement_history:
                df_per_measurement = pd.DataFrame(self.per_measurement_history)
                df_per_measurement.insert(0, 'epoch', self.history['epoch'])
                df_per_measurement.to_excel(writer, sheet_name='Per_Measurement_MAE', index=False)
            
            # Sheet 3: Summary statistics
            if len(self.history['epoch']) > 0:
                summary = {
                    'Metric': ['Best Train Loss', 'Best Val Loss', 'Best Train MAE', 'Best Val MAE', 
                              'Final Train Loss', 'Final Val Loss', 'Final LR', 'Total Epochs'],
                    'Value': [
                        min(self.history['train_loss']),
                        min(self.history['val_loss']),
                        min(self.history['train_mae']),
                        min(self.history['val_mae']),
                        self.history['train_loss'][-1],
                        self.history['val_loss'][-1],
                        self.history['learning_rate'][-1],
                        len(self.history['epoch'])
                    ]
                }
                df_summary = pd.DataFrame(summary)
                df_summary.to_excel(writer, sheet_name='Summary', index=False)
        
        print(f"✓ Saved to Excel: {excel_path}")
    
    def plot_training_curves(self):
        """Generate and save training curve plots."""
        if len(self.history['epoch']) < 2:
            return
        
        epochs = self.history['epoch']
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training Progress - {self.experiment_name}', fontsize=16, fontweight='bold')
        
        # Plot 1: Loss curves
        ax1 = axes[0, 0]
        ax1.plot(epochs, self.history['train_loss'], label='Train Loss', marker='o', linewidth=2)
        ax1.plot(epochs, self.history['val_loss'], label='Val Loss', marker='s', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss (MSE)')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: MAE curves
        ax2 = axes[0, 1]
        ax2.plot(epochs, self.history['train_mae'], label='Train MAE', marker='o', linewidth=2, color='green')
        ax2.plot(epochs, self.history['val_mae'], label='Val MAE', marker='s', linewidth=2, color='red')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE (Normalized)')
        ax2.set_title('Mean Absolute Error')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Learning rate
        ax3 = axes[1, 0]
        ax3.plot(epochs, self.history['learning_rate'], marker='o', linewidth=2, color='purple')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_title('Learning Rate Schedule')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Overfitting indicator (Train/Val ratio)
        ax4 = axes[1, 1]
        train_val_ratio = [t/v if v > 0 else 1 for t, v in 
                          zip(self.history['train_loss'], self.history['val_loss'])]
        ax4.plot(epochs, train_val_ratio, marker='o', linewidth=2, color='orange')
        ax4.axhline(y=1.0, color='r', linestyle='--', label='Perfect fit')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Train Loss / Val Loss')
        ax4.set_title('Overfitting Indicator (closer to 1 is better)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.plots_dir / 'training_curves.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved training curves: {plot_path}")
    
    def plot_per_measurement_mae(self, measurement_names: List[str]):
        """
        Plot per-measurement MAE over time.
        
        Args:
            measurement_names: List of measurement names
        """
        if not self.per_measurement_history or len(self.history['epoch']) < 2:
            return
        
        epochs = self.history['epoch']
        n_measurements = len(measurement_names)
        
        # Create figure
        fig, axes = plt.subplots(4, 4, figsize=(20, 16))
        fig.suptitle(f'Per-Measurement MAE - {self.experiment_name}', fontsize=16, fontweight='bold')
        
        axes = axes.flatten()
        
        for i, measurement in enumerate(measurement_names):
            if measurement in self.per_measurement_history:
                ax = axes[i]
                mae_values = self.per_measurement_history[measurement]
                ax.plot(epochs, mae_values, marker='o', linewidth=2)
                ax.set_xlabel('Epoch')
                ax.set_ylabel('MAE (Normalized)')
                ax.set_title(measurement.replace('-', ' ').title())
                ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_measurements, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.plots_dir / 'per_measurement_mae.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved per-measurement MAE plot: {plot_path}")
    
    def plot_comparison(self, checkpoint_paths: List[str], labels: List[str]):
        """
        Compare multiple training runs.
        
        Args:
            checkpoint_paths: List of paths to training_history.xlsx files
            labels: Labels for each run
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        fig.suptitle('Training Runs Comparison', fontsize=16, fontweight='bold')
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(checkpoint_paths)))
        
        for i, (path, label) in enumerate(zip(checkpoint_paths, labels)):
            df = pd.read_excel(path, sheet_name='Training_Metrics')
            
            # Plot loss
            axes[0].plot(df['epoch'], df['val_loss'], label=label, 
                        marker='o', color=colors[i], linewidth=2)
            
            # Plot MAE
            axes[1].plot(df['epoch'], df['val_mae'], label=label, 
                        marker='s', color=colors[i], linewidth=2)
        
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Validation Loss')
        axes[0].set_title('Validation Loss Comparison')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Validation MAE')
        axes[1].set_title('Validation MAE Comparison')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = self.plots_dir / 'comparison.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved comparison plot: {plot_path}")
    
    def export_summary_report(self):
        """Export a comprehensive summary report."""
        if len(self.history['epoch']) == 0:
            return
        
        report_path = self.data_dir / 'training_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"TRAINING REPORT - {self.experiment_name}\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Training Duration: {len(self.history['epoch'])} epochs\n")
            f.write(f"Start Time: {self.history['timestamp'][0]}\n")
            f.write(f"End Time: {self.history['timestamp'][-1]}\n\n")
            
            f.write("Best Metrics:\n")
            f.write(f"  Best Train Loss: {min(self.history['train_loss']):.4f} (Epoch {self.history['train_loss'].index(min(self.history['train_loss']))+1})\n")
            f.write(f"  Best Val Loss:   {min(self.history['val_loss']):.4f} (Epoch {self.history['val_loss'].index(min(self.history['val_loss']))+1})\n")
            f.write(f"  Best Train MAE:  {min(self.history['train_mae']):.4f} (Epoch {self.history['train_mae'].index(min(self.history['train_mae']))+1})\n")
            f.write(f"  Best Val MAE:    {min(self.history['val_mae']):.4f} (Epoch {self.history['val_mae'].index(min(self.history['val_mae']))+1})\n\n")
            
            f.write("Final Metrics:\n")
            f.write(f"  Final Train Loss: {self.history['train_loss'][-1]:.4f}\n")
            f.write(f"  Final Val Loss:   {self.history['val_loss'][-1]:.4f}\n")
            f.write(f"  Final Train MAE:  {self.history['train_mae'][-1]:.4f}\n")
            f.write(f"  Final Val MAE:    {self.history['val_mae'][-1]:.4f}\n")
            f.write(f"  Final LR:         {self.history['learning_rate'][-1]:.2e}\n\n")
            
            if self.per_measurement_history:
                f.write("Per-Measurement Final MAE:\n")
                for measurement, mae_values in self.per_measurement_history.items():
                    if len(mae_values) > 0:
                        f.write(f"  {measurement:20s}: {mae_values[-1]:.4f}\n")
            
            f.write("\n" + "="*80 + "\n")
        
        print(f"✓ Saved training report: {report_path}")


# Quick test
if __name__ == "__main__":
    print("Testing TrainingLogger...")
    
    logger = TrainingLogger(experiment_name="test_run")
    
    # Simulate training for 10 epochs
    for epoch in range(10):
        train_loss = 1.0 - epoch * 0.08 + np.random.rand() * 0.1
        val_loss = 1.0 - epoch * 0.07 + np.random.rand() * 0.15
        train_mae = 0.8 - epoch * 0.06 + np.random.rand() * 0.08
        val_mae = 0.8 - epoch * 0.05 + np.random.rand() * 0.12
        lr = 1e-4 * (0.5 ** (epoch // 5))
        
        per_measurement = {
            'chest': 0.5 - epoch * 0.03,
            'waist': 0.6 - epoch * 0.04,
            'hip': 0.55 - epoch * 0.035
        }
        
        logger.log_epoch(epoch, train_loss, train_mae, val_loss, val_mae, lr, per_measurement)
    
    logger.plot_per_measurement_mae(['chest', 'waist', 'hip'])
    logger.export_summary_report()
    
    print("\n✅ Logger test complete! Check logs/ directory")
