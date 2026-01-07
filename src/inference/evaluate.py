"""
evaluate.py - Model Evaluation on Test Sets

Evaluate model performance on testA and testB datasets.
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from src.config.config import Config
from src.inference.predict import BodyMeasurementPredictor


class ModelEvaluator:
    """
    Evaluate trained model on test datasets.
    """
    
    def __init__(self, predictor: BodyMeasurementPredictor):
        """Initialize evaluator."""
        self.predictor = predictor
        self.config = predictor.config
    
    def evaluate_test_set(self, test_data: pd.DataFrame, test_name: str = "Test"):
        """
        Evaluate model on test set.
        
        Args:
            test_data: Test DataFrame
            test_name: Name of test set (for reporting)
        """
        print(f"\n{'='*80}")
        print(f"Evaluating on {test_name} Set")
        print(f"{'='*80}")
        
        # Get predictions
        results = self.predictor.predict_and_compare(test_data)
        
        if len(results) == 0:
            print(f"❌ No predictions generated for {test_name}")
            return None
        
        # Calculate metrics per measurement
        metrics = {}
        
        for measurement in self.config.measurement.MEASUREMENT_COLUMNS:
            error_col = f"{measurement}_abs_error"
            
            if error_col in results.columns:
                mae = results[error_col].mean()
                rmse = np.sqrt((results[error_col] ** 2).mean())
                max_error = results[error_col].max()
                
                metrics[measurement] = {
                    'MAE': mae,
                    'RMSE': rmse,
                    'Max Error': max_error
                }
        
        # Print results
        print(f"\n📊 {test_name} Set Results:")
        print(f"{'Measurement':<20} {'MAE (cm)':<12} {'RMSE (cm)':<12} {'Max Error (cm)':<15}")
        print("-" * 80)
        
        for measurement, values in metrics.items():
            print(f"{measurement:<20} {values['MAE']:<12.2f} {values['RMSE']:<12.2f} {values['Max Error']:<15.2f}")
        
        # Overall metrics
        all_mae = [v['MAE'] for v in metrics.values()]
        print("-" * 80)
        print(f"{'Overall Average':<20} {np.mean(all_mae):<12.2f}")
        
        return results, metrics
    
    def plot_error_distribution(self, results: pd.DataFrame, save_path: str = None):
        """Plot error distribution for all measurements."""
        error_cols = [col for col in results.columns if col.endswith('_abs_error')]
        
        fig, axes = plt.subplots(4, 4, figsize=(20, 16))
        fig.suptitle('Error Distribution by Measurement', fontsize=16, fontweight='bold')
        
        axes = axes.flatten()
        
        for i, col in enumerate(error_cols):
            measurement = col.replace('_abs_error', '')
            ax = axes[i]
            
            ax.hist(results[col], bins=30, edgecolor='black', alpha=0.7)
            ax.set_xlabel('Absolute Error (cm)')
            ax.set_ylabel('Frequency')
            ax.set_title(measurement.replace('-', ' ').title())
            ax.axvline(results[col].mean(), color='r', linestyle='--', 
                      label=f'MAE: {results[col].mean():.2f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(error_cols), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved error distribution plot: {save_path}")
        else:
            plt.show()
        
        plt.close()


def main():
    """Main evaluation function."""
    print("="*80)
    print("MODEL EVALUATION")
    print("="*80)
    
    config = Config()
    
    # Initialize predictor
    predictor = BodyMeasurementPredictor(
        checkpoint_path='checkpoints/best_model.pth',
        config=config
    )
    
    # Initialize evaluator
    evaluator = ModelEvaluator(predictor)
    
    # Create results directory
    results_dir = Path('evaluation_results')
    results_dir.mkdir(exist_ok=True)
    
    # Evaluate on validation set
    val_data_path = config.data.PROCESSED_DIR / 'val_data.csv'
    if val_data_path.exists():
        val_data = pd.read_csv(val_data_path)
        val_results, val_metrics = evaluator.evaluate_test_set(val_data, "Validation")
        
        if val_results is not None:
            # Export results
            predictor.export_results(val_results, results_dir / 'validation_results.xlsx')
            
            # Plot error distribution
            evaluator.plot_error_distribution(
                val_results, 
                save_path=results_dir / 'validation_error_distribution.png'
            )
    
    print("\n" + "="*80)
    print("✅ EVALUATION COMPLETE")
    print(f"Results saved to: {results_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
