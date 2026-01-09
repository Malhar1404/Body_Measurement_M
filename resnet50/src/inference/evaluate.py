import torch
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple

from src.config.config import Config
from src.inference.predict import BodyMeasurementPredictor


class ModelEvaluator:
    """
    Comprehensive model evaluator.
    """
    
    def __init__(self, predictor: BodyMeasurementPredictor):
        """Initialize evaluator."""
        self.predictor = predictor
        self.config = predictor.config
    
    def evaluate_test_set(self, test_data: pd.DataFrame, 
                         test_name: str = "Test") -> Tuple[pd.DataFrame, Dict]:
        """
        Evaluate model on test set with comprehensive metrics.
        
        Args:
            test_data: Test DataFrame
            test_name: Name of test set
            
        Returns:
            Tuple of (results_df, metrics_dict)
        """
        print(f"\n{'='*80}")
        print(f"Evaluating on {test_name} Set ({len(test_data)} samples)")
        print(f"{'='*80}")
        
        # Get predictions
        results = self.predictor.predict_and_compare(test_data, batch_size=32)
        
        if len(results) == 0:
            print(f"❌ No predictions generated for {test_name}")
            return None, None
        
        # Calculate comprehensive metrics
        metrics = {}
        
        for measurement in self.config.measurement.MEASUREMENT_COLUMNS:
            error_col = f"{measurement}_abs_error"
            pct_error_col = f"{measurement}_pct_error"
            
            if error_col in results.columns:
                mae = results[error_col].mean()
                rmse = np.sqrt((results[f"{measurement}_error"] ** 2).mean())
                max_error = results[error_col].max()
                median_error = results[error_col].median()
                std_error = results[f"{measurement}_error"].std()
                mean_pct_error = results[pct_error_col].mean()
                
                metrics[measurement] = {
                    'MAE': mae,
                    'RMSE': rmse,
                    'Median_Error': median_error,
                    'Max_Error': max_error,
                    'Std_Error': std_error,
                    'Mean_Pct_Error': mean_pct_error
                }
        
        # Print results
        self._print_metrics_table(metrics, test_name)
        
        return results, metrics
    
    def _print_metrics_table(self, metrics: Dict, test_name: str):
        """Print formatted metrics table."""
        print(f"\n📊 {test_name} Set Results:")
        print(f"{'Measurement':<20} {'MAE':<8} {'RMSE':<8} {'Median':<8} {'Max':<8} {'Std':<8} {'%Error':<8}")
        print("-" * 95)
        
        all_mae = []
        for measurement, values in metrics.items():
            mae = values['MAE']
            all_mae.append(mae)
            print(f"{measurement:<20} {mae:<8.2f} {values['RMSE']:<8.2f} "
                  f"{values['Median_Error']:<8.2f} {values['Max_Error']:<8.2f} "
                  f"{values['Std_Error']:<8.2f} {values['Mean_Pct_Error']:<8.2f}")
        
        print("-" * 95)
        print(f"{'Overall Average':<20} {np.mean(all_mae):<8.2f}")
        print()
    
    def plot_error_distribution(self, results: pd.DataFrame, 
                               test_name: str = "Test",
                               save_path: str = None):
        """Plot comprehensive error distribution."""
        error_cols = [col for col in results.columns if col.endswith('_abs_error')]
        
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle(f'Error Distribution - {test_name} Set', fontsize=18, fontweight='bold')
        
        for i, col in enumerate(error_cols, 1):
            measurement = col.replace('_abs_error', '')
            ax = fig.add_subplot(4, 4, i)
            
            errors = results[col]
            
            # Histogram
            ax.hist(errors, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
            
            # Stats lines
            mae = errors.mean()
            median = errors.median()
            ax.axvline(mae, color='r', linestyle='--', linewidth=2, label=f'MAE: {mae:.2f}')
            ax.axvline(median, color='g', linestyle='--', linewidth=2, label=f'Median: {median:.2f}')
            
            ax.set_xlabel('Absolute Error (cm)', fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.set_title(measurement.replace('-', ' ').title(), fontsize=11, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(error_cols) + 1, 17):
            fig.add_subplot(4, 4, i).axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved error distribution: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_scatter_comparison(self, results: pd.DataFrame,
                               test_name: str = "Test",
                               save_path: str = None):
        """Plot scatter plots of predictions vs ground truth."""
        measurements = self.config.measurement.MEASUREMENT_COLUMNS
        
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle(f'Predictions vs Ground Truth - {test_name} Set', 
                    fontsize=18, fontweight='bold')
        
        for i, measurement in enumerate(measurements, 1):
            pred_col = f"{measurement}_pred"
            true_col = f"{measurement}_true"
            
            if pred_col not in results.columns:
                continue
            
            ax = fig.add_subplot(4, 4, i)
            
            preds = results[pred_col]
            truths = results[true_col]
            
            # Scatter plot
            ax.scatter(truths, preds, alpha=0.5, s=20)
            
            # Perfect prediction line
            min_val = min(truths.min(), preds.min())
            max_val = max(truths.max(), preds.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
            
            # Calculate R²
            r_squared = np.corrcoef(truths, preds)[0, 1] ** 2
            
            ax.set_xlabel('Ground Truth (cm)', fontsize=10)
            ax.set_ylabel('Predicted (cm)', fontsize=10)
            ax.set_title(f"{measurement.replace('-', ' ').title()}\nR² = {r_squared:.3f}", 
                        fontsize=11, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved scatter comparison: {save_path}")
        else:
            plt.show()
        
        plt.close()


def main():
    """Main evaluation function."""
    print("="*80)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*80)
    
    config = Config()
    
    # Initialize predictor with auto-find best checkpoint
    predictor = BodyMeasurementPredictor(config=config, auto_find_best=True)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(predictor)
    
    # Create results directory
    results_dir = Path('evaluation_results')
    results_dir.mkdir(exist_ok=True)
    
    # Evaluate on validation set
    val_data_path = config.data.PROCESSED_DIR / 'val_data.csv'
    
    if val_data_path.exists():
        val_data = pd.read_csv(val_data_path)
        
        # Evaluate
        val_results, val_metrics = evaluator.evaluate_test_set(val_data, "Validation")
        
        if val_results is not None:
            # Export detailed results
            predictor.export_results(val_results, results_dir / 'validation_detailed_results.xlsx')
            
            # Plot error distribution
            evaluator.plot_error_distribution(
                val_results,
                test_name="Validation",
                save_path=results_dir / 'error_distribution.png'
            )
            
            # Plot scatter comparison
            evaluator.plot_scatter_comparison(
                val_results,
                test_name="Validation",
                save_path=results_dir / 'scatter_comparison.png'
            )
            
            # Save metrics to CSV
            metrics_df = pd.DataFrame(val_metrics).T
            metrics_df.to_csv(results_dir / 'metrics_summary.csv')
            print(f"✓ Metrics saved to: {results_dir / 'metrics_summary.csv'}")
    
    print("\n" + "="*80)
    print("✅ EVALUATION COMPLETE")
    print(f"📁 Results saved to: {results_dir.absolute()}")
    print("="*80)


if __name__ == "__main__":
    main()