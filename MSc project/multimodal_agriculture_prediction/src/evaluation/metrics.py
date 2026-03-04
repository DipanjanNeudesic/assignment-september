"""
Comprehensive Evaluation Module for Multimodal Agriculture Prediction

This module provides evaluation metrics, visualization tools, and analysis
functions for comparing different fusion strategies and model performance.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import cosine
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
import json
from dataclasses import dataclass
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

@dataclass
class EvaluationResults:
    """Container for evaluation results."""
    metrics: Dict[str, float]
    predictions: np.ndarray
    targets: np.ndarray
    uncertainties: Optional[np.ndarray] = None
    attention_weights: Optional[Dict[str, np.ndarray]] = None
    quality_scores: Optional[Dict[str, np.ndarray]] = None
    metadata: Optional[Dict[str, Any]] = None


class ModelEvaluator:
    """Comprehensive model evaluation and analysis."""
    
    def __init__(self, save_dir: str = "results/evaluation"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def evaluate_model(
        self,
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device,
        model_type: str,
        return_detailed: bool = True
    ) -> EvaluationResults:
        """
        Comprehensive model evaluation.
        
        Args:
            model: Trained model to evaluate
            data_loader: Data loader for evaluation
            device: Device to run evaluation on
            model_type: Type of model ('early_fusion', 'late_fusion', 'gmu')
            return_detailed: Whether to return detailed outputs
        
        Returns:
            EvaluationResults object containing all evaluation outputs
        """
        model.eval()
        
        all_predictions = []
        all_targets = []
        all_uncertainties = []
        all_attention_weights = []
        all_quality_scores = []
        all_metadata = []
        
        with torch.no_grad():
            for batch in data_loader:
                satellite = batch['satellite'].to(device)
                weather = batch['weather'].to(device)
                targets = batch['yield'].to(device)
                
                # Model forward pass
                if model_type == 'early_fusion':
                    predictions = model(satellite, weather)
                    batch_results = {'prediction': predictions}
                    
                elif model_type == 'late_fusion':
                    if return_detailed:
                        predictions, expert_outputs = model(
                            satellite, weather, return_expert_outputs=True
                        )
                        batch_results = {
                            'prediction': predictions,
                            'expert_outputs': expert_outputs
                        }
                    else:
                        predictions = model(satellite, weather)
                        batch_results = {'prediction': predictions}
                
                elif model_type == 'gmu':
                    batch_results = model(
                        satellite, weather,
                        return_attention=return_detailed,
                        return_expert_outputs=return_detailed
                    )
                    predictions = batch_results['prediction']
                
                # Collect results
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                
                if 'uncertainty' in batch_results:
                    all_uncertainties.append(batch_results['uncertainty'].cpu().numpy())
                
                if 'attention_weights' in batch_results:
                    all_attention_weights.append(batch_results['attention_weights'])
                
                if 'quality_scores' in batch_results:
                    batch_quality = {}
                    for key, value in batch_results['quality_scores'].items():
                        batch_quality[key] = value.cpu().numpy()
                    all_quality_scores.append(batch_quality)
                
                # Collect metadata
                batch_metadata = {
                    'county_fips': batch['county_fips'],
                    'year': batch['year'],
                    'area_harvested': batch['metadata']['area_harvested'],
                    'area_planted': batch['metadata']['area_planted']
                }
                all_metadata.append(batch_metadata)
        
        # Concatenate results
        predictions = np.concatenate(all_predictions, axis=0).flatten()
        targets = np.concatenate(all_targets, axis=0).flatten()
        
        # Process optional outputs
        uncertainties = None
        if all_uncertainties:
            uncertainties = np.concatenate(all_uncertainties, axis=0).flatten()
        
        attention_weights = None
        if all_attention_weights:
            attention_weights = self._process_attention_weights(all_attention_weights)
        
        quality_scores = None
        if all_quality_scores:
            quality_scores = self._process_quality_scores(all_quality_scores)
        
        metadata = self._process_metadata(all_metadata)
        
        # Calculate comprehensive metrics
        metrics = self._calculate_comprehensive_metrics(
            predictions, targets, uncertainties
        )
        
        return EvaluationResults(
            metrics=metrics,
            predictions=predictions,
            targets=targets,
            uncertainties=uncertainties,
            attention_weights=attention_weights,
            quality_scores=quality_scores,
            metadata=metadata
        )
    
    def _calculate_comprehensive_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        uncertainties: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        metrics = {}
        
        # Basic regression metrics
        metrics['mae'] = mean_absolute_error(targets, predictions)
        metrics['rmse'] = np.sqrt(mean_squared_error(targets, predictions))
        metrics['mse'] = mean_squared_error(targets, predictions)
        
        # Relative metrics
        metrics['mape'] = np.mean(np.abs((targets - predictions) / (targets + 1e-8))) * 100
        metrics['smape'] = np.mean(
            2 * np.abs(targets - predictions) / (np.abs(targets) + np.abs(predictions) + 1e-8)
        ) * 100
        
        # Correlation metrics
        metrics['pearson_r'], metrics['pearson_p'] = pearsonr(targets, predictions)
        metrics['spearman_r'], metrics['spearman_p'] = spearmanr(targets, predictions)
        
        # R² and adjusted R²
        metrics['r2'] = r2_score(targets, predictions)
        n = len(targets)
        p = 1  # number of predictors (simplified)
        metrics['adj_r2'] = 1 - (1 - metrics['r2']) * (n - 1) / (n - p - 1)
        
        # Normalized metrics
        target_std = np.std(targets)
        metrics['nrmse'] = metrics['rmse'] / target_std if target_std > 0 else np.inf
        metrics['nmae'] = metrics['mae'] / np.mean(targets) if np.mean(targets) > 0 else np.inf
        
        # Residual analysis
        residuals = targets - predictions
        metrics['residual_mean'] = np.mean(residuals)
        metrics['residual_std'] = np.std(residuals)
        metrics['residual_skewness'] = self._calculate_skewness(residuals)
        metrics['residual_kurtosis'] = self._calculate_kurtosis(residuals)
        
        # Uncertainty calibration metrics
        if uncertainties is not None:
            metrics.update(self._calculate_uncertainty_metrics(predictions, targets, uncertainties))
        
        # Quantile-based metrics
        errors = np.abs(targets - predictions)
        metrics['q1_error'] = np.percentile(errors, 25)
        metrics['q3_error'] = np.percentile(errors, 75)
        metrics['iqr_error'] = metrics['q3_error'] - metrics['q1_error']
        
        return metrics
    
    def _calculate_uncertainty_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        uncertainties: np.ndarray
    ) -> Dict[str, float]:
        """Calculate uncertainty calibration metrics."""
        metrics = {}
        
        errors = np.abs(targets - predictions)
        
        # Uncertainty-error correlation
        unc_err_corr, _ = pearsonr(uncertainties, errors)
        metrics['uncertainty_error_correlation'] = unc_err_corr
        
        # Calibration: sort by uncertainty and check if error increases
        sort_idx = np.argsort(uncertainties)
        sorted_uncertainties = uncertainties[sort_idx]
        sorted_errors = errors[sort_idx]
        
        # Calculate calibration slope
        calibration_slope = np.polyfit(sorted_uncertainties, sorted_errors, 1)[0]
        metrics['calibration_slope'] = calibration_slope
        
        # Expected Calibration Error (simplified version)
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            bin_mask = (uncertainties >= bin_boundaries[i]) & (uncertainties < bin_boundaries[i + 1])
            if np.sum(bin_mask) > 0:
                bin_uncertainty = np.mean(uncertainties[bin_mask])
                bin_error = np.mean(errors[bin_mask])
                ece += np.sum(bin_mask) / len(uncertainties) * np.abs(bin_uncertainty - bin_error)
        metrics['expected_calibration_error'] = ece
        
        return metrics
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _process_attention_weights(self, attention_list: List) -> Dict[str, np.ndarray]:
        """Process attention weights from batches."""
        # Implementation depends on attention structure
        # This is a placeholder for the specific attention format
        return {}
    
    def _process_quality_scores(self, quality_list: List[Dict]) -> Dict[str, np.ndarray]:
        """Process quality scores from batches."""
        processed = {}
        for modality in ['satellite', 'weather']:
            scores = []
            for batch_quality in quality_list:
                if modality in batch_quality:
                    scores.append(batch_quality[modality])
            if scores:
                processed[modality] = np.concatenate(scores, axis=0).flatten()
        return processed
    
    def _process_metadata(self, metadata_list: List[Dict]) -> Dict[str, List]:
        """Process metadata from batches."""
        processed = {}
        for key in ['county_fips', 'year', 'area_harvested', 'area_planted']:
            processed[key] = []
            for batch_meta in metadata_list:
                if isinstance(batch_meta[key], torch.Tensor):
                    processed[key].extend(batch_meta[key].cpu().numpy().tolist())
                else:
                    processed[key].extend(batch_meta[key])
        return processed
    
    def compare_models(
        self,
        results: Dict[str, EvaluationResults],
        save_plots: bool = True
    ) -> Dict[str, Any]:
        """
        Compare multiple model results.
        
        Args:
            results: Dictionary mapping model names to EvaluationResults
            save_plots: Whether to save comparison plots
        
        Returns:
            Comparison analysis results
        """
        comparison = {
            'metrics_comparison': {},
            'statistical_tests': {},
            'rankings': {}
        }
        
        # Extract metrics for comparison
        metrics_df = pd.DataFrame({
            model_name: result.metrics
            for model_name, result in results.items()
        }).T
        
        comparison['metrics_comparison'] = metrics_df
        
        # Statistical significance tests
        comparison['statistical_tests'] = self._perform_statistical_tests(results)
        
        # Model rankings
        comparison['rankings'] = self._rank_models(metrics_df)
        
        if save_plots:
            self._create_comparison_plots(results, metrics_df)
        
        # Save comparison results
        self._save_comparison_results(comparison)
        
        return comparison
    
    def _perform_statistical_tests(self, results: Dict[str, EvaluationResults]) -> Dict[str, Any]:
        """Perform statistical significance tests between models."""
        from scipy.stats import ttest_rel, wilcoxon
        
        tests = {}
        model_names = list(results.keys())
        
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names):
                if i < j:  # Avoid duplicate comparisons
                    # Get absolute errors for both models
                    errors1 = np.abs(results[model1].targets - results[model1].predictions)
                    errors2 = np.abs(results[model2].targets - results[model2].predictions)
                    
                    # Paired t-test
                    t_stat, t_pvalue = ttest_rel(errors1, errors2)
                    
                    # Wilcoxon signed-rank test
                    w_stat, w_pvalue = wilcoxon(errors1, errors2)
                    
                    tests[f'{model1}_vs_{model2}'] = {
                        'paired_ttest': {'statistic': t_stat, 'pvalue': t_pvalue},
                        'wilcoxon': {'statistic': w_stat, 'pvalue': w_pvalue}
                    }
        
        return tests
    
    def _rank_models(self, metrics_df: pd.DataFrame) -> Dict[str, List[str]]:
        """Rank models based on different metrics."""
        rankings = {}
        
        # Key metrics for ranking (lower is better for most)
        lower_is_better = ['mae', 'rmse', 'mse', 'mape', 'smape']
        higher_is_better = ['r2', 'adj_r2', 'pearson_r', 'spearman_r']
        
        for metric in lower_is_better:
            if metric in metrics_df.columns:
                rankings[metric] = metrics_df[metric].sort_values().index.tolist()
        
        for metric in higher_is_better:
            if metric in metrics_df.columns:
                rankings[metric] = metrics_df[metric].sort_values(ascending=False).index.tolist()
        
        # Overall ranking (weighted combination)
        weights = {'mae': 0.3, 'rmse': 0.3, 'r2': 0.4}
        overall_score = 0
        
        for model in metrics_df.index:
            score = 0
            for metric, weight in weights.items():
                if metric in metrics_df.columns:
                    if metric in lower_is_better:
                        # Normalize by best score (lower is better)
                        best_score = metrics_df[metric].min()
                        model_score = best_score / (metrics_df.loc[model, metric] + 1e-8)
                    else:
                        # Higher is better
                        best_score = metrics_df[metric].max()
                        model_score = metrics_df.loc[model, metric] / (best_score + 1e-8)
                    score += weight * model_score
            overall_score = pd.concat([overall_score, pd.Series([score], index=[model])])
        
        rankings['overall'] = overall_score.sort_values(ascending=False).index.tolist()
        
        return rankings
    
    def create_evaluation_plots(self, results: EvaluationResults, model_name: str):
        """Create comprehensive evaluation plots."""
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Predictions vs Targets',
                'Residuals vs Predictions',
                'Residuals Distribution',
                'Error vs Uncertainty',
                'Attention Weights',
                'Quality Scores'
            ],
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "histogram"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "box"}]]
        )
        
        # 1. Predictions vs Targets
        fig.add_trace(
            go.Scatter(
                x=results.targets,
                y=results.predictions,
                mode='markers',
                name='Predictions',
                opacity=0.6
            ),
            row=1, col=1
        )
        
        # Perfect prediction line
        min_val = min(results.targets.min(), results.predictions.min())
        max_val = max(results.targets.max(), results.predictions.max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(dash='dash', color='red')
            ),
            row=1, col=1
        )
        
        # 2. Residuals vs Predictions
        residuals = results.targets - results.predictions
        fig.add_trace(
            go.Scatter(
                x=results.predictions,
                y=residuals,
                mode='markers',
                name='Residuals',
                opacity=0.6
            ),
            row=1, col=2
        )
        
        # Zero line
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=2)
        
        # 3. Residuals Distribution
        fig.add_trace(
            go.Histogram(
                x=residuals,
                name='Residuals Distribution',
                nbinsx=30
            ),
            row=2, col=1
        )
        
        # 4. Error vs Uncertainty (if available)
        if results.uncertainties is not None:
            errors = np.abs(residuals)
            fig.add_trace(
                go.Scatter(
                    x=results.uncertainties,
                    y=errors,
                    mode='markers',
                    name='Error vs Uncertainty',
                    opacity=0.6
                ),
                row=2, col=2
            )
        
        # 5. Quality Scores (if available)
        if results.quality_scores is not None:
            modalities = list(results.quality_scores.keys())
            for i, modality in enumerate(modalities):
                fig.add_trace(
                    go.Box(
                        y=results.quality_scores[modality],
                        name=f'{modality} Quality',
                        boxpoints='outliers'
                    ),
                    row=3, col=2
                )
        
        # Update layout
        fig.update_layout(
            title=f'{model_name} - Comprehensive Evaluation',
            height=1200,
            showlegend=True
        )
        
        # Save plot
        fig.write_html(self.save_dir / f'{model_name}_evaluation.html')
        
        # Also create static plots
        self._create_static_plots(results, model_name)
    
    def _create_static_plots(self, results: EvaluationResults, model_name: str):
        """Create static matplotlib plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Predictions vs Targets
        axes[0, 0].scatter(results.targets, results.predictions, alpha=0.6)
        axes[0, 0].plot([results.targets.min(), results.targets.max()], 
                       [results.targets.min(), results.targets.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Yield')
        axes[0, 0].set_ylabel('Predicted Yield')
        axes[0, 0].set_title('Predictions vs Actual')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add R² to the plot
        axes[0, 0].text(0.05, 0.95, f'R² = {results.metrics["r2"]:.3f}', 
                       transform=axes[0, 0].transAxes, fontsize=12,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
        
        # Residuals plot
        residuals = results.targets - results.predictions
        axes[0, 1].scatter(results.predictions, residuals, alpha=0.6)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predicted Yield')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residuals vs Predicted')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Residuals histogram
        axes[1, 0].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(x=0, color='r', linestyle='--')
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of Residuals')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Error vs Uncertainty (if available)
        if results.uncertainties is not None:
            errors = np.abs(residuals)
            axes[1, 1].scatter(results.uncertainties, errors, alpha=0.6)
            axes[1, 1].set_xlabel('Predicted Uncertainty')
            axes[1, 1].set_ylabel('Absolute Error')
            axes[1, 1].set_title('Error vs Uncertainty')
            axes[1, 1].grid(True, alpha=0.3)
            
            # Add correlation to the plot
            if 'uncertainty_error_correlation' in results.metrics:
                corr = results.metrics['uncertainty_error_correlation']
                axes[1, 1].text(0.05, 0.95, f'Correlation = {corr:.3f}', 
                               transform=axes[1, 1].transAxes, fontsize=12,
                               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
        else:
            axes[1, 1].text(0.5, 0.5, 'No Uncertainty Data', 
                           transform=axes[1, 1].transAxes, fontsize=14,
                           horizontalalignment='center', verticalalignment='center')
        
        plt.suptitle(f'{model_name} - Model Evaluation', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.save_dir / f'{model_name}_evaluation.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_comparison_plots(self, results: Dict[str, EvaluationResults], metrics_df: pd.DataFrame):
        """Create comparison plots between models."""
        # Metrics comparison bar plot
        key_metrics = ['mae', 'rmse', 'r2', 'mape']
        available_metrics = [m for m in key_metrics if m in metrics_df.columns]
        
        if available_metrics:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.flatten()
            
            for i, metric in enumerate(available_metrics[:4]):
                metrics_df[metric].plot(kind='bar', ax=axes[i])
                axes[i].set_title(f'{metric.upper()}')
                axes[i].set_ylabel(metric.upper())
                axes[i].tick_params(axis='x', rotation=45)
                axes[i].grid(True, alpha=0.3)
            
            plt.suptitle('Model Performance Comparison', fontsize=16)
            plt.tight_layout()
            plt.savefig(self.save_dir / 'models_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Predictions scatter plot comparison
        fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 5))
        if len(results) == 1:
            axes = [axes]
        
        for i, (model_name, result) in enumerate(results.items()):
            axes[i].scatter(result.targets, result.predictions, alpha=0.6)
            axes[i].plot([result.targets.min(), result.targets.max()], 
                        [result.targets.min(), result.targets.max()], 'r--', lw=2)
            axes[i].set_xlabel('Actual Yield')
            axes[i].set_ylabel('Predicted Yield')
            axes[i].set_title(f'{model_name}\nR² = {result.metrics["r2"]:.3f}')
            axes[i].grid(True, alpha=0.3)
        
        plt.suptitle('Model Predictions Comparison', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.save_dir / 'predictions_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_comparison_results(self, comparison: Dict[str, Any]):
        """Save comparison results to files."""
        # Save metrics comparison
        comparison['metrics_comparison'].to_csv(self.save_dir / 'metrics_comparison.csv')
        
        # Save detailed comparison
        with open(self.save_dir / 'comparison_analysis.json', 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {key: convert_numpy(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                else:
                    return obj
            
            json.dump(convert_numpy(comparison), f, indent=2)
        
        logger.info(f"Comparison results saved to {self.save_dir}")


def evaluate_and_compare_models(
    models: Dict[str, nn.Module],
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    save_dir: str = "results/evaluation"
) -> Dict[str, Any]:
    """
    Convenience function to evaluate and compare multiple models.
    
    Args:
        models: Dictionary mapping model names to trained models
        data_loader: Data loader for evaluation
        device: Device to run evaluation on
        save_dir: Directory to save results
    
    Returns:
        Complete comparison analysis
    """
    evaluator = ModelEvaluator(save_dir)
    
    results = {}
    for model_name, model in models.items():
        logger.info(f"Evaluating {model_name}...")
        
        # Determine model type from name (simple heuristic)
        if 'early' in model_name.lower():
            model_type = 'early_fusion'
        elif 'late' in model_name.lower():
            model_type = 'late_fusion'
        elif 'gmu' in model_name.lower():
            model_type = 'gmu'
        else:
            model_type = 'early_fusion'  # default
        
        result = evaluator.evaluate_model(model, data_loader, device, model_type)
        results[model_name] = result
        
        # Create individual evaluation plots
        evaluator.create_evaluation_plots(result, model_name)
    
    # Compare models
    comparison = evaluator.compare_models(results)
    
    logger.info("Evaluation completed for all models")
    
    return {
        'individual_results': results,
        'comparison': comparison,
        'evaluator': evaluator
    }


if __name__ == "__main__":
    # Example usage
    print("Evaluation module created successfully!")
    
    # Create dummy data for testing
    dummy_results = EvaluationResults(
        metrics={'mae': 5.2, 'rmse': 7.1, 'r2': 0.85},
        predictions=np.random.normal(150, 20, 1000),
        targets=np.random.normal(150, 20, 1000)
    )
    
    evaluator = ModelEvaluator("test_results")
    print("ModelEvaluator initialized successfully!")