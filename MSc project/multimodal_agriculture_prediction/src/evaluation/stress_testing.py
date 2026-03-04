"""
Stress Testing Framework for Multimodal Agriculture Prediction

This module implements comprehensive stress testing to evaluate model robustness
to various real-world data quality issues including cloudy satellite imagery,
missing weather data, sensor failures, and noise.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable, Any
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from enum import Enum
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

logger = logging.getLogger(__name__)

class DataQualityIssue(Enum):
    """Types of data quality issues to simulate."""
    CLOUDY_SATELLITE = "cloudy_satellite"
    MISSING_WEATHER = "missing_weather"
    SENSOR_NOISE = "sensor_noise"
    SATELLITE_CORRUPTION = "satellite_corruption"
    WEATHER_OUTLIERS = "weather_outliers"
    TEMPORAL_GAPS = "temporal_gaps"
    SPATIAL_ARTIFACTS = "spatial_artifacts"
    MIXED_QUALITY = "mixed_quality"

@dataclass
class StressTestConfig:
    """Configuration for stress testing."""
    issue_type: DataQualityIssue
    severity_levels: List[float] = None  # 0.0 to 1.0
    sample_fraction: float = 1.0  # Fraction of samples to affect
    num_trials: int = 5  # Number of random trials per severity level
    save_results: bool = True
    save_dir: str = "results/stress_testing"
    plot_results: bool = True

class DataCorruptor:
    """Utilities for simulating various data quality issues."""
    
    @staticmethod
    def add_cloud_cover(satellite_data: torch.Tensor, severity: float) -> torch.Tensor:
        """
        Simulate cloud cover in satellite imagery.
        
        Args:
            satellite_data: Satellite tensor (batch, channels, height, width)
            severity: Cloud coverage severity (0.0 to 1.0)
        
        Returns:
            Corrupted satellite data
        """
        batch_size, channels, height, width = satellite_data.shape
        corrupted = satellite_data.clone()
        
        # Create cloud mask
        cloud_mask = torch.rand(batch_size, 1, height, width) < severity
        
        # Apply cloud effects
        for i in range(batch_size):
            if cloud_mask[i].any():
                # Reduce RGB values (clouds appear white/gray)
                if channels >= 3:
                    cloudy_values = torch.rand_like(corrupted[i, :3]) * 0.7 + 0.3
                    corrupted[i, :3] = torch.where(
                        cloud_mask[i].expand(3, -1, -1),
                        cloudy_values,
                        corrupted[i, :3]
                    )
                
                # Affect vegetation indices (reduce NDVI due to blocked NIR)
                if channels > 3:
                    for c in range(3, channels):
                        noise_factor = torch.rand_like(corrupted[i, c:c+1]) * 0.5 + 0.25
                        corrupted[i, c:c+1] = torch.where(
                            cloud_mask[i],
                            corrupted[i, c:c+1] * noise_factor,
                            corrupted[i, c:c+1]
                        )
        
        return corrupted
    
    @staticmethod
    def add_sensor_noise(data: torch.Tensor, noise_std: float) -> torch.Tensor:
        """Add Gaussian noise to simulate sensor errors."""
        noise = torch.randn_like(data) * noise_std
        return data + noise
    
    @staticmethod
    def create_missing_weather_data(weather_data: torch.Tensor, missing_fraction: float) -> torch.Tensor:
        """
        Simulate missing weather data by setting values to NaN and then interpolating.
        
        Args:
            weather_data: Weather tensor (batch, sequence_length, features)
            missing_fraction: Fraction of time steps to make missing
        
        Returns:
            Weather data with simulated missing values (interpolated)
        """
        batch_size, seq_len, features = weather_data.shape
        corrupted = weather_data.clone()
        
        for i in range(batch_size):
            # Randomly select time steps to make missing
            n_missing = int(seq_len * missing_fraction)
            missing_indices = torch.randperm(seq_len)[:n_missing]
            
            # Set missing values to NaN
            corrupted[i, missing_indices, :] = float('nan')
            
            # Simple linear interpolation
            for f in range(features):
                series = corrupted[i, :, f]
                
                # Find valid (non-NaN) values
                valid_mask = ~torch.isnan(series)
                
                if valid_mask.sum() > 1:  # Need at least 2 points to interpolate
                    valid_indices = torch.where(valid_mask)[0]
                    valid_values = series[valid_mask]
                    
                    # Interpolate missing values
                    for idx in missing_indices:
                        if idx < valid_indices[0]:
                            # Forward fill
                            corrupted[i, idx, f] = valid_values[0]
                        elif idx > valid_indices[-1]:
                            # Backward fill
                            corrupted[i, idx, f] = valid_values[-1]
                        else:
                            # Linear interpolation
                            left_idx = valid_indices[valid_indices < idx].max()
                            right_idx = valid_indices[valid_indices > idx].min()
                            
                            left_val = series[left_idx]
                            right_val = series[right_idx]
                            
                            weight = (idx - left_idx).float() / (right_idx - left_idx).float()
                            corrupted[i, idx, f] = left_val + weight * (right_val - left_val)
                else:
                    # If not enough valid values, use mean
                    mean_val = series[valid_mask].mean() if valid_mask.sum() > 0 else 0.0
                    corrupted[i, missing_indices, f] = mean_val
        
        return corrupted
    
    @staticmethod
    def add_weather_outliers(weather_data: torch.Tensor, outlier_fraction: float, outlier_magnitude: float = 5.0) -> torch.Tensor:
        """Add extreme outliers to weather data."""
        batch_size, seq_len, features = weather_data.shape
        corrupted = weather_data.clone()
        
        # Calculate number of outliers
        total_values = batch_size * seq_len * features
        n_outliers = int(total_values * outlier_fraction)
        
        # Randomly select positions for outliers
        outlier_positions = []
        for _ in range(n_outliers):
            b = torch.randint(0, batch_size, (1,)).item()
            s = torch.randint(0, seq_len, (1,)).item()
            f = torch.randint(0, features, (1,)).item()
            outlier_positions.append((b, s, f))
        
        # Add outliers
        for b, s, f in outlier_positions:
            current_val = corrupted[b, s, f]
            std = weather_data[:, :, f].std()
            
            # Create outlier (either very high or very low)
            if torch.rand(1) < 0.5:
                outlier_val = current_val + outlier_magnitude * std
            else:
                outlier_val = current_val - outlier_magnitude * std
            
            corrupted[b, s, f] = outlier_val
        
        return corrupted
    
    @staticmethod
    def add_spatial_artifacts(satellite_data: torch.Tensor, artifact_fraction: float) -> torch.Tensor:
        """Add spatial artifacts like stripes or dead pixels."""
        batch_size, channels, height, width = satellite_data.shape
        corrupted = satellite_data.clone()
        
        for i in range(batch_size):
            if torch.rand(1) < artifact_fraction:
                artifact_type = torch.randint(0, 3, (1,)).item()
                
                if artifact_type == 0:  # Horizontal stripes
                    stripe_rows = torch.randint(0, height, (height // 10,))
                    corrupted[i, :, stripe_rows, :] = 0
                
                elif artifact_type == 1:  # Vertical stripes
                    stripe_cols = torch.randint(0, width, (width // 10,))
                    corrupted[i, :, :, stripe_cols] = 0
                
                else:  # Random dead pixels
                    n_dead = height * width // 20
                    dead_rows = torch.randint(0, height, (n_dead,))
                    dead_cols = torch.randint(0, width, (n_dead,))
                    corrupted[i, :, dead_rows, dead_cols] = 0
        
        return corrupted
    
    @staticmethod
    def create_temporal_gaps(weather_data: torch.Tensor, gap_fraction: float, max_gap_length: int = 5) -> torch.Tensor:
        """Create temporal gaps in weather data."""
        batch_size, seq_len, features = weather_data.shape
        corrupted = weather_data.clone()
        
        for i in range(batch_size):
            if torch.rand(1) < gap_fraction:
                # Create a temporal gap
                gap_length = torch.randint(1, max_gap_length + 1, (1,)).item()
                gap_start = torch.randint(0, seq_len - gap_length, (1,)).item()
                gap_end = gap_start + gap_length
                
                # Set gap values as interpolation between boundaries
                if gap_start > 0 and gap_end < seq_len:
                    start_val = corrupted[i, gap_start - 1, :]
                    end_val = corrupted[i, gap_end, :]
                    
                    for t in range(gap_start, gap_end):
                        weight = (t - gap_start + 1) / (gap_length + 1)
                        corrupted[i, t, :] = start_val + weight * (end_val - start_val)
                else:
                    # Forward or backward fill
                    if gap_start == 0:
                        corrupted[i, gap_start:gap_end, :] = corrupted[i, gap_end, :]
                    else:
                        corrupted[i, gap_start:gap_end, :] = corrupted[i, gap_start - 1, :]
        
        return corrupted

class StressTester:
    """Main stress testing framework."""
    
    def __init__(self, save_dir: str = "results/stress_testing"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = {}
        self.corruptions = DataCorruptor()
    
    def run_stress_test(
        self,
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device,
        config: StressTestConfig,
        model_type: str = 'early_fusion'
    ) -> Dict[str, Any]:
        """
        Run comprehensive stress test on a model.
        
        Args:
            model: Trained model to test
            data_loader: Clean data loader
            device: Device for inference
            config: Stress test configuration
            model_type: Type of model for forward pass
        
        Returns:
            Stress test results
        """
        logger.info(f"Running stress test: {config.issue_type.value}")
        
        # Default severity levels if not provided
        if config.severity_levels is None:
            config.severity_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        
        results = {
            'issue_type': config.issue_type.value,
            'severity_levels': config.severity_levels,
            'metrics_by_severity': {},
            'detailed_results': {}
        }
        
        model.eval()
        
        # Baseline performance (no corruption)
        baseline_metrics = self._evaluate_clean_data(model, data_loader, device, model_type)
        results['baseline_metrics'] = baseline_metrics
        
        # Test each severity level
        for severity in config.severity_levels:
            logger.info(f"Testing severity level: {severity}")
            
            severity_results = {
                'trials': [],
                'mean_metrics': {},
                'std_metrics': {}
            }
            
            # Run multiple trials for each severity
            for trial in range(config.num_trials):
                trial_metrics = self._run_single_trial(
                    model, data_loader, device, config, severity, model_type
                )
                severity_results['trials'].append(trial_metrics)
            
            # Calculate statistics across trials
            self._calculate_trial_statistics(severity_results)
            results['metrics_by_severity'][severity] = severity_results
        
        # Calculate robustness metrics
        results['robustness_analysis'] = self._analyze_robustness(results)
        
        if config.save_results:
            self._save_results(results, config)
        
        if config.plot_results:
            self._plot_results(results, config)
        
        return results
    
    def _evaluate_clean_data(
        self,
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device,
        model_type: str
    ) -> Dict[str, float]:
        """Evaluate model on clean data."""
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in data_loader:
                satellite = batch['satellite'].to(device)
                weather = batch['weather'].to(device)
                targets = batch['yield'].to(device)
                
                # Forward pass based on model type
                predictions = self._model_forward(model, satellite, weather, model_type)
                
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        predictions = np.concatenate(all_predictions).flatten()
        targets = np.concatenate(all_targets).flatten()
        
        return self._calculate_metrics(predictions, targets)
    
    def _run_single_trial(
        self,
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device,
        config: StressTestConfig,
        severity: float,
        model_type: str
    ) -> Dict[str, float]:
        """Run a single trial with corrupted data."""
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in data_loader:
                satellite = batch['satellite'].to(device)
                weather = batch['weather'].to(device)
                targets = batch['yield'].to(device)
                
                # Apply corruption based on issue type
                corrupted_satellite, corrupted_weather = self._apply_corruption(
                    satellite, weather, config.issue_type, severity, config.sample_fraction
                )
                
                # Forward pass
                predictions = self._model_forward(
                    model, corrupted_satellite, corrupted_weather, model_type
                )
                
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        predictions = np.concatenate(all_predictions).flatten()
        targets = np.concatenate(all_targets).flatten()
        
        return self._calculate_metrics(predictions, targets)
    
    def _apply_corruption(
        self,
        satellite: torch.Tensor,
        weather: torch.Tensor,
        issue_type: DataQualityIssue,
        severity: float,
        sample_fraction: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply specified corruption to data."""
        corrupted_satellite = satellite.clone()
        corrupted_weather = weather.clone()
        
        # Determine which samples to corrupt
        batch_size = satellite.size(0)
        n_corrupt = int(batch_size * sample_fraction)
        corrupt_indices = torch.randperm(batch_size)[:n_corrupt]
        
        if issue_type == DataQualityIssue.CLOUDY_SATELLITE:
            corrupted_satellite[corrupt_indices] = self.corruptions.add_cloud_cover(
                satellite[corrupt_indices], severity
            )
        
        elif issue_type == DataQualityIssue.MISSING_WEATHER:
            corrupted_weather[corrupt_indices] = self.corruptions.create_missing_weather_data(
                weather[corrupt_indices], severity
            )
        
        elif issue_type == DataQualityIssue.SENSOR_NOISE:
            # Apply noise to both modalities
            noise_std = severity * 0.1  # Adjust noise level
            corrupted_satellite[corrupt_indices] = self.corruptions.add_sensor_noise(
                satellite[corrupt_indices], noise_std
            )
            corrupted_weather[corrupt_indices] = self.corruptions.add_sensor_noise(
                weather[corrupt_indices], noise_std
            )
        
        elif issue_type == DataQualityIssue.SATELLITE_CORRUPTION:
            corrupted_satellite[corrupt_indices] = self.corruptions.add_spatial_artifacts(
                satellite[corrupt_indices], severity
            )
        
        elif issue_type == DataQualityIssue.WEATHER_OUTLIERS:
            corrupted_weather[corrupt_indices] = self.corruptions.add_weather_outliers(
                weather[corrupt_indices], severity
            )
        
        elif issue_type == DataQualityIssue.TEMPORAL_GAPS:
            corrupted_weather[corrupt_indices] = self.corruptions.create_temporal_gaps(
                weather[corrupt_indices], severity
            )
        
        elif issue_type == DataQualityIssue.SPATIAL_ARTIFACTS:
            corrupted_satellite[corrupt_indices] = self.corruptions.add_spatial_artifacts(
                satellite[corrupt_indices], severity
            )
        
        elif issue_type == DataQualityIssue.MIXED_QUALITY:
            # Apply multiple corruptions with reduced severity
            mixed_severity = severity * 0.5
            corrupted_satellite[corrupt_indices] = self.corruptions.add_cloud_cover(
                satellite[corrupt_indices], mixed_severity
            )
            corrupted_weather[corrupt_indices] = self.corruptions.create_missing_weather_data(
                weather[corrupt_indices], mixed_severity
            )
        
        return corrupted_satellite, corrupted_weather
    
    def _model_forward(
        self,
        model: nn.Module,
        satellite: torch.Tensor,
        weather: torch.Tensor,
        model_type: str
    ) -> torch.Tensor:
        """Forward pass through model."""
        if model_type == 'early_fusion':
            return model(satellite, weather)
        elif model_type == 'late_fusion':
            return model(satellite, weather)
        elif model_type == 'gmu':
            output = model(satellite, weather)
            return output['prediction'] if isinstance(output, dict) else output
        else:
            return model(satellite, weather)
    
    def _calculate_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        return {
            'mae': mean_absolute_error(targets, predictions),
            'rmse': np.sqrt(mean_squared_error(targets, predictions)),
            'r2': r2_score(targets, predictions),
            'mape': np.mean(np.abs((targets - predictions) / (targets + 1e-8))) * 100
        }
    
    def _calculate_trial_statistics(self, severity_results: Dict):
        """Calculate mean and std across trials."""
        metrics = ['mae', 'rmse', 'r2', 'mape']
        
        for metric in metrics:
            values = [trial[metric] for trial in severity_results['trials']]
            severity_results['mean_metrics'][metric] = np.mean(values)
            severity_results['std_metrics'][metric] = np.std(values)
    
    def _analyze_robustness(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Analyze model robustness from stress test results."""
        baseline_mae = results['baseline_metrics']['mae']
        baseline_r2 = results['baseline_metrics']['r2']
        
        # Calculate degradation metrics
        mae_degradations = []
        r2_degradations = []
        
        for severity, severity_data in results['metrics_by_severity'].items():
            if severity > 0:  # Skip clean data
                mae_degradation = (
                    severity_data['mean_metrics']['mae'] - baseline_mae
                ) / baseline_mae
                r2_degradation = (
                    baseline_r2 - severity_data['mean_metrics']['r2']
                ) / baseline_r2
                
                mae_degradations.append(mae_degradation)
                r2_degradations.append(r2_degradation)
        
        # Robustness metrics
        robustness = {
            'mean_mae_degradation': np.mean(mae_degradations),
            'max_mae_degradation': np.max(mae_degradations),
            'mean_r2_degradation': np.mean(r2_degradations),
            'max_r2_degradation': np.max(r2_degradations),
            'robustness_score': 1.0 / (1.0 + np.mean(mae_degradations))  # Higher is better
        }
        
        return robustness
    
    def _save_results(self, results: Dict[str, Any], config: StressTestConfig):
        """Save stress test results."""
        filename = f"stress_test_{config.issue_type.value}_{len(config.severity_levels)}levels.json"
        filepath = self.save_dir / filename
        
        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        with open(filepath, 'w') as f:
            json.dump(convert_numpy(results), f, indent=2)
        
        logger.info(f"Stress test results saved to {filepath}")
    
    def _plot_results(self, results: Dict[str, Any], config: StressTestConfig):
        """Plot stress test results."""
        severity_levels = results['severity_levels']
        
        # Extract metrics for plotting
        mae_means = [results['metrics_by_severity'][s]['mean_metrics']['mae'] for s in severity_levels]
        mae_stds = [results['metrics_by_severity'][s]['std_metrics']['mae'] for s in severity_levels]
        r2_means = [results['metrics_by_severity'][s]['mean_metrics']['r2'] for s in severity_levels]
        r2_stds = [results['metrics_by_severity'][s]['std_metrics']['r2'] for s in severity_levels]
        
        # Create plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # MAE plot
        ax1.errorbar(severity_levels, mae_means, yerr=mae_stds, marker='o', capsize=5, capthick=2)
        ax1.set_xlabel('Corruption Severity')
        ax1.set_ylabel('Mean Absolute Error')
        ax1.set_title(f'MAE vs {config.issue_type.value.replace("_", " ").title()} Severity')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=results['baseline_metrics']['mae'], color='r', linestyle='--', label='Baseline')
        ax1.legend()
        
        # R² plot
        ax2.errorbar(severity_levels, r2_means, yerr=r2_stds, marker='s', capsize=5, capthick=2)
        ax2.set_xlabel('Corruption Severity')
        ax2.set_ylabel('R² Score')
        ax2.set_title(f'R² vs {config.issue_type.value.replace("_", " ").title()} Severity')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=results['baseline_metrics']['r2'], color='r', linestyle='--', label='Baseline')
        ax2.legend()
        
        plt.suptitle(f'Stress Test Results: {config.issue_type.value.replace("_", " ").title()}')
        plt.tight_layout()
        
        # Save plot
        plot_filename = f"stress_test_{config.issue_type.value}.png"
        plt.savefig(self.save_dir / plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Stress test plot saved to {self.save_dir / plot_filename}")

    def compare_model_robustness(
        self,
        models: Dict[str, nn.Module],
        data_loader: torch.utils.data.DataLoader,
        device: torch.device,
        issue_types: List[DataQualityIssue],
        model_types: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Compare robustness of multiple models across different data quality issues.
        
        Args:
            models: Dictionary of model_name -> model
            data_loader: Test data loader
            device: Device for inference
            issue_types: List of data quality issues to test
            model_types: Dictionary mapping model names to their types
        
        Returns:
            Comprehensive robustness comparison
        """
        logger.info("Starting comprehensive robustness comparison...")
        
        comparison_results = {
            'models': list(models.keys()),
            'issue_types': [issue.value for issue in issue_types],
            'individual_results': {},
            'comparison_matrix': {},
            'robustness_rankings': {}
        }
        
        # Test each model on each issue type
        for model_name, model in models.items():
            logger.info(f"Testing model: {model_name}")
            comparison_results['individual_results'][model_name] = {}
            
            model_type = model_types.get(model_name, 'early_fusion')
            
            for issue_type in issue_types:
                logger.info(f"  Testing {issue_type.value}...")
                
                config = StressTestConfig(
                    issue_type=issue_type,
                    severity_levels=[0.0, 0.2, 0.4, 0.6, 0.8],
                    num_trials=3,  # Reduced for comparison
                    save_results=False,
                    plot_results=False
                )
                
                results = self.run_stress_test(model, data_loader, device, config, model_type)
                comparison_results['individual_results'][model_name][issue_type.value] = results
        
        # Create comparison matrices
        self._create_comparison_matrices(comparison_results)
        
        # Rank models by robustness
        self._rank_models_by_robustness(comparison_results)
        
        # Create comparison plots
        self._plot_robustness_comparison(comparison_results)
        
        # Save comparison results
        with open(self.save_dir / 'robustness_comparison.json', 'w') as f:
            def convert_for_json(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.integer, np.floating)):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {key: convert_for_json(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_for_json(item) for item in obj]
                else:
                    return obj
            
            json.dump(convert_for_json(comparison_results), f, indent=2)
        
        logger.info("Robustness comparison completed!")
        return comparison_results
    
    def _create_comparison_matrices(self, comparison_results: Dict[str, Any]):
        """Create comparison matrices for different metrics."""
        models = comparison_results['models']
        issue_types = comparison_results['issue_types']
        
        # Initialize matrices
        matrices = {
            'robustness_score': np.zeros((len(models), len(issue_types))),
            'max_mae_degradation': np.zeros((len(models), len(issue_types))),
            'max_r2_degradation': np.zeros((len(models), len(issue_types)))
        }
        
        # Fill matrices
        for i, model_name in enumerate(models):
            for j, issue_type in enumerate(issue_types):
                robustness_data = comparison_results['individual_results'][model_name][issue_type]['robustness_analysis']
                
                matrices['robustness_score'][i, j] = robustness_data['robustness_score']
                matrices['max_mae_degradation'][i, j] = robustness_data['max_mae_degradation']
                matrices['max_r2_degradation'][i, j] = robustness_data['max_r2_degradation']
        
        comparison_results['comparison_matrix'] = {
            metric: matrix.tolist() for metric, matrix in matrices.items()
        }
    
    def _rank_models_by_robustness(self, comparison_results: Dict[str, Any]):
        """Rank models by overall robustness."""
        models = comparison_results['models']
        
        # Calculate overall robustness scores
        overall_scores = {}
        for model_name in models:
            scores = []
            for issue_type in comparison_results['issue_types']:
                robustness_score = comparison_results['individual_results'][model_name][issue_type]['robustness_analysis']['robustness_score']
                scores.append(robustness_score)
            overall_scores[model_name] = np.mean(scores)
        
        # Rank models
        ranked_models = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)
        comparison_results['robustness_rankings']['overall'] = [model for model, score in ranked_models]
        comparison_results['robustness_rankings']['scores'] = dict(ranked_models)
    
    def _plot_robustness_comparison(self, comparison_results: Dict[str, Any]):
        """Create robustness comparison plots."""
        models = comparison_results['models']
        issue_types = comparison_results['issue_types']
        
        # Robustness heatmap
        robustness_matrix = np.array(comparison_results['comparison_matrix']['robustness_score'])
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            robustness_matrix,
            annot=True,
            fmt='.3f',
            xticklabels=[issue.replace('_', ' ').title() for issue in issue_types],
            yticklabels=models,
            cmap='RdYlGn',
            cbar_kws={'label': 'Robustness Score (Higher is Better)'}
        )
        plt.title('Model Robustness Comparison')
        plt.ylabel('Models')
        plt.xlabel('Data Quality Issues')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.save_dir / 'robustness_comparison_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Overall robustness ranking
        scores = comparison_results['robustness_rankings']['scores']
        models_ranked = comparison_results['robustness_rankings']['overall']
        
        plt.figure(figsize=(10, 6))
        scores_values = [scores[model] for model in models_ranked]
        bars = plt.bar(range(len(models_ranked)), scores_values)
        plt.xlabel('Models')
        plt.ylabel('Overall Robustness Score')
        plt.title('Overall Model Robustness Ranking')
        plt.xticks(range(len(models_ranked)), models_ranked, rotation=45)
        
        # Color bars based on performance
        for i, bar in enumerate(bars):
            if scores_values[i] > 0.8:
                bar.set_color('green')
            elif scores_values[i] > 0.6:
                bar.set_color('yellow')
            else:
                bar.set_color('red')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.save_dir / 'overall_robustness_ranking.png', dpi=300, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    # Example usage
    print("Stress testing framework created successfully!")
    
    # Test data corruption
    corruptions = DataCorruptor()
    
    # Test cloud cover simulation
    dummy_satellite = torch.randn(4, 7, 64, 64)
    cloudy_satellite = corruptions.add_cloud_cover(dummy_satellite, 0.5)
    print(f"Original satellite shape: {dummy_satellite.shape}")
    print(f"Cloudy satellite shape: {cloudy_satellite.shape}")
    
    # Test weather corruption
    dummy_weather = torch.randn(4, 32, 6)
    corrupted_weather = corruptions.create_missing_weather_data(dummy_weather, 0.3)
    print(f"Original weather shape: {dummy_weather.shape}")
    print(f"Corrupted weather shape: {corrupted_weather.shape}")
    
    print("Stress testing components working correctly!")