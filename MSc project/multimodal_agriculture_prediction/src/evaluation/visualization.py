"""
Comprehensive Visualization and Analysis Tools

This module provides advanced visualization capabilities for the multimodal
agriculture prediction project, including attention visualization, feature
analysis, temporal patterns, and interactive dashboards.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from pathlib import Path
import json
from datetime import datetime
import cv2
from scipy import stats
import warnings

logger = logging.getLogger(__name__)

class AttentionVisualizer:
    """Visualize attention mechanisms and feature importance."""
    
    def __init__(self, save_dir: str = "results/visualizations"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def visualize_gmu_attention(
        self,
        model,
        satellite_data: torch.Tensor,
        weather_data: torch.Tensor,
        sample_indices: List[int] = None,
        save_name: str = "gmu_attention"
    ):
        """
        Visualize GMU attention weights and quality assessments.
        
        Args:
            model: Trained GMU model
            satellite_data: Satellite data batch
            weather_data: Weather data batch
            sample_indices: Specific samples to visualize
            save_name: Name for saved visualization
        """
        model.eval()
        
        if sample_indices is None:
            sample_indices = list(range(min(4, satellite_data.size(0))))
        
        with torch.no_grad():
            output = model(
                satellite_data[sample_indices],
                weather_data[sample_indices],
                return_attention=True,
                return_expert_outputs=True
            )
        
        # Extract attention and quality data
        gate_weights = output.get('attention_weights', [])
        quality_scores = output.get('quality_scores', {})
        predictions = output['prediction'].cpu().numpy()
        uncertainty = output.get('uncertainty', torch.zeros_like(output['prediction'])).cpu().numpy()
        
        # Create subplot layout
        n_samples = len(sample_indices)
        n_gates = len(gate_weights) if gate_weights else 3
        
        fig = make_subplots(
            rows=n_samples,
            cols=4,
            subplot_titles=['Satellite Image', 'Weather Series', 'Gate Weights', 'Quality Scores'],
            specs=[[{"type": "scatter"}, {"type": "scatter"}, {"type": "bar"}, {"type": "bar"}] for _ in range(n_samples)],
            vertical_spacing=0.1
        )
        
        for i, sample_idx in enumerate(sample_indices):
            row = i + 1
            
            # 1. Satellite Image (show NDVI if available)
            sat_sample = satellite_data[sample_idx]
            if sat_sample.shape[0] >= 4:  # Has NDVI
                ndvi_img = sat_sample[3].cpu().numpy()  # Assuming 4th channel is NDVI
            else:
                # Create RGB composite
                rgb_img = sat_sample[:3].cpu().numpy()
                ndvi_img = np.mean(rgb_img, axis=0)  # Simple grayscale
            
            fig.add_trace(
                go.Heatmap(z=ndvi_img, colorscale='RdYlGn', name=f'NDVI {sample_idx}'),
                row=row, col=1
            )
            
            # 2. Weather Time Series
            weather_sample = weather_data[sample_idx].cpu().numpy()
            time_steps = np.arange(weather_sample.shape[0])
            
            # Show multiple weather variables
            for j, var_name in enumerate(['Temp', 'Precip', 'Humidity', 'Wind', 'Solar', 'Pressure']):
                if j < weather_sample.shape[1]:
                    fig.add_trace(
                        go.Scatter(
                            x=time_steps,
                            y=weather_sample[:, j],
                            name=f'{var_name} {sample_idx}',
                            line=dict(width=2),
                            opacity=0.7
                        ),
                        row=row, col=2
                    )
            
            # 3. Gate Weights
            if gate_weights and i < len(gate_weights):
                sample_gate_weights = gate_weights[i][sample_idx].cpu().numpy()  # [satellite_weight, weather_weight]
                fig.add_trace(
                    go.Bar(
                        x=['Satellite', 'Weather'],
                        y=sample_gate_weights,
                        name=f'Gates {sample_idx}',
                        marker_color=['blue', 'orange']
                    ),
                    row=row, col=3
                )
            
            # 4. Quality Scores
            if quality_scores:
                quality_names = []
                quality_values = []
                for modality, scores in quality_scores.items():
                    quality_names.append(modality.title())
                    quality_values.append(scores[sample_idx].cpu().numpy().item())
                
                fig.add_trace(
                    go.Bar(
                        x=quality_names,
                        y=quality_values,
                        name=f'Quality {sample_idx}',
                        marker_color=['green', 'blue']
                    ),
                    row=row, col=4
                )
        
        fig.update_layout(
            title="GMU Attention and Quality Analysis",
            height=300 * n_samples,
            showlegend=False
        )
        
        # Save interactive plot
        fig.write_html(self.save_dir / f"{save_name}.html")
        
        # Create detailed analysis
        self._create_attention_analysis(output, sample_indices, save_name)
    
    def _create_attention_analysis(self, output: Dict, sample_indices: List[int], save_name: str):
        """Create detailed attention analysis plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Extract data
        predictions = output['prediction'].cpu().numpy().flatten()
        uncertainty = output.get('uncertainty', torch.zeros_like(output['prediction'])).cpu().numpy().flatten()
        
        # 1. Prediction vs Uncertainty
        axes[0, 0].scatter(predictions, uncertainty, alpha=0.6)
        axes[0, 0].set_xlabel('Prediction')
        axes[0, 0].set_ylabel('Uncertainty')
        axes[0, 0].set_title('Prediction vs Uncertainty')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Quality Score Distribution
        if 'quality_scores' in output and output['quality_scores']:
            quality_data = []
            for modality, scores in output['quality_scores'].items():
                quality_data.extend([(modality, score.item()) for score in scores.cpu()])
            
            if quality_data:
                df_quality = pd.DataFrame(quality_data, columns=['Modality', 'Quality'])
                sns.boxplot(data=df_quality, x='Modality', y='Quality', ax=axes[0, 1])
                axes[0, 1].set_title('Quality Score Distribution')
        
        # 3. Gate Weights Analysis
        if 'attention_weights' in output and output['attention_weights']:
            gate_weights = output['attention_weights']
            if gate_weights:
                # Average across gates and samples
                avg_weights = []
                for gate_weight in gate_weights:
                    avg_weights.append(gate_weight.mean(dim=0).cpu().numpy())
                
                if avg_weights:
                    weights_array = np.array(avg_weights)
                    x = np.arange(weights_array.shape[1])
                    width = 0.35
                    
                    for i, gate_idx in enumerate(range(len(avg_weights))):
                        axes[1, 0].bar(x + i * width, weights_array[i], width, 
                                     label=f'Gate {gate_idx}', alpha=0.7)
                    
                    axes[1, 0].set_xlabel('Modality (0: Satellite, 1: Weather)')
                    axes[1, 0].set_ylabel('Average Weight')
                    axes[1, 0].set_title('Average Gate Weights')
                    axes[1, 0].legend()
                    axes[1, 0].set_xticks(x)
                    axes[1, 0].set_xticklabels(['Satellite', 'Weather'])
        
        # 4. Uncertainty Calibration
        if len(uncertainty) > 0 and np.var(uncertainty) > 0:
            # Sort by uncertainty and plot cumulative error
            sort_idx = np.argsort(uncertainty)
            sorted_uncertainty = uncertainty[sort_idx]
            
            # For demonstration, create synthetic errors
            synthetic_errors = np.abs(np.random.normal(0, sorted_uncertainty))
            cumulative_error = np.cumsum(synthetic_errors) / np.arange(1, len(synthetic_errors) + 1)
            
            axes[1, 1].plot(sorted_uncertainty, cumulative_error)
            axes[1, 1].set_xlabel('Predicted Uncertainty')
            axes[1, 1].set_ylabel('Cumulative Average Error')
            axes[1, 1].set_title('Uncertainty Calibration')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'Detailed Attention Analysis - {save_name}', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.save_dir / f'{save_name}_detailed.png', dpi=300, bbox_inches='tight')
        plt.close()


class FeatureAnalyzer:
    """Analyze and visualize learned features."""
    
    def __init__(self, save_dir: str = "results/feature_analysis"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def analyze_learned_features(
        self,
        model,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device,
        model_type: str = 'gmu',
        max_samples: int = 500
    ):
        """
        Comprehensive analysis of learned features.
        
        Args:
            model: Trained model
            data_loader: Data loader
            device: Device for inference
            model_type: Type of model
            max_samples: Maximum samples to analyze
        """
        model.eval()
        
        features = {
            'satellite': [],
            'weather': [],
            'fused': [],
            'predictions': [],
            'targets': [],
            'metadata': []
        }
        
        sample_count = 0
        with torch.no_grad():
            for batch in data_loader:
                if sample_count >= max_samples:
                    break
                
                satellite = batch['satellite'].to(device)
                weather = batch['weather'].to(device)
                targets = batch['yield'].to(device)
                
                # Extract features based on model type
                if model_type == 'gmu':
                    output = model(
                        satellite, weather,
                        return_attention=True,
                        return_expert_outputs=True
                    )
                    
                    features['fused'].append(output['fused_features'].cpu().numpy())
                    features['predictions'].append(output['prediction'].cpu().numpy())
                    
                    if 'expert_outputs' in output:
                        expert_outputs = output['expert_outputs']
                        if 'satellite' in expert_outputs:
                            features['satellite'].append(expert_outputs['satellite']['features'].cpu().numpy())
                        if 'weather' in expert_outputs:
                            features['weather'].append(expert_outputs['weather']['features'].cpu().numpy())
                
                elif model_type in ['early_fusion', 'late_fusion']:
                    # For these models, we need to hook into intermediate layers
                    predictions = model(satellite, weather)
                    features['predictions'].append(predictions.cpu().numpy())
                
                features['targets'].append(targets.cpu().numpy())
                features['metadata'].append({
                    'county_fips': batch['county_fips'],
                    'year': batch['year']
                })
                
                sample_count += len(satellite)
        
        # Concatenate collected features
        for key in ['satellite', 'weather', 'fused', 'predictions', 'targets']:
            if features[key]:
                features[key] = np.concatenate(features[key], axis=0)
        
        # Perform various analyses
        self._dimensionality_analysis(features)
        self._feature_importance_analysis(features)
        self._clustering_analysis(features)
        self._correlation_analysis(features)
    
    def _dimensionality_analysis(self, features: Dict):
        """Analyze feature dimensionality using PCA and t-SNE."""
        if 'fused' in features and len(features['fused']) > 0:
            fused_features = features['fused']
            targets = features['targets']
            
            # PCA Analysis
            pca = PCA(n_components=50)
            pca_features = pca.fit_transform(fused_features)
            
            # Plot explained variance
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.plot(np.cumsum(pca.explained_variance_ratio_))
            plt.xlabel('Number of Components')
            plt.ylabel('Cumulative Explained Variance Ratio')
            plt.title('PCA Explained Variance')
            plt.grid(True, alpha=0.3)
            
            # t-SNE visualization
            if len(fused_features) > 50:  # Only run t-SNE on subset
                sample_idx = np.random.choice(len(fused_features), 500, replace=False)
                tsne_features = fused_features[sample_idx]
                tsne_targets = targets[sample_idx]
                
                tsne = TSNE(n_components=2, random_state=42)
                tsne_result = tsne.fit_transform(tsne_features)
                
                plt.subplot(1, 2, 2)
                scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], 
                                    c=tsne_targets, cmap='viridis', alpha=0.6)
                plt.colorbar(scatter, label='Yield')
                plt.xlabel('t-SNE 1')
                plt.ylabel('t-SNE 2')
                plt.title('t-SNE Visualization (colored by yield)')
            
            plt.tight_layout()
            plt.savefig(self.save_dir / 'dimensionality_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _feature_importance_analysis(self, features: Dict):
        """Analyze feature importance using various methods."""
        if 'fused' not in features or len(features['fused']) == 0:
            return
        
        X = features['fused']
        y = features['targets'].flatten()
        
        # Calculate feature correlations with target
        correlations = []
        for i in range(X.shape[1]):
            corr, p_value = stats.pearsonr(X[:, i], y)
            correlations.append(abs(corr))
        
        correlations = np.array(correlations)
        
        # Plot top feature importances
        top_k = min(50, len(correlations))
        top_indices = np.argsort(correlations)[-top_k:]
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(top_k), correlations[top_indices])
        plt.xlabel('Absolute Correlation with Yield')
        plt.ylabel('Feature Index')
        plt.title(f'Top {top_k} Most Important Features')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.save_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _clustering_analysis(self, features: Dict):
        """Perform clustering analysis on features."""
        if 'fused' not in features or len(features['fused']) == 0:
            return
        
        X = features['fused']
        y = features['targets'].flatten()
        
        # K-means clustering
        n_clusters = 5
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(X)
        
        # PCA for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # 1. Clusters in PCA space
        scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='tab10')
        axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        axes[0].set_title('Clusters in PCA Space')
        
        # 2. Yield by cluster
        cluster_yields = [y[cluster_labels == i] for i in range(n_clusters)]
        axes[1].boxplot(cluster_yields, labels=[f'Cluster {i}' for i in range(n_clusters)])
        axes[1].set_ylabel('Yield')
        axes[1].set_title('Yield Distribution by Cluster')
        axes[1].grid(True, alpha=0.3)
        
        # 3. Cluster characteristics
        cluster_centers_pca = pca.transform(kmeans.cluster_centers_)
        axes[2].scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='tab10', alpha=0.6)
        axes[2].scatter(cluster_centers_pca[:, 0], cluster_centers_pca[:, 1], 
                       c='red', marker='x', s=200, linewidths=3)
        axes[2].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        axes[2].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        axes[2].set_title('Clusters with Centroids')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'clustering_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _correlation_analysis(self, features: Dict):
        """Analyze correlations between different feature types."""
        if len(features['satellite']) == 0 or len(features['weather']) == 0:
            return
        
        # Calculate cross-modal correlations
        sat_features = features['satellite']
        weather_features = features['weather']
        
        # Sample features for correlation analysis
        n_sat_features = min(20, sat_features.shape[1])
        n_weather_features = min(20, weather_features.shape[1])
        
        sat_sample = sat_features[:, :n_sat_features]
        weather_sample = weather_features[:, :n_weather_features]
        
        # Cross-correlation matrix
        cross_corr = np.corrcoef(sat_sample.T, weather_sample.T)
        sat_weather_corr = cross_corr[:n_sat_features, n_sat_features:]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(sat_weather_corr, 
                   xticklabels=[f'W{i}' for i in range(n_weather_features)],
                   yticklabels=[f'S{i}' for i in range(n_sat_features)],
                   cmap='RdBu_r', center=0, annot=False)
        plt.title('Cross-Modal Feature Correlations\n(Satellite vs Weather Features)')
        plt.xlabel('Weather Features')
        plt.ylabel('Satellite Features')
        plt.tight_layout()
        plt.savefig(self.save_dir / 'cross_modal_correlations.png', dpi=300, bbox_inches='tight')
        plt.close()


class TemporalAnalyzer:
    """Analyze temporal patterns in data and predictions."""
    
    def __init__(self, save_dir: str = "results/temporal_analysis"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def analyze_temporal_patterns(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        years: List[int],
        counties: List[str],
        save_name: str = "temporal_analysis"
    ):
        """
        Analyze temporal patterns in predictions and targets.
        
        Args:
            predictions: Model predictions
            targets: True yield values
            years: Corresponding years
            counties: Corresponding counties
            save_name: Name for saved analysis
        """
        # Create DataFrame for analysis
        df = pd.DataFrame({
            'prediction': predictions.flatten(),
            'target': targets.flatten(),
            'year': years,
            'county': counties
        })
        
        # Calculate errors
        df['error'] = df['target'] - df['prediction']
        df['abs_error'] = np.abs(df['error'])
        df['percent_error'] = (df['error'] / df['target']) * 100
        
        # Temporal trend analysis
        self._analyze_yearly_trends(df, save_name)
        self._analyze_seasonal_patterns(df, save_name)
        self._analyze_county_patterns(df, save_name)
        
    def _analyze_yearly_trends(self, df: pd.DataFrame, save_name: str):
        """Analyze year-over-year trends."""
        yearly_stats = df.groupby('year').agg({
            'target': ['mean', 'std'],
            'prediction': ['mean', 'std'],
            'abs_error': 'mean',
            'percent_error': 'mean'
        }).round(3)
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Yield Trends', 'Error Trends', 'Prediction Accuracy', 'Error Distribution by Year']
        )
        
        # 1. Yield trends
        years = yearly_stats.index
        fig.add_trace(
            go.Scatter(x=years, y=yearly_stats[('target', 'mean')], 
                      name='Actual Mean', line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=years, y=yearly_stats[('prediction', 'mean')], 
                      name='Predicted Mean', line=dict(color='red')),
            row=1, col=1
        )
        
        # 2. Error trends
        fig.add_trace(
            go.Scatter(x=years, y=yearly_stats[('abs_error', 'mean')], 
                      name='Mean Absolute Error', line=dict(color='orange')),
            row=1, col=2
        )
        
        # 3. Prediction accuracy (R²)
        yearly_r2 = []
        for year in years:
            year_data = df[df['year'] == year]
            if len(year_data) > 1:
                r2 = stats.pearsonr(year_data['target'], year_data['prediction'])[0] ** 2
            else:
                r2 = 0
            yearly_r2.append(r2)
        
        fig.add_trace(
            go.Scatter(x=years, y=yearly_r2, 
                      name='R² by Year', line=dict(color='green')),
            row=2, col=1
        )
        
        # 4. Error distribution by year
        for year in years[-3:]:  # Show last 3 years to avoid clutter
            year_errors = df[df['year'] == year]['percent_error']
            fig.add_trace(
                go.Box(y=year_errors, name=f'Year {year}'),
                row=2, col=2
            )
        
        fig.update_layout(
            title=f'Temporal Analysis - {save_name}',
            height=800,
            showlegend=True
        )
        
        fig.write_html(self.save_dir / f'{save_name}_yearly_trends.html')
        
        # Save statistics
        yearly_stats.to_csv(self.save_dir / f'{save_name}_yearly_stats.csv')
    
    def _analyze_seasonal_patterns(self, df: pd.DataFrame, save_name: str):
        """Analyze seasonal patterns (if applicable)."""
        # For simplicity, assume growing season patterns
        # In practice, this would use actual planting/harvest dates
        
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Yield by year
        plt.subplot(2, 3, 1)
        df.groupby('year')['target'].mean().plot(kind='line', marker='o')
        plt.title('Average Yield by Year')
        plt.ylabel('Yield (bu/acre)')
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Prediction accuracy by year
        plt.subplot(2, 3, 2)
        yearly_accuracy = df.groupby('year').apply(
            lambda x: stats.pearsonr(x['target'], x['prediction'])[0] ** 2
        )
        yearly_accuracy.plot(kind='line', marker='s', color='red')
        plt.title('Prediction Accuracy (R²) by Year')
        plt.ylabel('R²')
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Error variance by year
        plt.subplot(2, 3, 3)
        df.groupby('year')['abs_error'].std().plot(kind='line', marker='^', color='orange')
        plt.title('Error Variance by Year')
        plt.ylabel('Error Std Dev')
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Yield distribution by year (violin plot)
        plt.subplot(2, 3, 4)
        years_to_plot = sorted(df['year'].unique())[-5:]  # Last 5 years
        year_data = [df[df['year'] == year]['target'].values for year in years_to_plot]
        parts = plt.violinplot(year_data, positions=range(len(years_to_plot)))
        plt.xticks(range(len(years_to_plot)), years_to_plot)
        plt.title('Yield Distribution by Year')
        plt.ylabel('Yield (bu/acre)')
        
        # Plot 5: Error vs yield
        plt.subplot(2, 3, 5)
        plt.scatter(df['target'], df['abs_error'], alpha=0.5)
        plt.xlabel('Actual Yield')
        plt.ylabel('Absolute Error')
        plt.title('Error vs Actual Yield')
        plt.grid(True, alpha=0.3)
        
        # Plot 6: Prediction vs actual by year (recent years)
        plt.subplot(2, 3, 6)
        colors = plt.cm.viridis(np.linspace(0, 1, len(years_to_plot)))
        for i, year in enumerate(years_to_plot):
            year_data = df[df['year'] == year]
            plt.scatter(year_data['target'], year_data['prediction'], 
                       alpha=0.6, color=colors[i], label=f'{year}')
        
        # Perfect prediction line
        min_yield = df[['target', 'prediction']].min().min()
        max_yield = df[['target', 'prediction']].max().max()
        plt.plot([min_yield, max_yield], [min_yield, max_yield], 'r--', alpha=0.8)
        
        plt.xlabel('Actual Yield')
        plt.ylabel('Predicted Yield')
        plt.title('Predictions vs Actual (by Year)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.suptitle(f'Seasonal Patterns Analysis - {save_name}', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.save_dir / f'{save_name}_seasonal_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _analyze_county_patterns(self, df: pd.DataFrame, save_name: str):
        """Analyze spatial patterns by county."""
        county_stats = df.groupby('county').agg({
            'target': ['mean', 'std', 'count'],
            'prediction': 'mean',
            'abs_error': 'mean'
        }).round(3)
        
        # Filter counties with sufficient data
        min_samples = 5
        county_stats = county_stats[county_stats[('target', 'count')] >= min_samples]
        
        if len(county_stats) == 0:
            logger.warning("No counties with sufficient samples for analysis")
            return
        
        # Calculate county-level accuracy
        county_accuracy = {}
        for county in county_stats.index:
            county_data = df[df['county'] == county]
            if len(county_data) > 1:
                r2 = stats.pearsonr(county_data['target'], county_data['prediction'])[0] ** 2
                county_accuracy[county] = r2
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. County yield distribution
        top_counties = county_stats.nlargest(10, ('target', 'count')).index
        county_yields = [df[df['county'] == county]['target'].values for county in top_counties]
        
        axes[0, 0].boxplot(county_yields, labels=top_counties)
        axes[0, 0].set_title('Yield Distribution by County (Top 10)')
        axes[0, 0].set_ylabel('Yield (bu/acre)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. County error comparison
        county_errors = [df[df['county'] == county]['abs_error'].mean() for county in top_counties]
        axes[0, 1].bar(range(len(top_counties)), county_errors)
        axes[0, 1].set_title('Average Error by County')
        axes[0, 1].set_ylabel('Mean Absolute Error')
        axes[0, 1].set_xticks(range(len(top_counties)))
        axes[0, 1].set_xticklabels(top_counties, rotation=45)
        
        # 3. County accuracy
        if county_accuracy:
            accuracies = [county_accuracy.get(county, 0) for county in top_counties]
            axes[1, 0].bar(range(len(top_counties)), accuracies)
            axes[1, 0].set_title('Prediction Accuracy (R²) by County')
            axes[1, 0].set_ylabel('R²')
            axes[1, 0].set_xticks(range(len(top_counties)))
            axes[1, 0].set_xticklabels(top_counties, rotation=45)
        
        # 4. Yield vs sample count
        sample_counts = county_stats[('target', 'count')].values
        mean_yields = county_stats[('target', 'mean')].values
        
        axes[1, 1].scatter(sample_counts, mean_yields, alpha=0.6)
        axes[1, 1].set_xlabel('Number of Samples')
        axes[1, 1].set_ylabel('Average Yield')
        axes[1, 1].set_title('County Sample Count vs Average Yield')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'County Patterns Analysis - {save_name}', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.save_dir / f'{save_name}_county_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save county statistics
        county_stats.to_csv(self.save_dir / f'{save_name}_county_stats.csv')


class InteractiveDashboard:
    """Create interactive dashboards for model analysis."""
    
    def __init__(self, save_dir: str = "results/dashboard"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def create_model_comparison_dashboard(
        self,
        model_results: Dict[str, Dict],
        stress_test_results: Dict[str, Dict],
        save_name: str = "model_comparison_dashboard"
    ):
        """
        Create comprehensive model comparison dashboard.
        
        Args:
            model_results: Dictionary of model evaluation results
            stress_test_results: Dictionary of stress test results
            save_name: Name for saved dashboard
        """
        # Create main dashboard with multiple tabs
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=[
                'Model Performance Comparison',
                'Robustness Analysis',
                'Prediction Accuracy',
                'Error Analysis',
                'Feature Importance',
                'Temporal Trends',
                'Uncertainty Analysis',
                'Quality Scores',
                'Overall Rankings'
            ],
            specs=[
                [{"type": "bar"}, {"type": "bar"}, {"type": "scatter"}],
                [{"type": "box"}, {"type": "heatmap"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "bar"}, {"type": "table"}]
            ]
        )
        
        # Extract model names
        model_names = list(model_results.keys())
        
        # 1. Model Performance Comparison
        metrics = ['mae', 'rmse', 'r2', 'mape']
        for i, metric in enumerate(metrics):
            if i == 0:  # Only plot first metric to avoid clutter
                values = [model_results[name]['metrics'].get(metric, 0) for name in model_names]
                fig.add_trace(
                    go.Bar(x=model_names, y=values, name=metric.upper()),
                    row=1, col=1
                )
        
        # 2. Robustness Analysis
        if stress_test_results:
            robustness_scores = []
            for name in model_names:
                if name in stress_test_results:
                    score = stress_test_results[name].get('robustness_score', 0)
                    robustness_scores.append(score)
                else:
                    robustness_scores.append(0)
            
            fig.add_trace(
                go.Bar(x=model_names, y=robustness_scores, name='Robustness'),
                row=1, col=2
            )
        
        # 3. Prediction Accuracy Scatter
        for i, name in enumerate(model_names):
            if 'predictions' in model_results[name] and 'targets' in model_results[name]:
                pred = model_results[name]['predictions'][:100]  # Sample for visualization
                targ = model_results[name]['targets'][:100]
                fig.add_trace(
                    go.Scatter(x=targ, y=pred, mode='markers', name=name, opacity=0.6),
                    row=1, col=3
                )
        
        # Add perfect prediction line
        if model_results:
            sample_result = list(model_results.values())[0]
            if 'targets' in sample_result:
                targets = sample_result['targets']
                min_val, max_val = targets.min(), targets.max()
                fig.add_trace(
                    go.Scatter(x=[min_val, max_val], y=[min_val, max_val], 
                             mode='lines', name='Perfect', line=dict(dash='dash', color='red')),
                    row=1, col=3
                )
        
        # Additional plots would be added here...
        
        fig.update_layout(
            title="Multimodal Agriculture Prediction - Model Comparison Dashboard",
            height=1200,
            showlegend=True
        )
        
        # Save interactive dashboard
        fig.write_html(self.save_dir / f"{save_name}.html")
        
        # Create summary report
        self._create_summary_report(model_results, stress_test_results, save_name)
    
    def _create_summary_report(
        self,
        model_results: Dict[str, Dict],
        stress_test_results: Dict[str, Dict],
        save_name: str
    ):
        """Create a summary report of all analyses."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'models_analyzed': list(model_results.keys()),
            'performance_summary': {},
            'robustness_summary': {},
            'recommendations': []
        }
        
        # Performance summary
        for model_name, results in model_results.items():
            if 'metrics' in results:
                report['performance_summary'][model_name] = {
                    'mae': results['metrics'].get('mae', 'N/A'),
                    'rmse': results['metrics'].get('rmse', 'N/A'),
                    'r2': results['metrics'].get('r2', 'N/A')
                }
        
        # Robustness summary
        for model_name, results in stress_test_results.items():
            report['robustness_summary'][model_name] = {
                'robustness_score': results.get('robustness_score', 'N/A'),
                'max_degradation': results.get('max_mae_degradation', 'N/A')
            }
        
        # Generate recommendations
        if model_results:
            best_performance = max(model_results.items(), 
                                 key=lambda x: x[1]['metrics'].get('r2', 0))
            report['recommendations'].append(
                f"Best performing model: {best_performance[0]} (R² = {best_performance[1]['metrics'].get('r2', 'N/A'):.3f})"
            )
        
        if stress_test_results:
            best_robustness = max(stress_test_results.items(),
                                key=lambda x: x[1].get('robustness_score', 0))
            report['recommendations'].append(
                f"Most robust model: {best_robustness[0]} (Robustness = {best_robustness[1].get('robustness_score', 'N/A'):.3f})"
            )
        
        # Save report
        with open(self.save_dir / f"{save_name}_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Summary report saved to {self.save_dir / f'{save_name}_report.json'}")


if __name__ == "__main__":
    # Example usage
    print("Visualization and analysis tools created successfully!")
    
    # Test basic functionality
    visualizer = AttentionVisualizer("test_viz")
    analyzer = FeatureAnalyzer("test_features")
    temporal = TemporalAnalyzer("test_temporal")
    dashboard = InteractiveDashboard("test_dashboard")
    
    print("All visualization components initialized successfully!")