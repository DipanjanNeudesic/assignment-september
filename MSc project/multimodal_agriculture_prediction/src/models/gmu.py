"""
Gated Multimodal Unit (GMU) - Adaptive Fusion Architecture

This module implements the core innovation of the project: a Gated Multimodal Unit
that dynamically weights different data modalities based on data quality and relevance.
The GMU acts as a "switch" that decides which modality to trust more at any given time.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)

class GatedMultimodalUnit(nn.Module):
    """
    Gated Multimodal Unit for adaptive fusion of satellite and weather data.
    
    The GMU implements a gating mechanism that:
    1. Assesses the quality/reliability of each modality
    2. Dynamically adjusts fusion weights based on data quality
    3. Provides interpretable attention weights showing which modality is emphasized
    
    Key innovation: During cloudy periods (noisy satellite data), the GMU emphasizes
    weather data more heavily, and vice versa during sensor failures.
    """
    
    def __init__(
        self,
        satellite_dim: int = 256,
        weather_dim: int = 256,
        hidden_dim: int = 128,
        num_gates: int = 3,  # Number of gating mechanisms
        temperature: float = 1.0,  # Temperature for softmax
        use_quality_estimation: bool = True,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.satellite_dim = satellite_dim
        self.weather_dim = weather_dim
        self.hidden_dim = hidden_dim
        self.num_gates = num_gates
        self.temperature = temperature
        self.use_quality_estimation = use_quality_estimation
        
        # Project both modalities to common dimension
        self.satellite_projection = nn.Sequential(
            nn.Linear(satellite_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        self.weather_projection = nn.Sequential(
            nn.Linear(weather_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Quality estimation networks
        if use_quality_estimation:
            self.satellite_quality_estimator = QualityEstimationNetwork(
                satellite_dim, hidden_dim, dropout
            )
            self.weather_quality_estimator = QualityEstimationNetwork(
                weather_dim, hidden_dim, dropout
            )
        
        # Multi-level gating mechanisms
        self.gates = nn.ModuleList([
            GatingMechanism(hidden_dim, dropout) for _ in range(num_gates)
        ])
        
        # Cross-modal attention
        self.cross_attention = CrossModalAttention(hidden_dim, dropout)
        
        # Feature integration
        self.feature_integration = FeatureIntegration(
            hidden_dim, hidden_dim * 2, dropout
        )
        
        # Final output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Uncertainty estimation
        self.uncertainty_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        satellite_features: torch.Tensor,
        weather_features: torch.Tensor,
        return_attention: bool = False,
        external_quality_scores: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through GMU.
        
        Args:
            satellite_features: Features from satellite expert (batch_size, satellite_dim)
            weather_features: Features from weather expert (batch_size, weather_dim)
            return_attention: Whether to return attention weights
            external_quality_scores: Optional external quality assessments
        
        Returns:
            Dictionary containing fused features, attention weights, and uncertainty
        """
        batch_size = satellite_features.size(0)
        
        # Project to common dimension
        satellite_proj = self.satellite_projection(satellite_features)
        weather_proj = self.weather_projection(weather_features)
        
        # Estimate data quality
        quality_scores = {}
        if self.use_quality_estimation:
            satellite_quality = self.satellite_quality_estimator(satellite_features)
            weather_quality = self.weather_quality_estimator(weather_features)
            quality_scores['satellite'] = satellite_quality
            quality_scores['weather'] = weather_quality
        
        # Use external quality scores if provided
        if external_quality_scores is not None:
            quality_scores.update(external_quality_scores)
        
        # Apply multi-level gating
        gated_features = []
        gate_weights = []
        
        for i, gate in enumerate(self.gates):
            gated_sat, gated_weather, weights = gate(
                satellite_proj, weather_proj, quality_scores, self.temperature
            )
            gated_features.append((gated_sat, gated_weather))
            gate_weights.append(weights)
        
        # Cross-modal attention
        attended_features = self.cross_attention(satellite_proj, weather_proj)
        
        # Integrate all gated features
        integrated_features = self.feature_integration(gated_features, attended_features)
        
        # Final output
        fused_features = self.output_projection(integrated_features)
        uncertainty = self.uncertainty_head(fused_features)
        
        # Prepare output
        output = {
            'fused_features': fused_features,
            'uncertainty': uncertainty,
            'quality_scores': quality_scores
        }
        
        if return_attention:
            output['gate_weights'] = gate_weights
            output['projected_features'] = {
                'satellite': satellite_proj,
                'weather': weather_proj
            }
        
        return output


class QualityEstimationNetwork(nn.Module):
    """Estimates the quality/reliability of input features."""
    
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        
        self.quality_estimator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Quality score between 0 and 1
        )
        
        # Feature statistics for quality assessment
        self.stats_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 4, 3)  # Mean, std, skewness indicators
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Estimate quality score for input features."""
        # Basic quality estimation
        quality_score = self.quality_estimator(x)
        
        # Additional statistical indicators
        stats = self.stats_network(x)
        
        # Combine quality indicators
        # High variance might indicate noise, extreme values might indicate corruption
        variance_penalty = torch.sigmoid(-torch.abs(stats[:, 1:2]))  # Penalize high variance
        range_penalty = torch.sigmoid(-torch.abs(stats[:, 2:3]))     # Penalize extreme values
        
        # Adjust quality score based on statistics
        adjusted_quality = quality_score * variance_penalty * range_penalty
        
        return adjusted_quality


class GatingMechanism(nn.Module):
    """Individual gating mechanism for modality weighting."""
    
    def __init__(self, feature_dim: int, dropout: float = 0.2):
        super().__init__()
        
        # Gating networks for each modality
        self.satellite_gate = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, feature_dim),
            nn.Sigmoid()
        )
        
        self.weather_gate = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, feature_dim),
            nn.Sigmoid()
        )
        
        # Cross-modal gating (how much each modality influences the other)
        self.cross_gate = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 2),  # Weight for [satellite, weather]
            nn.Softmax(dim=1)
        )
        
        # Quality-aware weighting
        self.quality_gate = nn.Sequential(
            nn.Linear(2, 32),  # 2 quality scores
            nn.ReLU(inplace=True),
            nn.Linear(32, 2),
            nn.Softmax(dim=1)
        )
    
    def forward(
        self,
        satellite_features: torch.Tensor,
        weather_features: torch.Tensor,
        quality_scores: Dict[str, torch.Tensor],
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply gating mechanism to features.
        
        Args:
            satellite_features: Satellite features
            weather_features: Weather features
            quality_scores: Quality scores for each modality
            temperature: Temperature for softmax
        
        Returns:
            Gated satellite features, gated weather features, and gate weights
        """
        # Self-gating (each modality gates itself)
        satellite_self_gate = self.satellite_gate(satellite_features)
        weather_self_gate = self.weather_gate(weather_features)
        
        gated_satellite = satellite_features * satellite_self_gate
        gated_weather = weather_features * weather_self_gate
        
        # Cross-modal gating
        combined_features = torch.cat([gated_satellite, gated_weather], dim=1)
        cross_weights = self.cross_gate(combined_features)
        
        # Quality-aware gating
        if 'satellite' in quality_scores and 'weather' in quality_scores:
            quality_input = torch.cat([
                quality_scores['satellite'],
                quality_scores['weather']
            ], dim=1)
            quality_weights = self.quality_gate(quality_input)
            
            # Combine cross-modal and quality weights
            final_weights = cross_weights * quality_weights
        else:
            final_weights = cross_weights
        
        # Apply temperature scaling
        final_weights = F.softmax(final_weights / temperature, dim=1)
        
        # Apply final weighting
        satellite_weight = final_weights[:, 0:1]
        weather_weight = final_weights[:, 1:2]
        
        final_satellite = gated_satellite * satellite_weight
        final_weather = gated_weather * weather_weight
        
        return final_satellite, final_weather, final_weights


class CrossModalAttention(nn.Module):
    """Cross-modal attention mechanism between satellite and weather features."""
    
    def __init__(self, feature_dim: int, dropout: float = 0.2):
        super().__init__()
        
        self.feature_dim = feature_dim
        
        # Attention networks
        self.satellite_to_weather_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        self.weather_to_satellite_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Feature normalization
        self.layer_norm = nn.LayerNorm(feature_dim)
        
    def forward(
        self,
        satellite_features: torch.Tensor,
        weather_features: torch.Tensor
    ) -> torch.Tensor:
        """Apply cross-modal attention."""
        # Add sequence dimension for attention
        satellite_seq = satellite_features.unsqueeze(1)  # (batch, 1, features)
        weather_seq = weather_features.unsqueeze(1)      # (batch, 1, features)
        
        # Satellite attends to weather
        sat_attended, _ = self.satellite_to_weather_attention(
            satellite_seq, weather_seq, weather_seq
        )
        
        # Weather attends to satellite
        weather_attended, _ = self.weather_to_satellite_attention(
            weather_seq, satellite_seq, satellite_seq
        )
        
        # Combine attended features
        attended_features = torch.cat([
            sat_attended.squeeze(1),
            weather_attended.squeeze(1)
        ], dim=1)
        
        return self.layer_norm(attended_features)


class FeatureIntegration(nn.Module):
    """Integrates features from multiple gating levels."""
    
    def __init__(self, feature_dim: int, output_dim: int, dropout: float = 0.2):
        super().__init__()
        
        self.integration_network = nn.Sequential(
            nn.Linear(feature_dim * 6, output_dim),  # 6 = 3 gates * 2 modalities
            nn.LayerNorm(output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(inplace=True)
        )
        
        # Attention over different gate levels
        self.gate_attention = nn.Sequential(
            nn.Linear(feature_dim * 6, 3),  # 3 attention weights for 3 gates
            nn.Softmax(dim=1)
        )
    
    def forward(
        self,
        gated_features: List[Tuple[torch.Tensor, torch.Tensor]],
        attended_features: torch.Tensor
    ) -> torch.Tensor:
        """Integrate features from multiple gating levels."""
        # Flatten all gated features
        all_features = []
        for satellite_gated, weather_gated in gated_features:
            all_features.extend([satellite_gated, weather_gated])
        
        combined = torch.cat(all_features, dim=1)
        
        # Compute attention weights over gate levels
        gate_weights = self.gate_attention(combined)
        
        # Weighted combination of gate outputs
        weighted_features = []
        for i, (sat, weather) in enumerate(gated_features):
            weight = gate_weights[:, i:i+1]
            weighted_sat = sat * weight
            weighted_weather = weather * weight
            weighted_features.extend([weighted_sat, weighted_weather])
        
        # Integrate all features
        final_combined = torch.cat(weighted_features, dim=1)
        integrated = self.integration_network(final_combined)
        
        # Residual connection with attended features
        integrated = integrated + attended_features
        
        return integrated


class GMUPredictor(nn.Module):
    """
    Complete GMU-based predictor combining expert networks with adaptive fusion.
    
    This is the main model that combines the satellite and weather expert networks
    with the Gated Multimodal Unit for adaptive fusion.
    """
    
    def __init__(
        self,
        satellite_expert: nn.Module,
        weather_expert: nn.Module,
        gmu_config: Dict,
        prediction_head_config: Optional[Dict] = None
    ):
        super().__init__()
        
        self.satellite_expert = satellite_expert
        self.weather_expert = weather_expert
        
        # Gated Multimodal Unit
        self.gmu = GatedMultimodalUnit(**gmu_config)
        
        # Prediction head
        if prediction_head_config is None:
            prediction_head_config = {'hidden_dims': [128, 64], 'dropout': 0.2}
        
        self.prediction_head = self._build_prediction_head(prediction_head_config)
        
    def _build_prediction_head(self, config: Dict) -> nn.Module:
        """Build prediction head from configuration."""
        layers = []
        input_dim = 256  # GMU output dimension
        
        for hidden_dim in config['hidden_dims']:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(config.get('dropout', 0.2))
            ])
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, 1))  # Final prediction
        
        return nn.Sequential(*layers)
    
    def forward(
        self,
        satellite: torch.Tensor,
        weather: torch.Tensor,
        return_attention: bool = False,
        return_expert_outputs: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through GMU predictor.
        
        Args:
            satellite: Satellite imagery
            weather: Weather time series
            return_attention: Whether to return attention weights
            return_expert_outputs: Whether to return expert outputs
        
        Returns:
            Dictionary containing predictions and optional additional outputs
        """
        # Get expert features
        satellite_output = self.satellite_expert(satellite, return_features=True)
        weather_output = self.weather_expert(weather, return_features=True)
        
        # Apply GMU
        gmu_output = self.gmu(
            satellite_output['features'],
            weather_output['features'],
            return_attention=return_attention
        )
        
        # Generate final prediction
        prediction = self.prediction_head(gmu_output['fused_features'])
        
        # Prepare output
        result = {
            'prediction': prediction,
            'uncertainty': gmu_output['uncertainty'],
            'quality_scores': gmu_output['quality_scores']
        }
        
        if return_attention:
            result['attention_weights'] = gmu_output['gate_weights']
            result['projected_features'] = gmu_output['projected_features']
        
        if return_expert_outputs:
            result['expert_outputs'] = {
                'satellite': satellite_output,
                'weather': weather_output
            }
        
        return result


def create_gmu_model(
    satellite_expert: nn.Module,
    weather_expert: nn.Module,
    config: Dict
) -> GMUPredictor:
    """Create GMU model from configuration."""
    gmu_config = {
        'satellite_dim': config.get('satellite_dim', 256),
        'weather_dim': config.get('weather_dim', 256),
        'hidden_dim': config.get('hidden_dim', 128),
        'num_gates': config.get('num_gates', 3),
        'temperature': config.get('temperature', 1.0),
        'use_quality_estimation': config.get('use_quality_estimation', True),
        'dropout': config.get('dropout', 0.2)
    }
    
    prediction_head_config = config.get('prediction_head', {
        'hidden_dims': [128, 64],
        'dropout': 0.2
    })
    
    return GMUPredictor(
        satellite_expert=satellite_expert,
        weather_expert=weather_expert,
        gmu_config=gmu_config,
        prediction_head_config=prediction_head_config
    )


if __name__ == "__main__":
    # Test GMU
    satellite_dim = 256
    weather_dim = 256
    batch_size = 4
    
    # Create dummy expert features
    satellite_features = torch.randn(batch_size, satellite_dim)
    weather_features = torch.randn(batch_size, weather_dim)
    
    # Create GMU
    gmu = GatedMultimodalUnit(
        satellite_dim=satellite_dim,
        weather_dim=weather_dim,
        hidden_dim=128,
        num_gates=3
    )
    
    # Forward pass
    output = gmu(satellite_features, weather_features, return_attention=True)
    
    print(f"Fused features shape: {output['fused_features'].shape}")
    print(f"Uncertainty shape: {output['uncertainty'].shape}")
    print(f"Number of gate weights: {len(output['gate_weights'])}")
    
    # Count parameters
    total_params = sum(p.numel() for p in gmu.parameters() if p.requires_grad)
    print(f"GMU parameters: {total_params:,}")