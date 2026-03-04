"""
Late Fusion Model Architecture

This module implements a late fusion approach for multimodal crop yield prediction.
The model trains separate expert networks for satellite imagery and weather data,
then fuses their high-level representations or predictions at the decision level.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import numpy as np
from typing import Dict, Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)

class SatelliteExpertNetwork(nn.Module):
    """Expert network specialized for satellite imagery analysis."""
    
    def __init__(
        self,
        input_channels: int = 7,
        spatial_size: Tuple[int, int] = (64, 64),
        hidden_dims: List[int] = [64, 128, 256, 512],
        output_dim: int = 256,
        dropout: float = 0.2,
        use_attention: bool = True
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.spatial_size = spatial_size
        
        # Multi-scale CNN backbone
        self.backbone = SatelliteCNN(
            input_channels=input_channels,
            hidden_dims=hidden_dims,
            spatial_size=spatial_size,
            dropout=dropout
        )
        
        # Calculate backbone output size
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, *spatial_size)
            backbone_output = self.backbone(dummy_input)
            backbone_size = backbone_output.size(-1)
        
        # Feature pyramid network for multi-scale features
        self.fpn = FeaturePyramidNetwork(hidden_dims)
        
        # Global average pooling with attention
        if use_attention:
            self.global_pool = AttentiveGlobalPooling(hidden_dims[-1])
        else:
            self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Feature refinement layers
        self.feature_refiner = nn.Sequential(
            nn.Linear(backbone_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True)
        )
        
        # Expert prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(output_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )
        
        # Confidence estimation
        self.confidence_head = nn.Sequential(
            nn.Linear(output_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor, return_features: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through satellite expert network.
        
        Args:
            x: Satellite imagery (batch_size, channels, height, width)
            return_features: Whether to return intermediate features
        
        Returns:
            Dictionary containing prediction, features, and confidence
        """
        # Extract features through backbone
        features = self.backbone(x)
        
        # Refine features
        refined_features = self.feature_refiner(features)
        
        # Generate prediction and confidence
        prediction = self.prediction_head(refined_features)
        confidence = self.confidence_head(refined_features)
        
        result = {
            'prediction': prediction,
            'confidence': confidence,
            'features': refined_features
        }
        
        if return_features:
            result['raw_features'] = features
        
        return result


class WeatherExpertNetwork(nn.Module):
    """Expert network specialized for weather time series analysis."""
    
    def __init__(
        self,
        input_features: int = 6,
        sequence_length: int = 32,
        hidden_dim: int = 128,
        output_dim: int = 256,
        num_layers: int = 3,
        architecture: str = 'transformer',  # 'transformer', 'lstm', 'gru'
        dropout: float = 0.2,
        use_attention: bool = True
    ):
        super().__init__()
        
        self.input_features = input_features
        self.sequence_length = sequence_length
        self.architecture = architecture
        
        # Input embedding
        self.input_embedding = nn.Sequential(
            nn.Linear(input_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Temporal encoder
        if architecture == 'transformer':
            encoder_layer = TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation='relu',
                batch_first=True
            )
            self.temporal_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.positional_encoding = PositionalEncoding(hidden_dim, sequence_length)
            encoder_output_dim = hidden_dim
            
        elif architecture == 'lstm':
            self.temporal_encoder = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True,
                bidirectional=True
            )
            encoder_output_dim = hidden_dim * 2  # Bidirectional
            
        elif architecture == 'gru':
            self.temporal_encoder = nn.GRU(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True,
                bidirectional=True
            )
            encoder_output_dim = hidden_dim * 2  # Bidirectional
        
        # Temporal attention mechanism
        if use_attention:
            self.temporal_attention = TemporalAttention(encoder_output_dim)
        else:
            self.temporal_attention = None
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(encoder_output_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(inplace=True)
        )
        
        # Expert prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(output_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )
        
        # Confidence estimation
        self.confidence_head = nn.Sequential(
            nn.Linear(output_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor, return_features: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through weather expert network.
        
        Args:
            x: Weather time series (batch_size, sequence_length, features)
            return_features: Whether to return intermediate features
        
        Returns:
            Dictionary containing prediction, features, and confidence
        """
        batch_size = x.size(0)
        
        # Embed input features
        embedded = self.input_embedding(x)
        
        # Encode temporal sequence
        if self.architecture == 'transformer':
            # Add positional encoding
            embedded = self.positional_encoding(embedded)
            encoded = self.temporal_encoder(embedded)
        else:
            encoded, _ = self.temporal_encoder(embedded)
        
        # Apply temporal attention if available
        if self.temporal_attention is not None:
            aggregated = self.temporal_attention(encoded)
        else:
            # Simple mean pooling
            aggregated = encoded.mean(dim=1)
        
        # Extract features
        features = self.feature_extractor(aggregated)
        
        # Generate prediction and confidence
        prediction = self.prediction_head(features)
        confidence = self.confidence_head(features)
        
        result = {
            'prediction': prediction,
            'confidence': confidence,
            'features': features
        }
        
        if return_features:
            result['encoded'] = encoded
            result['aggregated'] = aggregated
        
        return result


class LateFusionModel(nn.Module):
    """
    Late Fusion model that trains separate expert networks and fuses their outputs.
    
    This model implements the "expert" approach where specialized architectures
    master their own domain before combining their predictions.
    """
    
    def __init__(
        self,
        image_channels: int = 7,
        image_size: Tuple[int, int] = (64, 64),
        weather_features: int = 6,
        sequence_length: int = 32,
        satellite_hidden_dims: List[int] = [64, 128, 256, 512],
        weather_architecture: str = 'transformer',
        weather_hidden_dim: int = 128,
        weather_num_layers: int = 3,
        fusion_strategy: str = 'weighted_average',  # 'weighted_average', 'mlp', 'attention'
        expert_output_dim: int = 256,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.fusion_strategy = fusion_strategy
        
        # Expert networks
        self.satellite_expert = SatelliteExpertNetwork(
            input_channels=image_channels,
            spatial_size=image_size,
            hidden_dims=satellite_hidden_dims,
            output_dim=expert_output_dim,
            dropout=dropout
        )
        
        self.weather_expert = WeatherExpertNetwork(
            input_features=weather_features,
            sequence_length=sequence_length,
            hidden_dim=weather_hidden_dim,
            output_dim=expert_output_dim,
            num_layers=weather_num_layers,
            architecture=weather_architecture,
            dropout=dropout
        )
        
        # Fusion mechanisms
        if fusion_strategy == 'mlp':
            self.fusion_network = MLPFusion(expert_output_dim * 2, dropout)
        elif fusion_strategy == 'attention':
            self.fusion_network = AttentionFusion(expert_output_dim, dropout)
        elif fusion_strategy == 'weighted_average':
            self.fusion_network = WeightedAverageFusion()
        else:
            raise ValueError(f"Unknown fusion strategy: {fusion_strategy}")
        
        # Final prediction layer (only used for some fusion strategies)
        if fusion_strategy in ['mlp', 'attention']:
            self.final_prediction = nn.Linear(128, 1)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(
        self,
        satellite: torch.Tensor,
        weather: torch.Tensor,
        return_expert_outputs: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through late fusion model.
        
        Args:
            satellite: Satellite imagery (batch_size, channels, height, width)
            weather: Weather time series (batch_size, sequence_length, features)
            return_expert_outputs: Whether to return individual expert outputs
        
        Returns:
            Final prediction or tuple of (prediction, expert_outputs)
        """
        # Get expert predictions
        satellite_output = self.satellite_expert(satellite)
        weather_output = self.weather_expert(weather)
        
        # Fuse expert outputs
        if self.fusion_strategy == 'weighted_average':
            final_prediction = self.fusion_network(
                satellite_output['prediction'],
                weather_output['prediction'],
                satellite_output['confidence'],
                weather_output['confidence']
            )
        else:
            # Feature-level fusion
            fusion_input = torch.cat([
                satellite_output['features'],
                weather_output['features']
            ], dim=1)
            
            if self.fusion_strategy == 'mlp':
                fusion_output = self.fusion_network(fusion_input)
                final_prediction = self.final_prediction(fusion_output)
            elif self.fusion_strategy == 'attention':
                fusion_output = self.fusion_network(
                    satellite_output['features'],
                    weather_output['features']
                )
                final_prediction = self.final_prediction(fusion_output)
        
        if return_expert_outputs:
            return final_prediction, {
                'satellite': satellite_output,
                'weather': weather_output
            }
        
        return final_prediction


# Supporting modules

class SatelliteCNN(nn.Module):
    """CNN backbone for satellite imagery."""
    
    def __init__(self, input_channels: int, hidden_dims: List[int], spatial_size: Tuple[int, int], dropout: float):
        super().__init__()
        
        layers = []
        in_channels = input_channels
        
        for i, hidden_dim in enumerate(hidden_dims):
            # Convolutional block
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True)
            ])
            
            # Downsampling
            if i < len(hidden_dims) - 1:
                layers.extend([
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Dropout2d(dropout)
                ])
            
            in_channels = hidden_dim
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv_layers(x)
        pooled = self.global_pool(features)
        return pooled.flatten(1)


class FeaturePyramidNetwork(nn.Module):
    """Feature Pyramid Network for multi-scale features."""
    
    def __init__(self, hidden_dims: List[int]):
        super().__init__()
        # Implementation would go here for multi-scale feature extraction
        pass


class AttentiveGlobalPooling(nn.Module):
    """Global pooling with attention mechanism."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(channels, channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention_weights = self.attention(x)
        attended = x * attention_weights
        return F.adaptive_avg_pool2d(attended, 1).flatten(1)


class TemporalAttention(nn.Module):
    """Attention mechanism for temporal sequences."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, sequence_length, hidden_dim)
        attention_weights = self.attention(x)  # (batch_size, sequence_length, 1)
        attention_weights = F.softmax(attention_weights, dim=1)
        attended = torch.sum(x * attention_weights, dim=1)  # (batch_size, hidden_dim)
        return attended


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, hidden_dim: int, max_length: int = 1000):
        super().__init__()
        
        pe = torch.zeros(max_length, hidden_dim)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * 
                           (-np.log(10000.0) / hidden_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


# Fusion mechanisms

class MLPFusion(nn.Module):
    """MLP-based fusion of expert features."""
    
    def __init__(self, input_dim: int, dropout: float = 0.2):
        super().__init__()
        self.fusion_mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fusion_mlp(x)


class AttentionFusion(nn.Module):
    """Attention-based fusion of expert features."""
    
    def __init__(self, feature_dim: int, dropout: float = 0.2):
        super().__init__()
        self.query = nn.Linear(feature_dim, 64)
        self.key = nn.Linear(feature_dim, 64)
        self.value = nn.Linear(feature_dim, 128)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, satellite_features: torch.Tensor, weather_features: torch.Tensor) -> torch.Tensor:
        # Stack features
        features = torch.stack([satellite_features, weather_features], dim=1)  # (batch, 2, feature_dim)
        
        # Compute attention
        queries = self.query(features)  # (batch, 2, 64)
        keys = self.key(features)      # (batch, 2, 64)
        values = self.value(features)  # (batch, 2, 128)
        
        # Attention scores
        scores = torch.bmm(queries, keys.transpose(1, 2)) / (64 ** 0.5)  # (batch, 2, 2)
        attention_weights = F.softmax(scores, dim=-1)
        
        # Attended features
        attended = torch.bmm(attention_weights, values)  # (batch, 2, 128)
        fused = attended.mean(dim=1)  # (batch, 128)
        
        return self.dropout(fused)


class WeightedAverageFusion(nn.Module):
    """Confidence-weighted average of expert predictions."""
    
    def forward(
        self,
        satellite_pred: torch.Tensor,
        weather_pred: torch.Tensor,
        satellite_conf: torch.Tensor,
        weather_conf: torch.Tensor
    ) -> torch.Tensor:
        # Normalize confidence scores
        total_conf = satellite_conf + weather_conf + 1e-8
        satellite_weight = satellite_conf / total_conf
        weather_weight = weather_conf / total_conf
        
        # Weighted average
        fused_pred = satellite_weight * satellite_pred + weather_weight * weather_pred
        return fused_pred


def create_late_fusion_model(config: Dict) -> LateFusionModel:
    """Create late fusion model from configuration."""
    return LateFusionModel(
        image_channels=config.get('image_channels', 7),
        image_size=config.get('image_size', (64, 64)),
        weather_features=config.get('weather_features', 6),
        sequence_length=config.get('sequence_length', 32),
        satellite_hidden_dims=config.get('satellite_hidden_dims', [64, 128, 256, 512]),
        weather_architecture=config.get('weather_architecture', 'transformer'),
        weather_hidden_dim=config.get('weather_hidden_dim', 128),
        weather_num_layers=config.get('weather_num_layers', 3),
        fusion_strategy=config.get('fusion_strategy', 'weighted_average'),
        expert_output_dim=config.get('expert_output_dim', 256),
        dropout=config.get('dropout', 0.2)
    )


if __name__ == "__main__":
    # Test model
    model = LateFusionModel()
    
    # Create dummy inputs
    satellite = torch.randn(4, 7, 64, 64)
    weather = torch.randn(4, 32, 6)
    
    # Forward pass
    output = model(satellite, weather)
    print(f"Model output shape: {output.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")