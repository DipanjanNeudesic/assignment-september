"""
Early Fusion Model Architecture

This module implements an early fusion approach for multimodal crop yield prediction.
The model concatenates satellite imagery features with weather time series at the
feature level and learns joint spatio-temporal representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class SpatialFeatureExtractor(nn.Module):
    """Extract spatial features from satellite imagery using CNN."""
    
    def __init__(
        self,
        input_channels: int = 7,  # RGB + 4 vegetation indices
        hidden_dims: list = [64, 128, 256, 512],
        spatial_size: Tuple[int, int] = (64, 64),
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.spatial_size = spatial_size
        
        # Convolutional layers
        layers = []
        in_channels = input_channels
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout2d(dropout)
            ])
            in_channels = hidden_dim
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Calculate flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, *spatial_size)
            conv_output = self.conv_layers(dummy_input)
            self.flattened_size = conv_output.numel()
        
        # Spatial attention mechanism
        self.spatial_attention = SpatialAttention(hidden_dims[-1])
        
        # Final projection
        self.feature_projection = nn.Sequential(
            nn.Linear(self.flattened_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Satellite imagery tensor of shape (batch_size, channels, height, width)
        
        Returns:
            Spatial features of shape (batch_size, 256)
        """
        # Extract convolutional features
        conv_features = self.conv_layers(x)
        
        # Apply spatial attention
        attended_features = self.spatial_attention(conv_features)
        
        # Flatten and project
        flattened = attended_features.view(x.size(0), -1)
        spatial_features = self.feature_projection(flattened)
        
        return spatial_features


class TemporalFeatureExtractor(nn.Module):
    """Extract temporal features from weather time series using LSTM/Transformer."""
    
    def __init__(
        self,
        input_features: int = 6,  # Weather variables
        sequence_length: int = 32,
        hidden_dim: int = 128,
        num_layers: int = 2,
        use_transformer: bool = True,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.input_features = input_features
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.use_transformer = use_transformer
        
        # Input projection
        self.input_projection = nn.Linear(input_features, hidden_dim)
        
        if use_transformer:
            # Transformer encoder
            encoder_layer = TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation='relu',
                batch_first=True
            )
            self.encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
            
            # Positional encoding
            self.positional_encoding = PositionalEncoding(hidden_dim, sequence_length)
            
        else:
            # LSTM encoder
            self.encoder = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True,
                bidirectional=True
            )
            hidden_dim = hidden_dim * 2  # Bidirectional
        
        # Temporal attention mechanism
        self.temporal_attention = TemporalAttention(hidden_dim)
        
        # Final projection
        self.feature_projection = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 256)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Weather time series tensor of shape (batch_size, sequence_length, features)
        
        Returns:
            Temporal features of shape (batch_size, 256)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project input features
        x = self.input_projection(x)
        
        if self.use_transformer:
            # Add positional encoding
            x = self.positional_encoding(x)
            
            # Transformer encoding
            encoded = self.encoder(x)
        else:
            # LSTM encoding
            encoded, _ = self.encoder(x)
        
        # Apply temporal attention
        attended_features = self.temporal_attention(encoded)
        
        # Project to final feature space
        temporal_features = self.feature_projection(attended_features)
        
        return temporal_features


class EarlyFusionModel(nn.Module):
    """
    Early Fusion model that combines satellite and weather data at the feature level.
    
    The model extracts features from both modalities and fuses them early in the
    processing pipeline to learn joint spatio-temporal representations.
    """
    
    def __init__(
        self,
        image_channels: int = 7,
        image_size: Tuple[int, int] = (64, 64),
        weather_features: int = 6,
        sequence_length: int = 32,
        fusion_strategy: str = 'concat',  # 'concat', 'add', 'multiply'
        hidden_dim: int = 512,
        num_fusion_layers: int = 3,
        dropout: float = 0.2,
        use_transformer: bool = True
    ):
        super().__init__()
        
        self.fusion_strategy = fusion_strategy
        
        # Spatial feature extractor for satellite imagery
        self.spatial_extractor = SpatialFeatureExtractor(
            input_channels=image_channels,
            spatial_size=image_size,
            dropout=dropout
        )
        
        # Temporal feature extractor for weather data
        self.temporal_extractor = TemporalFeatureExtractor(
            input_features=weather_features,
            sequence_length=sequence_length,
            use_transformer=use_transformer,
            dropout=dropout
        )
        
        # Determine fusion input dimension
        if fusion_strategy == 'concat':
            fusion_input_dim = 512  # 256 + 256
        else:
            fusion_input_dim = 256  # Same size for element-wise operations
        
        # Fusion network
        fusion_layers = []
        current_dim = fusion_input_dim
        
        for i in range(num_fusion_layers):
            next_dim = hidden_dim // (2 ** i) if i < num_fusion_layers - 1 else hidden_dim // 4
            fusion_layers.extend([
                nn.Linear(current_dim, next_dim),
                nn.BatchNorm1d(next_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            current_dim = next_dim
        
        self.fusion_network = nn.Sequential(*fusion_layers)
        
        # Final prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(current_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )
        
        # Feature fusion attention (optional enhancement)
        self.fusion_attention = CrossModalAttention(256, 256)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(
        self, 
        satellite: torch.Tensor, 
        weather: torch.Tensor,
        return_features: bool = False
    ) -> torch.Tensor:
        """
        Forward pass of the early fusion model.
        
        Args:
            satellite: Satellite imagery tensor (batch_size, channels, height, width)
            weather: Weather time series tensor (batch_size, sequence_length, features)
            return_features: Whether to return intermediate features
        
        Returns:
            Predicted crop yield (batch_size, 1)
        """
        # Extract modality-specific features
        spatial_features = self.spatial_extractor(satellite)
        temporal_features = self.temporal_extractor(weather)
        
        # Apply cross-modal attention (optional)
        spatial_features, temporal_features = self.fusion_attention(
            spatial_features, temporal_features
        )
        
        # Fuse features based on strategy
        if self.fusion_strategy == 'concat':
            fused_features = torch.cat([spatial_features, temporal_features], dim=1)
        elif self.fusion_strategy == 'add':
            fused_features = spatial_features + temporal_features
        elif self.fusion_strategy == 'multiply':
            fused_features = spatial_features * temporal_features
        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")
        
        # Pass through fusion network
        fusion_output = self.fusion_network(fused_features)
        
        # Generate prediction
        prediction = self.prediction_head(fusion_output)
        
        if return_features:
            return prediction, {
                'spatial_features': spatial_features,
                'temporal_features': temporal_features,
                'fused_features': fused_features,
                'fusion_output': fusion_output
            }
        
        return prediction


class SpatialAttention(nn.Module):
    """Spatial attention mechanism for satellite imagery."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels // 8, 1)
        self.conv2 = nn.Conv2d(channels // 8, 1, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention = self.conv1(x)
        attention = F.relu(attention, inplace=True)
        attention = self.conv2(attention)
        attention = self.sigmoid(attention)
        return x * attention


class TemporalAttention(nn.Module):
    """Temporal attention mechanism for weather time series."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, sequence_length, hidden_dim)
        attention_weights = self.attention(x)  # (batch_size, sequence_length, 1)
        attention_weights = self.softmax(attention_weights)
        
        # Weighted sum
        attended = torch.sum(x * attention_weights, dim=1)  # (batch_size, hidden_dim)
        return attended


class CrossModalAttention(nn.Module):
    """Cross-modal attention between spatial and temporal features."""
    
    def __init__(self, spatial_dim: int, temporal_dim: int):
        super().__init__()
        self.spatial_proj = nn.Linear(spatial_dim, 256)
        self.temporal_proj = nn.Linear(temporal_dim, 256)
        self.attention = nn.MultiheadAttention(256, num_heads=8, batch_first=True)
        
    def forward(self, spatial_features: torch.Tensor, temporal_features: torch.Tensor):
        # Project to common dimension
        spatial_proj = self.spatial_proj(spatial_features).unsqueeze(1)  # (batch, 1, 256)
        temporal_proj = self.temporal_proj(temporal_features).unsqueeze(1)  # (batch, 1, 256)
        
        # Cross-attention
        spatial_attended, _ = self.attention(spatial_proj, temporal_proj, temporal_proj)
        temporal_attended, _ = self.attention(temporal_proj, spatial_proj, spatial_proj)
        
        # Remove sequence dimension and return
        return spatial_attended.squeeze(1), temporal_attended.squeeze(1)


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


def create_early_fusion_model(config: Dict) -> EarlyFusionModel:
    """Create early fusion model from configuration."""
    return EarlyFusionModel(
        image_channels=config.get('image_channels', 7),
        image_size=config.get('image_size', (64, 64)),
        weather_features=config.get('weather_features', 6),
        sequence_length=config.get('sequence_length', 32),
        fusion_strategy=config.get('fusion_strategy', 'concat'),
        hidden_dim=config.get('hidden_dim', 512),
        num_fusion_layers=config.get('num_fusion_layers', 3),
        dropout=config.get('dropout', 0.2),
        use_transformer=config.get('use_transformer', True)
    )


if __name__ == "__main__":
    # Test model
    model = EarlyFusionModel()
    
    # Create dummy inputs
    satellite = torch.randn(4, 7, 64, 64)
    weather = torch.randn(4, 32, 6)
    
    # Forward pass
    output = model(satellite, weather)
    print(f"Model output shape: {output.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")