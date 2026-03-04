"""
Model Inference Pipeline

This script provides a clean interface for running inference with trained
models on new data, including uncertainty estimation and attention visualization.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import json
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.data.dataset import CropNetDataset, create_inference_dataloader
from src.models.early_fusion import EarlyFusionModel
from src.models.late_fusion import LateFusionModel  
from src.models.gmu import GatedMultimodalUnit
from src.evaluation.visualization import ModelVisualizer, AttentionVisualizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelInference:
    """
    Handles model inference with uncertainty estimation and visualization.
    """
    
    def __init__(
        self,
        model_path: str,
        model_type: str,
        device: Optional[torch.device] = None,
        uncertainty_samples: int = 50
    ):
        """
        Initialize inference pipeline.
        
        Args:
            model_path: Path to trained model checkpoint
            model_type: Type of model ('early_fusion', 'late_fusion', 'gmu')
            device: Compute device (auto-detected if None)
            uncertainty_samples: Number of samples for MC dropout uncertainty
        """
        self.model_path = model_path
        self.model_type = model_type
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.uncertainty_samples = uncertainty_samples
        
        self.model = None
        self.model_config = None
        
        logger.info(f"Initializing inference for {model_type} on {self.device}")
        
    def load_model(self) -> None:
        """Load trained model from checkpoint."""
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Extract model configuration
            self.model_config = checkpoint.get('config', {})
            
            # Create model based on type
            if self.model_type == 'early_fusion':
                self.model = EarlyFusionModel(
                    satellite_channels=self.model_config.get('satellite_channels', 4),
                    satellite_size=self.model_config.get('satellite_size', 64),
                    weather_features=self.model_config.get('weather_features', 10),
                    weather_sequence_length=self.model_config.get('weather_sequence_length', 30),
                    hidden_dim=self.model_config.get('hidden_dim', 128),
                    num_classes=self.model_config.get('num_classes', 1),
                    dropout_rate=self.model_config.get('dropout_rate', 0.1)
                )
                
            elif self.model_type == 'late_fusion':
                self.model = LateFusionModel(
                    satellite_channels=self.model_config.get('satellite_channels', 4),
                    satellite_size=self.model_config.get('satellite_size', 64),
                    weather_features=self.model_config.get('weather_features', 10),
                    weather_sequence_length=self.model_config.get('weather_sequence_length', 30),
                    hidden_dim=self.model_config.get('hidden_dim', 128),
                    num_classes=self.model_config.get('num_classes', 1),
                    dropout_rate=self.model_config.get('dropout_rate', 0.1)
                )
                
            elif self.model_type == 'gmu':
                # For GMU, we need expert networks (simplified loading)
                satellite_expert = EarlyFusionModel(
                    satellite_channels=self.model_config.get('satellite_channels', 4),
                    satellite_size=self.model_config.get('satellite_size', 64),
                    weather_features=0,  # Satellite expert only
                    weather_sequence_length=0,
                    hidden_dim=self.model_config.get('hidden_dim', 128),
                    num_classes=self.model_config.get('hidden_dim', 128),  # Output features
                    dropout_rate=self.model_config.get('dropout_rate', 0.1)
                )
                
                weather_expert = EarlyFusionModel(
                    satellite_channels=0,  # Weather expert only  
                    satellite_size=0,
                    weather_features=self.model_config.get('weather_features', 10),
                    weather_sequence_length=self.model_config.get('weather_sequence_length', 30),
                    hidden_dim=self.model_config.get('hidden_dim', 128),
                    num_classes=self.model_config.get('hidden_dim', 128),  # Output features
                    dropout_rate=self.model_config.get('dropout_rate', 0.1)
                )
                
                self.model = GatedMultimodalUnit(
                    satellite_expert=satellite_expert,
                    weather_expert=weather_expert,
                    expert_output_dim=self.model_config.get('hidden_dim', 128),
                    hidden_dim=self.model_config.get('gmu_hidden_dim', 128),
                    output_dim=self.model_config.get('num_classes', 1),
                    num_experts=2,
                    dropout_rate=self.model_config.get('dropout_rate', 0.1)
                )
                
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
            
            # Load state dict
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"✓ Model loaded successfully from {self.model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def predict_single(
        self, 
        satellite_data: torch.Tensor, 
        weather_data: torch.Tensor,
        return_uncertainty: bool = True,
        return_attention: bool = False
    ) -> Dict[str, Any]:
        """
        Make prediction for a single sample.
        
        Args:
            satellite_data: Satellite imagery tensor [C, H, W]
            weather_data: Weather time series tensor [T, F]
            return_uncertainty: Whether to estimate uncertainty
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary containing predictions and optional uncertainty/attention
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Add batch dimension
        satellite_batch = satellite_data.unsqueeze(0).to(self.device)
        weather_batch = weather_data.unsqueeze(0).to(self.device)
        
        result = {}
        
        # Standard prediction
        with torch.no_grad():
            if self.model_type == 'gmu':
                output, attention_weights = self.model(
                    satellite_batch, 
                    weather_batch, 
                    return_attention=True
                )
                if return_attention:
                    result['attention_weights'] = attention_weights
            else:
                output = self.model(satellite_batch, weather_batch)
        
        result['prediction'] = output.squeeze().cpu().item()
        
        # Uncertainty estimation using MC Dropout
        if return_uncertainty:
            result['uncertainty'] = self._estimate_uncertainty(
                satellite_batch, weather_batch
            )
        
        return result
    
    def predict_batch(
        self, 
        dataloader: DataLoader,
        return_uncertainty: bool = True,
        save_results: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Make predictions for a batch of samples.
        
        Args:
            dataloader: DataLoader containing samples
            return_uncertainty: Whether to estimate uncertainty
            save_results: Path to save results (optional)
            
        Returns:
            Dictionary containing batch predictions and statistics
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        all_predictions = []
        all_uncertainties = [] if return_uncertainty else None
        all_targets = []
        all_attention_weights = []
        
        logger.info(f"Running inference on {len(dataloader)} batches...")
        
        with torch.no_grad():
            for batch_idx, (satellite, weather, targets) in enumerate(dataloader):
                satellite = satellite.to(self.device)
                weather = weather.to(self.device)
                
                # Forward pass
                if self.model_type == 'gmu':
                    outputs, attention = self.model(
                        satellite, weather, return_attention=True
                    )
                    all_attention_weights.extend(attention.cpu().numpy())
                else:
                    outputs = self.model(satellite, weather)
                
                predictions = outputs.squeeze().cpu().numpy()
                all_predictions.extend(predictions)
                all_targets.extend(targets.numpy())
                
                # Uncertainty estimation for batch
                if return_uncertainty:
                    batch_uncertainties = []
                    for i in range(len(satellite)):
                        uncertainty = self._estimate_uncertainty(
                            satellite[i:i+1], weather[i:i+1]
                        )
                        batch_uncertainties.append(uncertainty)
                    all_uncertainties.extend(batch_uncertainties)
                
                if (batch_idx + 1) % 10 == 0:
                    logger.info(f"  Processed {batch_idx + 1}/{len(dataloader)} batches")
        
        # Compile results
        results = {
            'predictions': np.array(all_predictions),
            'targets': np.array(all_targets),
            'model_type': self.model_type,
            'num_samples': len(all_predictions),
            'timestamp': datetime.now().isoformat()
        }
        
        if return_uncertainty:
            results['uncertainties'] = np.array(all_uncertainties)
            results['mean_uncertainty'] = np.mean(all_uncertainties)
            results['uncertainty_std'] = np.std(all_uncertainties)
        
        if all_attention_weights:
            results['attention_weights'] = np.array(all_attention_weights)
        
        # Calculate basic metrics
        mse = np.mean((results['predictions'] - results['targets']) ** 2)
        mae = np.mean(np.abs(results['predictions'] - results['targets']))
        
        results['metrics'] = {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(np.sqrt(mse))
        }
        
        # Save results if requested
        if save_results:
            self._save_results(results, save_results)
        
        logger.info(f"✓ Inference completed for {len(all_predictions)} samples")
        logger.info(f"  MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {np.sqrt(mse):.4f}")
        
        return results
    
    def _estimate_uncertainty(
        self, 
        satellite: torch.Tensor, 
        weather: torch.Tensor
    ) -> float:
        """Estimate uncertainty using MC Dropout."""
        if not any(isinstance(m, torch.nn.Dropout) for m in self.model.modules()):
            return 0.0  # No dropout layers
        
        # Enable dropout for uncertainty estimation
        self.model.train()
        
        predictions = []
        for _ in range(self.uncertainty_samples):
            with torch.no_grad():
                if self.model_type == 'gmu':
                    output, _ = self.model(satellite, weather, return_attention=False)
                else:
                    output = self.model(satellite, weather)
                predictions.append(output.squeeze().cpu().item())
        
        # Return to eval mode
        self.model.eval()
        
        # Calculate uncertainty as standard deviation
        return float(np.std(predictions))
    
    def _save_results(self, results: Dict[str, Any], save_path: str) -> None:
        """Save inference results to file."""
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                json_results[key] = value.tolist()
            else:
                json_results[key] = value
        
        with open(save_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Results saved to {save_path}")

def main():
    """Command line interface for model inference."""
    parser = argparse.ArgumentParser(description="Run model inference")
    
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--model-type', type=str, required=True,
                       choices=['early_fusion', 'late_fusion', 'gmu'],
                       help='Type of model to use')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Path to data directory')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help='Data split to use for inference')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for inference')
    parser.add_argument('--output-dir', type=str, default='inference_results',
                       help='Output directory for results')
    parser.add_argument('--uncertainty', action='store_true',
                       help='Estimate prediction uncertainty')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualization plots')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loader workers')
    
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Initialize inference
    inference = ModelInference(
        model_path=args.model_path,
        model_type=args.model_type,
        device=device
    )
    inference.load_model()
    
    # Create data loader
    try:
        dataloader = create_inference_dataloader(
            data_dir=args.data_dir,
            split=args.split,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        logger.info(f"Loaded {args.split} dataset with {len(dataloader.dataset)} samples")
    except Exception as e:
        logger.error(f"Failed to create data loader: {str(e)}")
        return
    
    # Run inference
    results_file = output_dir / f"inference_results_{args.model_type}_{args.split}.json"
    
    results = inference.predict_batch(
        dataloader=dataloader,
        return_uncertainty=args.uncertainty,
        save_results=str(results_file)
    )
    
    # Create visualizations if requested
    if args.visualize:
        logger.info("Creating visualization plots...")
        
        try:
            visualizer = ModelVisualizer(save_dir=str(output_dir))
            
            # Plot predictions vs targets
            visualizer.plot_predictions_vs_targets(
                predictions=results['predictions'],
                targets=results['targets'],
                model_name=args.model_type,
                save_name=f"predictions_{args.model_type}_{args.split}"
            )
            
            # Plot uncertainty if available
            if args.uncertainty and 'uncertainties' in results:
                visualizer.plot_uncertainty_analysis(
                    predictions=results['predictions'],
                    targets=results['targets'],
                    uncertainties=results['uncertainties'],
                    save_name=f"uncertainty_{args.model_type}_{args.split}"
                )
            
            # Plot attention weights if available (GMU model)
            if 'attention_weights' in results:
                attention_viz = AttentionVisualizer(save_dir=str(output_dir))
                attention_viz.plot_attention_heatmap(
                    attention_weights=results['attention_weights'][:10],  # First 10 samples
                    save_name=f"attention_{args.model_type}_{args.split}"
                )
            
            logger.info("✓ Visualizations created successfully")
            
        except Exception as e:
            logger.error(f"Visualization failed: {str(e)}")
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("INFERENCE SUMMARY")
    logger.info("="*50)
    logger.info(f"Model: {args.model_type}")
    logger.info(f"Dataset: {args.data_dir} ({args.split} split)")
    logger.info(f"Samples processed: {results['num_samples']}")
    logger.info(f"MSE: {results['metrics']['mse']:.4f}")
    logger.info(f"MAE: {results['metrics']['mae']:.4f}")
    logger.info(f"RMSE: {results['metrics']['rmse']:.4f}")
    
    if args.uncertainty:
        logger.info(f"Mean uncertainty: {results['mean_uncertainty']:.4f}")
        logger.info(f"Uncertainty std: {results['uncertainty_std']:.4f}")
    
    logger.info(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main()