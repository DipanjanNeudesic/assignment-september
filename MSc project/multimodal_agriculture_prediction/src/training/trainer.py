"""
Training Pipeline for Multimodal Agriculture Prediction Models

This module provides comprehensive training functionality for Early Fusion,
Late Fusion, and GMU models, including loss functions, metrics, and optimization.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import logging
import wandb
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
import json
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for training."""
    model_type: str  # 'early_fusion', 'late_fusion', 'gmu'
    num_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    optimizer: str = 'adam'  # 'adam', 'adamw', 'sgd'
    scheduler: str = 'reduce_lr_on_plateau'  # 'reduce_lr_on_plateau', 'cosine'
    patience: int = 10
    min_lr: float = 1e-6
    gradient_clipping: float = 1.0
    mixed_precision: bool = True
    save_dir: str = 'checkpoints'
    experiment_name: str = 'multimodal_crop_prediction'
    log_interval: int = 10
    eval_interval: int = 5
    save_interval: int = 10
    early_stopping_patience: int = 20
    use_wandb: bool = True


class MultimodalLoss(nn.Module):
    """Custom loss function for multimodal crop yield prediction."""
    
    def __init__(
        self,
        regression_weight: float = 1.0,
        uncertainty_weight: float = 0.1,
        diversity_weight: float = 0.05,
        quality_weight: float = 0.05
    ):
        super().__init__()
        self.regression_weight = regression_weight
        self.uncertainty_weight = uncertainty_weight
        self.diversity_weight = diversity_weight
        self.quality_weight = quality_weight
        
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        uncertainty: Optional[torch.Tensor] = None,
        expert_outputs: Optional[Dict[str, torch.Tensor]] = None,
        quality_scores: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multimodal loss.
        
        Args:
            predictions: Model predictions (batch_size, 1)
            targets: Target yields (batch_size, 1)
            uncertainty: Uncertainty estimates (batch_size, 1)
            expert_outputs: Individual expert predictions
            quality_scores: Quality scores for each modality
        
        Returns:
            Dictionary of loss components
        """
        losses = {}
        
        # Primary regression loss
        regression_loss = self.mse_loss(predictions, targets)
        losses['regression'] = regression_loss
        
        # Uncertainty-aware loss
        if uncertainty is not None:
            # Penalize high uncertainty when predictions are accurate
            accuracy = torch.abs(predictions - targets)
            uncertainty_loss = torch.mean(uncertainty * accuracy)
            losses['uncertainty'] = uncertainty_loss
        else:
            losses['uncertainty'] = torch.tensor(0.0, device=predictions.device)
        
        # Diversity loss (for ensemble-like approaches)
        if expert_outputs is not None and len(expert_outputs) > 1:
            expert_preds = list(expert_outputs.values())
            diversity_loss = 0.0
            for i in range(len(expert_preds)):
                for j in range(i + 1, len(expert_preds)):
                    # Encourage diversity between experts
                    similarity = torch.cosine_similarity(
                        expert_preds[i].flatten(),
                        expert_preds[j].flatten(),
                        dim=0
                    )
                    diversity_loss += similarity
            diversity_loss /= (len(expert_preds) * (len(expert_preds) - 1) / 2)
            losses['diversity'] = diversity_loss
        else:
            losses['diversity'] = torch.tensor(0.0, device=predictions.device)
        
        # Quality consistency loss
        if quality_scores is not None:
            # High quality data should have lower uncertainty
            quality_loss = 0.0
            for modality, quality in quality_scores.items():
                if uncertainty is not None:
                    # Higher quality should correlate with lower uncertainty
                    quality_loss += torch.mean((1 - quality) * uncertainty)
            losses['quality'] = quality_loss / len(quality_scores) if quality_scores else torch.tensor(0.0)
        else:
            losses['quality'] = torch.tensor(0.0, device=predictions.device)
        
        # Total loss
        total_loss = (
            self.regression_weight * losses['regression'] +
            self.uncertainty_weight * losses['uncertainty'] +
            self.diversity_weight * losses['diversity'] +
            self.quality_weight * losses['quality']
        )
        losses['total'] = total_loss
        
        return losses


class ModelTrainer:
    """Comprehensive trainer for multimodal agriculture prediction models."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
        test_loader: Optional[DataLoader] = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Setup loss function
        self.loss_fn = MultimodalLoss()
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup scheduler
        self.scheduler = self._create_scheduler()
        
        # Setup mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if config.mixed_precision else None
        
        # Metrics tracking
        self.train_metrics = defaultdict(list)
        self.val_metrics = defaultdict(list)
        
        # Early stopping
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.epochs_without_improvement = 0
        
        # Create save directory
        Path(config.save_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize wandb if enabled
        if config.use_wandb:
            wandb.init(
                project=config.experiment_name,
                config=config.__dict__,
                name=f"{config.model_type}_{int(time.time())}"
            )
        
        logger.info(f"Trainer initialized for {config.model_type} model")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer from configuration."""
        if self.config.optimizer.lower() == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        if self.config.scheduler == 'reduce_lr_on_plateau':
            return ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=self.config.patience,
                min_lr=self.config.min_lr,
                verbose=True
            )
        elif self.config.scheduler == 'cosine':
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs,
                eta_min=self.config.min_lr
            )
        else:
            return None
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = defaultdict(list)
        epoch_metrics = defaultdict(list)
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move data to device
            satellite = batch['satellite'].to(self.device, non_blocking=True)
            weather = batch['weather'].to(self.device, non_blocking=True)
            targets = batch['yield'].to(self.device, non_blocking=True).unsqueeze(1)
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
                output = self._model_forward(satellite, weather, training=True)
                losses = self.loss_fn(
                    predictions=output['prediction'],
                    targets=targets,
                    uncertainty=output.get('uncertainty'),
                    expert_outputs=output.get('expert_outputs'),
                    quality_scores=output.get('quality_scores')
                )
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.scaler is not None:
                self.scaler.scale(losses['total']).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clipping)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                losses['total'].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clipping)
                self.optimizer.step()
            
            # Track losses
            for key, value in losses.items():
                epoch_losses[key].append(value.item())
            
            # Calculate metrics
            metrics = self._calculate_metrics(output['prediction'], targets)
            for key, value in metrics.items():
                epoch_metrics[key].append(value)
            
            # Log batch progress
            if batch_idx % self.config.log_interval == 0:
                logger.info(
                    f'Epoch {epoch}, Batch {batch_idx}/{len(self.train_loader)}, '
                    f'Loss: {losses["total"].item():.4f}, '
                    f'MAE: {metrics["mae"]:.4f}'
                )
        
        # Average epoch metrics
        avg_losses = {key: np.mean(values) for key, values in epoch_losses.items()}
        avg_metrics = {key: np.mean(values) for key, values in epoch_metrics.items()}
        
        return {**avg_losses, **avg_metrics}
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        epoch_losses = defaultdict(list)
        epoch_metrics = defaultdict(list)
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move data to device
                satellite = batch['satellite'].to(self.device, non_blocking=True)
                weather = batch['weather'].to(self.device, non_blocking=True)
                targets = batch['yield'].to(self.device, non_blocking=True).unsqueeze(1)
                
                # Forward pass
                output = self._model_forward(satellite, weather, training=False)
                losses = self.loss_fn(
                    predictions=output['prediction'],
                    targets=targets,
                    uncertainty=output.get('uncertainty'),
                    expert_outputs=output.get('expert_outputs'),
                    quality_scores=output.get('quality_scores')
                )
                
                # Track losses
                for key, value in losses.items():
                    epoch_losses[key].append(value.item())
                
                # Calculate metrics
                metrics = self._calculate_metrics(output['prediction'], targets)
                for key, value in metrics.items():
                    epoch_metrics[key].append(value)
        
        # Average epoch metrics
        avg_losses = {key: np.mean(values) for key, values in epoch_losses.items()}
        avg_metrics = {key: np.mean(values) for key, values in epoch_metrics.items()}
        
        return {**avg_losses, **avg_metrics}
    
    def _model_forward(self, satellite: torch.Tensor, weather: torch.Tensor, training: bool = True) -> Dict[str, torch.Tensor]:
        """Forward pass through model (handles different model types)."""
        if self.config.model_type == 'early_fusion':
            prediction = self.model(satellite, weather)
            return {'prediction': prediction}
        
        elif self.config.model_type == 'late_fusion':
            if training:
                prediction, expert_outputs = self.model(
                    satellite, weather, return_expert_outputs=True
                )
                return {'prediction': prediction, 'expert_outputs': expert_outputs}
            else:
                prediction = self.model(satellite, weather)
                return {'prediction': prediction}
        
        elif self.config.model_type == 'gmu':
            output = self.model(
                satellite, weather,
                return_attention=training,
                return_expert_outputs=training
            )
            return output
        
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")
    
    def _calculate_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        predictions = predictions.cpu().numpy().flatten()
        targets = targets.cpu().numpy().flatten()
        
        mae = np.mean(np.abs(predictions - targets))
        rmse = np.sqrt(np.mean((predictions - targets) ** 2))
        mape = np.mean(np.abs((predictions - targets) / (targets + 1e-8))) * 100
        
        # R² score
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        
        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'r2': r2
        }
    
    def train(self) -> Dict[str, List[float]]:
        """Complete training loop."""
        logger.info(f"Starting training for {self.config.num_epochs} epochs")
        
        for epoch in range(1, self.config.num_epochs + 1):
            epoch_start_time = time.time()
            
            # Training
            train_metrics = self.train_epoch(epoch)
            
            # Validation
            if epoch % self.config.eval_interval == 0:
                val_metrics = self.validate_epoch(epoch)
                
                # Update learning rate scheduler
                if self.scheduler is not None:
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        self.scheduler.step(val_metrics['total'])
                    else:
                        self.scheduler.step()
                
                # Check for improvement
                if val_metrics['total'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['total']
                    self.best_epoch = epoch
                    self.epochs_without_improvement = 0
                    self._save_checkpoint(epoch, is_best=True)
                else:
                    self.epochs_without_improvement += 1
                
                # Log metrics
                epoch_time = time.time() - epoch_start_time
                logger.info(
                    f'Epoch {epoch}/{self.config.num_epochs} - '
                    f'Train Loss: {train_metrics["total"]:.4f}, '
                    f'Val Loss: {val_metrics["total"]:.4f}, '
                    f'Val MAE: {val_metrics["mae"]:.4f}, '
                    f'Val R²: {val_metrics["r2"]:.4f}, '
                    f'Time: {epoch_time:.1f}s'
                )
                
                # Store metrics
                for key, value in train_metrics.items():
                    self.train_metrics[f'train_{key}'].append(value)
                for key, value in val_metrics.items():
                    self.val_metrics[f'val_{key}'].append(value)
                
                # Wandb logging
                if self.config.use_wandb:
                    wandb.log({
                        **{f'train/{k}': v for k, v in train_metrics.items()},
                        **{f'val/{k}': v for k, v in val_metrics.items()},
                        'epoch': epoch,
                        'learning_rate': self.optimizer.param_groups[0]['lr']
                    })
                
                # Early stopping check
                if self.epochs_without_improvement >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch} epochs")
                    break
            
            # Save checkpoint periodically
            if epoch % self.config.save_interval == 0:
                self._save_checkpoint(epoch, is_best=False)
        
        logger.info(f"Training completed. Best epoch: {self.best_epoch}")
        
        # Final evaluation on test set
        if self.test_loader is not None:
            self._evaluate_test_set()
        
        return {**self.train_metrics, **self.val_metrics}
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'config': self.config.__dict__,
            'train_metrics': dict(self.train_metrics),
            'val_metrics': dict(self.val_metrics)
        }
        
        # Save regular checkpoint
        checkpoint_path = Path(self.config.save_dir) / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = Path(self.config.save_dir) / 'best_model.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved with validation loss: {self.best_val_loss:.4f}")
    
    def _evaluate_test_set(self):
        """Evaluate on test set."""
        logger.info("Evaluating on test set...")
        
        # Load best model
        best_model_path = Path(self.config.save_dir) / 'best_model.pth'
        if best_model_path.exists():
            checkpoint = torch.load(best_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded best model from epoch {checkpoint['epoch']}")
        
        self.model.eval()
        test_metrics = defaultdict(list)
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in self.test_loader:
                satellite = batch['satellite'].to(self.device, non_blocking=True)
                weather = batch['weather'].to(self.device, non_blocking=True)
                target = batch['yield'].to(self.device, non_blocking=True).unsqueeze(1)
                
                output = self._model_forward(satellite, weather, training=False)
                
                predictions.append(output['prediction'].cpu().numpy())
                targets.append(target.cpu().numpy())
                
                # Calculate metrics
                metrics = self._calculate_metrics(output['prediction'], target)
                for key, value in metrics.items():
                    test_metrics[key].append(value)
        
        # Average test metrics
        avg_test_metrics = {key: np.mean(values) for key, values in test_metrics.items()}
        
        logger.info("Test Results:")
        for key, value in avg_test_metrics.items():
            logger.info(f"  {key.upper()}: {value:.4f}")
        
        # Log to wandb
        if self.config.use_wandb:
            wandb.log({f'test/{k}': v for k, v in avg_test_metrics.items()})
        
        return avg_test_metrics


def create_trainer(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig,
    test_loader: Optional[DataLoader] = None
) -> ModelTrainer:
    """Create trainer from configuration."""
    return ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config
    )


if __name__ == "__main__":
    # Example usage
    from src.models.early_fusion import EarlyFusionModel
    from src.data.dataset import create_data_loaders
    
    # Create model
    model = EarlyFusionModel()
    
    # Create data loaders (would need actual data)
    # train_loader, val_loader, test_loader = create_data_loaders(...)
    
    # Create training configuration
    config = TrainingConfig(
        model_type='early_fusion',
        num_epochs=50,
        batch_size=16,
        learning_rate=1e-4,
        use_wandb=False  # Disable for testing
    )
    
    print("Training pipeline created successfully!")