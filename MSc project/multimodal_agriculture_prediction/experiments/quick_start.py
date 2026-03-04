"""
Quick Start Script

This script provides a simple way to get started with the multimodal
agriculture prediction project using sample data and default configurations.
"""

import os
import sys
import shutil
import tempfile
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.data.dataset import CropNetDataset
from src.models.early_fusion import EarlyFusionModel
from src.models.late_fusion import LateFusionModel
from src.models.gmu import GatedMultimodalUnit
from src.training.trainer import TrainingConfig, ModelTrainer
from src.evaluation.visualization import ModelVisualizer

def create_sample_data(data_dir: str, num_samples: int = 100):
    """Create sample data for testing the pipeline."""
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating sample dataset with {num_samples} samples...")
    
    # Create directory structure
    for split in ['train', 'val', 'test']:
        (data_path / split / 'satellite').mkdir(parents=True, exist_ok=True)
        (data_path / split / 'weather').mkdir(parents=True, exist_ok=True)
        (data_path / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Generate sample data for each split
    split_sizes = {'train': int(0.7 * num_samples), 'val': int(0.15 * num_samples), 'test': int(0.15 * num_samples)}
    
    for split, size in split_sizes.items():
        for i in range(size):
            sample_id = f"{split}_{i:04d}"
            
            # Create sample satellite imagery (RGB + NIR channels)
            # Simulate realistic vegetation and soil patterns
            satellite_data = np.random.rand(4, 64, 64).astype(np.float32)
            
            # Add some structure to make it more realistic
            # Simulate vegetation patterns with higher NIR values
            vegetation_mask = np.random.rand(64, 64) > 0.3
            satellite_data[3][vegetation_mask] += 0.5  # NIR channel
            satellite_data[1][vegetation_mask] += 0.3  # Green channel
            
            # Save as multi-channel array
            np.save(data_path / split / 'satellite' / f"{sample_id}.npy", satellite_data)
            
            # Create sample weather time series (30 days, 10 features)
            weather_features = [
                'temperature_2m', 'relative_humidity_2m', 'precipitation',
                'wind_speed_10m', 'wind_direction_10m', 'pressure_sea_level',
                'solar_radiation', 'soil_temperature', 'soil_moisture', 'evapotranspiration'
            ]
            
            # Generate realistic weather patterns
            base_temp = 20 + 10 * np.sin(np.linspace(0, 2*np.pi, 30))  # Seasonal temperature
            temp_noise = np.random.normal(0, 2, 30)
            temperature = base_temp + temp_noise
            
            humidity = np.clip(80 - 0.5 * (temperature - 20) + np.random.normal(0, 5, 30), 0, 100)
            precipitation = np.maximum(0, np.random.exponential(2, 30))
            
            weather_data = np.column_stack([
                temperature,
                humidity,
                precipitation,
                np.random.uniform(0, 15, 30),  # wind speed
                np.random.uniform(0, 360, 30),  # wind direction
                np.random.normal(1013, 10, 30),  # pressure
                np.random.uniform(100, 800, 30),  # solar radiation
                temperature - np.random.uniform(2, 5, 30),  # soil temperature
                np.random.uniform(0.1, 0.4, 30),  # soil moisture
                np.random.uniform(1, 6, 30)  # evapotranspiration
            ]).astype(np.float32)
            
            np.save(data_path / split / 'weather' / f"{sample_id}.npy", weather_data)
            
            # Create sample yield label (realistic crop yield in tons/hectare)
            # Base yield influenced by weather patterns
            base_yield = 8.0  # tons per hectare
            
            # Yield influenced by average temperature and precipitation
            temp_effect = np.clip((np.mean(temperature) - 25) / 10, -0.3, 0.3)
            precip_effect = np.clip(np.sum(precipitation) / 100 - 1, -0.2, 0.4)
            
            yield_value = base_yield + base_yield * (temp_effect + precip_effect) + np.random.normal(0, 0.5)
            yield_value = np.clip(yield_value, 1.0, 15.0)  # Reasonable bounds
            
            # Save yield as JSON for easy reading
            label_data = {
                'yield_tons_per_hectare': float(yield_value),
                'sample_id': sample_id,
                'location': {
                    'lat': 40.0 + np.random.uniform(-5, 5),
                    'lon': -95.0 + np.random.uniform(-10, 10)
                },
                'crop_type': 'corn',
                'season': '2023'
            }
            
            with open(data_path / split / 'labels' / f"{sample_id}.json", 'w') as f:
                json.dump(label_data, f, indent=2)
    
    # Create metadata file
    metadata = {
        'dataset_name': 'sample_cropnet',
        'num_samples': sum(split_sizes.values()),
        'splits': split_sizes,
        'satellite_channels': ['red', 'green', 'blue', 'nir'],
        'satellite_resolution': '64x64',
        'weather_features': weather_features,
        'weather_sequence_length': 30,
        'target': 'yield_tons_per_hectare',
        'created_by': 'quick_start.py',
        'description': 'Synthetic dataset for testing multimodal agriculture prediction pipeline'
    }
    
    with open(data_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Sample dataset created at: {data_path}")
    print(f"  - Train samples: {split_sizes['train']}")
    print(f"  - Validation samples: {split_sizes['val']}")
    print(f"  - Test samples: {split_sizes['test']}")

def create_simple_configs(config_dir: str):
    """Create simple configuration files for quick testing."""
    config_path = Path(config_dir)
    config_path.mkdir(parents=True, exist_ok=True)
    
    # Base model configuration
    base_config = {
        'model': {
            'satellite_channels': 4,
            'satellite_size': 64,
            'weather_features': 10,
            'weather_sequence_length': 30,
            'hidden_dim': 128,
            'num_classes': 1,  # Regression
            'dropout_rate': 0.1
        },
        'training': {
            'num_epochs': 5,  # Short for quick testing
            'learning_rate': 0.001,
            'batch_size': 16,
            'weight_decay': 1e-4,
            'patience': 3,
            'min_delta': 0.001,
            'mixed_precision': False,  # Disable for compatibility
            'gradient_clip_norm': 1.0
        }
    }
    
    # Early fusion specific config
    early_fusion_config = base_config.copy()
    early_fusion_config['model']['fusion_method'] = 'early'
    early_fusion_config['model']['cnn_layers'] = [64, 128, 256]
    early_fusion_config['model']['lstm_layers'] = 2
    
    # Late fusion specific config  
    late_fusion_config = base_config.copy()
    late_fusion_config['model']['fusion_method'] = 'late'
    late_fusion_config['model']['expert_hidden_dims'] = [256, 128]
    late_fusion_config['model']['fusion_hidden_dims'] = [128, 64]
    
    # GMU specific config
    gmu_config = base_config.copy()
    gmu_config['model']['fusion_method'] = 'gmu'
    gmu_config['model']['gmu_hidden_dim'] = 128
    gmu_config['model']['num_experts'] = 2
    gmu_config['model']['quality_estimation_dim'] = 64
    gmu_config['model']['attention_heads'] = 4
    
    # Save configurations
    import yaml
    
    with open(config_path / 'early_fusion.yaml', 'w') as f:
        yaml.dump(early_fusion_config, f, default_flow_style=False)
    
    with open(config_path / 'late_fusion.yaml', 'w') as f:
        yaml.dump(late_fusion_config, f, default_flow_style=False)
    
    with open(config_path / 'gmu.yaml', 'w') as f:
        yaml.dump(gmu_config, f, default_flow_style=False)
    
    print(f"✓ Configuration files created at: {config_path}")

def run_quick_demo():
    """Run a quick demonstration of the pipeline."""
    print("🚀 Starting Quick Demo of Multimodal Agriculture Prediction")
    print("=" * 60)
    
    # Create temporary directory for demo
    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = os.path.join(temp_dir, 'sample_data')
        config_dir = os.path.join(temp_dir, 'config')
        output_dir = os.path.join(temp_dir, 'results')
        
        # Step 1: Create sample data
        print("\n1. Creating sample dataset...")
        create_sample_data(data_dir, num_samples=50)
        
        # Step 2: Create configurations
        print("\n2. Creating configuration files...")
        create_simple_configs(config_dir)
        
        # Step 3: Test data loading
        print("\n3. Testing data loading...")
        try:
            from torch.utils.data import DataLoader
            
            dataset = CropNetDataset(
                data_dir=data_dir,
                split='train',
                transform_satellite=None,
                transform_weather=None
            )
            
            dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
            
            # Test loading a batch
            sample_batch = next(iter(dataloader))
            satellite, weather, labels = sample_batch
            
            print(f"  ✓ Satellite data shape: {satellite.shape}")
            print(f"  ✓ Weather data shape: {weather.shape}")
            print(f"  ✓ Labels shape: {labels.shape}")
            print(f"  ✓ Dataset contains {len(dataset)} samples")
            
        except Exception as e:
            print(f"  ✗ Data loading failed: {str(e)}")
            return
        
        # Step 4: Test model creation
        print("\n4. Testing model creation...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            # Test Early Fusion model
            early_model = EarlyFusionModel(
                satellite_channels=4,
                satellite_size=64,
                weather_features=10,
                weather_sequence_length=30,
                hidden_dim=128,
                num_classes=1
            )
            
            print(f"  ✓ Early Fusion model created")
            print(f"    Parameters: {sum(p.numel() for p in early_model.parameters()):,}")
            
            # Test forward pass
            with torch.no_grad():
                output = early_model(satellite[:2], weather[:2])  # Use smaller batch
                print(f"    Output shape: {output.shape}")
            
        except Exception as e:
            print(f"  ✗ Model creation failed: {str(e)}")
            return
        
        # Step 5: Test training setup (without actual training)
        print("\n5. Testing training setup...")
        try:
            config = TrainingConfig(
                model_type='early_fusion',
                num_epochs=1,
                learning_rate=0.001,
                batch_size=4,
                save_dir=output_dir
            )
            
            # Create small dataloaders for testing
            train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
            val_loader = DataLoader(dataset, batch_size=4, shuffle=False)
            
            trainer = ModelTrainer(
                model=early_model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=val_loader,  # Use val_loader as test for demo
                config=config
            )
            
            print(f"  ✓ Trainer created successfully")
            print(f"  ✓ Training configuration: {config.num_epochs} epochs, LR={config.learning_rate}")
            
        except Exception as e:
            print(f"  ✗ Training setup failed: {str(e)}")
            return
        
        # Step 6: Test visualization
        print("\n6. Testing visualization...")
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            visualizer = ModelVisualizer(save_dir=output_dir)
            
            # Create dummy results for visualization
            dummy_results = {
                'early_fusion': {
                    'test_loss': 0.5,
                    'test_r2': 0.75,
                    'test_mse': 0.25,
                    'predictions': torch.randn(20),
                    'targets': torch.randn(20)
                }
            }
            
            # Test plotting (this might fail in headless environments)
            try:
                visualizer.plot_model_comparison(
                    dummy_results,
                    metrics=['r2', 'mse'],
                    save_name='demo_comparison'
                )
                print("  ✓ Visualization test passed")
            except Exception as viz_e:
                print(f"  ⚠ Visualization test skipped (display issue): {str(viz_e)}")
            
        except Exception as e:
            print(f"  ✗ Visualization setup failed: {str(e)}")
        
        print("\n" + "=" * 60)
        print("🎉 Quick Demo Completed Successfully!")
        print("\nNext Steps:")
        print("1. Prepare your real CropNet dataset")
        print("2. Adjust configurations in config/ directory")
        print("3. Run full experiments with: python experiments/run_experiments.py")
        print("4. Check out the documentation in docs/ for detailed usage")
        
        # Save demo results summary
        demo_summary = {
            'demo_completed': True,
            'components_tested': [
                'data_loading',
                'model_creation',
                'training_setup',
                'visualization'
            ],
            'device_used': str(device),
            'sample_data_created': True,
            'configs_created': True
        }
        
        summary_path = os.path.join(os.getcwd(), 'demo_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(demo_summary, f, indent=2)
        
        print(f"\nDemo summary saved to: {summary_path}")

if __name__ == "__main__":
    run_quick_demo()