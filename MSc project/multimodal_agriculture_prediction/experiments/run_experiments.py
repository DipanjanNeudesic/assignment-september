"""
Main Experiment Runner

This script orchestrates the complete experimental pipeline for the
multimodal agriculture prediction project, including training all models,
running stress tests, and generating comprehensive analysis.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
import torch
import yaml
import json
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.data.dataset import create_data_loaders
from src.models.early_fusion import create_early_fusion_model
from src.models.late_fusion import create_late_fusion_model
from src.models.gmu import create_gmu_model
from src.training.trainer import TrainingConfig, create_trainer
from src.evaluation.metrics import evaluate_and_compare_models
from src.evaluation.stress_testing import StressTester, DataQualityIssue, StressTestConfig
from src.evaluation.visualization import InteractiveDashboard

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiment.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """Load experiment configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_model(model_type: str, config: dict):
    """Create model based on type and configuration."""
    if model_type == 'early_fusion':
        return create_early_fusion_model(config)
    elif model_type == 'late_fusion':
        return create_late_fusion_model(config)
    elif model_type == 'gmu':
        # For GMU, we need expert networks first
        # This is a simplified version - in practice, you'd load pre-trained experts
        satellite_expert = create_late_fusion_model(config).satellite_expert
        weather_expert = create_late_fusion_model(config).weather_expert
        return create_gmu_model(satellite_expert, weather_expert, config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def train_single_model(
    model_type: str, 
    config: dict, 
    train_loader, 
    val_loader, 
    test_loader,
    device: torch.device,
    save_dir: str
) -> dict:
    """Train a single model and return results."""
    logger.info(f"Training {model_type} model...")
    
    # Create model
    model = create_model(model_type, config['model'])
    
    # Create training config
    training_config = TrainingConfig(
        model_type=model_type,
        **config['training']
    )
    training_config.save_dir = os.path.join(save_dir, model_type)
    
    # Create trainer
    trainer = create_trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=training_config
    )
    
    # Train model
    training_metrics = trainer.train()
    
    # Load best model for evaluation
    best_model_path = Path(training_config.save_dir) / 'best_model.pth'
    if best_model_path.exists():
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    return {
        'model': model,
        'training_metrics': training_metrics,
        'config': config
    }

def run_experiments(args):
    """Run complete experimental pipeline."""
    logger.info("Starting multimodal agriculture prediction experiments...")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configurations
    config_dir = Path(args.config_dir)
    model_configs = {}
    
    for model_type in ['early_fusion', 'late_fusion', 'gmu']:
        config_path = config_dir / f"{model_type}.yaml"
        if config_path.exists():
            model_configs[model_type] = load_config(config_path)
        else:
            logger.warning(f"Config file not found: {config_path}")
    
    if not model_configs:
        logger.error("No model configurations found!")
        return
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Train all models
    trained_models = {}
    training_results = {}
    
    for model_type, config in model_configs.items():
        try:
            result = train_single_model(
                model_type=model_type,
                config=config,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                device=device,
                save_dir=str(output_dir / "models")
            )
            trained_models[model_type] = result['model']
            training_results[model_type] = result['training_metrics']
            
            logger.info(f"✓ {model_type} training completed successfully")
            
        except Exception as e:
            logger.error(f"✗ {model_type} training failed: {str(e)}")
            continue
    
    if not trained_models:
        logger.error("No models trained successfully!")
        return
    
    # Comprehensive evaluation
    logger.info("Running comprehensive evaluation...")
    evaluation_results = evaluate_and_compare_models(
        models=trained_models,
        data_loader=test_loader,
        device=device,
        save_dir=str(output_dir / "evaluation")
    )
    
    # Stress testing
    if args.run_stress_tests:
        logger.info("Running stress tests...")
        stress_tester = StressTester(save_dir=str(output_dir / "stress_tests"))
        
        stress_test_results = {}
        model_types_map = {name: name.replace('_', '_') for name in trained_models.keys()}
        
        # Test different data quality issues
        issues_to_test = [
            DataQualityIssue.CLOUDY_SATELLITE,
            DataQualityIssue.MISSING_WEATHER,
            DataQualityIssue.SENSOR_NOISE,
            DataQualityIssue.MIXED_QUALITY
        ]
        
        for model_name, model in trained_models.items():
            model_stress_results = {}
            
            for issue in issues_to_test:
                try:
                    config = StressTestConfig(
                        issue_type=issue,
                        severity_levels=[0.0, 0.2, 0.4, 0.6, 0.8],
                        num_trials=3,
                        save_results=True,
                        plot_results=True
                    )
                    
                    result = stress_tester.run_stress_test(
                        model=model,
                        data_loader=test_loader,
                        device=device,
                        config=config,
                        model_type=model_types_map[model_name]
                    )
                    
                    model_stress_results[issue.value] = result
                    logger.info(f"✓ {model_name} - {issue.value} stress test completed")
                    
                except Exception as e:
                    logger.error(f"✗ {model_name} - {issue.value} stress test failed: {str(e)}")
                    continue
            
            stress_test_results[model_name] = model_stress_results
        
        # Compare robustness across models
        if len(stress_test_results) > 1:
            logger.info("Generating robustness comparison...")
            comparison_result = stress_tester.compare_model_robustness(
                models=trained_models,
                data_loader=test_loader,
                device=device,
                issue_types=issues_to_test,
                model_types=model_types_map
            )
    
    # Create interactive dashboard
    if args.create_dashboard:
        logger.info("Creating interactive dashboard...")
        dashboard = InteractiveDashboard(save_dir=str(output_dir / "dashboard"))
        
        try:
            dashboard.create_model_comparison_dashboard(
                model_results=evaluation_results['individual_results'],
                stress_test_results=stress_test_results if args.run_stress_tests else {},
                save_name="complete_analysis"
            )
            logger.info("✓ Interactive dashboard created successfully")
        except Exception as e:
            logger.error(f"✗ Dashboard creation failed: {str(e)}")
    
    # Save experiment summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'models_trained': list(trained_models.keys()),
        'evaluation_completed': True,
        'stress_tests_completed': args.run_stress_tests,
        'dashboard_created': args.create_dashboard,
        'output_directory': str(output_dir),
        'device_used': str(device)
    }
    
    if evaluation_results:
        summary['best_model'] = evaluation_results['comparison']['rankings']['overall'][0]
    
    with open(output_dir / 'experiment_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("🎉 Experiments completed successfully!")
    logger.info(f"Results saved to: {output_dir}")
    
    if 'best_model' in summary:
        logger.info(f"Best performing model: {summary['best_model']}")

def main():
    parser = argparse.ArgumentParser(description="Run multimodal agriculture prediction experiments")
    
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Path to CropNet dataset directory')
    parser.add_argument('--config-dir', type=str, default='config',
                       help='Directory containing model configuration files')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory for results')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training and evaluation')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loader workers')
    parser.add_argument('--run-stress-tests', action='store_true',
                       help='Run stress testing for robustness evaluation')
    parser.add_argument('--create-dashboard', action='store_true',
                       help='Create interactive dashboard')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode with verbose logging')
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        run_experiments(args)
    except KeyboardInterrupt:
        logger.info("Experiment interrupted by user")
    except Exception as e:
        logger.error(f"Experiment failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    main()