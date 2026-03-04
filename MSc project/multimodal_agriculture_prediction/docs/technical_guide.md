# Technical Implementation Guide

## Getting Started

### Prerequisites
```bash
# Python 3.8+ required
python --version

# CUDA-compatible GPU recommended
nvidia-smi
```

### Installation
```bash
# Clone repository
git clone <repository-url>
cd multimodal_agriculture_prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Quick Start
```bash
# 1. Download data
python src/data/download_cropnet.py --synthetic

# 2. Train models
python experiments/train_all_models.py

# 3. Run evaluation
python experiments/evaluate_models.py

# 4. Generate visualizations
python experiments/create_analysis.py
```

## Project Architecture

### Core Components

#### Data Pipeline (`src/data/`)
- **`dataset.py`**: CropNet dataset loader with preprocessing
- **`download_cropnet.py`**: Data download and preparation utilities

#### Model Architectures (`src/models/`)
- **`early_fusion.py`**: Feature-level fusion with 3D CNN
- **`late_fusion.py`**: Decision-level fusion with expert networks
- **`gmu.py`**: Gated Multimodal Unit (core innovation)

#### Training Infrastructure (`src/training/`)
- **`trainer.py`**: Unified training pipeline for all models
- Mixed precision support, gradient clipping, early stopping

#### Evaluation Suite (`src/evaluation/`)
- **`metrics.py`**: Comprehensive performance evaluation
- **`stress_testing.py`**: Robustness testing framework
- **`visualization.py`**: Advanced plotting and analysis tools

### Configuration System

All experiments use YAML configuration files:

```yaml
# config/early_fusion.yaml
model:
  type: early_fusion
  image_channels: 7
  weather_features: 6
  fusion_strategy: concat

training:
  num_epochs: 100
  batch_size: 32
  learning_rate: 1e-4
  optimizer: adam

data:
  crop_type: corn
  image_size: [64, 64]
  sequence_length: 32
```

## Model Implementations

### 1. Early Fusion Architecture

**Key Features:**
- Combines satellite and weather data at feature level
- 3D CNN for spatio-temporal processing
- Cross-modal attention mechanism

**Usage:**
```python
from src.models.early_fusion import EarlyFusionModel

model = EarlyFusionModel(
    image_channels=7,
    weather_features=6,
    fusion_strategy='concat'
)

# Forward pass
prediction = model(satellite_data, weather_data)
```

### 2. Late Fusion Architecture

**Key Features:**
- Separate expert networks for each modality
- Confidence-weighted prediction fusion
- Modular design for expert specialization

**Usage:**
```python
from src.models.late_fusion import LateFusionModel

model = LateFusionModel(
    satellite_hidden_dims=[64, 128, 256, 512],
    weather_architecture='transformer',
    fusion_strategy='weighted_average'
)

# Get expert outputs
prediction, expert_outputs = model(
    satellite_data, weather_data, 
    return_expert_outputs=True
)
```

### 3. Gated Multimodal Unit (GMU)

**Key Features:**
- Multi-level gating mechanisms
- Quality estimation networks
- Adaptive feature weighting

**Usage:**
```python
from src.models.gmu import GMUPredictor, create_gmu_model

# Create GMU with expert networks
gmu_model = create_gmu_model(
    satellite_expert, 
    weather_expert,
    config={'num_gates': 3, 'use_quality_estimation': True}
)

# Forward pass with attention
output = gmu_model(
    satellite_data, weather_data,
    return_attention=True
)
```

## Training Pipeline

### Custom Loss Function
The `MultimodalLoss` combines multiple objectives:

```python
# Regression + Uncertainty + Diversity + Quality
total_loss = (
    regression_weight * mse_loss +
    uncertainty_weight * uncertainty_loss +
    diversity_weight * expert_diversity_loss +
    quality_weight * quality_consistency_loss
)
```

### Training Configuration
```python
from src.training.trainer import TrainingConfig, ModelTrainer

config = TrainingConfig(
    model_type='gmu',
    num_epochs=100,
    batch_size=32,
    mixed_precision=True,
    gradient_clipping=1.0,
    early_stopping_patience=20
)

trainer = ModelTrainer(model, train_loader, val_loader, config)
metrics = trainer.train()
```

## Evaluation Framework

### Comprehensive Metrics
```python
from src.evaluation.metrics import ModelEvaluator

evaluator = ModelEvaluator()
results = evaluator.evaluate_model(
    model, test_loader, device, model_type='gmu'
)

print(f"MAE: {results.metrics['mae']:.3f}")
print(f"RMSE: {results.metrics['rmse']:.3f}")
print(f"R²: {results.metrics['r2']:.3f}")
```

### Stress Testing
```python
from src.evaluation.stress_testing import StressTester, DataQualityIssue

stress_tester = StressTester()

# Test cloud cover robustness
config = StressTestConfig(
    issue_type=DataQualityIssue.CLOUDY_SATELLITE,
    severity_levels=[0.2, 0.4, 0.6, 0.8],
    num_trials=5
)

results = stress_tester.run_stress_test(
    model, test_loader, device, config, 'gmu'
)
```

### Model Comparison
```python
from src.evaluation.metrics import evaluate_and_compare_models

models = {
    'Early Fusion': early_fusion_model,
    'Late Fusion': late_fusion_model,
    'GMU': gmu_model
}

comparison = evaluate_and_compare_models(
    models, test_loader, device
)
```

## Advanced Features

### Attention Visualization
```python
from src.evaluation.visualization import AttentionVisualizer

visualizer = AttentionVisualizer()
visualizer.visualize_gmu_attention(
    gmu_model, satellite_batch, weather_batch
)
```

### Feature Analysis
```python
from src.evaluation.visualization import FeatureAnalyzer

analyzer = FeatureAnalyzer()
analyzer.analyze_learned_features(
    model, data_loader, device, model_type='gmu'
)
```

### Interactive Dashboard
```python
from src.evaluation.visualization import InteractiveDashboard

dashboard = InteractiveDashboard()
dashboard.create_model_comparison_dashboard(
    model_results, stress_test_results
)
```

## Experiment Scripts

### Complete Model Training
```bash
# experiments/train_all_models.py
python experiments/train_all_models.py \
    --data-dir data/cropnet \
    --config-dir config \
    --output-dir results/models
```

### Robustness Evaluation
```bash
# experiments/stress_test_all.py
python experiments/stress_test_all.py \
    --models-dir results/models \
    --test-data data/cropnet/test \
    --output-dir results/stress_tests
```

### Analysis Generation
```bash
# experiments/generate_analysis.py
python experiments/generate_analysis.py \
    --results-dir results \
    --output-dir results/analysis
```

## Customization Guide

### Adding New Model Architecture
1. Create new file in `src/models/`
2. Inherit from `nn.Module`
3. Implement `forward()` method
4. Add to model factory in training pipeline

### Adding New Evaluation Metric
1. Extend `ModelEvaluator` class
2. Add metric calculation in `_calculate_comprehensive_metrics()`
3. Update visualization functions

### Adding New Data Corruption
1. Add new corruption function to `DataCorruptor` class
2. Update `_apply_corruption()` method in `StressTester`
3. Add corresponding `DataQualityIssue` enum value

## Performance Optimization

### Memory Optimization
```python
# Use gradient checkpointing
model = torch.utils.checkpoint.checkpoint_sequential(
    model, segments=2, input=x
)

# Mixed precision training
with torch.cuda.amp.autocast():
    output = model(input)
```

### Computational Efficiency
```python
# Compile model (PyTorch 2.0+)
model = torch.compile(model)

# Use efficient data loading
data_loader = DataLoader(
    dataset, 
    batch_size=32,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
)
```

## Troubleshooting

### Common Issues

#### CUDA Out of Memory
```python
# Reduce batch size
config.batch_size = 16

# Use gradient accumulation
accumulation_steps = 2
```

#### Slow Training
```python
# Enable mixed precision
config.mixed_precision = True

# Increase num_workers
data_loader = DataLoader(..., num_workers=8)
```

#### Poor Convergence
```python
# Adjust learning rate
config.learning_rate = 5e-5

# Enable learning rate scheduling
config.scheduler = 'cosine'
```

### Debug Mode
```python
# Enable detailed logging
logging.getLogger().setLevel(logging.DEBUG)

# Save intermediate outputs
model.forward(..., return_features=True)
```

## Testing

### Unit Tests
```bash
# Run all tests
pytest tests/

# Run specific module
pytest tests/test_models.py
```

### Integration Tests
```bash
# Test full pipeline
python tests/test_pipeline.py
```

### Benchmark Tests
```bash
# Performance benchmarking
python tests/benchmark_models.py
```

## Deployment

### Model Export
```python
# Export to ONNX
torch.onnx.export(model, dummy_input, "model.onnx")

# Export to TorchScript
scripted_model = torch.jit.script(model)
```

### Inference Pipeline
```python
from src.inference.predictor import CropYieldPredictor

predictor = CropYieldPredictor('path/to/model.pth')
yield_prediction = predictor.predict(
    satellite_image, weather_data
)
```

This technical guide provides comprehensive information for implementing, training, and evaluating the multimodal agriculture prediction models. For specific use cases or advanced configurations, refer to the example scripts in the `experiments/` directory.