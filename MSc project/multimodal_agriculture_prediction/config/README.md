# Sample Configuration Files

This directory contains example configuration files for different model architectures and experimental setups.

## Configuration Files

### Model Configurations

1. **`early_fusion.yaml`** - Configuration for Early Fusion model
2. **`late_fusion.yaml`** - Configuration for Late Fusion model  
3. **`gmu.yaml`** - Configuration for Gated Multimodal Unit (GMU) model

### Experimental Configurations

4. **`full_experiment.yaml`** - Complete experimental pipeline settings
5. **`stress_testing.yaml`** - Robustness testing parameters
6. **`quick_test.yaml`** - Fast testing configuration for development

## Usage

Copy these configurations to your working directory and modify them according to your specific requirements:

```bash
# Copy all configs to your experiment directory
cp config/*.yaml /path/to/your/experiment/config/

# Or copy specific configs
cp config/gmu.yaml /path/to/your/experiment/
```

## Configuration Structure

Each model configuration file contains:

- **model**: Model architecture parameters
- **training**: Training hyperparameters and settings
- **data**: Data loading and preprocessing parameters
- **evaluation**: Evaluation and metric settings

## Customization

Modify the following key parameters based on your dataset and requirements:

- **Data dimensions**: `satellite_channels`, `satellite_size`, `weather_features`
- **Model capacity**: `hidden_dim`, `num_layers`, `dropout_rate`
- **Training**: `learning_rate`, `batch_size`, `num_epochs`
- **Hardware**: `mixed_precision`, `num_workers`

See the individual configuration files for detailed parameter descriptions and recommended ranges.