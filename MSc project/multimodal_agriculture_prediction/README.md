# Multimodal Fusion for Sustainable Agriculture Yield Prediction

## Project Overview

This MSc research project investigates the effectiveness of different multimodal fusion strategies for crop yield prediction using satellite imagery and meteorological data. The project compares Early Fusion vs. Late Fusion architectures and implements a novel Gated Multimodal Unit (GMU) to dynamically weight different data modalities based on data quality.

## Research Questions

1. **Which fusion strategy is more effective?** Early Fusion (feature-level) vs. Late Fusion (decision-level)
2. **How robust are these approaches to data quality issues?** Testing with cloudy satellite imagery and missing weather data
3. **Can adaptive gating improve multimodal learning?** Using GMU to dynamically weight modalities

## Dataset

- **CropNet Framework**: 2,200+ U.S. counties
- **Satellite Data**: Sentinel-2 (RGB + NDVI vegetation indices)
- **Weather Data**: WRF-HRRR meteorological data (temperature, precipitation, humidity)
- **Ground Truth**: USDA crop yield records

## Model Architectures

### 1. Early Fusion
- 3D CNN processing combined image-weather data cubes
- Learns joint spatio-temporal representations

### 2. Late Fusion
- Separate expert models: CNN for imagery + LSTM/Transformer for weather
- Decision-level fusion of predictions

### 3. Gated Multimodal Unit (GMU)
- Adaptive weighting mechanism
- Emphasizes reliable modalities during data quality issues

## Project Structure

```
├── data/                           # Dataset storage and management
├── src/                           # Source code
│   ├── models/                    # Model architectures
│   ├── data/                      # Data processing pipelines
│   ├── training/                  # Training loops and optimization
│   └── evaluation/                # Evaluation metrics and analysis
├── experiments/                   # Experimental configurations
├── notebooks/                     # Jupyter notebooks for analysis
├── results/                       # Model outputs and visualizations
└── docs/                         # Documentation and research notes
```

## Getting Started

1. **Setup Environment**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Download Data**:
   ```bash
   python src/data/download_cropnet.py
   ```

3. **Run Experiments**:
   ```bash
   python experiments/train_models.py --config config/early_fusion.yaml
   ```

## Key Innovation

The project's primary innovation lies in implementing adaptive multimodal fusion that can handle real-world data heterogeneity - where sensors fail, clouds obscure satellites, and weather stations go offline. This addresses a critical gap between laboratory AI and field deployment.

## Expected Outcomes

1. **Comparative Analysis**: Quantitative comparison of fusion strategies
2. **Robustness Study**: Performance under various data quality scenarios
3. **Adaptive System**: GMU-based model that adjusts to data availability
4. **Research Publication**: Findings suitable for top-tier conferences (ICLR, NeurIPS)

## Research Significance

This work contributes to:
- **Climate AI**: Improving agricultural predictions for food security
- **Multimodal Learning**: Understanding fusion strategy effectiveness
- **Practical AI**: Addressing real-world data reliability challenges