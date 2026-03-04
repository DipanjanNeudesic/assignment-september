"""
CropNet Dataset Loader and Preprocessor

This module handles downloading and preprocessing of the CropNet dataset,
which includes satellite imagery and meteorological data for crop yield prediction.
"""

import os
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import rasterio
from rasterio.windows import Window
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CropNetDataset(Dataset):
    """
    CropNet Dataset for multimodal crop yield prediction.
    
    Combines satellite imagery and meteorological data for training
    early fusion, late fusion, and gated multimodal models.
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        counties: Optional[List[str]] = None,
        years: Optional[List[int]] = None,
        crop_type: str = 'corn',
        image_size: Tuple[int, int] = (64, 64),
        sequence_length: int = 32,  # Weather sequence length
        normalize: bool = True,
        augment: bool = True,
        load_indices: Optional[List[str]] = None
    ):
        """
        Initialize CropNet dataset.
        
        Args:
            data_dir: Root directory containing CropNet data
            split: Dataset split ('train', 'val', 'test')
            counties: List of county FIPS codes to include
            years: List of years to include
            crop_type: Type of crop ('corn', 'soy')
            image_size: Target size for satellite images
            sequence_length: Length of weather time series
            normalize: Whether to normalize features
            augment: Whether to apply data augmentation
            load_indices: Specific vegetation indices to load
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.crop_type = crop_type
        self.image_size = image_size
        self.sequence_length = sequence_length
        self.normalize = normalize
        self.augment = augment and (split == 'train')
        
        # Default vegetation indices
        if load_indices is None:
            self.load_indices = ['NDVI', 'EVI', 'SAVI', 'NDWI']
        else:
            self.load_indices = load_indices
            
        # Initialize transforms
        self._setup_transforms()
        
        # Load dataset metadata
        self.samples = self._load_dataset_samples(counties, years)
        
        # Initialize scalers for normalization
        if self.normalize:
            self.image_scaler = MinMaxScaler()
            self.weather_scaler = StandardScaler()
            self._fit_scalers()
        
        logger.info(f"Loaded {len(self.samples)} {split} samples")
    
    def _setup_transforms(self):
        """Setup image transformations."""
        transform_list = [
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
        ]
        
        if self.augment:
            transform_list.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            ])
        
        self.image_transform = transforms.Compose(transform_list)
    
    def _load_dataset_samples(self, counties: Optional[List[str]], years: Optional[List[int]]) -> List[Dict]:
        """Load dataset sample metadata."""
        samples = []
        
        # Load CropNet index file
        index_file = self.data_dir / f"{self.split}_index.csv"
        if not index_file.exists():
            raise FileNotFoundError(f"Index file not found: {index_file}")
        
        df = pd.read_csv(index_file)
        
        # Filter by counties if specified
        if counties:
            df = df[df['county_fips'].isin(counties)]
        
        # Filter by years if specified
        if years:
            df = df[df['year'].isin(years)]
        
        # Filter by crop type
        df = df[df['crop_type'] == self.crop_type]
        
        for _, row in df.iterrows():
            sample = {
                'county_fips': row['county_fips'],
                'year': int(row['year']),
                'yield': float(row['yield_bu_per_acre']),
                'satellite_path': self.data_dir / 'satellite' / f"{row['county_fips']}_{row['year']}.tif",
                'weather_path': self.data_dir / 'weather' / f"{row['county_fips']}_{row['year']}.nc",
                'area_harvested': float(row['area_harvested_acres']),
                'area_planted': float(row['area_planted_acres'])
            }
            samples.append(sample)
        
        return samples
    
    def _fit_scalers(self):
        """Fit scalers on training data."""
        if self.split != 'train':
            return
        
        logger.info("Fitting scalers on training data...")
        
        # Collect sample data for fitting
        sample_images = []
        sample_weather = []
        
        # Use subset for efficiency
        n_samples = min(100, len(self.samples))
        for i in range(0, len(self.samples), len(self.samples) // n_samples):
            try:
                img, weather, _ = self._load_raw_data(self.samples[i])
                sample_images.append(img.flatten())
                sample_weather.append(weather.flatten())
            except Exception as e:
                logger.warning(f"Failed to load sample {i}: {e}")
                continue
        
        # Fit image scaler
        if sample_images:
            sample_images = np.vstack(sample_images)
            self.image_scaler.fit(sample_images)
        
        # Fit weather scaler
        if sample_weather:
            sample_weather = np.vstack(sample_weather)
            self.weather_scaler.fit(sample_weather)
        
        logger.info("Scalers fitted successfully")
    
    def _load_satellite_data(self, satellite_path: Path) -> np.ndarray:
        """Load and preprocess satellite imagery."""
        try:
            with rasterio.open(satellite_path) as src:
                # Read all bands
                image = src.read()
                
                # Handle different band configurations
                if image.shape[0] >= 4:  # Assume RGB + NIR
                    rgb = image[:3]  # RGB bands
                    nir = image[3]   # Near-infrared
                    
                    # Calculate vegetation indices
                    indices = self._calculate_vegetation_indices(rgb, nir)
                    
                    # Combine RGB and indices
                    image = np.concatenate([rgb, indices], axis=0)
                
                # Transpose to (H, W, C) for torchvision transforms
                image = np.transpose(image, (1, 2, 0))
                
                # Handle invalid values
                image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
                
                return image.astype(np.float32)
                
        except Exception as e:
            logger.warning(f"Failed to load satellite data from {satellite_path}: {e}")
            # Return zeros if loading fails
            return np.zeros((*self.image_size, len(self.load_indices) + 3), dtype=np.float32)
    
    def _calculate_vegetation_indices(self, rgb: np.ndarray, nir: np.ndarray) -> np.ndarray:
        """Calculate vegetation indices from RGB and NIR bands."""
        red = rgb[0].astype(np.float32)
        green = rgb[1].astype(np.float32)
        blue = rgb[2].astype(np.float32)
        nir = nir.astype(np.float32)
        
        indices = []
        
        # Prevent division by zero
        eps = 1e-8
        
        if 'NDVI' in self.load_indices:
            # Normalized Difference Vegetation Index
            ndvi = (nir - red) / (nir + red + eps)
            indices.append(ndvi)
        
        if 'EVI' in self.load_indices:
            # Enhanced Vegetation Index
            evi = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1 + eps)
            indices.append(evi)
        
        if 'SAVI' in self.load_indices:
            # Soil Adjusted Vegetation Index
            L = 0.5  # soil brightness correction factor
            savi = (1 + L) * (nir - red) / (nir + red + L + eps)
            indices.append(savi)
        
        if 'NDWI' in self.load_indices:
            # Normalized Difference Water Index
            ndwi = (green - nir) / (green + nir + eps)
            indices.append(ndwi)
        
        # Stack indices
        return np.stack(indices, axis=0) if indices else np.array([]).reshape(0, *nir.shape)
    
    def _load_weather_data(self, weather_path: Path) -> np.ndarray:
        """Load and preprocess meteorological data."""
        try:
            with xr.open_dataset(weather_path) as ds:
                # Expected weather variables
                weather_vars = ['temperature_2m', 'precipitation', 'relative_humidity_2m', 
                              'wind_speed_10m', 'solar_radiation', 'pressure_msl']
                
                # Extract available variables
                data_arrays = []
                for var in weather_vars:
                    if var in ds.variables:
                        data = ds[var].values
                        data_arrays.append(data)
                    else:
                        # Fill with zeros if variable not available
                        time_steps = len(ds.time) if 'time' in ds.dims else self.sequence_length
                        data_arrays.append(np.zeros(time_steps))
                
                # Stack variables
                weather_data = np.stack(data_arrays, axis=1)  # (time, features)
                
                # Ensure correct sequence length
                if len(weather_data) > self.sequence_length:
                    # Take last sequence_length steps
                    weather_data = weather_data[-self.sequence_length:]
                elif len(weather_data) < self.sequence_length:
                    # Pad with last values
                    padding = np.tile(weather_data[-1:], (self.sequence_length - len(weather_data), 1))
                    weather_data = np.concatenate([weather_data, padding], axis=0)
                
                # Handle invalid values
                weather_data = np.nan_to_num(weather_data, nan=0.0)
                
                return weather_data.astype(np.float32)
                
        except Exception as e:
            logger.warning(f"Failed to load weather data from {weather_path}: {e}")
            # Return zeros if loading fails
            return np.zeros((self.sequence_length, 6), dtype=np.float32)
    
    def _load_raw_data(self, sample: Dict) -> Tuple[np.ndarray, np.ndarray, float]:
        """Load raw satellite and weather data."""
        satellite_data = self._load_satellite_data(sample['satellite_path'])
        weather_data = self._load_weather_data(sample['weather_path'])
        yield_value = sample['yield']
        
        return satellite_data, weather_data, yield_value
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample."""
        sample = self.samples[idx]
        
        try:
            # Load raw data
            satellite_data, weather_data, yield_value = self._load_raw_data(sample)
            
            # Apply image transformations
            if len(satellite_data.shape) == 3:
                # Convert to PIL-compatible format for transforms
                satellite_tensor = torch.from_numpy(satellite_data.transpose(2, 0, 1))
            else:
                satellite_tensor = torch.from_numpy(satellite_data)
            
            # Normalize if enabled
            if self.normalize:
                # Flatten, scale, and reshape satellite data
                orig_shape = satellite_tensor.shape
                satellite_flat = satellite_tensor.view(-1, 1).numpy()
                satellite_scaled = self.image_scaler.transform(satellite_flat)
                satellite_tensor = torch.from_numpy(satellite_scaled).view(orig_shape)
                
                # Scale weather data
                weather_flat = weather_data.reshape(-1, 1)
                weather_scaled = self.weather_scaler.transform(weather_flat)
                weather_data = weather_scaled.reshape(weather_data.shape)
            
            weather_tensor = torch.from_numpy(weather_data).float()
            yield_tensor = torch.tensor(yield_value, dtype=torch.float32)
            
            return {
                'satellite': satellite_tensor.float(),
                'weather': weather_tensor,
                'yield': yield_tensor,
                'county_fips': sample['county_fips'],
                'year': sample['year'],
                'metadata': {
                    'area_harvested': sample['area_harvested'],
                    'area_planted': sample['area_planted']
                }
            }
            
        except Exception as e:
            logger.error(f"Error loading sample {idx}: {e}")
            # Return dummy data in case of error
            return {
                'satellite': torch.zeros(len(self.load_indices) + 3, *self.image_size),
                'weather': torch.zeros(self.sequence_length, 6),
                'yield': torch.tensor(0.0, dtype=torch.float32),
                'county_fips': sample['county_fips'],
                'year': sample['year'],
                'metadata': {'area_harvested': 0.0, 'area_planted': 0.0}
            }


def create_data_loaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    val_split: float = 0.2,
    test_split: float = 0.1,
    **dataset_kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        data_dir: Root directory containing CropNet data
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
        val_split: Fraction of data to use for validation
        test_split: Fraction of data to use for testing
        **dataset_kwargs: Additional arguments for CropNetDataset
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    
    # Create datasets
    train_dataset = CropNetDataset(
        data_dir=data_dir,
        split='train',
        normalize=True,
        augment=True,
        **dataset_kwargs
    )
    
    val_dataset = CropNetDataset(
        data_dir=data_dir,
        split='val',
        normalize=True,
        augment=False,
        **dataset_kwargs
    )
    
    test_dataset = CropNetDataset(
        data_dir=data_dir,
        split='test',
        normalize=True,
        augment=False,
        **dataset_kwargs
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Example usage
    data_dir = "../../data/cropnet"
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir=data_dir,
        batch_size=16,
        crop_type='corn',
        image_size=(64, 64),
        sequence_length=32
    )
    
    # Test data loading
    for batch in train_loader:
        print("Satellite shape:", batch['satellite'].shape)
        print("Weather shape:", batch['weather'].shape)
        print("Yield shape:", batch['yield'].shape)
        break