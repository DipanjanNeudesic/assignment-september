"""
Data Download and Preparation Script for CropNet Dataset

This script downloads the CropNet dataset from Hugging Face and prepares
it for use in the multimodal agriculture prediction project.
"""

import os
import requests
import zipfile
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import List, Optional
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CropNetDownloader:
    """Downloads and prepares CropNet dataset."""
    
    def __init__(self, data_dir: str = "../../data/cropnet"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # CropNet dataset URLs (these would be actual URLs in practice)
        self.base_url = "https://huggingface.co/datasets/cropnet/resolve/main"
        self.dataset_files = {
            'train_index': 'train_index.csv',
            'val_index': 'val_index.csv', 
            'test_index': 'test_index.csv',
            'county_metadata': 'county_metadata.csv',
            'satellite_data': 'satellite_data.zip',
            'weather_data': 'weather_data.zip'
        }
    
    def download_file(self, filename: str, url: str) -> bool:
        """Download a single file."""
        filepath = self.data_dir / filename
        
        if filepath.exists():
            logger.info(f"{filename} already exists, skipping download")
            return True
        
        try:
            logger.info(f"Downloading {filename}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Downloaded {filename} successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download {filename}: {e}")
            return False
    
    def extract_zip(self, zip_path: Path, extract_dir: Path) -> bool:
        """Extract zip file."""
        try:
            logger.info(f"Extracting {zip_path.name}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            logger.info(f"Extracted {zip_path.name} successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to extract {zip_path}: {e}")
            return False
    
    def generate_synthetic_data(self):
        """Generate synthetic CropNet-like data for development."""
        logger.info("Generating synthetic CropNet data for development...")
        
        # Create directories
        (self.data_dir / 'satellite').mkdir(exist_ok=True)
        (self.data_dir / 'weather').mkdir(exist_ok=True)
        
        # Generate synthetic metadata
        counties = [f"{i:05d}" for i in range(10001, 10051)]  # 50 counties
        years = list(range(2018, 2023))  # 5 years
        crops = ['corn', 'soy']
        
        # Generate train/val/test splits
        np.random.seed(42)
        all_samples = []
        
        for county in counties:
            for year in years:
                for crop in crops:
                    # Generate synthetic yield data
                    base_yield = 150 if crop == 'corn' else 50
                    yield_val = base_yield + np.random.normal(0, 20)
                    
                    sample = {
                        'county_fips': county,
                        'year': year,
                        'crop_type': crop,
                        'yield_bu_per_acre': max(0, yield_val),
                        'area_harvested_acres': np.random.uniform(1000, 10000),
                        'area_planted_acres': np.random.uniform(1000, 10000)
                    }
                    all_samples.append(sample)
        
        # Create DataFrame and split
        df = pd.DataFrame(all_samples)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
        
        n_samples = len(df)
        n_train = int(0.7 * n_samples)
        n_val = int(0.15 * n_samples)
        
        train_df = df[:n_train]
        val_df = df[n_train:n_train + n_val]
        test_df = df[n_train + n_val:]
        
        # Save index files
        train_df.to_csv(self.data_dir / 'train_index.csv', index=False)
        val_df.to_csv(self.data_dir / 'val_index.csv', index=False)
        test_df.to_csv(self.data_dir / 'test_index.csv', index=False)
        
        logger.info(f"Generated {len(train_df)} training samples")
        logger.info(f"Generated {len(val_df)} validation samples")
        logger.info(f"Generated {len(test_df)} test samples")
        
        # Generate county metadata
        county_meta = []
        for county in counties:
            meta = {
                'county_fips': county,
                'state': 'IL',  # Illinois
                'latitude': 40 + np.random.uniform(-2, 2),
                'longitude': -89 + np.random.uniform(-2, 2),
                'elevation_m': np.random.uniform(100, 400),
                'climate_zone': np.random.choice(['humid_continental', 'humid_subtropical']),
                'soil_type': np.random.choice(['mollisol', 'alfisol', 'vertisol'])
            }
            county_meta.append(meta)
        
        pd.DataFrame(county_meta).to_csv(self.data_dir / 'county_metadata.csv', index=False)
        
        logger.info("Generated synthetic county metadata")
        
        # Note: Actual satellite and weather data files would be generated here
        # For now, we'll create placeholder files that the dataset loader can handle
        self._create_placeholder_data_files(all_samples)
    
    def _create_placeholder_data_files(self, samples: List[dict]):
        """Create placeholder satellite and weather data files."""
        logger.info("Creating placeholder data files...")
        
        # Create placeholder satellite images (small TIFF files)
        import rasterio
        from rasterio.transform import from_bounds
        
        for sample in samples[:10]:  # Just create a few examples
            county = sample['county_fips']
            year = sample['year']
            
            # Create synthetic satellite data
            satellite_path = self.data_dir / 'satellite' / f"{county}_{year}.tif"
            if not satellite_path.exists():
                # Create synthetic 4-band image (RGB + NIR)
                height, width = 64, 64
                data = np.random.randint(0, 255, (4, height, width), dtype=np.uint8)
                
                # Add realistic NDVI-like patterns
                red = data[0].astype(float)
                nir = data[3].astype(float)
                # Make vegetation areas have higher NIR
                vegetation_mask = np.random.random((height, width)) > 0.3
                data[3][vegetation_mask] = np.minimum(255, nir[vegetation_mask] * 1.5)
                
                transform = from_bounds(-90, 39, -88, 41, width, height)
                
                with rasterio.open(
                    satellite_path, 'w',
                    driver='GTiff',
                    height=height, width=width,
                    count=4, dtype=data.dtype,
                    crs='EPSG:4326',
                    transform=transform
                ) as dst:
                    dst.write(data)
            
            # Create synthetic weather data
            weather_path = self.data_dir / 'weather' / f"{county}_{year}.nc"
            if not weather_path.exists():
                import xarray as xr
                
                # Create 32 time steps of weather data
                time_steps = 32
                dates = pd.date_range(f'{year}-04-01', periods=time_steps, freq='W')
                
                # Generate realistic weather patterns
                base_temp = 15 + 10 * np.sin(np.linspace(0, np.pi, time_steps))  # Seasonal
                temperature = base_temp + np.random.normal(0, 5, time_steps)
                
                precipitation = np.random.exponential(2, time_steps)  # Rainfall
                humidity = 60 + 20 * np.random.random(time_steps)
                wind_speed = 5 + 3 * np.random.random(time_steps)
                solar_radiation = 200 + 100 * np.sin(np.linspace(0, np.pi, time_steps)) + np.random.normal(0, 20, time_steps)
                pressure = 1013 + np.random.normal(0, 10, time_steps)
                
                # Create xarray dataset
                ds = xr.Dataset({
                    'temperature_2m': (['time'], temperature),
                    'precipitation': (['time'], precipitation),
                    'relative_humidity_2m': (['time'], humidity),
                    'wind_speed_10m': (['time'], wind_speed),
                    'solar_radiation': (['time'], solar_radiation),
                    'pressure_msl': (['time'], pressure)
                }, coords={'time': dates})
                
                ds.to_netcdf(weather_path)
        
        logger.info("Created placeholder data files")
    
    def download_dataset(self, synthetic: bool = True):
        """Download the complete CropNet dataset."""
        if synthetic:
            self.generate_synthetic_data()
            return True
        
        # Download real dataset (when available)
        success = True
        for file_key, filename in self.dataset_files.items():
            url = f"{self.base_url}/{filename}"
            if not self.download_file(filename, url):
                success = False
        
        # Extract zip files
        for zip_file in ['satellite_data.zip', 'weather_data.zip']:
            zip_path = self.data_dir / zip_file
            if zip_path.exists():
                extract_dir = self.data_dir / zip_file.replace('.zip', '')
                if not self.extract_zip(zip_path, extract_dir):
                    success = False
        
        return success
    
    def verify_dataset(self) -> bool:
        """Verify dataset integrity."""
        logger.info("Verifying dataset...")
        
        required_files = [
            'train_index.csv',
            'val_index.csv', 
            'test_index.csv',
            'county_metadata.csv'
        ]
        
        for file in required_files:
            filepath = self.data_dir / file
            if not filepath.exists():
                logger.error(f"Missing required file: {file}")
                return False
            
            # Check if CSV files can be loaded
            try:
                df = pd.read_csv(filepath)
                logger.info(f"{file}: {len(df)} records")
            except Exception as e:
                logger.error(f"Cannot read {file}: {e}")
                return False
        
        # Check data directories
        satellite_dir = self.data_dir / 'satellite'
        weather_dir = self.data_dir / 'weather'
        
        if not satellite_dir.exists():
            logger.error("Satellite data directory missing")
            return False
        
        if not weather_dir.exists():
            logger.error("Weather data directory missing")
            return False
        
        logger.info("Dataset verification successful")
        return True


def main():
    parser = argparse.ArgumentParser(description="Download CropNet dataset")
    parser.add_argument('--data-dir', default='../../data/cropnet',
                       help='Directory to store dataset')
    parser.add_argument('--synthetic', action='store_true',
                       help='Generate synthetic data for development')
    parser.add_argument('--verify-only', action='store_true',
                       help='Only verify existing dataset')
    
    args = parser.parse_args()
    
    downloader = CropNetDownloader(args.data_dir)
    
    if args.verify_only:
        success = downloader.verify_dataset()
    else:
        success = downloader.download_dataset(synthetic=args.synthetic)
        if success:
            success = downloader.verify_dataset()
    
    if success:
        logger.info("Dataset preparation completed successfully")
    else:
        logger.error("Dataset preparation failed")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())