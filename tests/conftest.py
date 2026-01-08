import mlstac
import pytest
import os
import torch
import sen2sr

import numpy as np

from unittest.mock import MagicMock

import sen2sr_tools.get_sr_image as get_sr_image

# --- CONSTANTS FOR MOCK RETURNS ---
MOCK_CRS = 32630
MOCK_SAMPLE_DATE = "2025-01-01"
MOCK_DEVICE = torch.device("cpu")

# Mock Dask Array object with a compute method that returns a NumPy array
class MockCloudlessData:
    """
    Simulates an xarray.DataArray returned by cubo.
    Must support .band, .values, .compute(), and .attrs
    """
    def __init__(self):
        # 1. Add 'band' attribute to mimic xarray coordinates
        self.band = MagicMock()
        self.band.values = ["B04", "B08", "B02", "B03"] 
        
        # 2. Add 'values' (the actual image data)
        self.values = torch.rand(4, 128, 128).numpy()
        
        # 3. Add dictionary for attributes (proj:epsg, etc)
        self.attrs = {"epsg": MOCK_CRS}

    def compute(self):
        # In real xarray, compute() returns a DataArray with numpy backend.
        # We return self so the .band attribute is preserved for downstream checks.
        return self
    
    def to_numpy(self):
        # Helper if code calls .to_numpy() directly
        return self.values

    # Optional: If your code uses .sel(band=...)
    def isel(self, **kwargs):
        return self

# Mock the compiled model object
class MockCompiledModel:
    """Simulates model output: 4x upscaled tensor"""

    def __call__(self, X):
        return torch.rand(1, 4, X.shape[2]*4, X.shape[3]*4)


@pytest.fixture
def mock_sen2sr_dependencies(monkeypatch):
    """Mocks all external dependencies for get_sr_image."""

    # --- I/O & Model Download Mocks ---
    # Assume the model exists or downloads successfully
    monkeypatch.setattr(os.path, "exists", lambda x: True)
    monkeypatch.setattr(os, "listdir", lambda x: ["model_file.safetensor"])

    mlstac_download_mock = MagicMock()
    monkeypatch.setattr(mlstac, "download", mlstac_download_mock)

    mlstac_load_mock = MagicMock(return_value=MagicMock(
        compiled_model=lambda device: MockCompiledModel()))
    monkeypatch.setattr(mlstac, "load", mlstac_load_mock)

    # --- Data & Tensor Prep Mocks ---
    monkeypatch.setattr(get_sr_image, "lonlat_to_utm_epsg",
                        MagicMock(return_value=MOCK_CRS))
    monkeypatch.setattr(get_sr_image, "download_sentinel_cubo", MagicMock(
        return_value=(MockCloudlessData(), MOCK_SAMPLE_DATE)))

    # Force torch to run on CPU to avoid CUDA dependency in tests
    monkeypatch.setattr(torch, "cuda", MagicMock(is_available=lambda: False))
    monkeypatch.setattr(torch, "device", MagicMock(return_value=MOCK_DEVICE))

    # --- Model Prediction & Utility Mocks ---
    sen2sr_predict_large_mock = MagicMock(
        return_value=torch.rand(4, 512, 512))  # 4x upscaled mock
    monkeypatch.setattr(sen2sr, "predict_large", sen2sr_predict_large_mock)

    # Note: reorder_bands must return two things, both NumPy arrays
    reordered_mock = (torch.rand(4, 128, 128).numpy(),
                      torch.rand(4, 512, 512).numpy())
    reorder_bands_mock = MagicMock(return_value=reordered_mock)
    monkeypatch.setattr(get_sr_image, "reorder_bands", reorder_bands_mock)

    # Saving/Cleanup Mocks
    save_to_tif_mock = MagicMock()
    monkeypatch.setattr(get_sr_image, "save_to_tif", save_to_tif_mock)
    monkeypatch.setattr(get_sr_image, "save_to_png", MagicMock())
    monkeypatch.setattr(
        get_sr_image, "make_pixel_faithful_comparison", MagicMock())

    # Create a mock image array (3 bands, 128x128)
    mock_crop_array = np.zeros((3, 128, 128), dtype=np.float32)
    
    # Create mock metadata compatible with rasterio
    mock_crop_meta = {
        "driver": "GTiff", 
        "height": 128, 
        "width": 128, 
        "count": 3, 
        "dtype": "float32",
        "crs": MOCK_CRS,
        "transform": (10, 0, 0, 0, -10, 0) # Mock transform
    }
    
    # Return (array, metadata)
    crop_parcel_mock = MagicMock(return_value=(mock_crop_array, mock_crop_meta))
    monkeypatch.setattr(get_sr_image, "crop_parcel_from_tif", crop_parcel_mock)
    
    # Mock the rasterio module's open function to prevent actual file I/O
    mock_dst = MagicMock()
    mock_open = MagicMock()
    
    # Setup the Context Manager behavior: with rasterio.open(...) as dest:
    mock_open.return_value.__enter__.return_value = mock_dst
    
    # Patch rasterio inside the module under test
    monkeypatch.setattr(get_sr_image, "rasterio", MagicMock(open=mock_open))

    # Yield all the mocks you might want to assert calls on
    yield {
        "predict_large_mock": sen2sr_predict_large_mock,
        "crop_parcel_mock": crop_parcel_mock,
        "save_to_tif_mock": save_to_tif_mock,
        "mlstac_download_mock": mlstac_download_mock,
        "mock_device": MOCK_DEVICE
    }
