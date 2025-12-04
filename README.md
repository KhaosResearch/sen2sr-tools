# SEN2SR Tools

![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2FMiguelPadillaR%2Fsen2sr-tools%2Fmain%2Fpyproject.toml)
 [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
 ![Status: Active](https://img.shields.io/badge/Status-Active-00aa00.svg)
![Code style: Ruff](https://img.shields.io/badge/code%20style-autopep8-aa0000.svg)

This repository provides a list of comprehensive tools and utilities for the usage of the [SEN2SR](https://github.com/ESAOpenSR/SEN2SR.git) neural network super-resolution model, developed by the [ESAOpenSR](https://opensr.eu/) team. This model specializes in upscaling Sentinel-2's 10m/px images up to a x4 times improvement of 2.5m/px.

This package implements a feature to crop a polygon directly from the SR image, useful for close-up satellite observation of agricultural states.

## Installation
1. Clone the repository
    ```bash
    git clone https://github.com/KhaosResearch/sen2sr-tools.git
    cd sen2sr-tools
    ```
2. Create a virtual environment and install all dependencies
    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows use `venv\Scripts\activate`
    pip install -e .
    ```
## Quickstart
The overall workflow can be run directly, since it implements a simple demo in it:
```shell
cd sen2sr-tools
python -m sen2sr_tools.get_sr_image
```
It can also be launched as a console command:
```shell
python -m sen2sr_tools get-sr-image --latitude 42.465226 --longitude -2.292699 --start-date 2025-11-01 --end-date 2025-11-15 --geojson-path ./example.geojson
```
Output will be at `sen2sr_out`, where the SR and original images can be seen.
## Usage in Python

### Get a SR image

It runs the full workflow, from date and location data to super resolved image. Default `size` is the minimal requested 128 px. Default `bands` are Near-Infrared, Red, Blue, Green and the Scene Classification Layer, that contains cloud density information among others. Cropping a polygon from the image is optional, provided the GeoJSON filepath.

```python
from sen2sr_tools.get_sr_image import get_sr_image

lat = 37.265840
lng = -4.593406 
start_date = "2025-11-01"
end_date = "2025-11-15"
# bands = ["B08", "B02", "B03", "B04", "SCL"]       # Default bands
# size = 128                                        # Default size
# geojson_path = None                               # Default value

sr_filepath = get_sr_image(lat, lng, bands, start_date, end_date)

print(f"SR image successfully downloaded and saved at: {sr_filepath}")
```
### Download CUBO

It serves the requested bands' data directly. It uses the date range to get cloudless (<=1%) images. The `crs` can be provided (i.e, "EPSG:32630"), but it can also be automatically calculated from the `lat` and `lon` arguments. If no requested data was found within the date range, it retries with by expanding it backwards in time.

```python
from sen2sr_tools.get_sr_image import download_sentinel_cubo

lat = 37.265840
lng = -4.593406 
start_date = "2025-11-01"
end_date = "2025-11-15"
# crs = None                                   # Default CRS
# bands = ["B08", "B02", "B03", "B04", "SCL"]  # Default bands
# size = 128                                   # Default size
# cloud_threshold= 0.01                        # Default threshold density
# max_retries = 3                              # Default retries
# retry_days_shift = 15                        # Default shift

cloudless_image_data_array, sample_date = download_sentinel_cubo(lat, lon, start_date, end_date):
```
### Crop parcel from image

It crops the polygon from the given TIF file and returns the cropped image as PNG.

```python
from sen2sr_tools.get_sr_image import crop_png_from_tif

raster_path = "path/to/file.tif"
geojson_path = "path/to/polygon.geojson"
date = "2025-11-1"

out_png_path = crop_png_from_tif(raster_path, geojson_path, date)

print(f"Polygon successfully cropped and saved at: {out_png_path}")
```

## Attribution and License
We appreciate the work done by the [ESAOpenSR](https://opensr.eu/) team in upscaling and super-resolving satellite imagery and making SR models accessible through Open Source code. This module is built upon the core model and concepts of the `sen2sr` Python package.

- Original Repository: https://github.com/ESAOpenSR/SEN2SR.git

The original work is licensed under the **CC0 1.0 Universal License**.

## Trained model:
The model file (`model/model.safetensors`) is available for download from [HugginFace](https://huggingface.co/tacofoundation/sen2sr/resolve/main/SEN2SRLite/NonReference_RGBN_x4/mlm.json://huggingface.co/tacofoundation/sen2sr).

- Official SEN2SR HuggingFace model card: https://huggingface.co/tacofoundation/sen2sr.
