# SEN2SR Tools
This repository provides a list of comprehensive tools and utilities for the usage of the [SEN2SR](https://github.com/ESAOpenSR/SEN2SR.git) neural network super-resolution model, developed by the [ESAOpenSR](https://opensr.eu/) team. This model specializes in upscaling Sentinel-2's 10m/pc images up to a x4 times improvement of 2.5m/px.

This packages features a method to crop a polygon directly from the SR image, useful for close-up satellite observation of agricultural states.

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

## Usage in Python
### Get a SR image

TODO

## Attribution and License
We appreciate the work done by the [ESAOpenSR](https://opensr.eu/) team in upscaling and super-resolving satellite imagery and making SR models accessible through Open Source code. This module is built upon the core model and concepts of the `sen2sr` Python package.

- Original Repository: https://github.com/ESAOpenSR/SEN2SR.git

The original work is licensed under the **CC0 1.0 Universal License**.

## Trained model:
The model file (`model/model.safetensors`) is available for download from [HugginFace](https://huggingface.co/tacofoundation/sen2sr/resolve/main/SEN2SRLite/NonReference_RGBN_x4/mlm.json://huggingface.co/tacofoundation/sen2sr).

- Official SEN2SR HuggingFace model card: https://huggingface.co/tacofoundation/sen2sr.
