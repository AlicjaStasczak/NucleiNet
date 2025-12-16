<p align="center">
  <img src="images/NucleiNet_logo.png" alt="NucleiNet logo" width="300"/>
</p>

**NucleiNet** is a web-based application for **cell nucleus segmentation and quantitative analysis** of microscopy images. Built with **Python** and **Streamlit**, it supports both **single images** and **multi-frame stacks**, combines classical image processing with **U-Net–based deep learning**, and enables interactive exploration, batch processing, and metric export.

## Features

- Upload a microscopy image **or a stack** (multi-page TIFF, GIF)
- Interactive **cropping** and **pre-processing**, including CLAHE, bilateral filtering, non-local means denoising, unsharp masking, and adaptive thresholding
- Multiple segmentation methods:
  - **Otsu (binary)**
  - **Adaptive (binary)**
  - **Watershed**
  - **U-Net** (Keras `.keras/.h5`, SavedModel directory, or **TFLite `.tflite`**)
- Post-processing to improve separation of touching nuclei:
  - morphological operations
  - hole filling
  - small object removal
  - watershed splitting on binary masks
  - *(optional)* opening by reconstruction
  - *(optional)* skeletonization
- Quantitative metrics:
  - number of detected nuclei
  - mask coverage percentage
  - per-frame metrics for image stacks
- Export options:
  - segmented images (PNG)
  - processed stacks (TIFF)
  - metrics tables (CSV)

## Installation

It is recommended to use a **clean virtual environment** (venv or conda).

Clone the repository:
```bash
git clone https://github.com/your-username/NucleiNet.git
cd NucleiNet
Install dependencies:

Core dependencies:
pip install -r requirements.txt

Running the application
Standard run:
streamlit run app.py
The application will open in your browser at the local URL printed in the terminal.

Running with large images or stacks
When working with large microscopy images or large multi-frame TIFF stacks, Streamlit’s default upload and message size limits may be insufficient. Use the following command to increase these limits:
streamlit run app.py --server.maxUploadSize=2000 --server.maxMessageSize=2000
Values are specified in megabytes (MB). Adjust them according to available system memory.

Notes on large files
Uploaded images and stacks are written to a temporary file on disk.

Multi-frame TIFF stacks are processed frame by frame to reduce memory usage.

Ensure sufficient free disk space in your system temporary directory.

U-Net model support
The application supports the following model formats:

TFLite (.tflite)

Keras (.keras or .h5)

Models can be manually selected in the sidebar.

Metrics
Coverage is computed exclusively from the binary segmentation mask (values 0/255).

Cell counting is performed using connected components with optional hole filling, area filtering, and border exclusion.

In stack mode, metrics are exported per frame as a CSV file.

Limitations and recommendations
Segmentation quality depends strongly on image quality and preprocessing choices.

For very large images, consider downsampling or tiling before segmentation.

License
This project is released under the MIT License.
