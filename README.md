# Multi-Modal Pedestrian Detection with Cross-Attention Fusion

This repository contains the complete code and documentation for a deep learning project focused on robust pedestrian detection in adverse lighting conditions by fusing RGB and Infrared (IR) imagery.

## Overview

This project tackles the challenge of low-light object detection by leveraging the complementary strengths of visible-spectrum (RGB) and thermal (IR) sensors. The core of the model is a sophisticated fusion architecture that uses a bidirectional cross-attention mechanism to intelligently merge features from both modalities. The final detection is performed by an efficient, single-stage YOLO-style head. The entire pipeline is implemented in PyTorch and trained on the LLVIP dataset.

The project not only focuses on the final model but also details the iterative process of development, including debugging training instabilities like gradient explosion, implementing advanced learning rate schedulers, and analyzing model behavior through attention visualization.

## Key Features

* **Dual-Stream Backbone:** Uses two parallel ResNet-18 backbones to extract features from RGB and IR streams independently.
* **Bidirectional Cross-Attention:** A novel fusion module where each modality queries the other to create enhanced feature representations.
* **Gated Fusion:** A learned convolutional gating mechanism adaptively combines the original and enhanced features into a single, unified map.
* **YOLO-Style Detection Head:** An efficient, single-stage detector for fast and accurate predictions.
* **Advanced Training Strategy:** Implements progressive unfreezing, a cosine scheduler with warmup, differential learning rates, and gradient clipping to ensure stable training.

## Project Files

This repository includes the following key components:

* **`Pedestrian Detection.ipynb`**: The main Jupyter Notebook containing the complete, end-to-end Python code for the project. This includes data loading, model architecture definitions, training loops, evaluation logic, and visualization functions.
* **`Pedestrian Detection Report.pdf`**: A comprehensive academic report detailing the project's background, methodology, architecture, loss functions, training procedure, and a discussion of the results and future work.
* **`Pedestrian Detection Presentation.pdf`**: A presentation slide deck summarizing the project's key aspects, suitable for academic or technical presentations.

## Setup and Usage

### Dependencies

The project is built using Python 3 and PyTorch. The main dependencies are listed below and can be installed via pip:

```bash
pip install torch torchvision torchaudio
pip install timm
pip install numpy
pip install matplotlib
pip install opencv-python
pip install torchmetrics
pip install transformers