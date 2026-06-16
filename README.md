# Aerial Scene Classifier

A computer vision project using PyTorch and transfer learning to classify satellite imagery by terrain type.
Built as a portfolio project demonstrating image classification techniques applicable to autonomous systems and drone 
navigation.

## Project Overview

This project trains a ResNet18 convolutional neural network on the EuroSAT dataset — 27,000 satellite images across 
10 land use categories collected from the Sentinel-2 satellite. The model learns to distinguish terrain types including
forests, highways, residential areas, rivers, and more.

## Use Case

Autonomous drones and robotic systems need to understand what they are looking at.
This classifier simulates that capability — given an aerial image, the model identifies the terrain type below.
This has direct applications in autonomous navigation, aerial reconnaissance, and geospatial intelligence.

## Dataset

EuroSAT RGB dataset — 27,000 images, 10 classes, 64x64 pixels per image.  
Original research: Helber et al., IEEE Journal of Selected Topics in Applied Earth Observations and
Remote Sensing, 2019.  
Accessed via Kaggle: https://www.kaggle.com/datasets/apollo2506/eurosat-dataset

## Tech Stack

- Python 3.12
- PyTorch & Torchvision
- ResNet18 (pretrained, transfer learning)
- Matplotlib
- Streamlit

## Development Log

---

### Session 1 - Project Setup
**Date:** June 16, 2026
**Summary:** Environment configuration, dependency installation, and dataset acquisition.

- Initialized GitHub repository and cloned locally
- Configured virtual environment
- Installed dependencies: PyTorch, Torch, Torchvision, Matplotlib, Pandas, Pillow, Scikit-learn, Streamlit
- Downloaded EuroSAT RGB dataset and structured into data/ directory
- Added data/ to .gitignore to exclude large files from version control
- Created notebook.ipynb with formal lab report structure
- Wrote Abstract and Introduction