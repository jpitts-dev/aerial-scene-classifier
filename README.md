# Aerial Scene Classifier

A computer vision project using PyTorch and transfer learning to classify satellite imagery by terrain type.
Built as a portfolio project demonstrating image classification techniques applicable to autonomous systems and drone 
navigation.

## Project Overview

This project trains a ResNet18 convolutional neural network on the EuroSAT dataset - 27,000 satellite images across 
10 land use categories collected from the Sentinel-2 satellite. The model learns to distinguish terrain types including
forests, highways, residential areas, rivers, and more.

## Use Case

Autonomous drones and robotic systems need to understand what they are looking at.
This classifier simulates that capability, given an aerial image, the model identifies the terrain type below.
This has direct applications in autonomous navigation, aerial reconnaissance, and geospatial intelligence.

## Dataset

EuroSAT RGB dataset - 27,000 images, 10 classes, 64x64 pixels per image.  
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

### Session 2 - Data Analysis
**Date:** June 18, 2026

**Summary:** Began coding with exploratory data analysis of the EuroSAT dataset.

- Loaded dataset folder structure and confirmed all 10 terrain classes
- Counted images per class and identified slight class imbalance (2,000 to 3,000 images per class)
- Visualized class distribution as a bar chart
- Displayed sample images from each terrain class
- Identified AnnualCrop, PermanentCrop, and Pasture as visually similar classes likely to challenge the model
- Defined image transformation pipeline for ResNet18 compatibility
- Loaded dataset using ImageFolder and split into training, validation, and test sets (80/10/10)
- Loaded pretrained ResNet18 and replaced final classification layer for 10 terrain classes
- Froze pretrained layers to enable transfer of learning
- Defined CrossEntropyLoss loss function and Adam optimizer

### Session 3 - Model Training
**Date:** June 19, 2026

**Summary:** Wrote and ran the training loop, achieving 92.33% validation accuracy.

- Updated Methods section
- Revised Abstract and Introduction
- Added training section to notebook
- Trained ResNet18 transfer learning model for 5 rounds on CPU
- Achieved 91.60% training accuracy and 92.33% validation accuracy
- Saved trained model to model.pth

### Session 4 - Evaluation and Written Analysis
**Date:** June 20, 2026

**Summary:** Evaluated model performance and complete all written sections of the notebook.

- Evaluated model on test set, achieving 92.00% accuracy
- Generated per class accuracy breakdown
- Visualized per class accuracy with a bar chart
- Identified River and PermanentCrop as weakest performing classes
- Wrote Discussion section analyzing model strengths and weaknesses
- Wrote Conclusion section
- Added References section
- Added Environment section with library versions
- Cleaned up training and evaluation code for readability