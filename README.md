# Aerial Scene Classifier

A computer vision project using PyTorch and transfer learning to classify satellite imagery by terrain type.
Built as a portfolio project that demonstrates image classification techniques applicable to autonomous systems and 
drone navigation.

## Project Overview

This project trains ResNet18, a neural network designed to recognize patterns in images, on the EuroSAT dataset 
comprised of 27,000 satellite images across 10 land use categories collected from the Sentinel-2 satellite. 
The model learns to distinguish terrain types including forests, highways, residential areas, rivers, and more.

## Use Case

Identifying terrain from aerial imagery is a core challenge in autonomous drone navigation and geospatial intelligence. 
This project explores that problem using satellite imagery and transfer learning.

## Dataset

EuroSAT RGB dataset - 27,000 images, 10 classes, 64x64 pixels per image.  
Original research: Helber et al., IEEE Journal of Selected Topics in Applied Earth Observations and
Remote Sensing, 2019.  
Accessed via Kaggle: https://www.kaggle.com/datasets/apollo2506/eurosat-dataset

## Results

The model achieved 92.74% accuracy on the test set across 10 terrain categories. The strongest performance was on 
SeaLake (98.55%) and Forest (96.97%) categories. The most challenging category was Highway (85.87%), likely due to 
roads appearing as thin linear features that could be confused with other classes at this resolution.

## Tech Stack

- Python 3.12
- PyTorch & Torchvision
- ResNet18 (pretrained, transfer learning)
- Matplotlib

---

## Development Log

### Session 1 - Project Setup
**Date:** June 16, 2026

**Summary:** Configured environment/ notebook, and downloaded dataset.

- Created the GitHub repository and cloned it locally
- Set up a virtual environment for the project
- Installed dependencies: PyTorch, Torch, Torchvision, Matplotlib, Pandas, Pillow, Scikit-learn
- Downloaded EuroSAT RGB dataset and organized it into the data/ directory
- Added data/ to .gitignore to keep the dataset out of version control
- Created notebook.ipynb and established the lab report structure
- Wrote Introduction and Methods sections

### Session 2 - Data Analysis
**Date:** June 18, 2026

**Summary:** Began exploratory data analysis of the EuroSAT dataset.

- Loaded dataset folder and confirmed all 10 terrain classes
- Counted images per class and found a slight class imbalance (2,000 to 3,000 images per class)
- Visualized class distribution as a bar chart
- Displayed sample images from each terrain class
- Noticed that AnnualCrop, PermanentCrop, and Pasture look very similar, likely to confuse the model
- Defined image transformation pipeline to match what ResNet18 expects
- Loaded dataset and split into training, validation, and test sets at 80/10/10
- Loaded pretrained ResNet18 and replaced final classification layer for 10 terrain classes
- Froze all pretrained layers so only the final layer would be trained
- Set up CrossEntropyLoss and Adam optimizer

### Session 3 - Model Training
**Date:** June 19, 2026

**Summary:** Wrote and ran the training loop, achieving 92.33% validation accuracy.

- Revised Introduction and Methods section
- Added training section to notebook
- Trained the model for 5 rounds on CPU, which took ~30 mins
- Finished with 91.60% training accuracy and 92.33% validation accuracy
- Saved the trained model weights to model.pth

### Session 4 - Evaluation and Written Analysis
**Date:** June 20, 2026

**Summary:** Evaluated the model on the test set and completed all written sections of the notebook.

- Evaluated model on test set, achieving 92.00% accuracy
- Generated per class accuracy breakdown
- Visualized per class accuracy with a bar chart
- River and PermanentCrop were weakest performing classes
- Wrote Discussion, Conclusion, Abstract, References, and Environment sections
- Cleaned up all code cells for readability and clarity

### Session 5 - Code Cleanup and Standalone Script
**Date:** June 20, 2026

**Summary:** Finalized notebook comments, completed written sections review, and built standalone training script.

- Added comments to all code cells in the notebook
- Reviewed and finalized all written sections
- Built *train.py* as a standalone training script
- Verified *train.py* runs correctly from the terminal

### Session 6 - Revisited Review
**Date:** July 22, 2026

**Summary:** A thorough review of completed project

- Reviewed README and made edits for clarity
- Retrained model with fixed random seed for reproducibility
- Updated Results, Discussion, Conclusion, Abstract to reflect new test accuracy of 92.74%
- Updated README results section