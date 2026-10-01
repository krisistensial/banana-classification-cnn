# Banana Ripeness Classification using Convolutional Neural Network (CNN)

## Overview
This project implements a deep learning-based image classification system to identify the ripeness levels of bananas into four distinct categories: **Unripe (Mentah)**, **Ripe (Matang)**, **Overripe (Terlalu Matang)**, and **Rotten (Busuk)**. Built as an academic project for the Computer Vision course at Universitas Negeri Jakarta, this system leverages Transfer Learning with **MobileNetV2** and provides explainable AI insights using **Grad-CAM** activation maps to visualize the model's decision-making process.

## Dataset
* **Primary Dataset**: *Banalyzer — Banana Ripeness* containing images classified into four classes.
* **Data Cleaning**: Binarily identical duplicates (detected via MD5 hashing) and corrupt files (verified via PIL) were removed, filtering out 25 redundant/corrupted files (2.08% of the dataset).
* **Final Dataset Size**: 1175 high-quality images.
* **Class Distribution**:
  - Unripe (Mentah): 276 images
  - Ripe (Matang): 300 images
  - Overripe (Terlalu Matang): 299 images
  - Rotten (Busuk): 300 images

## Methodology
1. **Preprocessing & Augmentation**: Images were resized to 224x224. Training augmentations include Random Horizontal/Vertical Flips, Random Rotation (±15°), and Color Jitter to increase robustness against varying environmental conditions.
2. **Feature Engineering Exploration**: Explored color statistics showing a clear distinction in the Green-to-Red ratio between maturity states using PCA and t-SNE dimensional reduction.
3. **Model Architecture**: Transfer Learning using **MobileNetV2** pre-trained on ImageNet. A customized classification head was trained using Mixed Precision arithmetic.
4. **Loss Function**: Weighted Cross-Entropy Loss (weights adjusted: [0.981, 0.978, 0.978, 1.063]) combined with Label Smoothing (0.1) to counteract minor class imbalance.
5. **Optimization**: AdamW optimizer (learning rate: 0.0001, weight decay: 0.0005) with a Cosine Annealing learning rate scheduler over 10 epochs.

## Results
* **Overall Test Accuracy**: 94.35%
* **Precision (Macro Average)**: 94.59%
* **Recall (Macro Average)**: 94.44%
* **F1-Score (Macro Average)**: 94.31%

### Classification Report
```
                precision    recall  f1-score   support

Terlalu Matang     0.9737    0.8222    0.8916        45
        Matang     0.8958    0.9556    0.9247        45
         Busuk     0.9375    1.0000    0.9677        45
        Mentah     0.9767    1.0000    0.9882        42

      accuracy                         0.9435       177
     macro avg     0.9459    0.9444    0.9431       177
  weighted avg     0.9454    0.9435    0.9423       177
```

## Tech Stack
* **Programming Language**: Python
* **Deep Learning Framework**: PyTorch, Torchvision
* **Data Processing & Visualization**: Pandas, NumPy, OpenCV, PIL, Matplotlib, Seaborn, Scikit-Learn
* **Hardware Acceleration**: Mixed Precision (AMP) on NVIDIA Tesla T4 GPU
