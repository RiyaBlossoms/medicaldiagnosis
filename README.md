# Hybrid AI Medical Diagnosis System

A deep learning-based medical image classification project that uses a hybrid CNN–Transformer architecture to classify brain MRI scans into different tumor categories.

## Overview

This project aims to assist medical image analysis through an automated brain tumor classification system. It combines the local feature extraction capabilities of Convolutional Neural Networks (CNNs) with the global contextual understanding of Vision Transformers.

The system also incorporates Grad-CAM visualizations, prediction confidence scores, and AI-assisted clinical guidance to make the model's outputs easier to interpret.

## Key Features

- **Brain Tumor Classification:** Classifies MRI images into four categories.
- **Hybrid Deep Learning Architecture:** Combines EfficientNet-B3 and Vision Transformer.
- **Cross-Attention Fusion:** Integrates local and global image features.
- **Confidence Scores:** Displays prediction probabilities for the output classes.
- **Grad-CAM Visualization:** Highlights image regions that influence the model's predictions.
- **Clinical Guidance:** Provides AI-assisted guidance based on classification results.

## Tumor Categories

The model classifies MRI scans into:

1. Glioma
2. Meningioma
3. Pituitary Tumor
4. No Tumor

## Model Architecture

The proposed architecture consists of three main components:

**1. EfficientNet-B3 (CNN)**

Extracts local visual features such as edges, textures, shapes, and structural patterns from MRI images.

**2. Vision Transformer (ViT)**

Captures global contextual information by dividing images into patches and learning relationships between them using self-attention.

**3. Cross-Attention Fusion**

Combines the features extracted by EfficientNet-B3 and the Vision Transformer to create a unified feature representation for classification.

The fused features are passed through a classification head to generate the final prediction probabilities.

## Technology Stack

- Python 3.11
- PyTorch
- Torchvision
- TIMM
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn
- PIL (Pillow)

## Project Structure

```text
medicaldiagnosis/
│
├── data/
│   └── brain_tumor/
│       ├── train/
│       ├── val/
│       └── test/
│
├── results/
│   ├── Te-gl_1_gradcam.png
│   ├── Te-gl_1_probs.png
│   ├── Te-gl_22_gradcam.png
│   └── Te-gl_22_probs.png
│
├── clinical_guidance.py
├── config.py
├── dataset.py
├── download_datasets.py
├── gradcam.py
├── inference.py
├── main.py
├── model.py
├── train.py
├── utils.py
└── training_graphs.png
```

*Note: The dataset directory and trained model checkpoints are not included in this GitHub repository.*

## Installation

Clone the repository:

```bash
git clone https://github.com/RiyaBlossoms/medicaldiagnosis.git
```

Move into the project directory:

```bash
cd medicaldiagnosis
```

Install the required libraries:

```bash
pip install torch torchvision timm opencv-python numpy matplotlib scikit-learn pillow kagglehub
```

## Dataset

The project uses a Brain Tumor MRI dataset containing more than 7,000 images across four classes.

The dataset is divided into:

- Training set – Used to train the model.
- Validation set – Used to monitor model performance and tune hyperparameters.
- Testing set – Used for final evaluation.

Images are resized to 224 × 224 pixels and normalized before being passed to the model.

The dataset is not hosted in this repository. Download or obtain the dataset separately and organize it according to the expected directory structure.

## Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning Rate | 0.0001 |
| Batch Size | 2 |
| Loss Function | Cross Entropy |
| Scheduler | Cosine Annealing |
| Dropout | 0.3 |
| Image Resolution | 224 × 224 |
| Activation | GELU |

## Training and Inference

The main training and inference scripts are:

```bash
python train.py
```

```bash
python inference.py
```

Ensure that the dataset paths and trained model checkpoint paths are configured correctly before running these scripts. Refer to `config.py` and the respective scripts for any additional arguments or setup requirements.

## Explainability

Grad-CAM is used to visualize the regions of MRI images that contribute to the model's predictions.

Prediction probability visualizations are also generated to help understand the model's classification output.

Generated visualizations are stored in the `results/` directory.

## Future Scope

- Extend the framework to additional medical imaging datasets.
- Improve model generalization across different imaging sources.
- Develop an interactive interface for medical image analysis.
- Explore additional explainability techniques.

## Disclaimer

This project is developed for educational and research purposes. Its predictions and clinical guidance are not a substitute for professional medical diagnosis, radiological interpretation, or advice from qualified healthcare professionals.

## Author

**Riya Blossoms**

GitHub: [@RiyaBlossoms](https://github.com/RiyaBlossoms)
