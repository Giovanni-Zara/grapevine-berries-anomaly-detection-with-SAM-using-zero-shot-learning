# Anomaly Detection on Table Grapes Using SAM Segmentation and Open-Set Object Detection

## Overview

This repository contains the code and resources for my bachelor thesis, which focuses on anomaly detection in table grapes using advanced machine learning techniques. The project aims to develop an automated system capable of identifying anomalies within grape clusters by leveraging neural networks and various data pre-processing and post-processing techniques. The work is part of the European project "CANOPIES," which aims to develop a collaborative human-robot paradigm in precision agriculture for permanent crops.

## Project Structure

The project is divided into several key stages:

1. **Data Preparation and Analysis**: Initial exploration and preparation of the dataset.
2. **Segmentation and Feature Extraction**: Using Grounded Segment Anything (GSAM) for object recognition and segmentation.
3. **Classification and Anomaly Detection**: Training a neural network to classify grape clusters as healthy or anomalous.

## Dataset

The dataset consists of images of table grapes, categorized into healthy and anomalous samples. The images are organized in a hierarchical file system to separate the two types and various resolutions. The dataset is managed using a custom class created with the PyTorch framework.

### Dataset Structure

- **Healthy Samples**: Images of healthy grape clusters.
- **Anomalous Samples**: Images of grape clusters with visible anomalies.

### Dataset Splitting

The dataset is split into training (60%), validation (20%), and test (20%) sets to ensure robust model evaluation and prevent overfitting.

## Grounded Segment Anything (GSAM)

GSAM combines the capabilities of Grounding Dino (GD) for zero-shot object detection and Segment Anything (SAM) for image segmentation. This combination allows for precise object recognition and segmentation based on textual prompts.

### Grounding Dino (GD)

GD is a zero-shot detector that can classify and recognize objects not seen during training using textual prompts. It uses Vision Transformers (ViTs) to interpret textual inputs and generate precise labels and bounding boxes.

### Segment Anything (SAM)

SAM is a foundation model capable of segmenting any distinguishable entity within an image. It uses an image encoder, prompt encoder, and mask decoder to generate accurate image masks.

## Object Recognition and Segmentation

The project involves segmenting both healthy and anomalous grape clusters using GSAM. The process includes:

1. **Anomaly Segmentation**: Identifying and segmenting anomalous regions within the grape clusters.
2. **Healthy Grape Segmentation**: Segmenting healthy grape clusters using generic prompts.

### Example Illustrations

- **Anomaly Detection**: The algorithm successfully identifies and labels anomalous regions within the grape clusters.
- **Healthy Grape Detection**: The algorithm accurately segments healthy grape clusters.

## Feature Extraction

Features are extracted from the segmented image masks using a Convolutional Neural Network (CNN). The VGG16 architecture is employed for this purpose, with the top classification layer removed to focus on feature extraction.

### VGG16 Architecture

VGG16 is a deep CNN known for its performance in image recognition tasks. It consists of 13 convolutional layers, 5 max-pooling layers, and 3 fully connected layers. The model is pre-trained on ImageNet and fine-tuned for this specific task.

### Feature Extraction Process

1. **Image Preprocessing**: Convert single-channel masks to three-channel RGB images.
2. **Tensor Conversion**: Convert images to PyTorch tensors and then to NumPy arrays.
3. **Feature Extraction**: Pass the preprocessed images through VGG16 to extract features from the 'block5pool' layer.

## Classification - Anomaly Detection

The final stage involves classifying the extracted features using a simple feedforward neural network. The network consists of three fully connected layers with ReLU activation functions and a sigmoid output layer for binary classification.

### Simple Classifier Architecture

- **Input Layer**: 7x7x512 features from VGG16.
- **Hidden Layers**: Two fully connected layers with 512 and 128 neurons, respectively.
- **Output Layer**: Single neuron with a sigmoid activation function for binary classification.

### Training and Validation

The model is trained using the Binary Cross Entropy Loss (BCELoss) and optimized with the Adam optimizer. Performance is evaluated using metrics such as accuracy, precision, recall, and F1 score.

### Performance Metrics

- **Train Loss**: Loss on the training data.
- **Validation Loss**: Loss on the validation data.
- **Validation Accuracy**: Percentage of correctly classified samples in the validation set.
- **Confusion Matrix**: Detailed breakdown of true positives, false positives, true negatives, and false negatives.

## Improvements and Modifications

To address overfitting and improve generalization, dropout layers were added to the Simple Classifier. This modification resulted in more stable learning curves and better performance on unseen data.

### Dropout Implementation

- **Dropout Rate**: 50% dropout after the first and second fully connected layers.
- **Impact**: Reduced overfitting and improved model robustness.

## Conclusion

This project demonstrates the effectiveness of advanced machine learning techniques in anomaly detection for agricultural applications. The integration of GSAM for segmentation and a custom neural network for classification provides a robust solution for identifying anomalies in grape clusters. Future work includes further optimization of model parameters and extension to other crops.
