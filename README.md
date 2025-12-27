# 🔥 Forest Fire Prediction Using Deep Learning

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.4+](https://img.shields.io/badge/TensorFlow-2.4+-orange.svg)](https://tensorflow.org)
[![OpenCV 4.5+](https://img.shields.io/badge/OpenCV-4.5+-green.svg)](https://opencv.org)
[![Accuracy](https://img.shields.io/badge/Accuracy-94%25-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

*Early Detection Saves Lives and Ecosystems*

## 📌 Table of Contents
- [🎯 Overview](#-overview)
- [✨ Features](#-features)
- [📊 Results](#-results)
- [🛠️ Installation](#️-installation)
- [🏗️ Architecture](#️-architecture)
- [📁 Dataset](#-dataset)
- [👥 Contributors](#-team)
- [📄 Acknowledgment](#-Acknowledgment)
- [📜 License](#-license)
- [📞 Contact](#-contact)

## 🎯 Overview

This project implements a **Deep Learning-based Forest Fire Detection System** that achieves **94% accuracy** in early fire prediction. The system uses **Convolutional Neural Networks (CNN)** with **MobileNetV2** architecture and integrates **real-time WhatsApp alerts** for immediate response.

### **Academic Context**
- **Project Type**: Final Year BSc Project
- **Institution**: Loyola Academy (Autonomous), affiliated to Osmania University
- **Department**: Computer Data Science & Data Analytics Engineering
- **Academic Year**: 2019-2020
- **Guide**: Mrs. V. Shirisha Reddy
- **HOD**: Mrs. V. Theresa Vinayasheela

## ✨ Features

| Feature | Description | Technology Used |
|---------|-------------|-----------------|
| **Fire Detection** | Real-time detection from images/video | CNN + MobileNetV2 |
| **Alert System** | Automated WhatsApp notifications | Selenium WebDriver |
| **High Accuracy** | 94% classification accuracy | TensorFlow + Keras |
| **Data Augmentation** | Improved model generalization | ImageDataGenerator |
| **Web Interface** | Easy-to-use interface | OpenCV + Matplotlib |
| **Model Optimization** | Fast inference on edge devices | Model quantization |

## 📊 Results

| Metric | Value | Remark |
|--------|-------|--------|
| **Accuracy** | 94% | Test dataset |
| **F1-Score** | 0.95 | Fire class |
| **Precision** | 0.93 | Fire detection |
| **Recall** | 0.96 | Fire detection |
- **Inference Time**: ~0.2s (On CPU)
- **Model Size**: 85MB (Optimized for deployment)

## 🛠️ Installation

### **Prerequisites**
- Python 3.7 or higher
- 4GB RAM minimum
- Webcam (for real-time detection)
- WhatsApp Web access (for alerts)

## **Method 1: Using pip**
### **Clone the repository**
git clone https://github.com/yourusername/forest-fire-prediction-deep-learning.git
cd forest-fire-prediction-deep-learning

### **Install dependencies**
pip install -r requirements.txt

### **Download pre-trained model**
python src/download_model.py


## Method 2: Using Anaconda
### Create conda environment
conda create -n forest-fire python=3.8
conda activate forest-fire

### Install packages
conda install tensorflow-gpu opencv numpy pandas scikit-learn matplotlib
pip install selenium imutils


## 🏗️ Architecture
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Input Source  │───▶│   Preprocessing │───▶│  CNN Model     │
│  (Image/Video)  │    │   (224x224 RGB) │    │  (MobileNetV2) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  WhatsApp Alert │◀───│  Alert System   │◀───│  Classification │
│      Module     │    │    (Selenium)   │    │    (Fire/NoFire)│
└─────────────────┘    └─────────────────┘    └─────────────────┘

## CNN Architecture Details

- Base Model: MobileNetV2 (pre-trained on ImageNet)
- Input Size: 224 × 224 × 3
- Layers: 5 Convolution, 3 Pooling, 3 Fully Connected
- Dropout: 50% to prevent overfitting
- Optimizer: Adam (learning rate: 1e-4)
- Loss Function: Sparse Categorical Crossentropy

  ## 📁 Dataset
- Dataset Information
- Total Images: 10,000+
- Classes: Fire (5,000), No Fire (5,000)
- Sources: Real forest fire images, simulation data
- Format: JPG (224x224 pixels)
- The dataset is available at: [Google Drive](https://drive.google.com/drive/folders/1g7ovn6PPlJwBxj5cJDk8AKeLNigblPMO?usp=sharing)

  ## 👥 Contributors
- John Joshua Obed.P 
- Achyuth Kumar.U

  ## 🙏 Acknowledgments
- Mrs. V. Shirisha Reddy (Project Guide)
- Mrs. V. Theresa Vinayasheela (HOD)
- Loyola Academy Degree and PG College

  ## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Contact
For questions or collaborations:
Achyuth Kumar.U: achyuthundrakonda@gmail.com




