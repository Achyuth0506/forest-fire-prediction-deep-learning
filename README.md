# 🔥 Forest Fire Prediction Using Deep Learning

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.4+](https://img.shields.io/badge/TensorFlow-2.4+-orange.svg)](https://tensorflow.org)
[![OpenCV 4.5+](https://img.shields.io/badge/OpenCV-4.5+-green.svg)](https://opencv.org)
[![Accuracy](https://img.shields.io/badge/Accuracy-94%25-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub stars](https://img.shields.io/github/stars/yourusername/forest-fire-prediction-deep-learning?style=social)]()
[![GitHub forks](https://img.shields.io/github/forks/yourusername/forest-fire-prediction-deep-learning?style=social)]()

*Early Detection Saves Lives and Ecosystems*

</div>

## 📌 Table of Contents
- [🎯 Overview](#-overview)
- [✨ Features](#-features)
- [📊 Results](#-results)
- [🛠️ Installation](#️-installation)
- [🚀 Quick Start](#-quick-start)
- [🏗️ Architecture](#️-architecture)
- [📁 Dataset](#-dataset)
- [📈 Performance](#-performance)
- [📸 Screenshots](#-screenshots)
- [👥 Team](#-team)
- [📄 Documentation](#-documentation)
- [🤝 Contributing](#-contributing)
- [📜 License](#-license)
- [📞 Contact](#-contact)

## 🎯 Overview

This project implements a **Deep Learning-based Forest Fire Detection System** that achieves **94% accuracy** in early fire prediction. The system uses **Convolutional Neural Networks (CNN)** with **MobileNetV2** architecture and integrates **real-time WhatsApp alerts** for immediate response.

### **Key Innovations**
- ✅ Real-time fire detection from camera feeds
- ✅ Automated alert system via WhatsApp
- ✅ 94% accuracy on diverse test datasets
- ✅ Optimized for edge deployment
- ✅ Early warning system for disaster management

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
| **Inference Time** | ~0.2s | On CPU |
| **Model Size** | 85MB | Optimized for deployment |

## 🛠️ Installation

### **Prerequisites**
- Python 3.7 or higher
- 4GB RAM minimum
- Webcam (for real-time detection)
- WhatsApp Web access (for alerts)

### **Method 1: Using pip**
```bash
# Clone the repository
git clone https://github.com/yourusername/forest-fire-prediction-deep-learning.git
cd forest-fire-prediction-deep-learning

# Install dependencies
pip install -r requirements.txt

# Download pre-trained model
python src/download_model.py
