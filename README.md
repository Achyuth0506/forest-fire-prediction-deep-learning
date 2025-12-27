{\rtf1\ansi\ansicpg1252\cocoartf2867
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww38200\viewh21400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 # \uc0\u55357 \u56613  Forest Fire Prediction Using Deep Learning\
\
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)\
[![TensorFlow 2.4+](https://img.shields.io/badge/TensorFlow-2.4+-orange.svg)](https://tensorflow.org)\
[![OpenCV 4.5+](https://img.shields.io/badge/OpenCV-4.5+-green.svg)](https://opencv.org)\
[![Accuracy](https://img.shields.io/badge/Accuracy-94%25-brightgreen.svg)]()\
\
*Early Detection Saves Lives and Ecosystems*\
\
</div>\
\
## \uc0\u55357 \u56524  Table of Contents\
- [\uc0\u55356 \u57263  Overview](#-overview)\
- [\uc0\u10024  Features](#-features)\
- [\uc0\u55357 \u56522  Results](#-results)\
- [\uc0\u55357 \u57056 \u65039  Installation](#\u65039 -installation)\
- [\uc0\u55357 \u56960  Quick Start](#-quick-start)\
- [\uc0\u55356 \u57303 \u65039  Architecture](#\u65039 -architecture)\
- [\uc0\u55357 \u56513  Dataset](#-dataset)\
- [\uc0\u55357 \u56520  Performance](#-performance)\
- [\uc0\u55357 \u56568  Screenshots](#-screenshots)\
- [\uc0\u55357 \u56421  Team](#-team)\
- [\uc0\u55357 \u56516  Documentation](#-documentation)\
- [\uc0\u55358 \u56605  Contributing](#-contributing)\
- [\uc0\u55357 \u56540  License](#-license)\
- [\uc0\u55357 \u56542  Contact](#-contact)\
\
## \uc0\u55356 \u57263  Overview\
\
This project implements a **Deep Learning-based Forest Fire Detection System** that achieves **94% accuracy** in early fire prediction. The system uses **Convolutional Neural Networks (CNN)** with **MobileNetV2** architecture and integrates **real-time WhatsApp alerts** for immediate response.\
\
### **Key Innovations**\
- \uc0\u9989  Real-time fire detection from camera feeds\
- \uc0\u9989  Automated alert system via WhatsApp\
- \uc0\u9989  94% accuracy on diverse test datasets\
- \uc0\u9989  Optimized for edge deployment\
- \uc0\u9989  Early warning system for disaster management\
\
### **Academic Context**\
- **Project Type**: Final Year BSc Project\
- **Institution**: Loyola Academy (Autonomous), affiliated to Osmania University\
- **Department**: Computer Data Science & Data Analytics Engineering\
- **Academic Year**: 2019-2020\
- **Guide**: Mrs. V. Shirisha Reddy\
- **HOD**: Mrs. V. Theresa Vinayasheela\
\
## \uc0\u10024  Features\
\
| Feature | Description | Technology Used |\
|---------|-------------|-----------------|\
| **Fire Detection** | Real-time detection from images/video | CNN + MobileNetV2 |\
| **Alert System** | Automated WhatsApp notifications | Selenium WebDriver |\
| **High Accuracy** | 94% classification accuracy | TensorFlow + Keras |\
| **Data Augmentation** | Improved model generalization | ImageDataGenerator |\
| **Web Interface** | Easy-to-use interface | OpenCV + Matplotlib |\
| **Model Optimization** | Fast inference on edge devices | Model quantization |\
\
## \uc0\u55357 \u56522  Results\
\
| Metric | Value | Remark |\
|--------|-------|--------|\
| **Accuracy** | 94% | Test dataset |\
| **F1-Score** | 0.95 | Fire class |\
| **Precision** | 0.93 | Fire detection |\
| **Recall** | 0.96 | Fire detection |\
| **Inference Time** | ~0.2s | On CPU |\
| **Model Size** | 85MB | Optimized for deployment |\
\
## \uc0\u55357 \u57056 \u65039  Installation\
\
### **Prerequisites**\
- Python 3.7 or higher\
- 4GB RAM minimum\
- Webcam (for real-time detection)\
- WhatsApp Web access (for alerts)\
\
### **Method 1: Using pip**\
```bash\
# Clone the repository\
git clone https://github.com/yourusername/forest-fire-prediction-deep-learning.git\
cd forest-fire-prediction-deep-learning\
\
# Install dependencies\
pip install -r requirements.txt\
\
# Download pre-trained model\
python src/download_model.py\
\
## \uc0\u55356 \u57303 \u65039  Architecture\
\
### System Design\
\
\uc0\u9484 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9488     \u9484 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9488     \u9484 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9488 \
\uc0\u9474    Input Source  \u9474 \u9472 \u9472 \u9472 \u9654 \u9474    Preprocessing \u9474 \u9472 \u9472 \u9472 \u9654 \u9474   CNN Model     \u9474 \
\uc0\u9474   (Image/Video)  \u9474     \u9474    (224x224 RGB) \u9474     \u9474   (MobileNetV2) \u9474 \
\uc0\u9492 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9496     \u9492 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9496     \u9492 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9496 \
                                                       \uc0\u9474 \
\uc0\u9484 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9488     \u9484 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9488     \u9484 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9488 \
\uc0\u9474   WhatsApp Alert \u9474 \u9664 \u9472 \u9472 \u9472 \u9474   Alert System   \u9474 \u9664 \u9472 \u9472 \u9472 \u9474   Classification \u9474 \
\uc0\u9474       Module     \u9474     \u9474     (Selenium)   \u9474     \u9474     (Fire/NoFire)\u9474 \
\uc0\u9492 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9496     \u9492 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9496     \u9492 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9496 \
\
\
### **CNN Architecture Details**\
\
- **Base Model**: MobileNetV2 (pre-trained on ImageNet)\
- **Input Size**: 224 \'d7 224 \'d7 3\
- **Layers**: 5 Convolution, 3 Pooling, 3 Fully Connected\
- **Dropout**: 50% to prevent overfitting\
- **Optimizer**: Adam (learning rate: 1e-4)\
- **Loss Function**: Sparse Categorical Crossentropy\
\
## \uc0\u55357 \u56513  Dataset\
\
- Fire and No-Fire image dataset\
- Includes real fire scenes and simulated forest fire conditions  \
- Dataset link:  \
  https://drive.google.com/file/d/1v68ShtWeZWAL28I1kNzsR1-8IfECPV6m/view\
\
\
\
}