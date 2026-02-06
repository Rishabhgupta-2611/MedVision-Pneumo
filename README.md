# 🫁 MedVision-Pneumo  
## A Multi-Model Web-Based Deep Learning Framework for Automated Pneumonia Detection from Chest X-Ray Images

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-DeepLearning-orange)
![Flask](https://img.shields.io/badge/Flask-WebApp-black)
![Project](https://img.shields.io/badge/B.Tech-Final%20Year-success)

---

## 📌 Abstract

**MedVision-Pneumo** is a comprehensive deep learning–driven medical imaging system developed to assist in the automated detection of pneumonia using chest X-ray images. The project integrates five convolutional neural network (CNN) architectures—both custom-built and transfer learning–based—within a unified Flask-powered web application.

The system enables users to dynamically select a deep learning model, upload an X-ray image, and obtain a real-time prediction along with a confidence score. This project demonstrates the complete lifecycle of a machine learning solution, including data preprocessing, model training, evaluation, and deployment, making it suitable for academic, research, and educational purposes.

---

## 🔑 Keywords

Pneumonia Detection, Deep Learning, Medical Image Analysis, Convolutional Neural Networks, Transfer Learning, Flask Deployment, Chest X-Ray Classification

---

## 1️⃣ Introduction

Pneumonia is a serious respiratory infection that affects the lungs and can be life-threatening if not diagnosed at an early stage. Chest X-ray imaging is one of the most widely used diagnostic tools; however, manual interpretation requires experienced radiologists and is subject to human error and variability.

With recent advancements in deep learning and computer vision, automated medical image analysis has emerged as a powerful tool for assisting healthcare professionals. MedVision-Pneumo leverages these advancements to provide an intelligent, automated system capable of classifying chest X-ray images into **Pneumonia** and **Normal** categories using multiple CNN architectures.

---

## 2️⃣ Motivation

The motivation behind this project includes:

- Limited availability of expert radiologists in rural and underdeveloped regions  
- Increasing diagnostic workload in healthcare institutions  
- Delays in disease detection leading to higher mortality rates  
- Growing demand for AI-assisted decision support systems  
- Academic interest in applying artificial intelligence to healthcare  

---

## 3️⃣ Project Objectives

- Design and implement multiple CNN-based models for pneumonia detection  
- Compare the performance of different deep learning architectures  
- Develop a web-based interface for real-time inference  
- Demonstrate the use of transfer learning in medical imaging  
- Deploy trained models in a production-ready Flask environment  

---

## 4️⃣ Dataset Information

The dataset used in this project is publicly available on **Kaggle** and contains labeled chest X-ray images for pneumonia detection.

### Dataset Source
- **Name:** Chest X-Ray Images (Pneumonia)
- **Provided by:** Paul Timothy Mooney
- **Platform:** Kaggle
- **Dataset Link:**  
  https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

### Dataset Description
The dataset consists of frontal-view chest X-ray images categorized into two classes:
- **Pneumonia**
- **Normal**

The images are divided into training, validation, and test sets and are widely used for academic research in medical image classification.

### Dataset Usage
- The full dataset is **not included** in this repository due to size and license constraints.  
- Users must download the dataset manually from Kaggle.  
- The dataset is used strictly for **academic and research purposes**.
  
### How to Download the Dataset

1. Create or log in to your Kaggle account.
2. Open the dataset link provided above.
3. Click on the **Download** button to obtain the dataset as a ZIP file.
4. Extract the dataset on your local system.

### Recommended Directory Structure

After downloading and extracting the dataset, organize it as follows:

```text
dataset/
│
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
│
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
│
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

- **Dataset Type:** Chest X-ray Images  
- **Classification Type:** Binary Classification  
- **Classes:**  
  - Pneumonia  
  - Normal  
- **Image Format:** JPEG / PNG  
- **Image View:** Frontal Chest X-rays  

### Data Preprocessing
- Image resizing to fixed input dimensions  
- Pixel normalization  
- RGB conversion  
- Batch-wise processing for efficient training  

> The dataset used is publicly available and utilized strictly for academic and research purposes.

---

## 5️⃣ System Architecture Overview

### Frontend Layer
- Developed using HTML, CSS, JavaScript, and Bootstrap  
- Enables image upload and model selection  
- Provides a clean, modern, and user-friendly interface  

### Backend Layer
- Implemented using Flask  
- Handles routing, request processing, and inference logic  
- Manages image uploads and preprocessing  

### Model Layer
- Contains multiple trained CNN models stored in .h5 format  
- Models are lazily loaded via a centralized loader for efficiency  
- Optimized for CPU-based inference to run on standard machines

---

## 6️⃣ Deep Learning Models Implemented

| Model | Description |
|------|------------|
| **Custom CNN** | Lightweight CNN built from scratch for fast inference |
| **VGG16** | Pretrained deep CNN with strong hierarchical feature extraction |
| **VGG19** | Deeper VGG architecture capable of learning complex features |
| **ResNet50** | Residual network with skip connections for deeper representations |
| **MobileNetV2** | Efficient, lightweight model suitable for real‑time inference |

Each model is trained independently and stored in `.h5` format.

---

## 7️⃣ Model Training Methodology

- **Loss Function:** Binary Cross-Entropy  
- **Optimizer:** Adam  
- **Training Strategy:**  
  - Transfer learning for pretrained models  
  - Fine-tuning of top layers  
- **Regularization:** Data augmentation and early stopping (if configured)
- **Validation:** Continuous monitoring using validation datasets  

---

## 8️⃣ Evaluation Metrics

To ensure reliable medical predictions, the following metrics were used:

- Accuracy  
- Precision  
- Recall  
- F1-Score  

These metrics provide balanced evaluation, especially important where false negatives can have serious consequences.

---

## 9️⃣ User Interface Overview

### Upload Page
- Chest X‑ray image upload with drag‑and‑drop support and preview  
- Model selection dropdown listing all available CNN architectures
- Progress indicator and loader animation during inference 
- Keyboard shortcut hints for power‑users (optional)  

### Result Page
- Display of the uploaded X‑ray image  
- Prediction result (Normal / Pneumonia)  
- Selected model name  
- Confidence score visualization  
- Option to analyze another image  

---

## 🔧 Technology Stack

- **Programming Language:** Python 3.10  
- **Deep Learning:** TensorFlow, Keras  
- **Web Framework:** Flask  
- **Frontend:** HTML5, CSS3, JavaScript  
- **Model Format:** `.h5`  
- **Runtime:** Local (CPU-based)  

---

## 🧱 Project Structure

```text
MedVision-Pneumo/
├─ app.py                         # Flask application entry point
├─ requirements.txt               # Python dependencies
├─ README.md
│
├─ dataset/                       # Local dataset (not committed in full)
│  ├─ chest_xray/                 # Original Kaggle chest X‑ray dataset
│  │  ├─ train/
│  │  ├─ val/
│  │  └─ test/
│  └─ sample_chest_xray_data/     # Small sample subset (NORMAL / PNEUMONIA)
│
├─ models/                        # Trained deep learning models (.h5)
│  └─ Model1_CNN.h5
│
├─ notebooks/                     # Jupyter / Anaconda experiments
│  └─ MedVision_Pneumo_MultiModel_Training_Evaluation.ipynb
│
├─ static/                        # Frontend assets served by Flask
│  ├─ css/
│  │  └─ styles.css               # Custom UI styling
│  ├─ gradcam/                    # Grad‑CAM result images (optional)
│  ├─ img/
│  │  ├─ gradcam_placeholder.png
│  │  └─ hero-xray.png
│  └─ uploads/                    # User‑uploaded X‑ray images (runtime)
│
├─ templates/                     # HTML templates (Jinja2)
│  ├─ index.html                  # Home + upload interface
│  ├─ result.html                 # Prediction and Grad‑CAM view
│  ├─ error.html
│  └─ 404.html
│
├─ utils/                         # Helper modules
│  ├─ gradcam.py                  # Grad‑CAM generation utilities
│  └─ model_loader.py             # Lazy loading of selected models
│
└─ venv/                          # Local virtual environment (not committed)
```

---

## 🔬 System Workflow

1. User uploads a chest X‑ray image from the web interface.  
2. The image is validated, resized, and normalized. 
3. The selected deep learning model is loaded (if not already in memory).  
4. The model performs inference on the processed image.  
5. Prediction probabilities and class label are generated. 
6. The result, confidence score, and visualization are rendered on the result page.  

---

## ▶️ How to Run the Project Locally

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/MedVision-Pneumo.git
cd MedVision-Pneumo
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run Flask Application

```bash
python app.py
```
### Step 5: Open in Browser

```text
http://127.0.0.1:5000
```

---

## 📊 Output Interpretation

| Output        | Meaning                         |
|--------------|---------------------------------|
| Normal       | No pneumonia detected           |
| Pneumonia    | Pneumonia detected              |
| Confidence % | Model prediction certainty      |

---

## ⚠️ Disclaimer

This project is an academic prototype and is not intended for clinical diagnosis or treatment decision‑making.
Predictions must not be used as a substitute for professional medical judgment. Always consult qualified medical practitioners for diagnosis and care.

---

## 🙌 Acknowledgements

1.Kaggle dataset: Chest X‑Ray Images (Pneumonia) by Paul Timothy Mooney
2.Open‑source communities behind TensorFlow, Keras, and Flask
3.Academic guides and mentors supporting this B.Tech final‑year project

---
