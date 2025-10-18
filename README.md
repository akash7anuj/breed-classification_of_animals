# 🐄 Indian Bovine Breeds Detection System

## Overview

This project is an **AI-powered cattle breed detection system** that identifies Indian bovine breeds (cows and buffaloes) from images and displays the results on a **web-based dashboard** in real-time. It combines **deep learning (ResNet50)** for breed classification and **YOLOv8** for object detection.

![Indian Bovine Breeds Detection System](image\dashboard.png)
---

## 🎯 Objective

* Automatically classify the breed of Indian cattle from images.
* Display results on a **user-friendly web dashboard**.
* Handle multiple cattle in a single image.
* Provide detection confidence control for filtering results.

---

## 📦 Dataset

* Source: Kaggle ([Indian Bovine Breeds Dataset](https://www.kaggle.com/datasets/lukex9442/indian-bovine-breeds))
* Structure: Each subfolder represents a unique breed (e.g., Gir, Sahiwal, Tharparkar, Rathi, etc.)

### Data Cleaning & Preprocessing

* **Duplicate Removal** – Using MD5 hashing.
* **Blur Detection** – Laplacian variance method.
* **Corrupt & Small Image Filtering** – Remove unreadable or low-resolution images.
* **Image Standardization** – Resized to 224×224 pixels, RGB format.
* **Class Balancing** – Augmentation: flips, rotations, brightness changes.

---

## 🧮 Dataset Splitting

* **Training:** 70%
* **Validation:** 20%
* **Testing:** 10%

This ensures proper training, hyperparameter tuning, and evaluation on unseen data.

---

## 🧠 Model

* **Base Model:** ResNet50 (pretrained on ImageNet)
* **Configuration:**

  * Freeze all base layers.
  * Replace final fully connected layer with output for all bovine breeds.
  * Loss function: CrossEntropyLoss
  * Optimizer: Adam
* **Training Device:** GPU (CUDA) if available
* **Accuracy:** 95–98% depending on dataset quality

---

## ⚙️ Deployment

### Models

1. **Breed Classification:** ResNet50 trained on bovine dataset.
2. **Object Detection:** YOLOv8 for detecting cattle in images.

### Web Application

* Built with **Flask** and **Bootstrap 5**
* Features:

  * Image upload
  * Detection confidence slider
  * Displays detected breeds
  * Displays other objects detected
  * Shows result image with bounding boxes

---

## 🔧 Usage

### 1. Install dependencies

```bash
pip install torch torchvision ultralytics flask opencv-python pillow
```

### 2. Run the Flask app

```bash
python app.py
```

### 3. Open Dashboard

* Visit: [http://127.0.0.1:5000](http://127.0.0.1:5000)
* Upload an image of cattle.
* Adjust **confidence threshold** if needed.
* Click **Analyze Image** to get results.

---

## 🖥️ Screenshot

*(Add a screenshot of your dashboard here with detection results)*

---

## 🧩 How It Works

1. **Upload Image** → Flask receives it.
2. **YOLOv8** detects all objects in the image.
3. **Filter cattle** → crop detected cows/buffaloes.
4. **ResNet50** classifies breed for each crop.
5. **Draw bounding boxes** with breed name + confidence.
6. **Display results** on dashboard with detected breeds and other objects.

---

## 🚀 Key Features

* Handles multiple cattle in one image.
* Confidence threshold adjustment.
* Displays other objects detected.
* Clean and interactive web dashboard.
* High classification accuracy (~95–98%).

---

## 🛠️ Folder Structure

```
project/
│
├─ app.py                   # Flask web application
├─ best_model.pth           # Trained ResNet50 model
├─ static/                  # Folder to store uploaded and output images
├─ templates/
│   └─ index.html           # Web dashboard template
├─ dataset/                 # Raw and cleaned dataset
└─ README.md
```

---

## ⚠️ Notes

* Ensure **CUDA** is available for faster inference.
* The **confidence slider** helps filter out low-confidence detections.
* Only cows, buffaloes, and other bovine animals are classified; other objects are listed separately.

---

## 📚 References

* ResNet50: [https://arxiv.org/abs/1512.03385](https://arxiv.org/abs/1512.03385)
* YOLOv8: [https://ultralytics.com/](https://ultralytics.com/)
* Kaggle Dataset: [Indian Bovine Breeds](https://www.kaggle.com/datasets)
