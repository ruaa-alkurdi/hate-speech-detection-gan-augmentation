# Hate Speech Detection using GAN-based Data Augmentation

## 📌 Overview

This project addresses the class imbalance problem in hate speech detection using GAN-based data augmentation techniques.

The original dataset contains severe imbalance between hate and non-hate samples, leading to misleading high accuracy but poor minority-class detection.

To overcome this issue, the project implements:

- Vanilla GAN
- Conditional GAN (CGAN)
- Wasserstein GAN (WGAN)

Synthetic minority samples were generated to balance the dataset and improve classification performance.

---

## 📊 Dataset

- Source: Kaggle – Hate Speech and Offensive Language Dataset
- Total samples: 24,783
- Severe class imbalance:
  - Not Hate: 23,353
  - Hate: 1,435

Text data was converted to TF-IDF feature representations.

---

## 🧠 Methodology

1. Preprocessing and TF-IDF feature extraction
2. Training GAN models to generate synthetic minority samples
3. Balancing dataset using generated samples
4. Training MLP classifier under three scenarios:
   - Original Imbalanced
   - Vanilla GAN Balanced
   - CGAN Balanced
5. Performance evaluation using:
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - AUC-ROC
   - Confusion Matrix

---

## 📈 Results

Original Imbalanced Dataset:
- High accuracy
- Very low recall for hate speech (~0.07)

GAN Balanced Dataset:
- Recall improved to ~0.94
- Significant F1-score improvement
- Better minority class detection

GAN-based augmentation significantly improved classification robustness.

---

## 🛠 Tech Stack

- Python
- TensorFlow / Keras
- Scikit-learn
- TF-IDF
- GAN Architectures

---

This project was developed as part of the Special Topics in Data Science course at the University of Jordan.
# hate-speech-detection-gan-augmentation
