# 💳 Credit Card Fraud Detection using Machine Learning & Flask

This project is an end-to-end **Machine Learning System** that detects fraudulent credit card transactions.  
It includes **data preprocessing, EDA, model training, SMOTE oversampling**, and a **Flask web application** for real-time predictions.

---

## 📌 Project Overview

Credit card fraud detection is a highly imbalanced classification problem.  
This project uses the popular **Credit Card Fraud Dataset** to build multiple ML models and select the best-performing one.

---

## 🚀 Features

✔ Data Cleaning & Preprocessing  
✔ Outlier Detection using Boxplots  
✔ Imbalanced Data Handling using **SMOTE**  
✔ Multiple ML Model Training:  
- Logistic Regression  
- KNN  
- Decision Tree  
- Random Forest  

✔ Performance Evaluation (Accuracy, ROC-AUC, Classification Report)  
✔ Confusion Matrix Heatmaps  
✔ Full ML Pipeline using Scikit-Learn  
✔ Best Model Selected → **Random Forest Classifier**  
✔ Flask Web App for real-time fraud prediction  
✔ Model saved using `pickle`  

---

## 🧠 Technologies Used

| Category | Tools |
|---------|-------|
| Programming | Python |
| ML Libraries | scikit-learn, imbalanced-learn, pandas, numpy |
| Visualization | Matplotlib, Seaborn |
| Deployment | Flask |
| Model Saving | Pickle |

---

## 📊 Model Training

Models trained with SMOTE-balanced dataset:

| Model | Accuracy | ROC-AUC |
|-------|----------|----------|
| Logistic Regression | xx | xx |
| Decision Tree | xx | xx |
| KNN | xx | xx |
| **Random Forest** | **(best)** | **(best)** |

> Random Forest achieved the highest performance and is used inside the Flask application.

---

## 🧪 Dataset

The dataset used is the **Credit Card Fraud Dataset**:  
Contains 284,807 transactions with only 492 fraud cases (highly imbalanced).  
Features are anonymized using PCA transformations (V1–V28).

---

## 🖥 Flask Web Application

The web interface allows users to input transaction details and get an instant fraud prediction.

### 🔧 How to Run the Web App

```bash
pip install -r requirements.txt
python app.py
