# Flipkart Customer Satisfaction Prediction Dashboard

## 🎥 Video Demo

🎬 [Click here to watch the 5-minute walkthrough on Loom]([https://www.loom.com/share/YOUR_VIDEO_ID](https://www.loom.com/share/a007cc4420e146019c1d7e06e4a12d26?sid=56abab36-9a8c-4cc1-a4ec-be213ec7b79e)

## 🧠 Project Overview

This project presents an interactive, ML-powered **Customer Satisfaction (CSAT) Prediction Dashboard** tailored for Flipkart support analytics. It helps business teams and customer experience strategists understand the key drivers behind satisfaction scores and agent performance.

The solution leverages:

- **Machine Learning**: Imputation, classification, and predictive modeling using Random Forest and Logistic Regression.
- **Data Visualization**: Seaborn and Matplotlib for user behavior modeling (UBM) and correlation discovery.
- **Streamlit**: Fully deployable Streamlit app for real-time data exploration and model evaluation.

---

## ✨ Features

1. **CSV Upload Interface**: Upload raw customer support logs for analysis.
2. **Missing Value Imputation**: Smart filling of missing `Item_price` using trained regression models.
3. **Automated Feature Engineering**: Converts CSAT scores into sentiment labels and encodes categorical data.
4. **UBM Charts**: Insightful univariate, bivariate, and multivariate visualizations.
5. **Sentiment Mapping**: Visual verification of engineered sentiment labels.
6. **Balanced Model Training**: Includes handling class imbalance in CSAT scores.
7. **Model Evaluation Metrics**: Accuracy, Precision, F2 Score, Classification Report, and Confusion Matrix.
8. **Streamlit Deployment**: Ready for browser-based usage and cloud sharing.

---

## 🚀 Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/Flipkart-Customer-satisfaction-prediction.git
cd Flipkart-Customer-satisfaction-prediction
```
### 2. Install Dependencies
```bash
Make sure you have Python 3.8+ installed. Then run:
Copy
pip install -r requirements.
```
### 3. Run the Streamlit App
```bash
Copy
Edit
streamlit run streamlit_app.py
```
