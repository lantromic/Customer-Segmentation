
# 📊 Customer Segmentation & Prediction App

An end-to-end Machine Learning solution designed to categorize prospective customers into distinct segments based on demographic and behavioral data. The project features an interactive Streamlit web application and dimensionality reduction visualizations using **PCA**.

---

## 🎯 Project Overview

Understanding customer segments allows businesses to tailor marketing strategies, optimize product offerings, and improve customer retention. This project processes raw tabular data, cleans and imputes missing features, handles categorical variable encoding via a unified Scikit-Learn pipeline, and classifies customers into target categories using a **Random Forest Classifier**.

---

## 🛠️ Key Features

- **Automated Data Preprocessing:** Implements `SimpleImputer` and `OneHotEncoder` within a Scikit-Learn `ColumnTransformer` to handle missing values and high-cardinality categorical data seamlessly.
- **Interactive Web UI:** Built with **Streamlit**, allowing real-time single-customer segment inference via user-input forms.

---
## To Run the app run the following commands

pip install -r requirements.txt

streamlit run app.py
