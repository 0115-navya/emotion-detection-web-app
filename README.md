# Emotion Detection Web App 😊

A machine learning powered web application that detects emotions from user-input text.

## Features
- Predicts emotions such as **anger, fear, joy, love, sadness, surprise**
- Displays prediction confidence and probability distribution
- Handles uncertain predictions using confidence thresholding
- Clean and interactive UI using Streamlit

## Tech Stack
- Python
- Scikit-learn
- TF-IDF Vectorizer
- Logistic Regression
- Streamlit

##  How it Works
1. Text is preprocessed and vectorized using TF-IDF
2. A trained Logistic Regression model predicts emotion probabilities
3. The app displays the most likely emotion or shows uncertainty if confidence is low

##  Use Case
Demonstrates integration of **Machine Learning with a Python-based web backend**, including model deployment and real-time inference.

##  Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
