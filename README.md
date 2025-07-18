#  Iris Flower Classifier

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue)](https://www.python.org/)  

This project implements a simple machine learning pipeline to classify iris flowers using the **Random Forest** algorithm. The model is trained on the classic **Iris dataset** from scikit-learn.

##  Project Overview

The goal of this project is to:
- Load and preprocess the Iris dataset
- Train a Random Forest Classifier
- Save the trained model
- Make predictions on new data
- Visualize model performance

##  Algorithms Used

- **Random Forest Classifier** from `scikit-learn`

##  Dataset Details

- **Source:** `sklearn.datasets.load_iris()`
- **Features (4):**
  - Sepal Length
  - Sepal Width
  - Petal Length
  - Petal Width
- **Classes (3):**
  - Setosa (0)
  - Versicolor (1)
  - Virginica (2)

  ## Files

- `1_train_iris_model.py`: Train and save the model.
- `2_predict_iris.py`: Load the model and make predictions on new samples.
- `3_visualize_results.py`: Visualize model performance with a confusion matrix.
- `iris_model.pkl`: Saved trained model.

## How to run

1. Run `1_train_iris_model.py` to train and save the model.
2. Run `2_predict_iris.py` to test predictions.
3. Run `3_visualize_results.py` to see model accuracy and confusion matrix.

## Requirements

- Python 3.x
- scikit-learn
- joblib
- matplotlib
- seaborn
