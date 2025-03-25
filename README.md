# Diabetes-Prediction-Model
## Overview
This script implements a machine learning approach to predict diabetes outcomes using a diabetes dataset (diabetes-dataset.csv). It compares the performance of Logistic Regression and Linear Discriminant Analysis (LDA) for classification, evaluating how dropping certain features impacts the model, calculating expected loss with different thresholds, and comparing ROC curves for both models.

## Features
- **Logistic Regression:** A model used to predict the probability of diabetes (Outcome) based on various input features.
- **Linear Discriminant Analysis (LDA):** Another classification model that is compared with Logistic Regression using ROC curves and AUC scores.

## Steps

### Part A: Evaluating the Impact of Dropping a Feature
- **Objective**: Determine whether removing the `SkinThickness` feature improves the performance of the Logistic Regression model.
- **Approach**:
  - Load the dataset and create two feature sets:
    1. All features, including `SkinThickness`.
    2. All features except `SkinThickness`.
  - Standardize the features using `StandardScaler`.
  - Perform 10-fold cross-validation for both models to compare their performance based on mean accuracy.

### Part B: Calculating and Comparing Loss
- **Objective**: Calculate the expected loss for different thresholds based on false positives (500) and false negatives (2000) and compare it with different decision thresholds.
- **Approach**:
  - Train a Logistic Regression model using the better-performing feature set from Part A.
  - Compute the expected loss for:
    1. A baseline model (predicting all zeros).
    2. A model with a threshold of 0.5.
    3. A model with a calculated "breakeven" threshold.
  - Compare the losses on the test set.

### Part C: Comparing Model Performance with ROC Curves
- **Objective**: Compare the performance of Logistic Regression and LDA using ROC curves and AUC scores.
- **Approach**:
  - Train both Logistic Regression and LDA models on the test set.
  - Plot the ROC curves for both models and calculate the AUC to compare their effectiveness in classifying diabetes.

## Requirements
- pandas
- numpy
- scikit-learn
- matplotlib

To install the required libraries, run:

```bash
pip install pandas numpy scikit-learn matplotlib
