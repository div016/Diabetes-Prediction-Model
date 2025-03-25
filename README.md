# Diabetes-Prediction-Model
##Overview
This script implements a machine learning approach to predict diabetes outcomes using the diabetes dataset. It compares the performance of Logistic Regression and Linear Discriminant Analysis (LDA) for classification, evaluating how dropping certain features impacts the model, calculating expected loss with different thresholds, and comparing ROC curves for both models.

Features

Logistic Regression: A classification model used to predict the likelihood of diabetes (Outcome) based on various input features.
Linear Discriminant Analysis (LDA): A classifier that is also evaluated for predicting diabetes, compared with logistic regression using ROC curves and AUC scores.
Steps

Part A: Cross-validation and Feature Comparison
Load the diabetes dataset and prepare two versions of the feature set:
All features, including SkinThickness.
All features except SkinThickness.
Standardize the features using StandardScaler.
Perform 10-fold cross-validation for both feature sets.
Compare the mean accuracy and standard deviation of cross-validation results to evaluate the effect of dropping SkinThickness from the dataset.
Part B: Loss Function Evaluation
Split the dataset into training and testing sets.
Train a Logistic Regression model on the training set.
Predict probabilities for the test set and compute the loss for three different thresholds:
A naive baseline model (all zeros).
Threshold at 0.5.
A threshold of 0.2 (breakeven probability).
The loss is computed based on the cost of false positives (500) and false negatives (2000).
Part C: ROC Curve and AUC Comparison
Train both Logistic Regression and LDA models on the scaled training set.
Predict the probabilities of the positive class (diabetes = 1) for both models.
Compute the ROC curve for both models and calculate the AUC (Area Under the Curve).
Plot the ROC curve comparison for both models.
Requirements

pandas
numpy
scikit-learn
matplotlib
To install the required libraries, run:

pip install pandas numpy scikit-learn matplotlib
Dataset

This script uses the diabetes_dataset.csv dataset, which contains information about various health parameters and the outcome (whether the individual has diabetes or not). The dataset should include the following columns:

Pregnancies
Glucose
BloodPressure
SkinThickness
Insulin
BMI
DiabetesPedigreeFunction
Age
Outcome (target variable)
Output

Part A: Cross-validation results showing the accuracy and standard deviation for both models (with and without SkinThickness).
Part B: The computed loss for three different thresholds, including the naive baseline model.
Part C: ROC curve comparison between Logistic Regression and LDA, including AUC scores.
