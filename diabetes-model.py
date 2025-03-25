import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
################################ Part A: #############################
print("############################ Part 3A: #############################")

df = pd.read_csv('ieor142/diabetes_dataset.csv')

X_all = df.drop(columns=["Outcome"]) 

# exclude SkinThickness
X_without_skinthickness = df.drop(columns=["Outcome", "SkinThickness"])  
y = df["Outcome"]

scaler = StandardScaler()
X_all_scaled = scaler.fit_transform(X_all)
X_without_skinthickness_scaled = scaler.fit_transform(X_without_skinthickness)

logreg = LogisticRegression(max_iter=10000)

# perform 10-fold cross-validation for both models
cv_scores_all = cross_val_score(logreg, X_all_scaled, y, cv=10, scoring='accuracy')
cv_scores_without_skinthickness = cross_val_score(logreg, X_without_skinthickness_scaled, y, cv=10, scoring='accuracy')

mean_score_all = np.mean(cv_scores_all)
std_score_all = np.std(cv_scores_all)
mean_score_without_skinthickness = np.mean(cv_scores_without_skinthickness)
std_score_without_skinthickness = np.std(cv_scores_without_skinthickness)

print(f"Model with all features (including SkinThickness): Mean accuracy = {mean_score_all:.4f}, Std deviation = {std_score_all:.4f}")
print(f"Model without SkinThickness: Mean accuracy = {mean_score_without_skinthickness:.4f}, Std deviation = {std_score_without_skinthickness:.4f}")

if mean_score_without_skinthickness > mean_score_all:
    print("Dropping SkinThickness improves the model performance.")
else:
    print("Dropping SkinThickness does not improve the model performance.")

################################ Part B: #############################
print("############################ Part 3B: #############################")
df2 = pd.read_csv('ieor142/diabetes_dataset.csv')

X = df2.drop(columns=["Outcome", "SkinThickness"])  
y = df2["Outcome"]

# training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

y_prob = model.predict_proba(X_test)[:, 1]  

def calculate_loss(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()

    loss = FP * 500 + FN * 2000  
    return loss

# compute expected losses for different models
p_breakeven = 0.2
loss_baseline = calculate_loss(y_test, np.zeros_like(y_test), threshold=0.5)  
loss_p_0_5 = calculate_loss(y_test, y_prob, threshold=0.5)  
loss_p_breakeven = calculate_loss(y_test, y_prob, threshold=0.2) 

print(f"Loss of Naive Baseline Model: {loss_baseline}")
print(f"Loss of Model with p = 0.5: {loss_p_0_5}")
print(f"Loss of Model with p = {p_breakeven}: {loss_p_breakeven}")

print("############################ Part 3C: #############################")
df3 = pd.read_csv('ieor142/diabetes_dataset.csv')


# fefine features and target
X = df3.drop(columns=['Outcome', 'SkinThickness']) 
y = df3['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_reg = LogisticRegression(random_state=42)
lda = LinearDiscriminantAnalysis()

log_reg.fit(X_train_scaled, y_train)
lda.fit(X_train_scaled, y_train)

# get predicted probabilities for the positive class (diabetes = 1)
log_reg_probs = log_reg.predict_proba(X_test_scaled)[:, 1]
lda_probs = lda.predict_proba(X_test_scaled)[:, 1]

# ROC curves
fpr_log_reg, tpr_log_reg, _ = roc_curve(y_test, log_reg_probs)
fpr_lda, tpr_lda, _ = roc_curve(y_test, lda_probs)

roc_auc_log_reg = auc(fpr_log_reg, tpr_log_reg)
roc_auc_lda = auc(fpr_lda, tpr_lda)

plt.figure(figsize=(8, 6))
plt.plot(fpr_log_reg, tpr_log_reg, color='blue', lw=2, label=f'Logistic Regression (AUC = {roc_auc_log_reg:.2f})')
plt.plot(fpr_lda, tpr_lda, color='green', lw=2, label=f'LDA (AUC = {roc_auc_lda:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2) 
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison: Logistic Regression vs. LDA')
plt.legend(loc='lower right')
plt.show()

print(f'AUC for Logistic Regression: {roc_auc_log_reg:.2f}')
print(f'AUC for LDA: {roc_auc_lda:.2f}')
