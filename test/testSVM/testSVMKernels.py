import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

current_path = os.path.dirname(os.path.abspath(__file__))
test_path = os.path.dirname(current_path)
root_path = os.path.dirname(test_path)
sys.path.append(root_path)

from src.algorithm.SVM.svm_smo_classifier import SVMClassifier


X, y = make_moons(n_samples=200, noise=0.1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

svm_rbf = SVMClassifier(C=10.0, kernel="rbf", gamma=0.1)
svm_rbf.fit(X_train, y_train)
y_test_pred = svm_rbf.predict(X_test)
print(classification_report(y_test, y_test_pred))
print("="*60)
svm_poly = SVMClassifier(C=10.0, kernel="poly", degree=3)
svm_poly.fit(X_train, y_train)
y_test_pred = svm_poly.predict(X_test)
print(classification_report(y_test, y_test_pred))

plt.figure(figsize=(14, 10))
plt.subplot(221)
svm_rbf.plt_svm(X_train, y_train, is_show=False)
plt.subplot(222)
svm_rbf.plt_loss_curve(is_show=False)
plt.subplot(223)
svm_poly.plt_svm(X_train, y_train, is_show=False)
plt.subplot(224)
svm_poly.plt_loss_curve(is_show=False)
plt.show()
