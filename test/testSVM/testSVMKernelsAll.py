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


X, y = make_classification(
    n_samples=100, n_features=2, n_classes=2,
    n_informative=1, n_redundant=0, n_repeated=0,
    n_clusters_per_class=1, class_sep=0.8, random_state=21
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

svm_rbf1 = SVMClassifier(C=10.0, kernel="rbf", gamma=0.1)
svm_rbf1.fit(X_train, y_train)
y_test_pred = svm_rbf1.predict(X_test)
print(classification_report(y_test, y_test_pred))
print("="*60)
svm_rbf2 = SVMClassifier(C=10.0, kernel="rbf", gamma=0.1)
svm_rbf2.fit(X_train, y_train)
y_test_pred = svm_rbf2.predict(X_test)
print(classification_report(y_test, y_test_pred))
print("="*60)

svm_poly1 = SVMClassifier(C=10.0, kernel="poly", degree=3)
svm_poly1.fit(X_train, y_train)
y_test_pred = svm_poly1.predict(X_test)
print(classification_report(y_test, y_test_pred))
print("="*60)
svm_poly2 = SVMClassifier(C=10.0, kernel="poly", degree=6)
svm_poly2.fit(X_train, y_train)
y_test_pred = svm_poly2.predict(X_test)
print(classification_report(y_test, y_test_pred))


X, y = make_classification(
    n_samples=100, n_features=2, n_classes=2,
    n_informative=1, n_redundant=0, n_repeated=0,
    n_clusters_per_class=1, class_sep=0.8, random_state=21
)

X_train1, X_test1, y_train1, y_test1 = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

svm_linear1 = SVMClassifier(C=10.0, kernel="linear")
svm_linear1.fit(X_train1, y_train1)
y_test_pred1 = svm_linear1.predict(X_test1)
print(classification_report(y_test1, y_test_pred1))
print("="*60)
svm_linear2 = SVMClassifier(C=10.0, kernel="linear")
svm_linear2.fit(X_train1, y_train1)
y_test_pred2 = svm_linear2.predict(X_test1)
print(classification_report(y_test1, y_test_pred2))

plt.figure(figsize=(21, 10))
plt.subplot(231)
svm_rbf1.plt_svm(X_train, y_train, is_show=False)
plt.subplot(232)
svm_rbf2.plt_svm(X_train, y_train, is_show=False)
plt.subplot(233)
svm_poly1.plt_svm(X_train, y_train, is_show=False)
plt.subplot(234)
svm_poly2.plt_svm(X_train, y_train, is_show=False)
plt.subplot(235)
svm_linear1.plt_svm(X_train1, y_train1, is_show=False, is_margin=True)
plt.subplot(236)
svm_linear2.plt_svm(X_train1, y_train1, is_show=False, is_margin=True)
plt.subplots_adjust(wspace=0.3, hspace=0.3)
plt.show()
