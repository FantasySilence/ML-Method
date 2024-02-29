import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, load_iris
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

current_path = os.path.dirname(os.path.abspath(__file__))
test_path = os.path.dirname(current_path)
root_path = os.path.dirname(test_path)
sys.path.append(root_path)

from src.algorithm.SVM.svm_smo_classifier import SVMClassifier


# X, y = make_classification(
#     n_samples=200, n_features=2, n_classes=2,
#     n_informative=1, n_redundant=0, n_repeated=0,
#     n_clusters_per_class=1, class_sep=1.5, random_state=42
# )
iris = load_iris()
X, y = iris.data[:100, 2:], iris.target[:100]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

svm = SVMClassifier(C=100)
svm.fit(X_train, y_train)
y_test_pred = svm.predict(X_test)
print(classification_report(y_test, y_test_pred))
plt.figure(figsize=(16, 6))
plt.subplot(121)
svm.plt_svm(X_train, y_train, is_show=False, is_margin=True)
plt.subplot(122)
svm.plt_loss_curve(is_show=False)
plt.show()
