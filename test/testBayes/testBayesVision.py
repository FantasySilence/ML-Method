import os
import sys
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

current_path = os.path.dirname(os.path.abspath(__file__))
test_path = os.path.dirname(current_path)
root_path = os.path.dirname(test_path)
sys.path.append(root_path)

from src.algorithm.Bayes.naiveBayes import NaiveBayesClassifier
from src.common.utilsBayes.decisionBorder import plt_decision_border


X, y = make_blobs(
    n_samples=500, centers=4, cluster_std=0.85, random_state=0,
    n_features=2
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y
)

nbc = NaiveBayesClassifier(is_binned=True, max_bins=20, is_feature_all_R=True)
nbc.fit(X_train, y_train)
y_pred = nbc.predict(X_test)
print(classification_report(y_test, y_pred))
plt.figure(figsize=(14, 5), facecolor="white")
plt.subplot(121)
plt_decision_border(X_train, y_train, nbc, is_show=False)

nbc = NaiveBayesClassifier(is_binned=False, max_bins=20, feature_R_idx=[0, 1])
nbc.fit(X_train, y_train)
y_pred = nbc.predict(X_test)
print(classification_report(y_test, y_pred))
plt.subplot(122)
plt_decision_border(X_train, y_train, nbc, is_show=False)
plt.show()
