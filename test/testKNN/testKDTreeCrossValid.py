import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

current_path = os.path.dirname(os.path.abspath(__file__))
test_path = os.path.dirname(current_path)
root_path = os.path.dirname(test_path)
sys.path.append(root_path)

from src.algorithm.KNN.kdTree import KNearestNeighborKDTree



iris = load_iris()
X, y = iris.data, iris.target

accuracy_scores = []    # 储存每个alpha阀值下的交叉验证均分
k_neighbors = np.arange(3, 21)
acc = []
for k in k_neighbors:
    scores = []
    k_Fold = StratifiedKFold(n_splits=10).split(X, y)
    for train_idx, test_idx in k_Fold:
        knn = KNearestNeighborKDTree(k=k)
        knn.fit(X[train_idx], y[train_idx])
        y_test_pred = knn.predict(X[test_idx])
        scores.append(accuracy_score(y[test_idx], y_test_pred))
        del knn
    print("k = %d:"%k, np.mean(scores))
    accuracy_scores.append(np.mean(scores))

plt.figure(figsize=(10, 6))
plt.plot(k_neighbors, accuracy_scores, 'ko-', linewidth=1)
plt.title("KNN(KDTree) Test Scores under Different K_Neighbors", fontdict={"fontsize":14})
plt.xlabel("K_Neighbors", fontdict={"fontsize":12})
plt.ylabel("Accuracy Scores", fontdict={"fontsize":12})
plt.grid(":")
plt.show()