import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

current_path = os.path.dirname(os.path.abspath(__file__))
test_path = os.path.dirname(current_path)
root_path = os.path.dirname(test_path)
sys.path.append(root_path)
np.random.seed(0)

from src.algorithm.KNN.kdTree import KNearestNeighborKDTree

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                    random_state=0, stratify=y)

k_neighbors = np.arange(3, 21)
acc = []
for k in k_neighbors:
    knn = KNearestNeighborKDTree(k=k)
    knn.fit(X_train, y_train)
    y_test_hat = knn.predict(X_test)
    acc.append(accuracy_score(y_test, y_test_hat))

plt.figure(figsize=(10, 6))
plt.plot(k_neighbors, acc, 'ko-', linewidth=1)
plt.title("KNN(KDTree) Test Scores under Different K_Neighbors", fontdict={"fontsize":14})
plt.xlabel("K_Neighbors", fontdict={"fontsize":12})
plt.ylabel("Accuracy Scores", fontdict={"fontsize":12})
plt.grid(":")
plt.show()

knn = KNearestNeighborKDTree(k=3)
knn.fit(X_train, y_train)
y_test_hat = knn.predict(X_test)
print(classification_report(y_test, y_test_hat))
knn.draw_kd_tree()
