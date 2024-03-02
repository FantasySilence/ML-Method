import os
import sys
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

current_path = os.path.dirname(os.path.abspath(__file__))
test_path = os.path.dirname(current_path)
root_path = os.path.dirname(test_path)
sys.path.append(root_path)

from src.algorithm.LDA.multiLDAClassify import MultiLDAClassifier


iris = load_iris()
X, y = iris.data, iris.target
X = StandardScaler().fit_transform(X)

lda = MultiLDAClassifier(n_components=2)
X_new = lda.fit_transform(X, y)
print(lda.variance_explained())

plt.figure(figsize=(14, 5))
plt.subplot(121)
plt.scatter(X_new[:, 0], X_new[:, 1], c=y, marker="o")
plt.title("LDA Dimension Reduction(Myself)", fontdict={"fontsize":14})
plt.xlabel("PC1", fontdict={"fontsize":12})
plt.xlabel("PC2", fontdict={"fontsize":12})
plt.grid(ls=":")

lda_sk = LinearDiscriminantAnalysis(n_components=2)
lda_sk.fit(X, y)
X_sk = lda_sk.transform(X)

plt.subplot(122)
plt.scatter(X_sk[:, 0], X_sk[:, 1], c=y, marker="o")
plt.title("LDA Dimension Reduction(Sklearn)", fontdict={"fontsize":14})
plt.xlabel("PC1", fontdict={"fontsize":12})
plt.xlabel("PC2", fontdict={"fontsize":12})
plt.grid(ls=":")
plt.show()
