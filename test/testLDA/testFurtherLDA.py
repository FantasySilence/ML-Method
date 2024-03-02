import os
import sys
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

current_path = os.path.dirname(os.path.abspath(__file__))
test_path = os.path.dirname(current_path)
root_path = os.path.dirname(test_path)
sys.path.append(root_path)

from src.algorithm.LDA.multiLDAClassify import MultiLDAClassifier


X, y = make_classification(
    n_samples=2000, n_features=20, n_informative=3, n_classes=5,
    n_redundant=0, n_clusters_per_class=1, class_sep=2, random_state=42
)

X = StandardScaler().fit_transform(X)
lda = MultiLDAClassifier(n_components=3)
X_new = lda.fit_transform(X, y)
print(lda.variance_explained())

plt.figure(figsize=(14, 10))
plt.subplot(221)
plt.scatter(X_new[:, 0], X_new[:, 1], c=y, marker="o")
plt.title("LDA Dimension Reduction(Myself)", fontdict={"fontsize":14})
plt.xlabel("PC1", fontdict={"fontsize":12})
plt.xlabel("PC2", fontdict={"fontsize":12})
plt.grid(ls=":")

plt.subplot(222)
plt.scatter(X_new[:, 1], X_new[:, 2], c=y, marker="o")
plt.title("LDA Dimension Reduction(Myself)", fontdict={"fontsize":14})
plt.xlabel("PC2", fontdict={"fontsize":12})
plt.xlabel("PC3", fontdict={"fontsize":12})
plt.grid(ls=":")

lda_sk = LinearDiscriminantAnalysis(n_components=3)
lda_sk.fit(X, y)
X_sk = lda_sk.transform(X)

plt.subplot(223)
plt.scatter(X_sk[:, 0], X_sk[:, 1], c=y, marker="o")
plt.title("LDA Dimension Reduction(Sklearn)", fontdict={"fontsize":14})
plt.xlabel("PC1", fontdict={"fontsize":12})
plt.xlabel("PC2", fontdict={"fontsize":12})
plt.grid(ls=":")
plt.subplot(224)
plt.scatter(X_sk[:, 1], X_sk[:, 2], c=y, marker="o")
plt.title("LDA Dimension Reduction(Sklearn)", fontdict={"fontsize":14})
plt.xlabel("PC2", fontdict={"fontsize":12})
plt.xlabel("PC3", fontdict={"fontsize":12})
plt.grid(ls=":")
plt.subplots_adjust(hspace=0.3, wspace=0.3)
plt.show()