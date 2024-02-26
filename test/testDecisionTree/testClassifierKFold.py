import os
import sys
import copy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report

current_path = os.path.dirname(os.path.abspath(__file__))
test_path = os.path.dirname(current_path)
root_path = os.path.dirname(test_path)
sys.path.append(root_path)

from src.algorithm.decisiontree.decisionTree_C import DecisionTreeClassifier



bc_data = load_breast_cancer()
X, y = bc_data.data, bc_data.target
alphas = np.linspace(0, 10, 30)
accuracy_scores = []    # 储存每个alpha阀值下的交叉验证均分
cart = DecisionTreeClassifier(criterion="cart", is_all_feature_R=True, max_bins=10)

# 交叉验证
for alpha in alphas:
    scores = []
    k_fold = StratifiedKFold(n_splits=10).split(X,y)
    for train_idx, test_idx in k_fold:
        tree = copy.deepcopy(cart)
        tree.fit(X[train_idx], y[train_idx])
        tree.prune(alpha=alpha)
        y_test_pred = tree.predict(X[test_idx])
        scores.append(accuracy_score(y[test_idx], y_test_pred))
        del tree
    print(alpha, ":", np.mean(scores))
    accuracy_scores.append(np.mean(scores))

plt.figure(figsize=(8, 6))
plt.plot(alphas, accuracy_scores, "ko-", lw=1)
plt.grid(ls=":")
plt.xlabel("Alpha", fontdict={"fontsize":12})
plt.ylabel("Accuracy_Scores", fontdict={"fontsize":12})
plt.title("Cross Validation Scores under Different Pruning Alpha", fontdict={"fontsize":14})
plt.show()