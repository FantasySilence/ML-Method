import os
import sys
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report, accuracy_score

current_path = os.path.dirname(os.path.abspath(__file__))
test_path = os.path.dirname(current_path)
root_path = os.path.dirname(test_path)
sys.path.append(root_path)

from src.algorithm.decisiontree.decisionTree_C import DecisionTreeClassifier
from src.common.utilsDecisionTree.pltTree import plot_decision_func



data, target = make_classification(n_samples=200, n_features=2, n_classes=2, n_informative=1, n_redundant=0,
                                n_clusters_per_class=1, class_sep=1, random_state=21)
cart_tree = DecisionTreeClassifier(is_all_feature_R=True)
cart_tree.fit(data, target)
y_test_pred = cart_tree.predict(data)
print(classification_report(target, y_test_pred))
plt.figure(figsize=(16, 12))
plt.subplot(221)
acc = accuracy_score(target, y_test_pred)
plot_decision_func(data, target, cart_tree, acc=acc, is_show=False, title_info="By CART UnPrune")

# 剪枝处理
alpha = [1, 3, 5]
for i in range(3):
    cart_tree.prune(alpha=alpha[i])
    y_test_pred = cart_tree.predict(data)
    acc = accuracy_score(target, y_test_pred)
    plt.subplot(222 + i)
    plot_decision_func(data, target, cart_tree, acc=acc, is_show=False, title_info="By CART Prune(a=%.1f)"%alpha[i])
plt.subplots_adjust(wspace=0.3, hspace=0.5)
plt.show()