import os 
import sys
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

current_path = os.path.dirname(os.path.abspath(__file__))
test_path = os.path.dirname(current_path)
root_path = os.path.dirname(test_path)
root_path = os.path.dirname(root_path)
sys.path.append(root_path)

# 基学习器，决策树，逻辑回归，支持向量机
from src.algorithm.logistic_regression.singleLogisticRegression import LogisticRegressor
from src.algorithm.decisiontree.decisionTree_C import DecisionTreeClassifier
from src.algorithm.SVM.svm_smo_classifier import SVMClassifier

from src.algorithm.ensemble.boost.adaboost_C import AdaBoostClassifier
from src.common.utilsBoost.decisionBorder import plt_decision_border


X, y = make_classification(
    n_samples=300, n_features=2, n_informative=1, n_redundant=0,
    n_repeated=0, n_classes=2, n_clusters_per_class=1, class_sep=1,
    random_state=42
)

# 同质：同种基学习器
base_tree = DecisionTreeClassifier(max_depth=3, is_all_feature_R=True, max_bins=20)
ada_bc = AdaBoostClassifier(base_estimators=base_tree, n_estimators=15, learning_rate=0.9)
ada_bc.fit(X, y)
print("基学习器的权重系数：\n", ada_bc.estimators_weights)
y_pred = ada_bc.predict(X)
print(classification_report(y, y_pred))
plt_decision_border(X, y, ada_bc)

# 异质：不同种基学习器
log_reg = LogisticRegressor(batch_size=20, max_epoches=20)
cart = DecisionTreeClassifier(max_depth=4, is_all_feature_R=True)
svm = SVMClassifier(C=5.0, max_epochs=20)
ada_bc2 = AdaBoostClassifier(base_estimators=[cart, svm], learning_rate=1.0)
ada_bc2.fit(X, y)
print("基学习器的权重系数：\n", ada_bc2.estimators_weights)
y_pred = ada_bc2.predict(X)
print(classification_report(y, y_pred))
plt_decision_border(X, y, ada_bc2)
