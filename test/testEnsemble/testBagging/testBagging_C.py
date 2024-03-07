import os 
import sys
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

current_path = os.path.dirname(os.path.abspath(__file__))
test_path = os.path.dirname(current_path)
root_path = os.path.dirname(test_path)
root_path = os.path.dirname(root_path)
sys.path.append(root_path)

from src.algorithm.ensemble.bagging.bagging_C_R import BaggingClassifierRegressor
from src.algorithm.decisiontree.decisionTree_C import DecisionTreeClassifier


iris = load_iris()
X, y = iris.data, iris.target
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, shuffle=True, random_state=42
)

base_es = DecisionTreeClassifier(max_bins=50, max_depth=10, is_all_feature_R=True)
bagcr = BaggingClassifierRegressor(
    base_estimator=base_es, n_estimators=20, task="C", OOB=True
)
bagcr.fit(X_train, y_train)
y_pred = bagcr.predict(X_test)
print(classification_report(y_test, y_pred))
print("包外估计精度：", bagcr.oob_score)
