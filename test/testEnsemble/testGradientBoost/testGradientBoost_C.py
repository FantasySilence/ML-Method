import os 
import sys
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

current_path = os.path.dirname(os.path.abspath(__file__))
test_path = os.path.dirname(current_path)
root_path = os.path.dirname(test_path)
root_path = os.path.dirname(root_path)
sys.path.append(root_path)

from src.algorithm.ensemble.gradientBoost.gradientBoost_C\
import GradientBoostClassifier
from src.algorithm.decisiontree.decisionTree_R import DecisionTreeRegressor


iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

base_es = DecisionTreeRegressor(max_bins=50, max_depth=3)

gbc = GradientBoostClassifier(base_estimator=base_es, n_estimators=20)
gbc.fit(X_train, y_train)
y_pred = gbc.predict(X_test)
print(classification_report(y_test, y_pred))
