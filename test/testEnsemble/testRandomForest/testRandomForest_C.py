import os 
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report   
from sklearn.model_selection import train_test_split

current_path = os.path.dirname(os.path.abspath(__file__))
test_path = os.path.dirname(current_path)
root_path = os.path.dirname(test_path)
root_path = os.path.dirname(root_path)
sys.path.append(root_path)

from src.algorithm.ensemble.randomForest.randomForest_C_R\
import RandomForestClassifierRegressor
from src.algorithm.decisiontree.decisionTree_C import DecisionTreeClassifier


wine = load_wine()
iris = load_iris()
X, y = iris.data, iris.target
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, shuffle=True
)

base_se = DecisionTreeClassifier(max_bins=50, max_depth=10, is_all_feature_R=True)
rf_model = RandomForestClassifierRegressor(
    base_estimator=base_se, n_estimators=20, task="c",
    OOB=True, feature_importance=True
)
rf_model.fit(X_train, y_train)
y_hat = rf_model.predict(X_test)
print(classification_report(y_test, y_hat))
print("包外估计的精度：", rf_model.oob_score)
print("特征重要性：", rf_model.feature_importance_scores)

plt.figure(figsize=(10, 6))
data_iris = pd.DataFrame([iris.feature_names, rf_model.feature_importance_scores]).T
data_iris.columns = ["Feature Name", "Importance"]
sns.barplot(x="Importance", y="Feature Name", data=data_iris)
plt.title("Iris DataSet Feature Importance", fontdict={"fontsize":14})
plt.show()
