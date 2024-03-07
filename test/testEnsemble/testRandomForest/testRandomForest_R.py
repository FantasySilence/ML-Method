import os 
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

current_path = os.path.dirname(os.path.abspath(__file__))
test_path = os.path.dirname(current_path)
root_path = os.path.dirname(test_path)
root_path = os.path.dirname(root_path)
sys.path.append(root_path)

from src.common.utilsFile.filesio import FilesIO
from src.algorithm.ensemble.randomForest.randomForest_C_R\
import RandomForestClassifierRegressor
from src.algorithm.decisiontree.decisionTree_R import DecisionTreeRegressor


boston = pd.read_csv(FilesIO.getDataPath("Boston.csv"))
X, y = boston.iloc[:, 1:-1].values, boston.iloc[:, -1].values
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

base_es = DecisionTreeRegressor(max_bins=50, max_depth=10)
rf_model = RandomForestClassifierRegressor(
    base_estimator=base_es, n_estimators=50, task="r", 
    OOB=True, feature_sampling_rate=0.3, feature_importance=True
)
rf_model.fit(X_train, y_train)
idx = np.argsort(y_test)
y_pred = rf_model.predict(X_test)

plt.figure(figsize=(10, 6))
plt.plot(y_test[idx], "k-", lw=1.5, label="Test True Values")
plt.plot(y_pred[idx], "r-", lw=1, label="Predict Values")
plt.title(
    "Boston House (R2 = %.5f, MSE = %.5f)"%(
        r2_score(y_test, y_pred), ((y_test - y_pred) ** 2).mean()
    )
)
plt.xlabel("Observation Serial Number", fontdict={"fontsize":12})
plt.ylabel("Test True VS Predict Values", fontdict={"fontsize":12})
plt.legend(frameon=False)
plt.grid(ls=":")
plt.show()

plt.figure(figsize=(10, 6))
data_boston = pd.DataFrame([boston.columns[1:-1], rf_model.feature_importance_scores]).T
data_boston.columns = ["Feature Name", "Importance"]
sns.barplot(x="Importance", y="Feature Name", data=data_boston)
plt.title("Boston DataSet Feature Importance", fontdict={"fontsize":14})
plt.show()
