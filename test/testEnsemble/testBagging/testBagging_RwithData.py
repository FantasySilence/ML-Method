import os 
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

current_path = os.path.dirname(os.path.abspath(__file__))
test_path = os.path.dirname(current_path)
root_path = os.path.dirname(test_path)
root_path = os.path.dirname(root_path)
sys.path.append(root_path)
np.random.seed(0)

from src.common.utilsFile.filesio import FilesIO
from src.algorithm.ensemble.bagging.bagging_C_R import BaggingClassifierRegressor
from src.algorithm.decisiontree.decisionTree_R import DecisionTreeRegressor


boston = pd.read_csv(FilesIO.getDataPath("Boston.csv"))
X, y = np.asarray(boston.iloc[:, :-1]), np.asarray(boston.iloc[:, -1])
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

base_es = DecisionTreeRegressor(max_bins=50, max_depth=8)
bcr = BaggingClassifierRegressor(base_estimator=base_es, n_estimators=20, task="r")
bcr.fit(X_train, y_train)

idx = np.argsort(y_test)
y_hat = bcr.predict(X_test)

plt.figure(figsize=(7, 5))
plt.plot(y_hat[idx], "r-", lw=1, label="Bagging Prediction")
plt.plot(y_test[idx], "k-", lw=1, label="Test True Values")
plt.title(
    "Bagging(20 estimators) Regressor(R2 = %.5f)"%r2_score(y_test, y_hat), 
    fontdict={"fontsize":14}
)
plt.xlabel("x", fontdict={"fontsize":12})
plt.ylabel("y", fontdict={"fontsize":12})
plt.legend(frameon=False)
plt.grid(ls=":")
plt.show()