import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

current_path = os.path.dirname(os.path.abspath(__file__))
test_path = os.path.dirname(current_path)
root_path = os.path.dirname(test_path)
sys.path.append(root_path)
np.random.seed(0)

from src.algorithm.decisiontree.decisionTree_R import DecisionTreeRegressor
from src.common.utilsFile.filesio import FilesIO



data = pd.read_csv(FilesIO.getDataPath("Boston.csv"))
X, y = data.iloc[:, 1:-1].values, data.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

tree = DecisionTreeRegressor(max_bins=100)
tree.fit(X_train, y_train)
y_test_pred = tree.predict(X_test)
mse, r2 = tree.cal_mse_r2(y_test, y_test_pred)

plt.figure(figsize=(16,6), facecolor="white", dpi=80)

plt.subplot(121)
idx = np.argsort(y_test)
plt.plot(y_test[idx], "k-", lw=2, label="Test True Values")
plt.plot(y_test_pred[idx], "r-", lw=1.5, label="Test Predictions")
plt.title("Decision Tree Regressor(MSE = %.5f,R2 = %.5f)(Unprune)"%(mse, r2))
plt.xlabel("x", fontdict={"fontsize":12, "color":"b"})
plt.ylabel("y", fontdict={"fontsize":12, "color":"b"})
plt.legend(frameon=False)
plt.grid(ls=":")

plt.subplot(122)
tree.prune(100)
y_test_pred2 = tree.predict(X_test)
mse, r2 = tree.cal_mse_r2(y_test, y_test_pred2)
idx = np.argsort(y_test)
plt.plot(y_test[idx], "k-", lw=2, label="Test True Values")
plt.plot(y_test_pred2[idx], "r-", lw=1.5, label="Test Predictions")
plt.title("Decision Tree Regressor(MSE = %.5f,R2 = %.5f)(prune)"%(mse, r2))
plt.xlabel("x", fontdict={"fontsize":12, "color":"b"})
plt.ylabel("y", fontdict={"fontsize":12, "color":"b"})
plt.legend(frameon=False)
plt.grid(ls=":")

plt.show()