import os
import sys
import numpy as np
import matplotlib.pyplot as plt

current_path = os.path.dirname(os.path.abspath(__file__))
test_path = os.path.dirname(current_path)
root_path = os.path.dirname(test_path)
sys.path.append(root_path)
np.random.seed(0)

from src.algorithm.decisiontree.decisionTree_R import DecisionTreeRegressor



obj_fun = lambda x: np.sin(x)
n = 100    # 样本量
x = np.linspace(0, 10, n)
target = obj_fun(x) + 0.2 * np.random.rand(n)
data = x[:, np.newaxis]     # 二维数组

tree = DecisionTreeRegressor(max_bins=50, max_depth=10)
tree.fit(data, target)
x_test = np.linspace(0, 10, 200)
y_test_pred = tree.predict(x_test[:, np.newaxis])
mse, r2 = tree.cal_mse_r2(obj_fun(x_test), y_test_pred)

plt.figure(figsize=(16, 6), facecolor="white", dpi=80)

plt.subplot(121)
plt.scatter(data, target, s=15, c="k", label="Raw Data")
plt.plot(x_test, y_test_pred, "r-", lw=1.5, label="Fit Model")
plt.title("Decision Tree Regressor(MSE = %.5f,R2 = %.5f)(Unprune)"%(mse, r2))
plt.xlabel("x", fontdict={"fontsize":12, "color":"b"})
plt.ylabel("y", fontdict={"fontsize":12, "color":"b"})
plt.legend(frameon=False)
plt.grid(ls=":")

plt.subplot(122)
tree.prune(0.2)
y_test_pred = tree.predict(x_test[:, np.newaxis])
mse, r2 = tree.cal_mse_r2(obj_fun(x_test), y_test_pred)
plt.scatter(data, target, s=15, c="k", label="Raw Data")
plt.plot(x_test, y_test_pred, "r-", lw=1.5, label="Fit Model")
plt.title("Decision Tree Regressor(MSE = %.5f,R2 = %.5f)(prune)"%(mse, r2))
plt.xlabel("x", fontdict={"fontsize":12, "color":"b"})
plt.ylabel("y", fontdict={"fontsize":12, "color":"b"})
plt.legend(frameon=False)
plt.grid(ls=":")
plt.show()
