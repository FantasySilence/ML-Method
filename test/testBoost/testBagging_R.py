import os 
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

current_path = os.path.dirname(os.path.abspath(__file__))
test_path = os.path.dirname(current_path)
root_path = os.path.dirname(test_path)
sys.path.append(root_path)
np.random.seed(0)

from src.algorithm.boost.bagging_C_R import BaggingClassifierRegressor
from src.algorithm.decisiontree.decisionTree_R import DecisionTreeRegressor


f = lambda x: 0.5 * np.exp(-(x + 3) ** 2) +\
      np.exp(-x ** 2) + 1.5 * np.exp(-(x - 3) ** 2)
N = 200
X = np.random.rand(N) * 10 - 5
X = np.sort(X)
y = f(X) + np.random.randn(N) * 0.05
X = X.reshape(-1, 1)

base_es = DecisionTreeRegressor(max_bins=30, max_depth=8)
base_es.fit(X, y)
bcr = BaggingClassifierRegressor(base_estimator=base_es, n_estimators=100, task="r")
bcr.fit(X, y)

X_test = np.linspace(1.1 * X.min(axis=0), 1.1 * X.max(axis=0), 1000).reshape(-1, 1)
y_bagging_hat = bcr.predict(X_test)
y_cart_hat = base_es.predict(X_test)

plt.figure(figsize=(7, 5), facecolor="white")
plt.scatter(X, y, s=10, c="k", label="Raw Data")
plt.plot(X_test, f(X_test), "k-", lw=1.5, label="True F(x)")
plt.plot(
    X_test, y_bagging_hat, "r-", label="Bagging(R2 = %.5f)"%r2_score(
        f(X_test), y_bagging_hat
    )
)
plt.plot(
    X_test, y_cart_hat, "b-", label="CART(R2 = %.5f)"%r2_score(
        f(X_test), y_cart_hat
    )
)
plt.title("Bagging(100 estimators) vs CART Regressor", fontdict={"fontsize":14})
plt.xlabel("x", fontdict={"fontsize":12})
plt.ylabel("y", fontdict={"fontsize":12})
plt.legend(frameon=False)
plt.grid(ls=":")
plt.show()
