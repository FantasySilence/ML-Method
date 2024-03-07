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

from src.common.utilsFile.filesio import FilesIO
from src.algorithm.ensemble.boost.adaboost_R import AdaBoostRegressor
from src.algorithm.decisiontree.decisionTree_R import DecisionTreeRegressor


boston = pd.read_csv(FilesIO.getDataPath("Boston.csv"))
X, y = boston.iloc[:, :-1].values, boston.iloc[:, -1].values
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

base_ht = DecisionTreeRegressor(max_bins=50, max_depth=5)
plt.figure(figsize=(14, 15), facecolor="white")
def train_plot(cs, loss, i):
    ada_ht = AdaBoostRegressor(
        base_estimators=base_ht, n_estimators=20, 
        comb_strategy=cs, loss=loss
    )
    ada_ht.fit(X_train, y_train)
    y_pred = ada_ht.predict(X_test)
    plt.subplot(231 + i)
    idx = np.argsort(y_test)
    plt.plot(y_test[idx], "k-", lw=1.5, label="Test True")
    plt.plot(y_pred[idx], "r-", lw=1, label="Predict")
    plt.legend(frameon=False)
    plt.title(
        "%s, %s, \n R2 = %.5f, MSE = %.5f"%(
            cs, loss, r2_score(y_test, y_pred), ((y_test - y_pred) ** 2).mean()
        )
    )
    plt.xlabel("Test Samples Serial Number", fontdict={"fontsize":12})
    plt.ylabel("True VS Predict", fontdict={"fontsize":12})
    plt.grid(ls=":")
    print(cs, loss)

loss_func = ["linear", "square", "exp"]
comb_strategy = ["weight_mean", "weight_median"]
i = 0
for loss in loss_func:
    for cs in comb_strategy:
        train_plot(cs, loss, i)
        i += 1
plt.subplots_adjust(wspace=0.3, hspace=0.5)
plt.show()
