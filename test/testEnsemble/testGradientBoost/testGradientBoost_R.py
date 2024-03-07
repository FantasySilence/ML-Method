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
from src.algorithm.decisiontree.decisionTree_R import DecisionTreeRegressor
from src.algorithm.ensemble.gradientBoost.gradientBoost_R import\
GradientBoostRegressor


boston = pd.read_csv(FilesIO.getDataPath("Boston.csv"))
X, y = boston.iloc[:, :-1].values, boston.iloc[:, -1].values
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

loss_func = ["lae", "huber", "quantile", "logcosh"]
base_es = DecisionTreeRegressor(max_bins=50, max_depth=5)
plt.figure(figsize=(14, 15), facecolor="white")

for i, loss in enumerate(loss_func):
    gbr = GradientBoostRegressor(base_estimators=base_es, n_estimators=20, loss=loss)
    gbr.fit(X_train, y_train)
    y_pred = gbr.predict(X_test)
    idx = np.argsort(y_test)
    plt.subplot(221 + i)
    plt.plot(y_test[idx], "k-", lw=1.5, label="Test True Values")
    plt.plot(y_pred[idx], "r-", lw=1, label="Predicted Values")
    plt.legend(frameon=False)
    plt.xlabel("Observation Serial Number", fontdict={"fontsize":12})
    plt.ylabel("Test True VS Predict", fontdict={"fontsize":12})
    plt.title(
        "Boston House Price (R2 = %.5f, MSE = %.5f, loss = %s)"%(
            r2_score(y_test, y_pred),((y_test - y_pred)**2).mean(), loss
        )
    )
    plt.grid(ls=":")
plt.subplots_adjust(wspace=0.3, hspace=0.3)
plt.show()
