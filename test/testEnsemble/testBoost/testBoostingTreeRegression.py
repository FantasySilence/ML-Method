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
from src.algorithm.ensemble.boost.boostingTree_R import BoostTreeRegressor
from src.algorithm.decisiontree.decisionTree_R import DecisionTreeRegressor


boston = pd.read_csv(FilesIO.getDataPath("Boston.csv"))
X, y = boston.iloc[:, :-1].values, boston.iloc[:, -1].values
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

base_ht = DecisionTreeRegressor(max_bins=50, max_depth=5)

btr = BoostTreeRegressor(
    base_estimators=base_ht, n_estimators=20
)
btr.fit(X_train, y_train)
y_pred = btr.predict(X_test)
idx = np.argsort(y_test)
plt.figure(figsize=(7, 5), facecolor="white")
plt.plot(y_test[idx], "k-", lw=1.5, label="Test True")
plt.plot(y_pred[idx], "r-", lw=1, label="Predict")
plt.legend(frameon=False)
plt.title(
    "Boosting Tree Regression, \n R2 = %.5f, MSE = %.5f"%(
        r2_score(y_test, y_pred), ((y_test - y_pred) ** 2).mean()
    )
)
plt.xlabel("Test Samples Serial Number", fontdict={"fontsize":12})
plt.ylabel("True VS Predict", fontdict={"fontsize":12})
plt.grid(ls=":")
plt.show()
