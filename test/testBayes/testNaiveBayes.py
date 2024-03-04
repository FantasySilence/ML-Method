import os
import sys
import numpy as np
import pandas as pd

current_path = os.path.dirname(os.path.abspath(__file__))
test_path = os.path.dirname(current_path)
root_path = os.path.dirname(test_path)
sys.path.append(root_path)

from src.algorithm.Bayes.naiveBayes import NaiveBayesClassifier
from src.common.utilsFile.filesio import FilesIO


wm = pd.read_csv(FilesIO.getDataPath("watermelon.csv"), encoding="gbk")
X, y = np.asarray(wm.iloc[:, 1:-1]), np.asarray(wm.iloc[:, -1])

nbc = NaiveBayesClassifier(is_binned=True, feature_R_idx=[6, 7], max_bins=5)
nbc.fit(X, y)
y_prob = nbc.predict_proba(X)
print(y_prob)
y_hat = nbc.predict(X)
print(y_hat)
