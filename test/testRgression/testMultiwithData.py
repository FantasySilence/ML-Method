import os
import sys
import pandas as pd

current_path = os.path.dirname(os.path.abspath(__file__))
test_path = os.path.dirname(current_path)
root_path = os.path.dirname(test_path)
sys.path.append(root_path)

from src.common.utilsFile.filesio import FilesIO
from src.algorithm.regression.multiLinearRegression import MultiVarLinearRegression


url = FilesIO.getDataPath("Boston.csv")
data = pd.read_csv(url)
attribute_list, target_y = data.columns[1:-1].to_list(), data.columns[-1]
model = MultiVarLinearRegression(url, attribute_list, target_y)
model.format_output()
