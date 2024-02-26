import os
import sys

current_path = os.path.dirname(os.path.abspath(__file__))
test_path = os.path.dirname(current_path)
root_path = os.path.dirname(test_path)
sys.path.append(root_path)

from src.common.utilsFile.filesio import FilesIO
from src.algorithm.regression.multiLinearRegression import MultiVarLinearRegression



url = FilesIO.getDataPath("Advertising.csv")
attribute_list = ['TV', 'radio', 'newspaper']
target_y = "sales"
model = MultiVarLinearRegression(url, attribute_list, target_y)
model.format_output()