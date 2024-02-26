import os
import sys
import numpy as np

current_path = os.path.dirname(os.path.abspath(__file__))
test_path = os.path.dirname(current_path)
root_path = os.path.dirname(test_path)
sys.path.append(root_path)

from src.common.utilsFile.filesio import FilesIO
from src.algorithm.regression.singlelinearRegression import SingleVarLinearRegression



url = FilesIO.getDataPath('datas.csv')
model = SingleVarLinearRegression(url, 'x', 'ln_y')
model.one_varLRM()
model.format_output()
print()
print("-"*100)
xp = np.array([5.5,6.6,7.7])
model.predict(xp)
print("-"*100)
# 可视化
model.plt_LRM()
model.regression_diagnostics()
model.residual_analysis()