"""
-----------------------------------------
@Author: Fantasy_Silence
@Time: 2024-03-10
@IDE: Visual Studio Code  Python: 3.9.7
-----------------------------------------
这是主程序入口，在这里同一调用各个模块解决问题
"""

import pandas as pd
from src.common.utilsFile.filesio import FilesIO
from src.algorithm.regression.multiLinearRegression import MultiVarLinearRegression

url = FilesIO.getDataPath("Boston.csv")
data = pd.read_csv(url)
attribute_list, target_y = data.columns[1:-1].to_list(), data.columns[-1]
model = MultiVarLinearRegression(url, attribute_list, target_y)
model.format_output()
