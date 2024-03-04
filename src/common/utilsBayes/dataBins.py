import numpy as np

class DataBinsWrapper:

    """
    连续特征数据的离散化，分箱(分段)操作,根据ma_bins，计算分位数，以分位数分箱(分段),
    然后根据样本特征取值所在的区间段(哪个箱)位置索引标记当前值
    """

    def __init__(self, max_bins=10):
        self.max_bins = max_bins    # 最大分箱数量, 10%, 20%,...,90%
        self.XrangeMap = None    # 箱(区间段)
    
    def fit(self, x_samples):

        """
        根据样本进行分箱
        x_samples：样本，或者一个特征属性的值
        """

        if x_samples.ndim == 1:     # 一个特征属性的值
            n_feature = 1
            x_samples = x_samples[:, np.newaxis]    # 增加维度，转换为二维数组
        else:
            n_feature = x_samples.shape[1]
        
        # 构建分箱，区间段
        self.XrangeMap = [[] for _ in range(n_feature)]
        for idx in range(n_feature):
            x_sorted = sorted(x_samples[:, idx])    # 按特征索引，从小到大排序
            for bin in range(1, self.max_bins):
                p = (bin/self.max_bins)*100//1
                p_val = np.percentile(x_sorted, p)
                self.XrangeMap[idx].append(p_val)
            self.XrangeMap[idx] = sorted(list(set(self.XrangeMap[idx])))
    
    def transform(self, x_samples, XrangeMap=None):

        """
        根据已存在的箱，将数据分成max_bins类
        x_samples：样本，或者一个特征属性的值
        """

        if x_samples.ndim == 1:     # 一个特征属性的值
            if XrangeMap is not None:
                return np.asarray(np.digitize(x_samples, XrangeMap[0])).reshape(-1)
            else:
                return np.asarray(np.digitize(x_samples, self.XrangeMap[0])).reshape(-1)
        else:
            return np.asarray([np.digitize(x_samples[:, i], self.XrangeMap[i]) for i in range(x_samples.shape[1])]).T

if __name__ == "__main__":
    
    import pandas as pd
    data = pd.read_csv("D:\python_project\some_code\Study\Machine_Learning\data\watermelon.csv",encoding="gbk")
    x = np.asarray(data.loc[:,["密度","含糖率"]])
    dbw = DataBinsWrapper(max_bins=8)
    dbw.fit(x)
    print(dbw.transform(x))
