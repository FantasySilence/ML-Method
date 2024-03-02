import numpy as np
import scipy as sp


class MultiLDAClassifier:

    """
    线性判别分析多分类降维
    """

    def __init__(self, n_components=2):

        """
        必要的参数初始化
        :param n_components: 降维后的维数
        """

        self.n_components = n_components
        self.Sw, self.Sb = None, None
        self.eig_values = None  # 广义特征值
        self.W = None   # 投影矩阵

    
    def fit(self, x_samples, y_target):

        """
        线性判别分析多分类降维，计算投影矩阵
        """

        x_samples, y_target = np.asarray(x_samples), np.asarray(y_target)
        class_values = np.sort(np.unique(y_target))      # 不同类别取值
        n_samples, n_features = x_samples.shape
        self.Sw = np.zeros((n_features, n_features))    # 类内散度矩阵
        for i in range(len(class_values)):
            class_x = x_samples[y_target == class_values[i]]   # 按样本类别划分
            mu = np.mean(class_x, axis=0)   # 对特征取均值构成均值向量
            self.Sw += (class_x - mu).T.dot(class_x - mu)    # 累加计算类内散度矩阵
        mu_t = np.mean(x_samples, axis=0)
        self.Sb = (x_samples - mu_t).T.dot(x_samples - mu_t) - self.Sw    # 类间散度矩阵
        self.eig_values, eig_vec = sp.linalg.eig(self.Sb, self.Sw)
        idx = np.argsort(self.eig_values)[::-1]     # 从大到小
        self.eig_values = self.eig_values[idx]
        eig_sort = eig_vec[:, idx]
        self.W = eig_sort[:, :self.n_components]
        return self.W
    

    def transform(self, x_samples):

        """
        根据投影矩阵计算降维后的新样本数据
        """

        x_samples = np.asarray(x_samples)
        if self.W is not None:
            return x_samples.dot(self.W)
        else:
            raise ValueError("请先fit,构造投影矩阵再降维...")
        

    def fit_transform(self, x_samples, y_target):

        """
        调用fit和transform
        """

        self.fit(x_samples, y_target)
        return self.transform(x_samples)
    

    def variance_explained(self):

        """
        计算方差解释率
        """

        idx = np.argwhere(np.imag(self.eig_values) != 0)
        if len(idx) == 0:
            self.eig_values = np.real(self.eig_values)
        ratio = self.eig_values / np.sum(self.eig_values)
        return ratio[:self.n_components]