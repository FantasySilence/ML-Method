import numpy as np


class SingleLDAClassifier:

    """
    线性判别分析二分类模型
    """

    def __init__(self):

        """
        必要的参数初始化
        """

        self.mu = None  # 各个类别的均值向量
        self.Sw_i = None  # 各类内散度矩阵
        self.Sw = None  # 类内散度矩阵
        self.weight = None  # 模型系数，投影的方向
        self.w0 = None  # 阀值


    def fit(self, x_train, y_train):

        """
        训练模型
        :param x_train: 训练数据
        :param y_train: 训练标签
        """

        x_train, y_train = np.asarray(x_train), np.asarray(y_train)
        class_values = np.sort(np.unique(y_train))      # 不同类别取值
        n_samples, n_features = x_train.shape
        if len(class_values) != 2:
            raise ValueError('仅限于二分类且线性可分数据集!')
        
        # 1.计算类均值，Sw散度矩阵，Sb散度矩阵
        class_size = []     # 计算各个类别的样本量
        self.Sw_i = dict()    # 字典，类别取值为键，对应类别样本内的散度矩阵为值
        self.mu = dict()    # 字典，类别取值为键，对应类别样本的均值向量为值
        self.Sw = np.zeros((n_features, n_features))    # 类内散度矩阵
        for label_val in class_values:
            class_x = x_train[y_train == label_val]   # 按样本类别划分
            class_size.append(class_x.shape[0])   # 计算各类别样本量
            self.mu[label_val] = np.mean(class_x, axis=0)   # 对特征取均值构成均值向量
            self.Sw_i[label_val] =\
                  (class_x - self.mu[label_val]).T.dot(class_x - self.mu[label_val])
            self.Sw += self.Sw_i[label_val]    # 累加计算类内散度矩阵
        
        # 2.计算投影方向w
        u, sigma, v = np.linalg.svd(self.Sw)
        inv_sw = v * np.linalg.inv(np.diag(sigma)) * u.T     # 计算逆矩阵
        self.weight = inv_sw.dot(self.mu[0] - self.mu[1])     # 投影方向

        # 3.计算阀值w0
        self.w0 = (class_size[0] * self.weight.dot(self.mu[0]) +\
                   class_size[1] * self.weight.dot(self.mu[1])) / n_samples
        
        return self.weight
        

    def predict(self, x_test):

        """
        预测
        :param x_test: 测试数据
        """

        x_test = np.asarray(x_test)
        y_pred = self.weight.dot(x_test.T) - self.w0
        y_test_pred = np.zeros(x_test.shape[0], dtype=int)  # 初始化测试样本的类别值
        y_test_pred[y_pred < 0] = 1     # 小于阀值为负类
        return y_test_pred