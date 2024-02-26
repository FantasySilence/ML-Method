from os import error
import numpy as np



class SquareErrorUtils:

    """
    平方误差最小化准则
    选择其中最优的一个作为切分点
    对特征属性进行分箱处理
    """

    @staticmethod
    def __set_sample_weight__(sample_weight, n_samples):
        
        """
        设置样本权重
        sample_weight: 样本权重
        n_samples: 样本个数
        """

        if sample_weight is None:
            sample_weight = np.asarray([1.0] * n_samples)
        return sample_weight
    

    @staticmethod
    def square_error(y, sample_weight):
        
        """
        y: 当前划分趋于的目标值集合
        sample_weight: 当前样本的权重
        """

        y = np.asarray(y)
        return np.sum((y - y.mean())**2 * sample_weight)
    

    def cond_square_error(self, x, y, sample_weight):
        
        """
        计算根据特征x划分的趋于中y的误差值
        x: 某个特征划分区域所办含的样本
        y: x对应的目标值
        sample_weight: 当前x的权重
        """

        x, y = np.asarray(x), np.asarray(y)
        error = 0.0
        for x_val in set(x):
            x_idx = np.where(x == x_val)    # 按区域计算误差
            new_y = y[x_idx]    # 对应区域的目标值
            new_sample_weight = sample_weight[x_idx]
            error += self.square_error(new_y, new_sample_weight)
        return error
    

    def square_error_gain(self, x, y, sample_weight):

        """
        平方误差带来的增益值
        x: 某个特征变量
        y: 对应的目标值
        sample_weight: 样本权重
        """

        sample_weight = self.__set_sample_weight__(sample_weight, len(x))
        return self.square_error(y, sample_weight) -\
              self.cond_square_error(x, y, sample_weight)