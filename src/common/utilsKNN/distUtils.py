import numpy as np



class DistanceUtils:

    """
    度量距离的工具类，实现闵可夫斯基距离
    """

    def __init__(self, p=2):

        self.p = p      # 默认欧氏距离，p=1时曼哈顿距离，p=np.inf是切比雪夫距离
    

    def distance_func(self, xi, xj):

        """
        特征空间中两个样本实例的距离计算
        xi: k维空间中某个样本实例
        xj: k维空间中某个样本实例
        """

        xi, xj = np.asarray(xi), np.asarray(xj)
        if self.p == 1 or self.p == 2:
            return (((np.abs(xi - xj)) ** self.p).sum()) ** (1 / self.p)
        elif self.p == np.inf:
            return np.max(np.abs(xi - xj))
        elif self.p == "cos":
            return xi.dot(xj) / np.sqrt((xi)**2).sum() / np.sqrt((xj)**2).sum()
        else:
            raise ValueError("仅支持曼哈顿距离(p=1),欧式距离(p=2)和切比雪夫距离(p=np.inf)")
        