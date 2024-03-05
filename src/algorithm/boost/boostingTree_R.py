import copy
import numpy as np

from src.algorithm.decisiontree.decisionTree_R import DecisionTreeRegressor


class BoostTreeRegressor():

    """
    提升树回归算法，采用平方误差损失
    """

    def __init__(self, base_estimators=None, n_estimators=10, learning_rate=1.0,):

        """
        base_estimators：基学习器
        n_estimators：基学习器的个数
        learning_rate：学习率，降低后续基学习器的权重避免过拟合
        """

        self.base_estimators = base_estimators
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        # 如果没有基学习器则以深度为2的决策树作为基学习器
        if self.base_estimators is None:
            self.base_estimators = DecisionTreeRegressor(max_depth=2)
        if type(self.base_estimators) != list:
            # 同质学习器
            self.base_estimators = [copy.deepcopy(self.base_estimators) 
                                    for _ in range(self.n_estimators)]
        else:
            # 异质学习器
            self.n_estimators = len(self.base_estimators)
    

    def fit(self, x_train, y_train):

        """
        回归提升树的训练，针对每个基决策树算法，拟合上一轮的残差
        x_train: 训练集特征
        y_train: 训练集标签
        """

        x_train, y_train = np.asarray(x_train), np.asarray(y_train)
        # 1.训练第一颗回归决策树并预测
        self.base_estimators[0].fit(x_train, y_train)
        y_hat = self.base_estimators[0].predict(x_train)
        y_residual = y_train - y_hat    # 残差,MSE的负梯度
        # 2.从第二棵树开始，每次拟合上一轮的残差
        for idx in range(1, self.n_estimators):
            self.base_estimators[idx].fit(x_train, y_residual)
            y_hat += self.base_estimators[idx].predict(x_train) * self.learning_rate
            y_residual = y_train - y_hat


    def predict(self, x_test):

        """
        回归提升树的预测
        x_test: 测试集特征
        """

        x_test = np.asarray(x_test)
        y_hat_mat = np.sum(
            [self.base_estimators[0].predict(x_test)] + 
            [
                np.power(self.learning_rate, i) * self.base_estimators[i].predict(x_test)
                for i in range(1, self.n_estimators - 1)
            ] +
            [self.base_estimators[-1].predict(x_test)]
        , axis=0)
        return y_hat_mat
