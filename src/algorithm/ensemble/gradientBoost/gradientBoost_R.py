import copy
import numpy as np

from src.algorithm.decisiontree.decisionTree_R import DecisionTreeRegressor


class GradientBoostRegressor():

    """
    梯度提升回归算法，以损失函数在当前模型的负梯度近似为残差
    1.假设回归决策树以MSE构建的，针对不同的损失函数，计算不同的基尼指数划分标准
    2.预测，集成，也根据不同的损失函数，预测叶子节点的输出...
    """

    def __init__(self, base_estimators=None, n_estimators=10, learning_rate=1.0,
                 loss="ls", huber_threshold=0.1, quantile_threshold=0.5):

        """
        base_estimators：基学习器
        n_estimators：基学习器的个数
        learning_rate：学习率，降低后续基学习器的权重避免过拟合
        loss：损失函数的类型
        huber_threshold：仅对huber损失函数有效
        quantile_threshold：仅对分位数损失函数有效
        """

        self.base_estimators = base_estimators
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.loss = loss
        self.huber_threshold = huber_threshold
        self.quantile_threshold = quantile_threshold
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
    

    def _cal_negetive_gradient(self, y_true, y_pred):

        """
        计算损失函数在当前模型的负梯度
        y_true: 真实值
        y_pred: 预测值
        """

        if self.loss.lower() == "ls":
            return y_true - y_pred      # MSE
        elif self.loss.lower() == "lae":
            return np.sign(y_true - y_pred)      # MAE
        elif self.loss.lower() == "huber":
            return np.where(
                np.abs(y_true - y_pred) > self.huber_threshold,
                self.huber_threshold * np.sign(y_true - y_pred),
                y_true - y_pred
            )       # 平滑平均绝对损失
        elif self.loss.lower() == "quantile":
            return np.where(
                y_true > y_pred, self.quantile_threshold,
                self.quantile_threshold - 1
            )       # 分位数损失
        elif self.loss.lower() == "logcosh":
            return -np.tanh(y_true - y_pred)      # 双曲余弦的对数的负梯度
        else:
            raise ValueError("仅限于ls, lae, huber, quantile, logcosh...")
    

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
        y_residual = self._cal_negetive_gradient(y_train, y_hat)    # 计算负梯度
        # 2.从第二棵树开始，每次拟合上一轮的残差
        for idx in range(1, self.n_estimators):
            self.base_estimators[idx].fit(x_train, y_residual)
            y_hat += self.base_estimators[idx].predict(x_train) * self.learning_rate
            y_residual = self._cal_negetive_gradient(y_train, y_hat)    # 计算负梯度


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
