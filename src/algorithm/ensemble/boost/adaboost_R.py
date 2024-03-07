import copy
import numpy as np

from src.algorithm.decisiontree.decisionTree_R import DecisionTreeRegressor


class AdaBoostRegressor():

    """
    AdaBoost回归算法，结合(继承策略)，加权中位数，预测值的加权平均
    1.同质学习器，异质学习器
    2.回归误差率依赖于相对误差：平方误差，线性误差，指数误差
    """

    def __init__(self, base_estimators=None, n_estimators=10, learning_rate=1.0,
                 loss="square", comb_strategy="weight_median"):

        """
        base_estimators：基学习器
        n_estimators：基学习器的个数
        learning_rate：学习率，降低后续基学习器的权重避免过拟合
        loss：损失函数, linear, square, exp
        comb_strategy：结合策略，weight_median, weight_mean
        """

        self.base_estimators = base_estimators
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.loss = loss
        self.comb_strategy = comb_strategy
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
        self.estimator_weights = []    # 每个基学习器的权重系数
    

    def _cal_loss(self, y_true, y_hat):

        """
        根据损失函数计算相对误差
        y_true: 真实值
        y_hat: 预测值
        """

        errors = np.abs(y_true - y_hat)     # 绝对误差
        if self.loss.lower() == "linear":
            return errors / np.max(errors)     # 线性误差
        elif self.loss.lower() == "square":
            error_s = (y_true - y_hat) ** 2     
            return error_s / np.max(error_s) ** 2     # 平方误差
        elif self.loss.lower() == "exp":
            return 1 - np.exp(-errors / np.max(errors))     # 指数误差
        else:
            raise ValueError("仅支持linear, square和exp...")


    def fit(self, x_train, y_train):

        """
        AdaBoost回归算法，基学习器的训练
        1.基学习器基于带权重分布的训练集训练
        2.计算最大绝对误差、相对误差、回归误差率
        3.计算当前置信度
        4.更新下一轮的权重分布
        x_train: 训练集特征 
        y_train: 训练集标签
        """

        x_train, y_train = np.asarray(x_train), np.asarray(y_train)
        n_samples, n_class = x_train.shape[0], len(set(y_train))    # 样本量
        sample_weights = np.ones(n_samples)     # 初始化权重为1.0
        for idx in range(self.n_estimators):
            # 1.基学习器基于带权重分布的训练集训练以及预测
            self.base_estimators[idx].fit(x_train, y_train, sample_weights)
            y_hat = self.base_estimators[idx].predict(x_train)
            # 2.计算最大绝对误差、相对误差、回归误差率
            errors = self._cal_loss(y_train, y_hat)     # 相对误差
            error_rate = np.dot(errors, sample_weights / n_samples)    # 回归误差率
            # 3.计算当前置信度，基学习器的权重参数
            alpha_rate = error_rate / (1.0 - error_rate)
            self.estimator_weights.append(alpha_rate)
            # 4.更新下一轮的权重分布
            sample_weights *= np.power(alpha_rate, 1.0 - errors)
            sample_weights = sample_weights / np.sum(sample_weights) * n_samples
        # 5.计算基学习器的权重系数以及考虑学习率
        self.estimator_weights = np.log(1 / np.asarray(self.estimator_weights))
        for i in range(self.n_estimators):
            self.estimator_weights[i] *= np.power(self.learning_rate, i)
    

    def predict(self, x_test):

        """
        AdaBoost回归算法预测
        按照加权中位数以及加权平均两种结合策略
        x_test：测试样本
        """

        x_test = np.asarray(x_test)
        if self.comb_strategy == "weight_mean":     # 加权平均
            self.estimator_weights /= np.sum(self.estimator_weights)
            y_hat_mat = np.array([
                self.estimator_weights[i] * self.base_estimators[i].predict(x_test)
                for i in range(self.n_estimators)
            ])
            return np.sum(y_hat_mat, axis=0)
        elif self.comb_strategy == "weight_median":     # 加权中位数
            y_hat_mat = np.array([
                self.base_estimators[i].predict(x_test)
                for i in range(self.n_estimators)
            ]).T
            sorted_idx = np.argsort(y_hat_mat, axis=1)
            weight_cdf = np.cumsum(self.estimator_weights[sorted_idx], axis=1)
            median_or_above = weight_cdf >= 0.5 * weight_cdf[:, -1][:, np.newaxis]
            median_idx = np.argmax(median_or_above, axis=1)
            median_estimators = sorted_idx[np.arange(x_test.shape[0]), median_idx]
            return y_hat_mat[np.arange(x_test.shape[0]), median_estimators]
