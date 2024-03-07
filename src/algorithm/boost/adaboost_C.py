import copy
import numpy as np

from src.algorithm.decisiontree.decisionTree_C import DecisionTreeClassifier



class AdaBoostClassifier:

    """
    adaboost分类算法，既可以二分类也可以多分类，取决于基学习器
    1.同质学习器：非列表形式，按同种基学习器构造
    2.异质学习器：列表传递[lg, svm, cart, ...]
    """

    def __init__(self, base_estimators=None, n_estimators=10, learning_rate=1.0):

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
            self.base_estimators = DecisionTreeClassifier(max_depth=2)
        if type(self.base_estimators) != list:
            # 同质学习器
            self.base_estimators = [copy.deepcopy(self.base_estimators) 
                                    for _ in range(self.n_estimators)]
        else:
            # 异质学习器
            self.n_estimators = len(self.base_estimators)
        self.estimators_weights = []    # 每个基学习器的权重系数


    def fit(self, x_train, y_train):

        """
        训练AdaBoost每个基学习器，计算权重分布，
        每个基学习器的权重系数α和误差率
        x_train: 训练集特征 m*k的二维数组
        y_train: 训练集标签
        """

        x_train, y_train = np.asarray(x_train), np.asarray(y_train)
        n_samples, n_class = x_train.shape[0], len(set(y_train))    # 样本量
        sample_weights = np.ones(n_samples)     # 初始化权重为1.0
        # 针对每个基学习器，根据带有权重分布的训练集训练基学习器，计算相关参数
        for idx in range(self.n_estimators):
            # 1.使用带有权重分布的训练集学习并预测
            self.base_estimators[idx].fit(x_train, y_train, sample_weights)
            # 只关心分类错误的，如果错误则为0，否则为1
            y_hat_0 = (self.base_estimators[idx].predict(x_train)==y_train).astype(int)
            # 2.计算误差率
            error_rate = sample_weights.dot(1.0 - y_hat_0) / n_samples
            if error_rate > 0.5:
                # 当前基学习器没用
                self.estimators_weights.append(0)
                continue
            # 3.计算基学习器的权重分布
            alpha_rate = 0.5 * np.log((1.0 - error_rate) / (error_rate + 1e-8)) \
                + np.log(n_class - 1.0)
            alpha_rate = min(10.0, alpha_rate)  # 限制最大值为10.0, 避免过大
            self.estimators_weights.append(alpha_rate)
            # 4.更新样本权重分布
            sample_weights *= np.exp(-1.0 * alpha_rate * np.power(-1.0, 1 - y_hat_0))
            sample_weights = sample_weights / np.sum(sample_weights) * n_samples
        # 5.更新estimators的权重系数，按照学习率
        for i in range(self.n_estimators):
            self.estimators_weights[i] *= np.power(self.learning_rate, i)
    

    def predict_proba(self, x_test):

        """
        预测测试样本所属类别的概率，软投票
        x_test：测试样本
        """

        x_test = np.asarray(x_test)
        # 按照加法模型，线性组合基学习器
        y_hat_prob = np.sum([
            self.base_estimators[i].predict_proba(x_test) * self.estimators_weights[i]
            for i in range(self.n_estimators)
        ], axis=0)
        return y_hat_prob / y_hat_prob.sum(axis=1, keepdims=True)
    

    def predict(self, x_test):

        """
        预测测试样本所属类别
        x_test：测试样本
        """
        
        return np.argmax(self.predict_proba(x_test), axis=1)    
        