import copy
import numpy as np

from src.algorithm.decisiontree.decisionTree_R import DecisionTreeRegressor
from src.common.utilsBoost.funcutils import one_hot_encoding
from src.common.utilsBoost.funcutils import softmax


class GradientBoostClassifier:

    """
    梯度提升多分类算法：采用回归树，即训练与类别数相同的几组回归树
    每一组代表一个类别，然后对所有组输出进行softmax操作将其转换为概率分布
    在通过交叉熵损失函数求每棵树相应的负梯度，指导下一轮的训练
    """

    def __init__(self, base_estimator=None, n_estimators=10, learning_rate=1.0):
        
        """
        base_estimators：基学习器
        n_estimators：基学习器的个数
        learning_rate：学习率，降低后续基学习器的权重避免过拟合
        """

        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        # 如果不提供学习器，默认按照深度为2的决策树作为基学习器
        if self.base_estimator is None:
            self.base_estimator = DecisionTreeRegressor(max_depth=2)
        if type(self.base_estimator) != list:
            # 同质学习器
            self.base_estimator = [copy.deepcopy(self.base_estimator) 
                                    for _ in range(self.n_estimators)]
        else:
            # 异质学习器
            self.n_estimators = len(self.base_estimator)
        self.base_estimators = []   # 扩展到class_num组分类器

    
    def fit(self, x_train, y_train):

        """
        梯度提升多分类模型的训练
        x_train: 训练集特征
        y_train: 训练集标签
        """

        x_train, y_train = np.asarray(x_train), np.asarray(y_train)
        class_num = len(np.unique(y_train))     # 类别数
        y_encoded = one_hot_encoding(y_train)   # one-hot编码

        # 拷贝组分类器
        self.base_estimators = [
            copy.deepcopy(self.base_estimator) for _ in range(class_num)
        ]

        # 初始化第一轮基学习器，针对每个类别分别训练一个基学习器
        y_hat_scores = []     # 用于存储每个类别的预测值
        for c_idx in range(class_num):
            self.base_estimators[c_idx][0].fit(x_train, y_encoded[:, c_idx])
            y_hat_scores.append(self.base_estimators[c_idx][0].predict(x_train))
        y_hat_scores = np.c_[y_hat_scores].T            # 转置为列向量
        grad_y = y_encoded - softmax(y_hat_scores)    # 按类别计算负梯度

        # 训练后续基学习器，每轮针对每个类别，分别训练一个基学习器
        for idx in range(1, self.n_estimators):
            y_hat_values = []
            for c_idx in range(class_num):
                self.base_estimators[c_idx][idx].fit(x_train, grad_y[:, c_idx])
                y_hat_values.append(self.base_estimators[c_idx][idx].predict(x_train))
            y_hat_scores = y_hat_scores + np.c_[y_hat_values].T * self.learning_rate
            grad_y = y_encoded - softmax(y_hat_scores)    # 按类别计算负梯度
    

    def predict_proba(self, x_test):

        """
        预测测试样本所属类别的概率
        x_test: 测试集特征
        """

        x_test = np.asarray(x_test)
        y_hat_scores = []
        for c_idx in range(len(self.base_estimators)):
            # 取当前类别的M个基学习器
            estimator = self.base_estimators[c_idx]
            y_hat_scores.append(
                np.sum(
                    [estimator[0].predict(x_test)] +
                    [
                    self.learning_rate * estimator[i].predict(x_test)
                    for i in range(1, self.n_estimators - 1)
                    ] +
                    [estimator[-1].predict(x_test)], axis=0
                )
            )
        return softmax(np.c_[y_hat_scores].T)
    

    def predict(self, x_test):

        """
        预测测试样本所属类别
        x_test: 测试集特征
        """

        return np.argmax(self.predict_proba(x_test), axis=1)
        