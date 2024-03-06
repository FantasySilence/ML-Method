import copy
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score

from src.algorithm.decisiontree.decisionTree_R import DecisionTreeRegressor
from src.algorithm.decisiontree.decisionTree_C import DecisionTreeClassifier


class BaggingClassifierRegressor:

    """
    1.Bagging基本流程：采样出T个含m个训练样本的采样集，然后基于每个采样集训练一个基学习器
    最后再集成
    2.预测输出进行结合：Bagging通常对分类任务采用简单投票法，对回归任务采用简单平均法
    3.把回归任务和分类任务集成到一个算法中，有参数task来控制，包外估计OOB控制
    """

    def __init__(self, base_estimator=None, n_estimators=10, task="C", OOB=False):

        """
        base_estimators：基学习器
        n_estimators：基学习器的个数
        task：任务，C表示分类，R表示回归
        OOB：是否进行OOB估计，True表示进行
        """

        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        if task.lower() not in ["c", "r"]:
            raise ValueError("Bagging任务仅限分类(C/c)和回归(R/r)")
        self.task = task
        # 如果不提供学习器，默认按照深度为2的决策树作为基学习器
        if self.base_estimator is None:
            if self.task.lower() == "c":
                self.base_estimator = DecisionTreeClassifier()
            elif self.task.lower() == "r":
                self.base_estimator = DecisionTreeRegressor()
        if type(self.base_estimator) != list:
            # 同质学习器
            self.base_estimator = [copy.deepcopy(self.base_estimator) 
                                    for _ in range(self.n_estimators)]
        else:
            # 异质学习器
            self.n_estimators = len(self.base_estimator)
        self.OOB = OOB
        self.oob_indices = []   # 保存每次又放回的采用未被使用的样本索引
        self.y_oob_hat = None   # 包括估计样本预测值(回归)或预测类别概率(分类)
        self.oob_score = None   # OOB估计的评分
    

    def fit(self, x_train, y_train):

        """
        Bagging算法(包含回归和分类)的训练
        x_train: 训练集特征
        y_train: 训练集标签
        """

        x_train, y_train = np.asarray(x_train), np.asarray(y_train)
        n_samples = x_train.shape[0]    # 样本量
        for estimator in self.base_estimator:
            # 1.有放回的随机重采样训练集
            indices = np.random.choice(n_samples, n_samples, replace=True)
            indices = np.unique(indices)    # 采样样本索引
            x_bootstrap, y_bootstrap = x_train[indices, :], y_train[indices]
            # 2.基于采样数据，训练基学习器
            estimator.fit(x_bootstrap, y_bootstrap)
            # 存储每个基学习器未使用的样本索引
            n_indices = set(np.arange(n_samples)).difference(set(indices))
            self.oob_indices.append(list(n_indices)) # 每个基学习器未参与训练的样本索引
        # 3.包外估计
        if self.OOB:
            if self.task.lower() == "c":
                self._oob_score_classifier(x_train, y_train)
            else:
                self._oob_score_regressor(x_train, y_train)
    

    def _oob_score_classifier(self, x_train, y_train):

        """
        Bagging分类任务的OOB估计
        x_train: 训练集特征
        y_train: 训练集标签
        """

        self.y_oob_hat, y_true = [], []
        for i in range(x_train.shape[0]):   # 针对每个测试样本
            y_hat_i = []    # 当前样本在每个基学习器下的预测概率
            for idx in range(self.n_estimators):    # 针对每个基学习器
                if i in self.oob_indices[idx]:      # 如果该样本属于包外估计
                    y_hat = self.base_estimator[idx].predict_proba(
                        x_train[i, np.newaxis]
                    )
                    y_hat_i.append(y_hat[0])
            if y_hat_i:     # 非空，计算个基学习器预测类别概率的均值
                self.y_oob_hat.append(np.mean(np.c_[y_hat_i], axis=0))
                y_true.append(y_train[i])
        self.y_oob_hat = np.asarray(self.y_oob_hat)
        self.oob_score = accuracy_score(y_true, np.argmax(self.y_oob_hat, axis=1))


    def _oob_score_regressor(self, x_train, y_train):
        
        """
        Bagging回归任务的OOB估计
        x_train: 训练集特征
        y_train: 训练集标签
        """

        self.y_oob_hat, y_true = [], []
        for i in range(x_train.shape[0]):   # 针对每个测试样本
            y_hat_i = []    # 当前样本在每个基学习器下的预测概率
            for idx in range(self.n_estimators):    # 针对每个基学习器
                if i in self.oob_indices[idx]:      # 如果该样本属于包外估计
                    y_hat = self.base_estimator[idx].predict(
                        x_train[i, np.newaxis]
                    )
                    y_hat_i.append(y_hat[0])
            if y_hat_i:     # 非空，计算个基学习器预测类别概率的均值
                self.y_oob_hat.append(np.mean(y_hat_i))
                y_true.append(y_train[i])
        self.y_oob_hat = np.asarray(self.y_oob_hat)
        self.oob_score = r2_score(y_true, self.y_oob_hat)


    def predict_proba(self, x_test):

        """
        分类任务中预测样本所属类别的概率
        x_test: 测试集特征
        """

        if self.task.lower() != "c":
            raise ValueError("predict_proba方法仅限分类任务")
        x_test = np.asarray(x_test)
        y_test_hat = []     # 用于存储测试样本所属类别的概率
        for estimator in self.base_estimator:
            y_test_hat.append(estimator.predict_proba(x_test))
        return np.mean(y_test_hat, axis=0)
    

    def predict(self, x_test):

        """
        分类任务：预测测试样本所属类别，类别概率大者索引为所属类别
        回归任务：预测测试样本，对每个基学习器预测值简单平均
        x_test: 测试集特征
        """

        if self.task.lower() == "c":
            return np.argmax(self.predict_proba(x_test), axis=1)
        elif self.task.lower() == "r":
            y_hat = []      # 预测值
            for estimator in self.base_estimator:
                y_hat.append(estimator.predict(x_test))
            return np.mean(y_hat, axis=0)
