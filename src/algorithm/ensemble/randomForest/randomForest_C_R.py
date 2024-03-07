import copy
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score

from src.algorithm.decisiontree.decisionTree_R import DecisionTreeRegressor
from src.algorithm.decisiontree.decisionTree_C import DecisionTreeClassifier


class RandomForestClassifierRegressor:

    """
    随机森林是Bagging的一个扩展变体，随机森林在以决策树为基学习器构建Bagging集成的基础上
    进一步在决策树的训练中引入随机属性选择，即对训练样本和输入变量增加随机扰动。
    """

    def __init__(self, base_estimator=None, n_estimators=10, feature_sampling_rate=0.5, 
                 task="C", OOB=False, feature_importance=False):

        """
        base_estimators：基学习器
        n_estimators：基学习器的个数
        task：任务，C表示分类，R表示回归
        OOB：是否进行OOB估计，True表示进行
        feature_sampling_rate：特征变量的抽样率
        feature_importance：是否仅需特征重要性的评估
        """

        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.feature_sampling_rate = feature_sampling_rate
        if task.lower() not in ["c", "r"]:
            raise ValueError("Bagging任务仅限分类(C/c)和回归(R/r)")
        self.task = task
        # 如果不提供学习器，默认按照深度为2的决策树作为基学习器
        if self.base_estimator is None:
            if self.task.lower() == "c":
                base_estimator = DecisionTreeClassifier()
            elif self.task.lower() == "r":
                base_estimator = DecisionTreeRegressor()
        self.base_estimator = [copy.deepcopy(base_estimator) 
                                for _ in range(self.n_estimators)]
        self.n_estimators = len(self.base_estimator)
        self.OOB = OOB
        self.oob_indices = []   # 保存每次又放回的采用未被使用的样本索引
        self.y_oob_hat = None   # 包括估计样本预测值(回归)或预测类别概率(分类)
        self.oob_score = None   # OOB估计的评分
        self.feature_importance = feature_importance
        self.feature_importance_scores = None   # 特征重要性分数
        self.feature_sampling_indices = []     # 储存特征变量抽样索引
    

    def fit(self, x_train, y_train):

        """
        随机森林算法(包含回归和分类)的训练
        x_train: 训练集特征
        y_train: 训练集标签
        """

        x_train, y_train = np.asarray(x_train), np.asarray(y_train)
        n_samples, n_features = x_train.shape    # 样本量, 特征数
        for estimator in self.base_estimator:
            # 1.有放回的随机重采样训练集
            indices = np.random.choice(n_samples, n_samples, replace=True)
            indices = np.unique(indices)    # 采样样本索引
            x_bootstrap, y_bootstrap = x_train[indices, :], y_train[indices]
            # 2.对特征属性变量进行抽样
            fb_num = int(self.feature_sampling_rate * n_features)   # 抽样特征数
            feature_idx = np.random.choice(n_features, fb_num, replace=False)
            self.feature_sampling_indices.append(feature_idx)
            x_bootstrap = x_bootstrap[:, feature_idx]   # 获取特征变量抽样后的训练样本
            # 3.基于采样数据，训练基学习器
            estimator.fit(x_bootstrap, y_bootstrap)
            # 存储每个基学习器未使用的样本索引
            n_indices = set(np.arange(n_samples)).difference(set(indices))
            self.oob_indices.append(list(n_indices)) # 每个基学习器未参与训练的样本索引
        # 4.包外估计
        if self.OOB:
            if self.task.lower() == "c":
                self._oob_score_classifier(x_train, y_train)
            else:
                self._oob_score_regressor(x_train, y_train)
        # 5.特征重要性估计
        if self.feature_importance:
            if self.task.lower() == "c":
                self._feature_importance_score_classifier(x_train, y_train)
            else:
                self._feature_importance_score_regressor(x_train, y_train)        
    

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
                    x_sample = x_train[i, self.feature_sampling_indices[idx]]
                    y_hat = self.base_estimator[idx].predict_proba(
                        x_sample.reshape(1, -1)
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
                    x_sample = x_train[i, self.feature_sampling_indices[idx]]
                    y_hat = self.base_estimator[idx].predict(
                        x_sample.reshape(1, -1)
                    )
                    y_hat_i.append(y_hat[0])
            if y_hat_i:     # 非空，计算个基学习器预测类别概率的均值
                self.y_oob_hat.append(np.mean(y_hat_i))
                y_true.append(y_train[i])
        self.y_oob_hat = np.asarray(self.y_oob_hat)
        self.oob_score = r2_score(y_true, self.y_oob_hat)

    
    def _feature_importance_score_classifier(self, x_train, y_train):

        """
        分类任务的特征重要性估计
        x_train: 训练集特征
        y_train: 训练集标签
        """

        n_feature = x_train.shape[1]    # 特征变量数目
        self.feature_importance_scores = np.zeros(n_feature)    # 特征重要性评分
        for f_j in range(n_feature):
            f_j_scores = []     # 存储第j个特征在所有基学习器预测的OOB误差变化
            for idx, estimator in enumerate(self.base_estimator):
                # 获取当前基学习器的特征变量索引
                f_s_indices = list(self.feature_sampling_indices[idx])    
                if f_j in f_s_indices:     # 如果f_j在当前基学习器的特征变量中
                    # 1.计算基于OOB的测试误差error
                    # OOB样本以及特征抽样
                    x_samples = x_train[self.oob_indices[idx], :][:, f_s_indices]
                    y_hat = estimator.predict(x_samples)
                    error = 1 - accuracy_score(y_train[self.oob_indices[idx]], y_hat)
                    # 2.计算第j个特征随机打乱顺序后的测试误差
                    np.random.shuffle(x_samples[:, f_s_indices.index(f_j)])
                    y_hat_j = estimator.predict(x_samples)
                    error_j = 1 - accuracy_score(y_train[self.oob_indices[idx]], y_hat_j)
                    f_j_scores.append(error_j - error)
            # 3.计算所有基学习器对当前第j个特征评分的均值
            self.feature_importance_scores[f_j] = np.mean(f_j_scores)
        return self.feature_importance_scores


    def _feature_importance_score_regressor(self, x_train, y_train):

        """
        回归任务的特征重要性估计
        x_train: 训练集特征
        y_train: 训练集标签
        """

        n_feature = x_train.shape[1]    # 特征变量数目
        self.feature_importance_scores = np.zeros(n_feature)    # 特征重要性评分
        for f_j in range(n_feature):
            f_j_scores = []     # 存储第j个特征在所有基学习器预测的OOB误差变化
            for idx, estimator in enumerate(self.base_estimator):
                # 获取当前基学习器的特征变量索引
                f_s_indices = list(self.feature_sampling_indices[idx])    
                if f_j in f_s_indices:     # 如果f_j在当前基学习器的特征变量中
                    # 1.计算基于OOB的测试误差error
                    # OOB样本以及特征抽样
                    x_samples = x_train[self.oob_indices[idx], :][:, f_s_indices]
                    y_hat = estimator.predict(x_samples)
                    error = 1 - r2_score(y_train[self.oob_indices[idx]], y_hat)
                    # 2.计算第j个特征随机打乱顺序后的测试误差
                    np.random.shuffle(x_samples[:, f_s_indices.index(f_j)])
                    y_hat_j = estimator.predict(x_samples)
                    error_j = 1 - r2_score(y_train[self.oob_indices[idx]], y_hat_j)
                    f_j_scores.append(error_j - error)
            # 3.计算所有基学习器对当前第j个特征评分的均值
            self.feature_importance_scores[f_j] = np.mean(f_j_scores)
        return self.feature_importance_scores 


    def predict_proba(self, x_test):

        """
        分类任务中预测样本所属类别的概率
        x_test: 测试集特征
        """

        if self.task.lower() != "c":
            raise ValueError("predict_proba方法仅限分类任务")
        x_test = np.asarray(x_test)
        y_test_hat = []     # 用于存储测试样本所属类别的概率
        for idx, estimator in enumerate(self.base_estimator):
            x_test_bootstrap = x_test[:, self.feature_sampling_indices[idx]]
            y_test_hat.append(estimator.predict_proba(x_test_bootstrap))
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
            for idx, estimator in enumerate(self.base_estimator):
                x_test_bootstrap = x_test[:, self.feature_sampling_indices[idx]]
                y_hat.append(estimator.predict(x_test_bootstrap))
            return np.mean(y_hat, axis=0)
