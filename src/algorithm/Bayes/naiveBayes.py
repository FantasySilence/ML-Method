import numpy as np
import collections as cc
from scipy.stats import norm

from src.common.utilsBayes.dataBins import DataBinsWrapper


class NaiveBayesClassifier:

    """
    朴素贝叶斯分类器，对于连续属性采用两种方式操作
    1.分箱处理
    2.直接进行高斯分布的参数估计
    """

    def __init__(self, is_binned=False, is_feature_all_R=False, 
                 feature_R_idx=None, max_bins=10):
        
        """
        必要的参数初始化
        is_binned：连续特征变量是否进行分箱处理，离散化
        is_feature_all_R：是否所有特征变量都是连续数值
        feature_R_idx：混合式数据中，连续特征变量的索引
        max_bins：最大分箱数量
        """

        self.is_binned = is_binned
        if is_binned:
            self.is_feature_all_R = is_feature_all_R
            self.max_bins = max_bins
            self.dbw = DataBinsWrapper()    # 分箱对象
            self.dbw_XrangeMap = dict()    # 储存训练样本特征分箱的段点
        self.feature_R_idx = feature_R_idx
        self.class_values, self.n_class = None, 0  # 类别取值以及类别数量
        self.prior_prob = dict()    # 先验分布，键为类别取值，值为先验概率
        # 存储每个类所对应的特征变量取值频次或者连续属性的高斯分布参数
        self.classified_feature_prob = dict()  
        # 训练样本集每个特征不同的取值，针对离散数据
        self.feature_values_num = dict()      
        self.class_values_num = dict()    # 目标集中每个类别的样本量
    

    def _data_bin_wrapper(self, x_samples):
        
        """
        针对特定的连续特征属性索引分别进行分箱
        x_samples：训练样本
        """
        
        self.feature_R_idx = np.asarray(self.feature_R_idx)
        x_samples_prep = []     # 存储分箱后的数据
        if not self.dbw_XrangeMap:
            for i in range(x_samples.shape[1]):
                if i in self.feature_R_idx:
                    self.dbw.fit(x_samples[:, i])
                    self.dbw_XrangeMap[i] = self.dbw.XrangeMap
                    x_samples_prep.append(self.dbw.transform(x_samples[:, i]))
                else:
                    x_samples_prep.append(x_samples[:, i])
        else:
            for i in range(x_samples.shape[1]):
                if i in self.feature_R_idx:
                    x_samples_prep.append(self.dbw.transform(x_samples[:, i],
                                                             self.dbw_XrangeMap[i]))
                else:
                    x_samples_prep.append(x_samples[:, i])
        return np.asarray(x_samples_prep).T


    def fit(self, x_train, y_train):

        """
        训练模型,将涉及到的所有概率计算好并存储
        x_train：训练样本特征
        y_train：训练样本类别
        """

        x_train, y_train = np.asarray(x_train), np.asarray(y_train)
        self.class_values = np.unique(y_train)      # 类别取值
        self.n_class = len(self.class_values)       # 类别数量
        if self.n_class < 2:
            print("仅有一个类别，不进行贝叶斯分类器估计...")
            exit(0)
        self._prior_probability(y_train)    # 计算先验概率
        # 每个特征变量不同的取值数，类条件概率的分母
        for i in range(x_train.shape[1]):
            self.feature_values_num[i] = len(np.unique(x_train[:, i]))
        if self.is_binned:
            self._binned_fit(x_train, y_train)    # 分箱处理
        else:
            self._gaussian_fit(x_train, y_train)    # 高斯分布参数估计
    

    def _prior_probability(self, y_train):

        """
        计算类别的先验概率
        y_train：训练样本类别
        """

        n_samples = len(y_train)    # 总样本量
        self.class_values_num = cc.Counter(y_train) # 例如：Counter({'否': 9, '是': 8})
        for key in self.class_values_num.keys():
            self.prior_prob[key] = (self.class_values_num[key] + 1) \
                                / (n_samples + self.n_class) 


    def _binned_fit(self, x_train, y_train):

        """
        连续特征变量分箱处理,然后计算各概率值
        x_train：训练样本特征
        y_train：训练样本类别
        """

        if self.is_feature_all_R:
            self.dbw.fit(x_train)
            x_train = self.dbw.transform(x_train)
        elif self.feature_R_idx is not None:
            x_train = self._data_bin_wrapper(x_train)

        for c in self.class_values:
            class_x = x_train[y_train == c]     # 获取对应类别的样本
            # 每个离散变量特征中特定值出现的频次，连续特征变量存u, sigma
            feature_counter = dict()
            for i in range(x_train.shape[1]):
                feature_counter[i] = cc.Counter(class_x[:, i])
            self.classified_feature_prob[c] = feature_counter


    def _gaussian_fit(self, x_train, y_train):

        """
        连续特征变量不进行分箱处理,直接进行高斯分布估计，离散特征变量取值除外
        x_train：训练样本特征
        y_train：训练样本类别
        """

        for c in self.class_values:
            class_x = x_train[y_train == c]     # 获取对应类别的样本
            # 每个离散变量特征中特定值出现的频次，连续特征变量存u, sigma
            feature_counter = dict()
            for i in range(x_train.shape[1]):
                # 连续特征
                if self.feature_R_idx is not None and (i in self.feature_R_idx):
                    # 极大似然估计均值和方差
                    mu,sigma = norm.fit(np.asarray(class_x[:, i], dtype=np.float64))
                    feature_counter[i] = {"mu": mu, "sigma": sigma}
                # 离散特征
                else:
                    feature_counter[i] = cc.Counter(class_x[:, i])
            self.classified_feature_prob[c] = feature_counter
    

    def predict_proba(self, x_test):

        """
        预测样本类别的概率
        x_test：测试样本特征
        """

        x_test = np.asarray(x_test)
        if self.is_binned:
            return self._binned_predict_proba(x_test)
        else:
            return self._gaussian_predict_proba(x_test)
        
    
    def _binned_predict_proba(self, x_test):

        """
        连续特征变量分箱离散化，预测
        x_test：测试样本特征
        """

        if self.is_feature_all_R:
            x_test = self.dbw.transform(x_test)
        elif self.feature_R_idx is not None:
            x_test = self._data_bin_wrapper(x_test)
        y_test_hat = np.zeros((x_test.shape[0], self.n_class))  # 存储样本所属类别概率
        for i in range(x_test.shape[0]):
            test_sample = x_test[i, :]  # 当前测试样本
            y_hat = []  # 当前测试样本所属各个类别的概率
            for c in self.class_values:
                prob_ln = np.log(self.prior_prob[c])    # 当前类别的先验概率
                # 当前类别下不同特征变量不同取值的频次，构成字典
                feature_frequency = self.classified_feature_prob[c]    
                for j in range(x_test.shape[1]):
                    value = test_sample[j]
                    cur_feature_frequency = feature_frequency[j]
                    # 按照拉普拉斯修正计算
                    prob_ln += np.log(
                        (cur_feature_frequency.get(value, 0)+ 1)/\
                            (self.class_values_num[c]+self.feature_values_num[j])
                    )
                y_hat.append(prob_ln)       # 输入第c个类别的概率
            y_test_hat[i, :] = self.__softmax_fun__(np.asarray(y_hat))
        return y_test_hat


    @staticmethod
    def __softmax_fun__(x):

        """
        softmax函数，避免溢出对x做了处理
        """

        exps = np.exp(x - np.max(x))
        return exps / np.sum(exps)


    def _gaussian_predict_proba(self, x_test):

        """
        高斯分布参数估计，预测
        x_test：测试样本特征
        """

        y_test_hat = np.zeros((x_test.shape[0], self.n_class))  # 存储样本所属类别概率
        for i in range(x_test.shape[0]):
            test_sample = x_test[i, :]  # 当前测试样本
            y_hat = []  # 当前测试样本所属各个类别的概率
            for c in self.class_values:
                prob_ln = np.log(self.prior_prob[c])    # 当前类别的先验概率
                # 当前类别下不同特征变量不同取值的频次，构成字典
                feature_frequency = self.classified_feature_prob[c]    
                for j in range(x_test.shape[1]):
                    value = test_sample[j]
                    if self.feature_R_idx is not None and (j in self.feature_R_idx):
                        # 取极大似然估计的均值和方差
                        mu, sigma = feature_frequency[j].values()
                        prob_ln += np.log(norm.pdf(value, mu, sigma) + 1e-8)
                    else:
                        cur_feature_frequency = feature_frequency[j]
                        # 按照拉普拉斯修正计算
                        prob_ln += np.log(
                            (cur_feature_frequency.get(value, 0)+ 1)/\
                                (self.class_values_num[c]+self.feature_values_num[j])
                        )
                y_hat.append(prob_ln)       # 输入第c个类别的概率
            y_test_hat[i, :] = self.__softmax_fun__(np.asarray(y_hat))
        return y_test_hat
    

    def predict(self, x_test):

        """
        预测样本类别
        x_test：测试样本特征
        """

        return np.argmax(self.predict_proba(x_test), axis=1)
        