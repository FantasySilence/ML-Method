import numpy as np
import math

class EntropyUtils:

    """
    决策树中各种熵的计算，包括信息熵，信息增益，基尼系数
    统一要求，信息增益最大，信息增益率最大，基尼系数增益最大
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
        
        

    def cal_info_entropy(self, y_labels, sample_weight=None):

        """
        计算样本的信息熵
        y_label: 递归样本子集中的类别集合或特征取值
        sample_weight: 样本权重
        """

        y = np.asarray(y_labels)
        sample_weight = self.__set_sample_weight__(sample_weight, len(y))
        y_values = np.unique(y)     # 样本中不同类别值
        ent_y = 0.0
        for val in y_values:
            p_i = len(y[y == val])*np.mean(sample_weight[y==val]) / len(y)
            ent_y += -p_i * math.log2(p_i)
        return ent_y
    
    def conditional_entropy(self, feature_x, y_labels, sample_weight=None):

        """
        给定条件下信息熵的计算
        feature_x: 某个样本特征值
        y_label: 递归样本子集中的类别集合或特征取值
        sample_weight: 各样本权重
        """

        x, y = np.asarray(feature_x), np.asarray(y_labels)
        sample_weight = self.__set_sample_weight__(sample_weight, len(y))
        cond_ent = 0.0
        for x_val in np.unique(x):
            x_idx = np.where(x == x_val)    # 某个特征取值的样本索引集合
            sub_x, sub_y = x[x_idx], y[x_idx]
            sub_sample_weight = sample_weight[x_idx]
            p_k = len(sub_y) / len(y)
            cond_ent += p_k * self.cal_info_entropy(sub_y, sub_sample_weight)
        return cond_ent

    def info_gain(self, feature_x, y_labels, sample_weight=None):

        """
        计算信息增益
        feature_x: 某个样本特征值
        y_label: 递归样本子集中的类别集合或特征取值
        sample_weight: 各样本权重
        """  

        return self.cal_info_entropy(y_labels) -\
               self.conditional_entropy(feature_x, y_labels, sample_weight)

    def info_gain_rate(self, feature_x, y_labels, sample_weight=None):

        """
        计算信息增益率
        feature_x: 某个样本特征值
        y_label: 递归样本子集中的类别集合或特征取值
        sample_weight: 各样本权重
        """

        return self.info_gain(feature_x, y_labels, sample_weight) / \
               self.cal_info_entropy(feature_x, sample_weight)
    
    def cal_gini(self, y_labels, sample_weight=None):

        """
        计算当前特征或类别集合的基尼值
        y_label: 递归样本子集中的类别集合或特征取值
        sample_weight: 各样本权重
        """

        y = np.asarray(y_labels)
        sample_weight = self.__set_sample_weight__(sample_weight, len(y))
        y_values = np.unique(y)     # 样本中不同类别值
        gini_value = 1.0
        for val in y_values:
            p_k = len(y[y == val])*np.mean(sample_weight[y==val]) / len(y)
            gini_value -= p_k ** 2
        return gini_value
    
    def conditional_gini(self, feature_x, y_labels, sample_weight=None):

        """
        给定条件下基尼值的计算
        feature_x: 某个样本特征值
        y_label: 递归样本子集中的类别集合或特征取值
        sample_weight: 各样本权重
        """

        x, y = np.asarray(feature_x), np.asarray(y_labels)
        sample_weight = self.__set_sample_weight__(sample_weight, len(y))
        cond_gini = 0.0
        for x_val in np.unique(x):
            x_idx = np.where(x == x_val)    # 某个特征取值的样本索引集合
            sub_x, sub_y = x[x_idx], y[x_idx]
            sub_sample_weight = sample_weight[x_idx]
            p_k = len(sub_y) / len(y)
            cond_gini += p_k * self.cal_gini(sub_y, sub_sample_weight)
        return cond_gini
    
    def gini_gain(self, feature_x, y_labels, sample_weight=None):

        """
        计算基尼指数增益
        feature_x: 某个样本特征值
        y_label: 递归样本子集中的类别集合或特征取值
        sample_weight: 各样本权重
        """

        return self.cal_gini(y_labels, sample_weight) -\
               self.conditional_gini(feature_x, y_labels, sample_weight)
    
if __name__ == "__main__":
    
    import pandas as pd
    data = pd.read_csv("D:\python_project\some_code\Study\Machine_Learning\data\watermelon.csv",encoding="gbk").iloc[:,1:]
    feat_names = data.columns[:6]
    y = data.iloc[:,-1]
    ent_obj = EntropyUtils()
    for feat in feat_names:
        print(feat, ":", ent_obj.info_gain(data.loc[:, feat], y))
    print("="*70)
    for feat in feat_names:
        print(feat, ":", ent_obj.info_gain_rate(data.loc[:, feat], y))
    print("="*70)
    for feat in feat_names:
        print(feat, ":", ent_obj.gini_gain(data.loc[:, feat], y))