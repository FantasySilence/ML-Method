import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy.linalg as lg
import scipy.stats as ss
from statsmodels.stats.outliers_influence import variance_inflation_factor

matplotlib.rcParams['font.sans-serif'] = ['STsong']
matplotlib.rcParams['axes.unicode_minus'] = False

class MultiVarLinearRegression:

    """
    多元线性回归,正规方程求解
    """

    def __init__(self, url, attribute_list, target_y, alpha=0.05):
        
        """
        参数初始化
        url: 测试集路径
        attribute_list: 自变量的样本特征名称
        target_y: 因变量的样本特征名称
        alpha: 置信度,默认为0.05
        """

        self.alpha = alpha  # 置信度,默认为0.05
        self.test_url = None    # 测试集路径
        self.data = pd.read_csv(url).dropna()   # 读取数据并去除异常值
        self.attr_label = attribute_list    # 自变量的样本特征名称
        self.target_label = target_y    # 因变量的样本特征名称
        self.x = np.array(self.data.loc[:, self.attr_label]) # 自变量样本数据
        self.X = np.hstack((np.ones((self.x.shape[0], 1)),self.x))     # 构造常数项1
        self.y = np.array(self.data.loc[:, self.target_label]) # 目标值
        self.n = self.x.shape[0]    # 样本数
        self.k = self.x.shape[1]    # 自变量个数
        self.df = self.n - self.k - 1    # 自由度
        self.w = np.empty((1, self.k+1))   # 回归系数
        self.w_ms = []  # 系数的标准误
        self.y_hat = np.zeros((self.n, 1))    # 预测值
        self.H = np.empty((self.k+1, self.k+1))   # H矩阵
        self.F = 0.0    # 整个回归的F检验
        self.fp = 0.0    # F检验的p值,与置信度相比
        self.t_stat = []    # 回归系数的t检验统计量
        self.tp = []   # t检验的p值,与置信度相比


    def cal_corr(self):

        """
        计算变量间的相关系数
        """

        self.attr_label.append(self.target_label)
        sample_x = self.data.loc[:, self.attr_label]
        corr = sample_x.corr(method="pearson")
        # 相关系数矩阵可视化，热力图
        sns.heatmap(corr, annot=True, cmap="YlGnBu")
        plt.title("相关系数矩阵热图", fontdict={"fontsize":14})
        plt.show()


    def cal_VIF(self):

        """
        多重共线性判断：基于方差膨胀因子
        vif小于5不存在共线性(共线性较弱)
        vif介于5到10之间，存在中等程度共线性
        vif大于10，存在强烈共线性，需要消除共线性，方法：去除变量，变量变换，岭回归，主成分分析
        """
        
        self.attr_label.append(self.target_label)
        sample_x = np.array(self.data.loc[:, self.attr_label[:-1]])
        vif = [variance_inflation_factor(sample_x, i) for i in range(sample_x.shape[1])]
        print("-"*100)
        print("变量间方差膨胀因子：")
        for i in range(sample_x.shape[1]):
            print(self.attr_label[i], ":", vif[i])


    def cal_coef(self):

        """
        计算回归系数,使用正规方程求解
        """

        self.H = self.X.T.dot(self.X)
        if np.all(np.linalg.eigvals(self.H) > 0):    # 正定，可逆
            self.w = lg.inv(self.H).dot(self.X.T).dot(self.y)
        else:
            print("H矩阵不是正定矩阵，无法求解")
        self.y_hat = self.X.dot(self.w.T)


    def cal_stats(self):

        """
        计算一些统计量
        """

        self.SST = ((self.y - self.y.mean())**2).sum()  # 总平方和
        self.SSR = ((self.y_hat - self.y)**2).sum()           # 残差平方和
        self.SSE = ((self.y_hat - self.y.mean())**2).sum()    # 回归平方和
        self.sigma_hat = np.sqrt(self.SSR/self.df)   # sigma的无偏估计


    def R_square(self):

        """
        R2拟合优度表示建立的模型拥有的变动程度能模拟总变动程度的百分比，剩下的1-百分比为未知变动
        修正的R2，给自变量个数添加了惩罚项，提出了自变量个数对R2的影响
        """

        self.rmse = np.sqrt(((self.y - self.y_hat)**2).sum()/self.n)
        self.r2 = 1 - self.SSR/self.SST
        self.r2_adj = 1 - (1-self.r2)*(self.n-1)/self.df


    def F_test(self):

        """
        用于检验总体回归关系的显著性
        """

        self.F = (self.SSE/self.k)/(self.SSR/self.df)
        self.fp = ss.f.sf(self.F, self.k, self.df)


    def t_test(self):

        """
        用于检验回归系数的显著性
        """

        H_inv = lg.inv(self.H)
        t = ss.t.isf(self.alpha/2, df=self.df)
        s = np.sqrt(self.SSR/self.df)
        self.confidence = []
        for i, val in enumerate(self.w):
            self.t_stat.append((val/np.sqrt(np.diag(H_inv)[i]))/np.sqrt(self.SSR/self.df))
            self.tp.append(2*ss.t.sf(np.abs(self.t_stat[i]), df=self.df))
            self.w_ms.append(self.sigma_hat*np.sqrt(np.diag(H_inv)[i]))
            self.confidence.append(t*s*np.sqrt(np.diag(H_inv)[i]))
  
    
    def format_output(self):

        """
        格式化输出回归分析报告
        """

        self.cal_coef()
        self.cal_stats()
        self.R_square()
        self.F_test()
        self.t_test()
        print("{:^25s}".format("回归统计"))
        print("="*30)
        print("{:10s}\t{:.8f}".format("相关系数", np.sqrt(self.r2)))
        print("{:10s}\t{:.8f}".format("拟合优度R2", self.r2))
        print("{:10s}\t{:.8f}".format("修正后的R2", self.r2_adj))
        print("{:10s}\t{:.8f}".format("标准误差", self.rmse))
        print("{:10s}\t{:d}".format("观测值", self.n))
        print("="*30)
        print()
        print("方差分析")
        print("="*95)
        print("\t\t{:10s}\t{:10s}\t{:10}\t{:10s}\t{:10s}".format(
            "平方和(SS)", "自由度(df)", "均方(MS)", "F统计量", "P值"
        ))
        print("{:10s}\t{:10f}\t{:10d}\t{:10f}\t{:10f}\t{:10e}".format(
            "回归分析", self.SSE, self.k, self.SSE/self.k, self.F,self.fp
        ))
        print("{:10s}\t{:10f}\t{:10d}\t{:10f}".format(
            "残差", self.SSR, self.df, self.SSR/self.df
        ))
        print("{:10s}\t{:10f}\t{:10d}".format(
            "总体", self.SST, self.n-1
        ))
        print("="*95)
        print()
        print("="*100)
        print("\t{:8s}\t{:8s}\t{:8s}\t{:8s}\t{:8s}\t{:8s}".format(
            "回归系数", "标准误差", "t统计量", "p值", "下限95%", "上限95%"
        ))
        print("{:4s}\t{:8f}\t{:8f}\t{:8f}\t{:8e}\t{:8f}\t{:8f}".format(
            "截距项", self.w[0], self.w_ms[0], self.t_stat[0],
            self.tp[0], self.w[0]-self.confidence[0], self.w[0]+self.confidence[0]
        ))
        for i in range(1, self.k+1):
            print("{:4s}\t{:8f}\t{:8f}\t{:8f}\t{:8e}\t{:8f}\t{:8f}".format(
                self.attr_label[i-1][:4], self.w[i], self.w_ms[i], self.t_stat[i],
                self.tp[i], self.w[i]-self.confidence[i], self.w[i]+self.confidence[i]
            ))
        print("="*100)
