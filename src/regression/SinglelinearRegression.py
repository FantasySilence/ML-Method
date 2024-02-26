# Powered by: @御河DE天街

import pandas as pd
import numpy as np
import scipy.stats as ss
import pingouin as pg
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['STsong']
matplotlib.rcParams['axes.unicode_minus'] = False

class SingleVarLinearRegression:

    """
    一元线性回归\n
    1.模型系数的求解，置信区间，各种检验\n
    2.回归诊断\n
    3.残差分析\n
    4.各种可视化\n
    """

    def __init__(self, url, x_label, y_label, alpha=0.05):
        
        """
        参数初始化\n
        url: 数据集路径\n
        x_label: 自变量标签\n
        y_label: 因变量标签\n
        alpha: 置信度(默认为0.05)
        """

        self.test_url = None    # 测试集路径
        self.data = pd.read_csv(url).dropna()   # 读取数据并去除异常值
        self.x_label = x_label
        self.y_label = y_label
        self.x = self.data.loc[:, self.x_label] # 自变量样本数据
        self.y = self.data.loc[:, self.y_label] # 因变量样本数据
        self.alpha = alpha  # 置信度

        self.n = len(self.x)    # 样本量
        self.w = 0              # 回归系数
        self.b = 0              # 截距(常数项)
        self.yhat = np.empty(self.n)    # 预测值
        self.rmse = 0.0     # 均方根误差

        self.df = self.n - 2    # 自由度
        self.r2 = 0.0     # 可决系数
        self.r2_adj = 0.0     # 调整后的可决系数
        self.F = 0.0     # F统计量值
        self.p = 0.0     # p值
        self.tp = 0.0     # t检验统计量的值


    def one_varLRM(self):

        """
        一元线性回归正规方程法求解模型系数
        回归方程：y=wx+b
        """

        self.w = (self.y * (self.x - self.x.mean())).sum()/((self.x**2).sum() - self.x.sum()**2/self.n)
        self.b = (self.y - self.w*self.x).sum()/self.n
        self.yhat = self.w * self.x + self.b    # 预测值


    def stats_cal(self):

        """
        各个统计量的计算
        """

        self.Lxx = ((self.x - self.x.mean())**2).sum()
        self.TSS = ((self.y - self.y.mean())**2).sum()
        self.RSS = ((self.y - self.yhat)**2).sum()
        self.ESS = ((self.yhat - self.y.mean())**2).sum()
        self.sigma_hat = np.sqrt(self.RSS/self.df)   # sigma的无偏估计
        self.tp = ss.t.isf(self.alpha/2, df=self.df)
        self.err = self.yhat - self.y    # 实际残差，向量化的元素
        self.mse = (self.err**2).mean()    # 均方误差
        self.rmse = np.sqrt(self.mse)    # 均方根误差
    

    def coef_test(self):

        """
        回归系数的检验，包括t值,p值,标准误，置信区间
        """

        confidence_w = self.tp * self.sigma_hat / np.sqrt(self.Lxx)
        self.w_t = self.w/(self.sigma_hat / np.sqrt(self.Lxx))
        self.w_p = 2*ss.t.sf(np.abs(self.w_t), df=self.df)
        self.w_ms = self.sigma_hat / np.sqrt(self.Lxx)
        self.w_internal = [self.w-confidence_w,self.w+confidence_w]
        confidence_b = self.tp * self.sigma_hat * np.sqrt(1/self.n + self.x.mean()**2/self.Lxx)
        self.b_t = self.b/(self.sigma_hat * np.sqrt(1/self.n + self.x.mean()**2/self.Lxx))
        self.b_p = 2*ss.t.sf(np.abs(self.b_t), df=self.df)
        self.b_ms = self.sigma_hat * np.sqrt(1/self.n + self.x.mean()**2/self.Lxx)
        self.b_internal = [self.b-confidence_b,self.b+confidence_b]


    def plt_LRM(self, is_save=False):

        """
        可视化：原始数据离散点，回归方程，回归系数置信区间，预测置信区间
        """

        plt.figure(figsize=(8,6))
        plt.plot(np.array(self.x), np.array(self.y), '+', label="原始数据离散点")
        plt.plot(np.array(self.x), np.array(self.yhat), 'r-', lw=2, label="回归方程")
        confidence = self.tp*self.sigma_hat*np.sqrt(1/self.n + (self.x - self.x.mean())**2/self.Lxx)
        plt.plot(np.array(self.x), np.array(self.yhat - confidence), 'g--', lw=1, label="置信区间")
        plt.plot(np.array(self.x), np.array(self.yhat + confidence), 'g--', lw=1)
        confidence_pred = self.tp*self.sigma_hat*np.sqrt(1 + 1/self.n + (self.x - self.x.mean())**2/self.Lxx)
        plt.plot(np.array(self.x), np.array(self.yhat - confidence_pred), 'b--', lw=1, label="预测置信区间")
        plt.plot(np.array(self.x), np.array(self.yhat + confidence_pred), 'b--', lw=1)
        plt.xlabel(self.x_label, fontdict={"fontsize":12})
        plt.ylabel(self.y_label, fontdict={"fontsize":12})
        plt.legend(fontsize=12)
        plt.title("$y = %.10f*x+%.10f$"%(self.w,self.b), fontdict={"fontsize":14})
        if is_save:
            plt.savefig('D:\python_project\some_code\瞎写的玩意儿\datas\回归.png')
        plt.show()
    

    def r_square(self):

        """
        R2拟合优度表示建立的模型拥有的变动程度能模拟总变动程度的百分比，剩下的1-百分比为未知变动
        修正的R2，给自变量个数添加了惩罚项，提出了自变量个数对R2的影响
        """

        self.r2 = 1 - self.RSS/self.TSS
        k = 1 # 一元线性回归中，自变量个数为1
        self.r2_adj = 1 - (1-self.r2)*(self.n-1)/(self.n-k-1)
    

    def F_test(self):

        """
        一元线性回归的显著性检验，F检验与t检验等价
        这里进行F检验
        """

        self.F = (self.ESS/1)/(self.RSS/self.df)
        self.p = ss.f.sf(self.F, 1, self.df)


    def regression_diagnostics(self, is_save=False):

        """
        回归诊断：异常点的识别和处理
        在回归模型中，异常点包括离群点，高杠杆值点和强影响点
        一般称严重偏离既定模型的数据点为离群点，远离数据主题的点为高杠杆值点
        对统计推断影响比较大的点为强影响点，其中离群点和高杠杆值点都有可能形成强影响点
        回归完成后进行回归诊断，删除既是离群点又是强影响点的可能会使拟合效果更好
        """

        zerr = self.err/self.rmse    # 标准化残差
        outliers = dict()   # 储存离群点(异常值)，高杠杆值，强影响值点
        # 标准化残差
        ind_zerr = zerr[np.abs(zerr)>2].index.values
        outliers['1.标准化残差绝对值大于2的样本序号'] = ind_zerr
        # 学生化残差
        hi = 1/self.n + (self.x - self.x.mean())**2/self.Lxx
        serr = self.err/self.rmse/np.sqrt(1 - hi)
        ind_serr = serr[np.abs(serr)>2].index.values
        outliers['2.学生化残差绝对值大于2的样本序号'] = ind_serr
        # 高杠杆值的识别
        p = 1   # 自变量个数
        ind_hi = hi[hi>2*(p+1)/self.n].index.values
        outliers['3.高杠杆值样本序号'] = ind_hi
        # 强影响点的识别
        Di =  self.err**2/((p+1)*self.mse)*(hi/(1-hi)**2)     # cook距离
        D = np.mean(Di)     # cook距离的均值
        ind_Di = Di[Di>3*D].index.values
        outliers['4.强影响点样本序号'] = ind_Di
        print("-"*100)
        for key in outliers.keys():
            print(key + ":")
            print(outliers[key])
        print("-"*100)

        # 回归诊断可视化
        plt.figure(figsize=(18,12))
        # 标准化残差和异常值点
        ax = plt.subplot(221)
        plt.plot(range(len(self.err)), np.array(zerr), "b+")
        plt.axhline(y=-2, ls="-.", c="r")
        plt.axhline(y=2, ls="-.", c="r")
        ax.set_xlabel("样本编号")
        ax.set_ylabel("标准化残差")
        ax.set_title("标准化残差和异常值点")
        # 学生化残差和异常值点
        ax = plt.subplot(222)
        plt.plot(range(len(self.err)), np.array(serr), "r+")
        plt.axhline(y=-2, ls="-.", c="k")
        plt.axhline(y=2, ls="-.", c="k")
        ax.set_xlabel("样本编号")
        ax.set_ylabel("学生化残差")
        ax.set_title("学生化残差和异常值点")
        # 高杠杆值图和高杠杆值点
        ax = plt.subplot(223)
        plt.plot(range(len(self.err)), np.array(hi), "g+")
        plt.axhline(y=2*(p+1)/self.n, ls="-.", c="r")
        ax.set_xlabel("样本编号")
        ax.set_ylabel("高杠杆值")
        ax.set_title("高杠杆值图和高杠杆值点")
        ax = plt.subplot(224)
        # 库克距离和强影响点
        plt.plot(range(len(self.err)), np.array(Di), "y+")
        plt.axhline(y=3*D, ls="-.", c="r")
        ax.set_xlabel("样本编号")
        ax.set_ylabel("库克距离")
        ax.set_title("库克距离和强影响点")
        plt.subplots_adjust(hspace=0.3, wspace=0.2)
        if is_save:
            plt.savefig('D:\python_project\some_code\瞎写的玩意儿\datas\回归诊断.png')
        plt.show()
    

    def residual_analysis(self, is_save=False):

        """
        残差分析
        1.残差值序列图：检验残差间独立性，各观测对应的残差随机地在水平轴上下无规则地波动，则说明残差间相互独立
        2.残差与拟合值图：检验残差的同方差性，残差基本分布在上下等宽的水平条带内，说明残差是同方差的;若呈现喇叭口形
          应对因变量y做某种变换(如取平方根，取对数，取倒数等)后重新拟合
        3.残差与自变量图：检验拟合优劣，残差基本分布在左右等宽的水平条带内，说明模型与数据拟合效果较好
          若残差分布在弯曲的条带内，说明拟合不好，此时可增加x的非线性项，然后重新拟合
        4.残差直方图：检验残差正态性
        5.残差正态概率图：检验是否服从正态分布
        6.残差与滞后残差图：检验残差间是否存在自相关性，散点均匀分布于四个象限内，说明不存在自相关性
        """

        plt.figure(figsize=(18,12))
        # 残差值序列图
        ax = plt.subplot(231)
        plt.plot(range(len(self.err)), np.array(self.err), 'r+')
        plt.axhline(y=0, ls="-.", c="k")
        ax.set_xlabel("样本编号")
        ax.set_ylabel("残差值")
        ax.set_title("残差值序列图")
        # 残差与拟合值图
        ax = plt.subplot(232)
        plt.plot(np.array(self.yhat), np.array(self.err), 'b+')
        plt.axhline(y=0, ls="-.", c="k")
        ax.set_xlabel("拟合值")
        ax.set_ylabel("残差")
        ax.set_title("残差与拟合值图")
        # 残差与自变量图
        ax = plt.subplot(233)
        plt.plot(np.array(self.x), np.array(self.err), 'c+')
        plt.axhline(y=0, ls="-.", c="k")
        ax.set_xlabel("自变量值")
        ax.set_ylabel("残差")
        ax.set_title("残差与自变量图")
        # 残差直方图
        ax = plt.subplot(234)
        plt.hist(self.err, bins=25, alpha=0.6, color='m', density=True)
        # 极大似然估计残差的均值和标准差
        err_m, err_std = ss.norm.fit(self.err)
        err_val = np.linspace(min(self.err), max(self.err), 500)
        err_pdf = np.exp(-(err_val-err_m)**2/(2*err_std**2))/(np.sqrt(2*np.pi)*err_std)
        plt.plot(err_val, err_pdf, 'b-')
        ax.set_xlabel("残差")
        ax.set_ylabel("频率/概率")
        ax.set_title("残差直方图")
        # 残差正态概率图
        ax = plt.subplot(235)
        pg.qqplot(self.err, ax=ax)
        ax.set_title("残差正态概率图")
        # 残差与滞后残差图
        ax = plt.subplot(236)
        plt.plot(np.array(self.err)[1:],np.diff(self.err), 'g+')
        plt.axhline(y=0, ls="-.", c="k")
        plt.axvline(x=0, ls="-.", c="k")
        ax.set_xlabel("残差")
        ax.set_ylabel("滞后残差")
        ax.set_title("残差与滞后残差图")
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3, wspace=0.2)
        if is_save:
            plt.savefig('D:\python_project\some_code\瞎写的玩意儿\datas\残差分析.png')
        plt.show()
    
    
    def predict(self, xp):

        """
        指定点的预测以及预测区间
        xp:指定点
        """

        y_pred = self.w*xp + self.b     # 预测值
        confidence_pred = self.tp*self.sigma_hat*np.sqrt(1+1/self.n+(xp-self.x.mean())**2/self.Lxx)
        y_pred_low, y_pred_up = y_pred - confidence_pred, y_pred + confidence_pred
        print("给定点的预测值和预测区间为：")
        if len(xp)>1:
            for i, val in enumerate(xp):
                print("指定点%f的预测值是：%f,置信区间[%f, %f]"%(val, y_pred[i], y_pred_low[i], y_pred_up[i]))
        else:
            print("指定点%f的预测值是：%f,置信区间[%f, %f]"%(xp, y_pred, y_pred_low, y_pred_up))
    
    
    def format_output(self):

        """
        格式化输出回归报告
        """

        self.one_varLRM()
        self.stats_cal()
        self.coef_test()
        self.r_square()
        self.F_test()
        
        print("-"*100)
        if self.b > 0:
            print("拟合方程为：y = %.10f*x + %.10f"%(self.w, self.b))   
        else:
            print("拟合方程为：y = %.10f*x%.10f"%(self.w, self.b))
        print("-"*100)
        print()
        print("\t    "+"回归统计")
        print("="*32)
        print("%3s%24s"%('相关系数',round(np.sqrt(self.r2),8)))
        print("%5s%22s"%('拟合优度R2',round(self.r2,8)))
        print("%5s%22s"%("修正后的R2",round(self.r2_adj,8)))
        print("%3s%24s"%('标准误差',round(self.rmse,8)))
        print("%3s%18s"%('观测值',self.n))
        print("="*32)
        print()
        print("方差分析")
        print("="*95)
        print("\t\t平方和(SS)\t自由度(df)\t均方(MS)\tF统计量\t\tP值")
        print("回归分析\t%.8f\t%d\t\t%.8f\t%.8f\t%e"%(self.ESS,1,self.ESS,self.F,self.p))
        print("残差\t\t%.8f\t%d\t\t%.8f"%(self.RSS,self.df,(self.RSS/self.df)))
        print("总体\t\t%.8f\t%d"%(self.TSS,self.n-1))
        print("="*95)
        print()
        print("="*100)
        print("  \t回归系数\t标准误差\tt统计量\t\tp值\t\t下限95%\t\t上限95%")
        print("截距项\t%.8f\t%.8f\t%.8f\t%e\t%.8f\t%.8f"%(self.b,self.b_ms,self.b_t,self.b_p,self.b_internal[0],self.b_internal[1]))
        print(self.x_label+"\t%.8f\t%.8f\t%.8f\t%e\t%.8f\t%.8f"%(self.w,self.w_ms,self.w_t,self.w_p,self.w_internal[0],self.w_internal[1]))
        print("="*100)
