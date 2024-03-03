import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['STsong']
matplotlib.rcParams['axes.unicode_minus'] = False

class ModelPerformanceMetrics:

    """
    模型性能度量,二分类和多分类模型的泛化性能度量
    1.计算混淆矩阵
    2.计算分类报告，模板采用sklearn.classification_report格式
    3.计算P(查准率)R(查全率)指标，并可视化P-R曲线，计算AP
    4.计算ROC的指标，真正例率，假正例率，并可视化ROC曲线，计算AUC
    5.计算代价曲线，归一化指标，正例概率代价，可视化代价曲线，并计算期望总体代价
    """

    def __init__(self, y_true, y_prob):

        """
        初始化参数
        y_true: 样本的真实类别
        y_prob: 样本的预测类别
        """

        self.y_true = np.asarray(y_true, dtype=np.int64)
        self.y_prob = np.asarray(y_prob, dtype=np.float64)
        self.n_samples, self.n_class = self.y_prob.shape      # 样本数量，类别数量
        if self.n_class > 2:
            self.y_true = self.__label_one_hot__()  # 对真实类别进行one-hot编码
        else:
            self.y_true = self.y_true.reshape(-1)
        self.cm = self.cal_confusion_matrix()
    
    def __label_one_hot__(self):

        """
        对真实类别标签进行one-hot编码，编码后的维度与模型预测概率维度相同
        """

        y_true_lab = np.zeros((self.n_samples, self.n_class))
        for i in range(self.n_samples):
            y_true_lab[i, self.y_true[i]] = 1
        return y_true_lab
    
    def cal_confusion_matrix(self):

        """
        计算并构建混淆矩阵
        """

        confusion_matrix = np.zeros((self.n_class, self.n_class), dtype=np.int64)
        for i in range(self.n_samples):
            idx = np.argmax(self.y_prob[i, :])  # 最大概率对应的索引
            if self.n_class == 2:
                idx_true = self.y_true[i]       # 第i个样本的真实类别
            else:
                idx_true = np.argmax(self.y_true[i, :])  # 第i个样本的真实类别
            if idx_true == idx:
                confusion_matrix[idx, idx] += 1     # 预测正确在对角线上加1
            else:
                confusion_matrix[idx_true, idx] += 1    # 预测错误，在真实类别行预测错误列加1
        return confusion_matrix

    def cal_classification_report(self, target_name=None):

        """
        计算并构造分类报告
        """

        precision = np.diag(self.cm)/np.sum(self.cm, axis=0)    # 查准率
        recall = np.diag(self.cm)/np.sum(self.cm, axis=1)      # 查全率
        f1_score = 2*precision*recall/(precision+recall)    # F1调和平均
        support = np.sum(self.cm, axis=1)                       # 各个类别的支持样本量
        support_all = np.sum(support)       # 总样本量
        accuracy = np.trace(self.cm)/support_all
        p_m, r_m = np.mean(precision), np.mean(recall)
        macro_avg = [p_m, r_m, 2*p_m*r_m/(p_m+r_m)]
        weight = support/support_all    # 各个类别所占总样本量的比例为权重
        weighted_avg = [np.sum(precision*weight), np.sum(recall*weight), np.sum(f1_score*weight)]

        # 构造分类报告
        metrics_1 = pd.DataFrame(np.array([precision,recall,f1_score,support]).T, 
                                 columns=["precision","recall","f1_score","support"])
        metrics_2 = pd.DataFrame([["","","",""],["","",accuracy, support_all],
                                 np.hstack([macro_avg, support_all]),
                                 np.hstack([weighted_avg, support_all])],
                                 columns=["precision","recall","f1_score","support"])
        c_report = pd.concat([metrics_1, metrics_2], ignore_index=False)
        if target_name is None:     # 未设置类别标签，则默认为0,1,2...
            target_name = [str(i) for i in range(self.n_class)]
        else:
            target_name = list(target_name)
        target_name.extend(["","accuracy","macro_avg","weighted_avg"])
        c_report.index = target_name
        return c_report
    
    @staticmethod
    def __sort_positive__(y_prob):

        """
        按照预测为正例的概率进行降序排序，并返回排序的索引向量
        """

        idx = np.argsort(y_prob)[::-1]  # 降序排列
        return idx

    def precision_recall_curve(self):

        """
        Precision和recall曲线，计算各个坐标点的值，可视化P-R曲线
        """

        pr_array = np.zeros((self.n_samples, 2))      # 用于存储每个样本预测概率作为阀值时的P和R指标
        if self.n_class == 2:   # 二分类
            idx = self.__sort_positive__(self.y_prob[:, 0])
            y_true = self.y_true[idx]       # 真值类别标签按照排序索引进行排序
            # 针对每个样本，把预测概率作为阀值，计算P和R指标
            for i in range(self.n_samples):
                tp, tn, fn, fp = self.__cal_sub_metrics__(y_true, i+1)
                pr_array[i,:] = tp/(tp+fn), tp/(tp+fp)
        else:   # 多分类
            precision = np.zeros((self.n_samples, self.n_class))    # 查准率
            recall = np.zeros((self.n_samples, self.n_class))       # 查全率
            for k in range(self.n_class):
                # 针对每个类别，分别计算P，R指标，然后平均
                idx = self.__sort_positive__(self.y_prob[:, k])
                y_true_k = self.y_true[:,k]     # 真值类别第k列
                y_true = y_true_k[idx]       # 真值类别第k列按照排序索引进行排序
                # 针对每个样本，把预测概率作为阀值，计算P和R指标
                for i in range(self.n_samples):
                    tp, tn, fn, fp = self.__cal_sub_metrics__(y_true, i+1)
                    precision[i,k] = tp/(tp+fp)     # 查准率
                    recall[i,k] = tp/(tp+fn)       # 查全率
            # 宏查准率和宏查全率
            pr_array = np.array([np.mean(recall, axis=1), np.mean(precision, axis=1)]).T
        return pr_array
    
    def roc_metrics_curve(self):

        """
        ROC曲线，计算真正例率和假正例率，并可视化
        """

        roc_array = np.zeros((self.n_samples, 2))      # 用于存储每个样本预测概率作为阀值时的TPR和FPR指标
        if self.n_class == 2:   # 二分类
            idx = self.__sort_positive__(self.y_prob[:, 0])
            y_true = self.y_true[idx]       # 真值类别标签按照排序索引进行排序
            # 针对每个样本，把预测概率作为阀值，计算TPR和FPR指标
            n_nums, p_nums = len(y_true[y_true==1]), len(y_true[y_true==0])     # 真实类别中反例与正例的样本量
            tp, tn, fn, fp = self.__cal_sub_metrics__(y_true, 1)
            roc_array[0,:] = fp/(tn+fp), tp/(tp+fn)
            for i in range(self.n_samples):
                if y_true[i] == 1:
                    roc_array[i,:] = roc_array[i-1,0] + 1/n_nums, roc_array[i-1,1]
                else:
                    roc_array[i,:] = roc_array[i-1,0], roc_array[i-1,1] + 1/p_nums
        else:   # 多分类
            fpr = np.zeros((self.n_samples, self.n_class))       # 假正例率
            tpr = np.zeros((self.n_samples, self.n_class))       # 真正例率
            for k in range(self.n_class):
                # 针对每个类别，分别计算TPR，FPR指标，然后平均
                idx = self.__sort_positive__(self.y_prob[:, k])
                y_true_k = self.y_true[:,k]     # 真值类别第k列
                y_true = y_true_k[idx]       # 真值类别第k列按照排序索引进行排序
                # 针对每个样本，把预测概率作为阀值，计算P和R指标
                for i in range(self.n_samples):
                    tp, tn, fn, fp = self.__cal_sub_metrics__(y_true, i+1)
                    fpr[i,k] = fp/(tn+fp)       # 假正例率
                    tpr[i,k] = tp/(tp+fn)       # 真正例率
            # 宏查准率和宏查全率
            roc_array = np.array([np.mean(fpr, axis=1), np.mean(tpr, axis=1)]).T
        return roc_array
    
    def fnr_fpr_metrics_curve(self):

        """
        代价曲线指标，假反例率FNR，假正例率FPR
        """

        fpr_fnr_array = self.roc_metrics_curve()
        fpr_fnr_array[:, 1] = 1 - fpr_fnr_array[:, 1]       # 计算假反例率
        return fpr_fnr_array

    def __cal_sub_metrics__(self, y_true_sort, n):

        """
        计算TP，NP，FP，TN
        y_true_sort: 排序后的真实值类别
        n: 以第n个样本的概率为阀值
        """

        if self.n_class == 2:
            pre_label = np.r_[np.zeros(n, dtype=np.int64), np.ones(self.n_samples-n, dtype=np.int64)]
            tp = len(pre_label[(pre_label==0)&(pre_label==y_true_sort)])    # 真正例
            tn = len(pre_label[(pre_label==1)&(pre_label==y_true_sort)])    # 真反例
            fp = np.sum(y_true_sort) - tn           # 假正例
            fn = self.n_samples - tp - tn - fp      # 假反例
        else:
            pre_label = np.r_[np.ones(n, dtype=np.int64), np.zeros(self.n_samples-n, dtype=np.int64)]
            tp = len(pre_label[(pre_label==1)&(pre_label==y_true_sort)])    # 真正例
            tn = len(pre_label[(pre_label==0)&(pre_label==y_true_sort)])    # 真反例
            fn = np.sum(y_true_sort) - tp           # 假正例
            fp = self.n_samples - tp - tn - fn      # 假反例

        return tp, tn, fn, fp
    
    @staticmethod
    def __cal_ap__(pr_val):

        """
        计算AP，PR曲线下面的面积
        """

        return (pr_val[1:, 0] - pr_val[:-1, 0]).dot(pr_val[1:, 1])
    
    @staticmethod
    def __cal_auc__(roc_val):

        """
        计算AUC，ROC曲线下面的面积
        """

        return (roc_val[1:, 0] - roc_val[:-1, 0]).dot((roc_val[:-1, 1]+roc_val[1:, 1])/2)
    
    @staticmethod
    def __cal_etc__(p_cost, cost_norm):

        """
        计算期望总体代价，代价曲线下方的面积
        """

        return (p_cost[1:] - p_cost[:-1]).dot((cost_norm[:-1]+cost_norm[1:])/2)
    
    def plt_PRcurve(self, pr_val, label=None, is_show=True):

        """
        可视化P-R曲线
        """

        ap = self.__cal_ap__(pr_val)
        if is_show:
            plt.figure(figsize=(8, 6))
        if label:
            plt.step(pr_val[:, 0], pr_val[:, 1], "-", where="post", lw=2, label=label+", AP = %.3f"%ap)
        else:
            plt.step(pr_val[:, 0], pr_val[:, 1], "-", where="post", lw=2)
        plt.title("P-R Curve of Test(AP=%.5f)"%ap, fontdict={"fontsize":14})
        plt.xlabel("Recall", fontdict={"fontsize":12})
        plt.ylabel("Precision", fontdict={"fontsize":12})
        plt.grid(ls=":")
        plt.legend(frameon=False,fontsize=12,loc="lower left")
        if is_show:
            plt.show()
                
    def plt_ROCcurve(self, roc_val, label=None, is_show=True):

        """
        可视化ROC曲线
        """

        auc = self.__cal_auc__(roc_val)
        if is_show:
            plt.figure(figsize=(8, 6))
        if label:
            plt.step(roc_val[:, 0], roc_val[:, 1], "-", where="post", lw=2, label=label+", AUC = %.3f"%auc)
        else:
            plt.step(roc_val[:, 0], roc_val[:, 1], "-", where="post", lw=2)
        plt.plot([0, 1], [0, 1], "--", color="navy")
        plt.title("ROC Curve(AUC=%.5f)"%auc, fontdict={"fontsize":14})
        plt.xlabel("FPR", fontdict={"fontsize":12})
        plt.ylabel("TPR", fontdict={"fontsize":12})
        plt.grid(ls=":")
        plt.legend(frameon=False,fontsize=12,loc="lower right")
        if is_show:
            plt.show()    

    def plt_cost_curve(self, fnr_fpr_vals, alpha, class_i=0):

        """
        可视化代价曲线
        fnr_fpr_vals: 假反例率和假正例率的二维数组
        alpha: alpha = cost10/cost01, 更侧重于正例预测为反例的代价，让cost01=1
        class_i: 指定绘制第i个类别的额代价曲线，如果是二分类，则默认为0
        """    

        plt.figure(figsize=(8, 6))
        fpr_s, fnr_s = fnr_fpr_vals[:, 0], fnr_fpr_vals[:, 1]       # 获取假正例率和假反例率
        cost01, cost10 = 1, alpha
        if self.n_class == 2:
            class_i = 0
        if 0<=class_i<self.n_class:
            p = np.sort(self.y_prob[:,class_i])
        else:
            p = np.sort(self.y_prob[:,0])   # 不满足条件默认为第一个类别
        positive_cost = p*cost01/(p*cost01+(1-p)*cost10)
        for fpr, fnr in zip(fpr_s, fnr_s):
            cost_norm = fnr*positive_cost + (1-positive_cost)*fpr
            plt.plot(positive_cost, cost_norm, "b-", lw=0.5)
        # 查找公共边界，计算期望总体代价
        public_cost = np.outer(fnr_s, positive_cost)+np.outer(fpr_s, (1-positive_cost))
        public_cost_min = public_cost.min(axis=0)
        plt.plot(positive_cost, public_cost_min, "r-", lw=1)    # 公共边界
        plt.fill_between(positive_cost, 0, public_cost_min, facecolor="g", alpha=0.5)
        cost_area = self.__cal_etc__(positive_cost, public_cost_min)
        plt.xlabel("Positive Probcost", fontdict={"fontsize":12})
        plt.ylabel("Normalize Cost", fontdict={"fontsize":12})
        plt.title("Cost Curve,ETC=%.8f"%cost_area, fontdict={"fontsize":14})
        plt.show()
