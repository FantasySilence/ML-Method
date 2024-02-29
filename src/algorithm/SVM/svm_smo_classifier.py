import copy
import random
import numpy as np
import matplotlib.pyplot as plt

from src.common.utilsSVM.kernelFunc import linearKernel
from src.common.utilsSVM.kernelFunc import polyKernel
from src.common.utilsSVM.kernelFunc import rbfKernel



class SVMClassifier:

    """
    支持向量机二分类算法, 基于SMO算法
    硬间隔，软间隔，核函数，可做线性可分、非线性可分
    1.训练样本的目标值限定编码为{0, 1}, SVM在fit时把"0"重编码为"-1"
    """

    def __init__(self, C=1.0, gamma=1.0, degree=3, coef0=1.0, kernel=None,
                 alpha_tol=1e-3, kkt_tol=1e-3, max_epochs=100):

        """
        一些必要的参数初始化
        C：软硬间隔参数，硬间隔适当增加，软间隔适当减小
        gamma：径向基函数/高斯核函数超参数
        degree：多项式核函数的阶次
        coef0：多项式核函数的常数项
        kernel：选择的核函数
        """

        self.C = C
        self.kernel = kernel
        if self.kernel is None or self.kernel.lower() == "linear":
            # 线性核函数
            self.kernel_func = linearKernel()
        elif self.kernel.lower() == "poly":
            # 多项式核函数
            self.kernel_func = polyKernel(degree, coef0)
        elif self.kernel.lower() == "rbf":
            self.kernel_func = rbfKernel(gamma)
        else:
            raise ValueError("仅限Linear,Poly或RBF，值为None时默认为Linear")
        
        self.alpha_tol = alpha_tol  # 支持向量的容忍度
        self.kkt_tol = kkt_tol      # 在精度内检查
        self.max_epochs = max_epochs
        self.alpha = None       # 松弛变量
        self.E = None           # 误差
        self.w, self.b = None, None     # 模型参数

        self.support_vectors = []      # 记录支持向量的索引
        self.support_vector_alpha = [] # 记录支持向量的松弛变量
        # 记录支持向量所对应的样本和目标
        self.support_vector_x, self.support_vector_y = [], []
        self.cross_entropy_loss = []  # 记录损失函数值
    

    def init_params(self, x_train, y_train):

        """
        初始化必要参数
        x_train: 训练集
        y_train: 编码后训练集的目标值
        """

        n_samples, n_features = x_train.shape
        self.w, self.b = np.zeros(n_features), 0.0      # 模型系数初始化为0
        self.alpha = np.zeros(n_samples)    # 松弛变量
        self.E = self.decision_func(x_train) - y_train   # 初始化误差，所有所有样本类别取反

    
    def decision_func(self, x):

        """
        SVM模型的计算
        x：可以是样本集，也可以是单个样本
        """

        if len(self.support_vectors) == 0:  # 当前没有支持向量
            if x.ndim <= 1:     # 单个样本
                return 0
            else:
                return np.zeros(x.shape[0])
        else:
            if x.ndim <= 1:
                wt_x = 0.0
            else:
                wt_x = np.zeros(x.shape[0])
            for i in range(len(self.support_vectors)):
                wt_x += self.support_vector_alpha[i] * self.support_vector_y[i] * \
                    self.kernel_func(x, self.support_vector_x[i])
        return wt_x + self.b    # 计算结果加上b值


    def _meet_kkt(self, x_i, y_i, alpha_i, sample_weight_i):

        """
        判定当前需要优化的alpha是否满足KKT条件
        x_i: 已选择的样本
        y_i: 已选择的目标
        alpha_i: 需要优化的alpha
        """

        if alpha_i < self.C * sample_weight_i:
            return y_i * self.decision_func(x_i) >= 1 - self.kkt_tol
        else:
            return y_i * self.decision_func(x_i) <= 1 + self.kkt_tol
    

    def _select_best_j(self, best_i):

        """
        基于已经选择的第一个alpha_i，寻找误差|E_i - E_j|最大的best_j
        best_i: 已经选择的第一个alpha_i的索引
        """

        valid_j_list = [j for j in range(len(self.alpha))
                        if self.alpha[j] > 0 and best_i != j]
        if len(valid_j_list) > 0:
            # 在可选的j列表中找到绝对误差最大的索引
            idx = np.argmax(np.abs(self.E[best_i] - self.E[valid_j_list]))
            best_j = valid_j_list[int(idx)]      # 最佳的j
        else:
            idx = list(range(len(self.alpha)))      # 所有样本的索引
            seq = idx[:best_i] + idx[best_i+1:]     # 排除best_i
            best_j = random.choice(seq)     # 随机选择
        return best_j
    

    def _clip_alpha_j(self, y_i, y_j, alpha_j_unc, 
                      alpha_i_old, alpha_j_old, sample_weight_j):

        """
        修剪第二个更新的alpha值
        y_i: 当前选择的第1个y目标值
        y_j: 当前选择的第2个y目标值
        alpha_j_unc: 当前为修剪的第2个alpha值
        alpha_i_old: 当前选择的第1个未更新的alpha值
        alpha_j_old: 当前选择的第2个未更新的alpha值
        """

        C = self.C * sample_weight_j
        if y_i == y_j:
            inf = max(0, alpha_i_old + alpha_j_old - C)    # L
            sup = min(C, alpha_i_old + alpha_j_old)    # U
        else:
            inf = max(0, alpha_j_old - alpha_i_old)    # L
            sup = min(C, C + alpha_j_old - alpha_i_old)    # U
        alpha_j_new = [inf if alpha_j_unc < inf 
                       else sup if alpha_j_unc > sup
                       else alpha_j_unc]
        return alpha_j_new[0]
    

    def _update_error(self, x_train, y_train, y):

        """
        更新误差，计算交叉熵损失
        x_train：训练集
        y_train：目标集
        y：编码后的目标集
        """

        y_predict = self.decision_func(x_train)     # 当前优化过程中的模型预测值
        self.E = y_predict - y     # 更新误差
        loss = -(y_train.T.dot(np.log(self.sigmoid(y_predict))) +\
                  (1 - y_train).T.dot(np.log(1 - self.sigmoid(y_predict))))
        self.cross_entropy_loss.append(loss / len(y))
        
    

    def fit(self, x_train, y_train, sample_weight=None):

        """
        SVM的训练：基于SMO算法实现
        1.按照启发式方法选择一对需要优化的alpha
        2.对参数alpha, b, w, E等仅限更新和修剪
        x_train: 训练集
        y_train: 训练集的目标值
        """

        x_train, y_train = np.asarray(x_train), np.asarray(y_train)
        if sample_weight is None:
            sample_weight = np.array([1] * x_train.shape[0])
        class_values = np.sort(np.unique(y_train))
        if class_values[0] != 0 or class_values[1] != 1:
            raise ValueError("仅限二分类，类别编码为{0, 1}...")
        y = copy.deepcopy(y_train)
        y[y == 0] = -1      # SVM类别限定为{-1, 1}
        self.init_params(x_train, y)    # 初始化参数
        n_samples = x_train.shape[0]    # 样本量
        for epoch in range(self.max_epochs):
            if_all_match_kkt_condition = True       # 表示所有样本都满足KKT条件

            # 1.选择一对需要优化的alpha_i和alpha_j
            for i in range(n_samples):      # 外层循环
                x_i, y_i = x_train[i, :], y[i]       # 当前选择的第一个需要优化的样本
                # 取当前为更新的alpha, 误差
                alpha_i_old, err_i_old = self.alpha[i], self.E[i] 
                # 不满足KKT条件
                if not self._meet_kkt(x_i, y_i, alpha_i_old, sample_weight[i]):
                    if_all_match_kkt_condition = False  # 表示存在需要优化的变量
                    j = self._select_best_j(i)  # 基于alpha_i选择alpha_j
                    # 当前第二个需优化的alpha和误差
                    alpha_j_old, err_j_old = self.alpha[j], self.E[j]
                    # 第二个需要优化的alpha所对应的样本
                    x_j, y_j = x_train[j, :], y[j]

                    # 2.基于已经选择的alpha_i和alpha_j, 对alpha,b,E,w进行更新
                    # 首先获取未修剪的第2个需要更新的alpha值
                    k_ii, k_jj, k_ij = self.kernel_func(x_i, x_i),\
                          self.kernel_func(x_j, x_j), self.kernel_func(x_i, x_j)
                    eta = k_ii + k_jj - 2 * k_ij
                    if np.abs(eta) < 1e-3:  # 避免分母过小，表示选择跟新的两个样本比较接近
                        continue
                    # 未修剪的alpha_j
                    alpha_j_unc = alpha_j_old - y_j * (err_j_old - err_i_old) / eta
                    # 修剪alpha_j使得满足0 < alpha_j < C
                    alpha_j_new = self._clip_alpha_j(y_i, y_j, alpha_j_unc, 
                                                     alpha_i_old, alpha_j_old,
                                                     sample_weight[j])
                    
                    # 3.通过修剪后的alpha_j_new更新alpha_i
                    alpha_j_delta = alpha_j_new - alpha_j_old
                    alpha_i_new = alpha_i_old - y_i * y_j * alpha_j_delta
                    self.alpha[i], self.alpha[j] = alpha_i_new, alpha_j_new   # 更新回存

                    # 4.更新模型系数w,b
                    alpha_i_delta = alpha_i_new - alpha_i_old
                    # w的更新仅与已更新的一对alpha有关
                    self.w += alpha_i_delta * y_i * x_i + alpha_j_delta * y_j * x_j
                    b_i_delta = -self.E[i] - y_i * k_ii * alpha_i_delta - y_j * k_ij * alpha_j_delta
                    b_j_delta = -self.E[j] - y_i * k_ij * alpha_i_delta - y_j * k_jj * alpha_j_delta
                    if 0 < alpha_i_new < self.C * sample_weight[i]:
                        self.b += b_i_delta
                    elif 0 < alpha_j_new < self.C * sample_weight[j]: 
                        self.b += b_j_delta
                    else:
                        self.b += (b_i_delta + b_j_delta) / 2
                    
                    # 5.更新误差
                    self._update_error(x_train, y_train, y)

                    # 6.更新支持向量相关的信息，即重新选取
                    self.support_vectors = np.where(self.alpha > self.alpha_tol)[0]
                    self.support_vector_x = x_train[self.support_vectors, :]
                    self.support_vector_y = y[self.support_vectors]
                    self.support_vector_alpha = self.alpha[self.support_vectors] 

            if if_all_match_kkt_condition:      # 没有需要优化的alpha，则提取停机
                break
    

    def get_params(self):

        """
        SVM模型的参数
        """

        return self.w, self.b
    

    def predict_proba(self, x_test):

        """
        预测函数
        x_test: 测试集
        """

        x_test = np.asarray(x_test)
        y_test_hat = np.zeros((x_test.shape[0], 2))
        y_test_hat[:, 1] = self.sigmoid(self.decision_func(x_test))
        y_test_hat[:, 0] = 1.0 - y_test_hat[:, 1]
        return y_test_hat
    

    def predict(self, x_test):

        """
        预测函数
        x_test: 测试集
        """

        return np.argmax(self.predict_proba(x_test), axis=1)
    

    @staticmethod
    def sigmoid(x):

        """
        sigmoid函数
        """

        x = np.asarray(x)
        x[x > 30.0] = 30.0      # 避免上溢
        x[x < -50.0] = -50.0    # 避免下溢
        return 1 / (1 + np.exp(-x))


    def plt_loss_curve(self, is_show=True):

        """
        可视化交叉熵损失曲线
        """

        if is_show:
            plt.figure(figsize=(7, 5))
        plt.plot(self.cross_entropy_loss, "k-", lw=1)
        plt.xlabel("Train Epochs", fontdict={"fontsize":12})
        plt.ylabel("The Mean of Cross Entropy Loss", fontdict={"fontsize":12})
        plt.title("The SVM Loss Curve of Cross Entropy")
        plt.grid(":")
        if is_show:
            plt.show()

    
    def plt_svm(self, X, y, is_show=True, is_margin=False):

        """
        可视化支持向量机模型，分类边界，样本，间隔，支持向量
        """

        # 可视化分类填充区域
        X, y = np.asarray(X), np.asarray(y)
        if is_show:
            plt.figure(figsize=(7, 5))
        if X.shape[1] != 2:
            raise ValueError("只支持二维数据...")
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xi, yi = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))
        zi = self.predict(np.c_[xi.ravel(), yi.ravel()])
        zi = zi.reshape(xi.shape)   # 等值线x,y,z的维度必须一致
        plt.contourf(xi, yi, zi, cmap="winter", alpha=0.3)

        # 可视化正例，负例样本
        positive, negative = X[y == 1.0], X[y == 0.0]
        plt.plot(positive[:, 0], positive[:, 1], "*", label="Positive Sample")
        plt.plot(negative[:, 0], negative[:, 1], "x", label="Negative Sample")

        # 可视化支持向量
        if len(self.support_vectors) != 0:
            plt.scatter(self.support_vector_x[:, 0], self.support_vector_x[:, 1],
                        s=80, c="none", edgecolors="k", label="Support Vectors")
        
        if is_margin and (self.kernel is None or self.kernel.lower() == "linear"):
            w, b = self.get_params()
            xi_ = np.linspace(x_min, x_max, 100)
            yi_ = -(w[0] * xi_ + b) / w[1]
            margin = 1 / w[1]
            plt.plot(xi_, yi_, "r-", lw=1.5, label="Decision Boundary")
            plt.plot(xi_, yi_ + margin, "k:", lw=1.5, label="Maximum Margin")
            plt.plot(xi_, yi_ - margin, "k:", lw=1.5)
        plt.title("Support Vector Machine Decision Boundary", fontdict={"fontsize":14})
        plt.xlabel("X1", fontdict={"fontsize":12})
        plt.ylabel("X2", fontdict={"fontsize":12})
        plt.legend(frameon=False)
        plt.axis([x_min, x_max, y_min, y_max])

        if is_show:
            plt.show()
