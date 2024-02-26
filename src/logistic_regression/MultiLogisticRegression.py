import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class MultiLogisticRegressor:

    """
    逻辑回归，采用梯度下降算法+正则化，交叉熵损失函数，实现多分类，softmax函数
    """

    def __init__(self, fit_intercept=True, normalized=True, alpha=0.05, eps=1e-10,
                 max_epoches=300, batch_size=20, l1_ratio=None, l2_ratio=None, en_rou=None):
        
        """
        fit_intercept: 是否训练截距项\n
        normalized: 是否按训练集的数据进行标准化\n
        alpha: 学习率\n
        eps: 提前停止训练的精度要求\n
        max_epoches: 最大迭代次数\n
        batch_size: 批量大小，若为1则为随机梯度下降，若为样本数量则为批量梯度下降，否则为小批量梯度下降\n
        l1_ratio: LASSO回归惩罚项系数\n
        l2_ratio: 岭回归惩罚项系数\n
        en_rou: 弹性网络权衡l1和l2的系数
        """

        self.fit_intercept = fit_intercept
        self.normalized = normalized
        self.alpha = alpha
        self.eps = eps
        if l1_ratio:
            if l1_ratio < 0:
                raise ValueError("惩罚项系数不能为负数")
        self.l1_ratio = l1_ratio
        if l2_ratio:
            if l2_ratio < 0:
                raise ValueError("惩罚项系数不能为负数")
        self.l2_ratio = l2_ratio
        if en_rou:
            if en_rou < 0 and en_rou > 1:
                raise ValueError("弹性网络权衡系数的范围在[0,1]")
        self.en_rou = en_rou
        self.max_epoches = max_epoches
        self.batch_size = batch_size
        self.theta = None    # 模型的系数
        if self.normalized:
            self.feature_mean, self.feature_std = None, None
        self.n_samples, self.n_classes = 0, 0   # 样本量、特征属性数量
        self.train_loss, self.test_loss = [], []    # 存储训练过程中的训练损失、测试损失


    def __init_theta__(self, n_feature, n_classes):
        
        """
        初始化模型参数
        n_feature: 特征属性数量
        n_class: 类别数
        """
        
        # 初始化为0
        self.theta = np.zeros((n_feature, n_classes))


    @staticmethod
    def __one_hot_encoding__(target):

        """
        将标签转换为one-hot编码
        """

        class_labels = np.unique(target)    # 类别标签，去除重复
        target_y = np.zeros((len(target), len(class_labels)), dtype=np.int64) 
        for i, label in enumerate(target):
            target_y[i, label] = 1
        return target_y


    @staticmethod
    def __softmax_fun__(x):

        """
        softmax函数，避免溢出对x做了处理
        """

        exps = np.exp(x - np.max(x))
        exp_sum = np.sum(exps, axis=1, keepdims=True)
        return exps / exp_sum


    @staticmethod
    def __sign_func__(z_values):

        """
        符号函数，针对l1正则化
        z_values: 模型系数, 二维数组
        """

        sign = np.zeros(z_values.shape)
        sign[z_values > 0] = 1.0
        sign[z_values < 0] = -1.0
        return sign


    @staticmethod
    def cal_cross_entropy(y_test, y_prob):

        """
        计算交叉熵损失
        y_test: 样本真实值
        y_prob: 模型预测类别概率
        """

        loss = -np.sum(y_test*np.log(y_prob + 1e-8), axis=1)
        loss -= np.sum((1 - y_test)*np.log(1-y_prob + 1e-8), axis=1)
        return np.mean(loss)


    def fit(self, x_train, y_train, x_test=None, y_test=None):

        """
        样本的预处理，模型系数的求解
        x_train: 训练集特征值
        y_train: 训练集目标值
        x_test: 测试集特征值
        y_test: 测试集目标值
        """

        y_train = self.__one_hot_encoding__(y_train)    # 将标签转换为one-hot编码
        self.n_classes = y_train.shape[1]    # 类别数量
        if y_test is not None:
            y_test = self.__one_hot_encoding__(y_test)
        if self.normalized:
            self.feature_mean = np.mean(x_train, axis=0)    # 特征值的均值
            self.feature_std = np.std(x_train, axis=0) + 1e-8      # 特征值的标准差
            x_train = (x_train - self.feature_mean)/self.feature_std    # 标准化
            if x_test is not None:
                x_test = (x_test - self.feature_mean)/self.feature_std    # 标准化
        if self.fit_intercept:
            x_train = np.c_[x_train, np.ones((len(y_train), 1))]     # 在样本后加一列1
            if x_test is not None and y_test is not None:
                x_test = np.c_[x_test, np.ones((len(y_test), 1))]
        self.__init_theta__(x_train.shape[1], self.n_classes)   # 初始化模型参数
        # 训练模型
        self.__fit_gradient_desc__(x_train, y_train, x_test, y_test)


    def __fit_gradient_desc__(self, x_train, y_train, x_test, y_test):

        """
        三种梯度下降算法实现外加正则化
        x_train: 训练集特征值
        y_train: 训练集目标值
        x_test: 测试集特征值
        y_test: 测试集目标值
        """

        train_samples  = np.c_[x_train, y_train]    # 组合训练集的特征和目标，以便于随机打乱样本顺序
        for epoch in range(self.max_epoches):
            self.alpha *=0.95
            np.random.shuffle(train_samples)    # 随机打乱样本顺序
            batch_num = train_samples.shape[0]//self.batch_size    # 批量数量
            for idx in range(batch_num):
                # 按照小批量大小，选取数据
                batch_xy = train_samples[idx*self.batch_size: (idx+1)*self.batch_size]
                # 选取样本和目标值,目标值不再是一列
                batch_x, batch_y = batch_xy[:, :x_train.shape[1]], batch_xy[:, x_train.shape[1]:]    
                # 计算权重的更新增量，包含截距项
                y_prob_batch = self.__softmax_fun__(batch_x.dot(self.theta))
                delta = ((y_prob_batch-batch_y).T.dot(batch_x)/self.batch_size).T
                # 计算并添加正则化部分，不包含截距项，最后一列是截距项
                dw_reg = np.zeros(shape=(x_train.shape[1]-1, self.n_classes))
                if self.l1_ratio and self.l2_ratio is None:
                    # LASSO回归, L1正则化
                    dw_reg = self.l1_ratio*self.__sign_func__(self.theta[:-1,:])
                if self.l2_ratio and self.l1_ratio is None:
                    # 岭回归, L2正则化
                    dw_reg = 2*self.l2_ratio*self.theta[:-1,:]
                if self.en_rou and self.l1_ratio and self.l2_ratio:
                    # 弹性网络正则化
                    dw_reg = self.l1_ratio*self.en_rou*self.__sign_func__(self.theta[:-1,:])
                    dw_reg += 2*self.l2_ratio*(1-self.en_rou)*self.theta[:-1,:]
                delta[:-1,:] += dw_reg/self.batch_size
                self.theta = self.theta - self.alpha*delta
            # 计算训练过程中的交叉熵损失值
            y_train_prob = self.__softmax_fun__(x_train.dot(self.theta))
            train_cost = self.cal_cross_entropy(y_train, y_train_prob)      # 训练集的交叉熵损失均值
            self.train_loss.append(train_cost)     # 交叉熵损失
            if x_test is not None and y_test is not None:
                y_test_prob = self.__softmax_fun__(x_test.dot(self.theta))
                test_cost = self.cal_cross_entropy(y_test, y_test_prob)    # 测试集的交叉熵损失均值
                self.test_loss.append(test_cost)    # 交叉熵损失
            # 两次交叉熵损失均值小于给定的精度，则停止
            if epoch > 10 and abs(self.train_loss[-1] - self.train_loss[-2]) <= self.eps:
                break
    

    def get_params(self):

        """
        获取模型的系数
        """

        if self.fit_intercept:
            weight, bias = self.theta[:-1,:], self.theta[-1,:]
        else:
            weight, bias = self.theta, np.array([0])
        if self.normalized:
            weight = weight/self.feature_std.reshape(-1,1)
            bias = bias - weight.T.dot(self.feature_mean)
        return weight, bias
    

    def predict_prob(self, x_test):

        """
        预测测试样本的概率，第一列为y=0的概率，第二列为y=1的概率
        x_test: 测试样本
        """

        if self.normalized:
            x_test = (x_test - self.feature_mean)/self.feature_std
        if self.fit_intercept:
            x_test = np.c_[x_test, np.ones(shape=x_test.shape[0])]
        y_prob = self.__softmax_fun__(x_test.dot(self.theta))
        return y_prob
    

    def predict(self, x):

        """
        预测样本类别
        x: 测试样本
        """

        y_prob = self.predict_prob(x)
        # 对应每个样本中所有类别概率，哪个概率最大，返回哪个类别所在的列索引，即为类别
        return np.argmax(y_prob, axis=1)
    

    def plt_loss_curve(self, lab=None, is_show=True):

        """
        可视化交叉熵损失曲线
        """

        if is_show:
            plt.figure(figsize=(8, 6))
        plt.plot(self.train_loss, 'b-', lw=1.2, label="Train Loss")
        if self.test_loss:
            plt.plot(self.test_loss, 'r-', lw=1.2, label="Test Loss")
        plt.xlabel("Train Times", fontdict={"fontsize":12})
        plt.ylabel("Cross Entropy Mean", fontdict={"fontsize":12})
        plt.title("%s: Cross Entropy Curve"%lab, fontdict={"fontsize":14})
        plt.grid(ls=":")
        plt.legend(frameon=False)
        if is_show:
            plt.show()


    @staticmethod
    def plt_confusion_matrix(confusion_matrix, label_name=None, is_show=True):

        """
        可视化混淆矩阵
        """

        sns.set_theme()
        cm = pd.DataFrame(confusion_matrix, columns=label_name, index=label_name)
        sns.heatmap(cm, annot=True, cbar=False)
        acc = np.diag(confusion_matrix).sum()/confusion_matrix.sum()
        plt.title("Confusion Matrix (Acc=%.5f)"%acc, fontdict={"fontsize":14})
        plt.xlabel("Predict", fontdict={"fontsize":12})
        plt.ylabel("True", fontdict={"fontsize":12})
        if is_show:
            plt.show()
