import copy
import numpy as np
from threading import Thread
from sklearn.preprocessing import LabelEncoder


class ClassifyThread(Thread):

    """
    继承Thread类来创建线程，每个子线程对应一个基学习器，提高训练核的
    预测效率，并发执行各个基学习器是独立训练划分好样本
    """

    def __init__(self, target, args, kwargs):
        Thread.__init__(self)
        self.target = target    # 指定子线程调用的目标方法
        # 为target提供输入参数，输入形式是元组
        self.args = args    # 主要指以实例化的二分类基学习器
        self.kwargs = kwargs    # 不同的二分类学习器模型，所独有的参数不同
        self.result = self.target(*self.args, **self.kwargs)    # 调用目标方法进行训练/预测
    


class MultiClassifierWrapper:

    """
    多分类学习方法包装，采用线程并发执行
    """

    def __init__(self, base_classifier, mode="ovo"):

        """
        必要的参数初始化
        base_classifier: 以实例化的基学习器，包含初始化参数
        mode: 0v0和0vR两种策略
        """

        self.base_classifier = base_classifier
        self.mode = mode
        self.n_class = 0    # 标记类别数，即几分类任务
        self.classifiers = None    # 根据不同的划分策略，所训练的基学习器

    
    @staticmethod
    def fit_base_Classifier(base_classifier, x_split, y_split, **kwargs):

        """
        二分类基学习器训练，针对单个拆分的二分类任务
        base_classifier：某个基学习器
        x_split：划分的训练样本集
        y_split：划分的目标集
        """

        try:
            return base_classifier.fit(x_split, y_split, **kwargs)
        except AttributeError:
            print("请检查基学习器是否实现fit方法")
            exit(0)


    @staticmethod
    def predict_proba_base_Classifier(base_classifier, x_test):

        """
        二分类基学习器预测，针对单个拆分的二分类任务
        base_classifier：某个基学习器
        x_test：划分的训练样本集
        """

        try:
            return base_classifier.predict_proba(x_test)
        except AttributeError:
            print("请检查基学习器是否实现predict_proba方法")
            exit(0)



    def fit(self, x_train, y_train, **kwargs):

        """
        以某个二分类基学习器实现多分类学习
        x_train：训练样本
        y_train：目标集
        **kwargs：基学习器的自由参数，字典形式
        """

        x_train, y_train = np.asarray(x_train), np.asarray(y_train)
        y_train = LabelEncoder().fit_transform(y_train)    # 重编码目标集0,1,2,...
        self.n_class = len(np.unique(y_train))    # 标记类别数，即几分类任务
        if self.mode.lower() == "ovo":
            # 以类别标记划分样本，以类别标记为键
            class_sample, class_y = dict(), dict()  
            for label in np.unique(y_train):
                class_sample[label] = x_train[y_train == label]
                class_y[label] = y_train[y_train == label]
            
            # 两两配对，以不同的两个类别为键，构建基学习器
            self.classifiers, thread_tasks = {}, {}     # thread_tasks存储以构建的子线程
            for i in range(self.n_class):
                for j in range(i + 1, self.n_class):
                    # 拷贝一份基学习器
                    self.classifiers[(i, j)] = copy.deepcopy(self.base_classifier)
                    # 每次取两个类别Ci，Cj的样本,并对其进行重编码为0,1
                    sample_paris = np.r_[class_sample[i], class_sample[j]]
                    target_paris =\
                          LabelEncoder().fit_transform(np.r_[class_y[i], class_y[j]])
                    # 构建子线程
                    task = ClassifyThread(
                        target=self.fit_base_Classifier,
                        args=(self.classifiers[(i, j)], sample_paris, target_paris),
                        kwargs=kwargs
                    )
                    task.start()    # 开启子线程
                    thread_tasks[(i, j)] = task    # 存储子线程
            for i in range(self.n_class):
                for j in range(i + 1, self.n_class):
                    thread_tasks[(i, j)].join()    # 加入，并发执行训练任务

        elif self.mode.lower() == "ovr":
            self.classifiers, thread_tasks = [], []    # thread_tasks存储以构建的子线程
            for i in range(self.n_class):
                self.classifiers.append(copy.deepcopy(self.base_classifier))
                y_encode = (y_train == i).astype(int)   # 当前类别为1，否则为0
                task = ClassifyThread(
                    target=self.fit_base_Classifier,
                    args=(self.classifiers[i], x_train, y_encode),
                    kwargs=kwargs
                )
                task.start()
                thread_tasks.append(task)
            for i in range(self.n_class):
                thread_tasks[i].join()
        else:
            print("仅限于一对一(ovo)一对多(ovr)两种策略")
            exit(0)


    def predict_proba(self, x_test, **kwargs):

        """
        预测测试样本类别概率
        x_test：测试样本
        """

        x_test = np.asarray(x_test)
        if self.mode.lower() == "ovo":
            # y_test_hat为每两个类别样本的预测概率，以类别标签为键
            y_test_hat, thread_tasks = {}, {}    # thread_tasks存储以构建的子线程
            for i in range(self.n_class):
                for j in range(i + 1, self.n_class):
                    # 构建子线程
                    task = ClassifyThread(
                        target=self.predict_proba_base_Classifier,
                        args=(self.classifiers[(i, j)], x_test),
                        kwargs=kwargs
                    )
                    task.start()    # 开启子线程
                    thread_tasks[(i, j)] = task    # 存储子线程
            for i in range(self.n_class):
                for j in range(i + 1, self.n_class):
                    thread_tasks[(i, j)].join()    # 加入，并发执行训练任务
            for i in range(self.n_class):
                for j in range(i + 1, self.n_class):
                    y_test_hat[(i, j)] = thread_tasks[(i, j)].result
            total_probability = np.zeros((x_test.shape[0], self.n_class))
            for i in range(self.n_class):
                for j in range(i + 1, self.n_class):
                    # 属于不同类别的概率累加
                    total_probability[:, i] += y_test_hat[(i, j)][:, 0]
                    total_probability[:, j] += y_test_hat[(i, j)][:, 1]
            return total_probability / total_probability.sum(axis=1, keepdims=True)

        elif self.mode.lower() == "ovr":
            y_test_hat, thread_tasks = [], []    # thread_tasks存储以构建的子线程
            for i in range(self.n_class):
                task = ClassifyThread(
                    target=self.predict_proba_base_Classifier,
                    args=(self.classifiers[i], x_test),
                    kwargs=kwargs
                )
                task.start()
                thread_tasks.append(task)
            for i in range(self.n_class):
                thread_tasks[i].join()
            for i in range(self.n_class):
                y_test_hat.append(thread_tasks[i].result)
            total_probability = np.zeros((x_test.shape[0], self.n_class))
            for i in range(self.n_class):
                total_probability[:, i] = y_test_hat[i][:, 1]   # 对应编码，1是正例
            return total_probability / total_probability.sum(axis=1, keepdims=True)
        
        else:
            print("仅限于一对一(ovo)一对多(ovr)两种策略")
            exit(0)
    

    def predict(self, x_test):

        """
        预测测试样本类别
        x_test：测试样本
        """

        return np.argmax(self.predict_proba(x_test), axis=1)