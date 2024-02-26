import numpy as np
import matplotlib.pyplot as plt

def plot_decision_func(X, y, clf, acc=None, title_info=None, is_show=True, support_vectors=None):
    
    """
    可视化分类的边界
    X, y: 测试样本和类别
    clf: 分类模型
    acc: 模型分类准确率，可以不传
    title_info: 可视化标题title的额外信息
    is_show: 是否显示，用于绘制子图
    support_vectors: 扩展支持向量机
    """

    if is_show:
        plt.figure(figsize=(8,6))
    # 根据特征变量的最小值，最大值，生成二维网格，用于绘制等值线
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xi, yi = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

    y_pred = clf.predict(np.c_[xi.ravel(), yi.ravel()])     # 模型预测值
    y_pred = y_pred.reshape(xi.shape)                      # 模型预测值转为二维数组
    plt.contourf(xi, yi, y_pred, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
    plt.xlabel("Feature 1", fontdict={"fontsize":12})
    plt.ylabel("Feature 2", fontdict={"fontsize":12})
    if acc:
        if title_info:
            plt.title("Model Classification Boundary %s \n(accuracy=%.5f)"%(title_info,acc), fontdict={"fontsize":14})
        else:
            plt.title("Model Classification Boundary"%acc, fontdict={"fontsize":14})
    else:
        if title_info:
            plt.title("Model Classification Boundary %s"%title_info, fontdict={"fontsize":14})
        else:
            plt.title("Model Classification Boundary", fontdict={"fontsize":14})
    # 可视化支持支持向量，针对SVM
    if support_vectors is not None:
        plt.scatter(X[support_vectors, 0], X[support_vectors, 1],
                    s=80, c="none", alpha=0.7, edgecolors="red")
    if is_show:
        plt.show()