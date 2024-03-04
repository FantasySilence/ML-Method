import numpy as np
import matplotlib.pyplot as plt


def plt_decision_border(X, y, clf, is_show=True):

    """
    可视化分类边界
    X, y: 测试样本与类别
    clf: 分类模型
    is_show: 是否显示图像
    """

    if is_show:
        plt.figure(figsize=(7, 5), facecolor="white")
    # 根据特征变量的最大值和最小值，生成二维网格，用于绘制等值线
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xi, yi = np.meshgrid(
        np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100)
    )
    y_pred = clf.predict(np.c_[xi.ravel(), yi.ravel()])     # 模型预测值
    y_pred = y_pred.reshape(xi.shape)
    plt.contourf(xi, yi, y_pred, cmap="winter", alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k")
    plt.xlabel("Feature 1", fontdict={"fontsize":12})
    plt.ylabel("Feature 2", fontdict={"fontsize":12})
    plt.title("Naive Bayes Classification Boundary", fontdict={"fontsize":14})
    if is_show:
        plt.show()
        