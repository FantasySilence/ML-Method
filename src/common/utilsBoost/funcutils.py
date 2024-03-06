import numpy as np


def one_hot_encoding(target):

    """
    one_hot编码
    """

    class_labels = np.unique(target)    # 类别标签
    target_y = np.zeros((len(target), len(class_labels)), dtype=np.int64)
    for i, lable in enumerate(target):
        target_y[i, lable] = 1  # 对应类别所在的列为1
    return target_y


def softmax(x):

    """
    softmax函数
    """

    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps, axis=1, keepdims=True)
