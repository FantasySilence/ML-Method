class R_TreeNode:

    """
    决策树回归算法，树的节点信息封装
    """

    def __init__(self, feature_idx: int=None, feature_val=None, y_hat=None,
                 square_error=None, n_samples: int=None, criterion_val=None,
                 left_child_Node=None, right_child_Node=None):

        """
        决策树节点信息封装
        feature_idx: 特征索引，如果指定特征属性名称，可以按索引取值
        feature_val: 特征取值
        y_hat: 当前节点的预测值
        square_error: 节点划分标准：当前节点的平方误差
        n_samples: 当前节点所包含的样本数量
        criterion_val: 节点划分的标准
        left_child_Node: 左子树
        right_child_Node: 右子树
        """

        self.feature_idx = feature_idx
        self.feature_val = feature_val
        self.y_hat = y_hat
        self.square_error = square_error
        self.n_samples = n_samples
        self.criterion_val = criterion_val
        self.left_child_Node = left_child_Node
        self.right_child_Node = right_child_Node