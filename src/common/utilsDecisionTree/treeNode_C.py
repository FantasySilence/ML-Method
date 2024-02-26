class C_TreeNode:

    """
    决策树分类算法，数的节点信息封装
    """

    def __init__(self, feature_idx:int=None, feature_val=None, criterion_val:float=None,
                 n_samples:int=None, target_dist:dict=None, weight_dist:dict=None,
                 left_child_Node=None, right_child_Node=None):
        
        """
        决策树节点信息封装
        feature_idx: 特征索引，如果指定特征名称，可以按照索引取值
        feature_val: 特征取值
        criterion_val: 节点划分的标准：信息增益(率)，基尼指数增益
        n_samples: 当前节点所包含的样本数量
        target_dist: 当前节点样本类别分布：
        weight_dist: 当前节点所包含的样本权重
        left_child_Node: 左子节点
        right_child_Node: 右子节点
        """
        
        self.feature_idx = feature_idx
        self.feature_val = feature_val
        self.criterion_val = criterion_val
        self.n_samples = n_samples
        self.target_dist = target_dist
        self.weight_dist = weight_dist
        self.left_child_Node = left_child_Node
        self.right_child_Node = right_child_Node

    def level_order(self):

        """
        按层次遍历树
        """

        pass