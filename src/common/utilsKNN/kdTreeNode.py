class KDTreeNode:

    """
    KD树节点信息封装
    """

    def __init__(self, instance_node=None, instance_label=None, instance_idx=None,
                 split_feature=None, left_child=None, right_child=None, kdt_depth=None):

        """
        用于封装KD树的节点信息
        instance_node: 实例点，一个样本
        instance_label: 实例点对应的类别标签
        instance_idx: 该实例点对用的样本索引，用于KD树的可视化
        split_feature: 划分的特征属性
        left_child: 左子树，小于切分点的
        right_child: 右子树，大于切分点的
        kdt_depth: KD树的深度
        """

        self.instance_node = instance_node
        self.instance_label = instance_label
        self.instance_idx = instance_idx
        self.split_feature = split_feature
        self.left_child = left_child
        self.right_child = right_child
        self.kdt_depth = kdt_depth
