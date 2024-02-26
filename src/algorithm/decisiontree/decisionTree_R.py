import numpy as np

from src.common.utilsDecisionTree.squareError import SquareErrorUtils
from src.common.utilsDecisionTree.treeNode_R import R_TreeNode
from src.common.utilsDecisionTree.dataBins import DataBinsWrapper


class DecisionTreeRegressor:

    """
    回归决策树CART算法，按照二叉树构造\n
    1.划分标准：平方误差最小化\n
    2.创建决策树，递归算法实现\n
    3.预测，预测概率，预测类别\n
    4.数据的预处理操作\n
    5.剪枝处理\n
    """

    def __init__(self, criterion="mse", 
                 max_depth=None, min_samples_split=2, min_sample_leaf=1,
                 min_target_std=1e-3, min_impurity_decrease=0, max_bins=10):
        
        """
        参数初始化\n
        criterion：节点划分标准\n
        is_all_feature_R：所有样本特征是否全是连续数据\n
        dbw_feature_idx：混合类型数据，可以指定连续特征属性的索引\n
        max_depth：树的最大深度，如果不传入参数则一直划分下去\n
        min_samples_split：最小的划分节点的样本数量，小于则不划分\n
        min_sample_leaf：叶子节点所包含的最小样本树，剩余样本小于这个值，标记叶子节点\n
        min_impurity_decrease：最小节点不纯度减少值，小于这个值，不足以划分\n
        max_bins：连续数据的分箱数，越大，则划分越细
        """
        
        self.utils = SquareErrorUtils()     # 节点划分类
        self.criterion = criterion          # 节点划分标准
        if criterion.lower() == "mse":
            self.criterion_func = self.utils.square_error_gain      # 平方误差增益
        else:
            raise ValueError("参数criterion仅限mse...")
        self.min_target_std = min_target_std   # 最小的目标值值标准差，小于这个值，则不划分
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_sample_leaf = min_sample_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.max_bins = max_bins
        self.root_node:R_TreeNode = None      # 回归决策树的根节点
        self.dbw = DataBinsWrapper(max_bins=max_bins)   # 连续数据离散化
        self.dbw_XrangeMap = {}   # 连续数据离散化后的分箱范围
    

    def fit(self, x_train, y_train, sample_weight=None):

        """
        回归决策树的创建，递归操作前的必要信息处理(分箱)\n
        x_train：训练样本\n
        y_train：目标集\n
        sample_weight：各样本的权重\n
        """

        x_train, y_train = np.asarray(x_train), np.asarray(y_train)
        self.class_values = np.unique(y_train)
        n_samples, n_features = x_train.shape   # 训练样本的样本量和特征属性数目
        if sample_weight is None:
            sample_weight = np.asarray([1.0]*n_samples)
        self.root_node = R_TreeNode()       # 创建一个空树
        self.dbw.fit(x_train)
        x_train = self.dbw.transform(x_train)
        self.__build_tree__(1, self.root_node, x_train, y_train, sample_weight)
    

    def __build_tree__(self, cur_depth, curr_node:R_TreeNode, x_train, y_train, sample_weight):

        """
        递归创建hi贵决策树，按照先序
        cur_depth：递归划分后的树的深度
        curr_node：递归后的当前根节点
        x_train：递归划分后的训练样本
        y_train：递归划分后的目标集合
        sample_weight：递归划分后的各样本的权重
        """

        n_samples, n_features = x_train.shape   # 当前样本子集中的样本量和特征属性数目
        # 计算当前树节点的预测值，即加权平均值
        curr_node.y_hat = np.dot(sample_weight/np.sum(sample_weight), y_train)
        curr_node.n_samples = n_samples
        
        # 递归出口判断
        curr_node.square_error = ((y_train - y_train.mean())**2).sum()
        # 所有样本目标值较为集中，样本方差很小不足以划分
        if curr_node.square_error <= self.min_target_std:
            # 如果为0，则表示当前样本集合为空，递归出口2，3
            return
        if n_samples < self.min_samples_split:   # 样本数量小于最小样本数量
            return
        if self.max_depth is not None and cur_depth > self.max_depth:   # 树的深度超过最大深度
            return
        
        # 划分标准，选择最佳的划分特征及取值
        best_idx, best_val, best_criterion_val = None, None, 0.0
        for k in range(n_features):     # 对当前样本集合中每个特征计算划分标准
            for f_val in sorted(np.unique(x_train[:,k])):   # 当前特征的不同取值
                region_x = (x_train[:,k] <= f_val).astype(int)   # 当前取值小于等于f_val则为1，否则为0
                criterion_val = self.criterion_func(region_x, y_train, sample_weight)
                if criterion_val > best_criterion_val:
                    best_criterion_val = criterion_val      # 最佳的划分标准值
                    best_idx, best_val = k, f_val       # 当前最佳的特征索引和取值
        
        # 递归出口判断
        if best_idx is None:    # 当前属性为空，或者所有的样本在所有属性上取值相同，无法划分
            return
        if best_criterion_val <= self.min_impurity_decrease:   # 最佳划分标准值小于最小不纯度减少值
            return
        curr_node.criterion_val = best_criterion_val
        curr_node.feature_idx = best_idx
        curr_node.feature_val = best_val

        # 创建左子树，并递归创建以当前节点为子树根节点的左子树
        left_idx = np.where(x_train[:,best_idx] <= best_val)      # 左子树包含的样本子集索引
        if len(left_idx) >= self.min_sample_leaf:
            left_child_Node = R_TreeNode()
            curr_node.left_child_Node = left_child_Node
            self.__build_tree__(cur_depth+1, left_child_Node, x_train[left_idx], 
                                y_train[left_idx], sample_weight[left_idx])
        
        # 创建右子树，并递归创建以当前节点为子树根节点的右子树
        right_idx = np.where(x_train[:,best_idx] > best_val)     # 右子树包含的样本子集索引
        if len(right_idx) >= self.min_sample_leaf:
            right_child_Node = R_TreeNode()
            curr_node.right_child_Node = right_child_Node
            self.__build_tree__(cur_depth+1, right_child_Node, x_train[right_idx], 
                                y_train[right_idx], sample_weight[right_idx])
    

    def __search_tree_predict__(self, cur_node:R_TreeNode, x_test):

        """
        根据测试样本从根节点到叶子节点搜索路径，判定所属区域(叶子节点)
        按后序遍历搜索
        x_test：测试样本
        """

        if cur_node.left_child_Node and x_test[cur_node.feature_idx] <= cur_node.feature_val:
            return self.__search_tree_predict__(cur_node.left_child_Node, x_test)
        elif cur_node.right_child_Node and x_test[cur_node.feature_idx] > cur_node.feature_val:
            return self.__search_tree_predict__(cur_node.right_child_Node, x_test)
        else:
            return cur_node.y_hat
            

    def predict(self, x_test):

        """
        预测样本x_test的预测值\n
        x_test：测试样本集
        """

        x_test = np.asarray(x_test)
        if self.dbw_XrangeMap is None:
            raise ValueError("请先创建回归决策树")
        x_test = self.dbw.transform(x_test)
        y_test_pred = []    # 存储样本的预测值
        for i in range(x_test.shape[0]):
            y_test_pred.append(self.__search_tree_predict__(self.root_node, x_test[i]))
        return np.asarray(y_test_pred)
    

    @staticmethod
    def cal_mse_r2(y_test, y_pred):

        """
        模型预测的均方误差MSE与判决系数R2\n
        y_test：测试样本的真实值\n
        y_pred：测试样本的预测值
        """

        y_test, y_pred = y_test.reshape(-1), y_pred.reshape(-1)
        mse = ((y_test - y_pred)**2).mean()     # 均方误差
        r2 = 1 - ((y_test - y_pred)**2).sum() / ((y_test - y_test.mean())**2).sum()    # 判决系数
        return mse, r2
    

    def __prune_node__(self, cur_node:R_TreeNode, alpha):

        """
        递归剪枝,针对决策树中的内部节点自底向上，逐个考察
        方法：后序遍历
        cur_node: 当前递归的决策树的内部节点
        alpha: 剪枝阀值，权衡模型对训练数据的拟合程度与模型复杂度
        """

        # 若左子树存在，递归剪枝左子树
        if cur_node.left_child_Node:
            self.__prune_node__(cur_node.left_child_Node, alpha) 
        # 若右子树存在，递归剪枝右子树   
        if cur_node.right_child_Node:
            self.__prune_node__(cur_node.right_child_Node, alpha)

        # 针对决策树的内部节点剪枝，非叶节点
        if cur_node.left_child_Node is not None or cur_node.right_child_Node is not None:
            for child_node in [cur_node.left_child_Node, cur_node.right_child_Node]:
                if child_node is None:
                    continue
                if child_node.left_child_Node is not None or child_node.right_child_Node is not None:
                    return
            # 计算剪枝前的损失值(平方误差)，2表示当前节点包含两个叶子节点
            pre_prune_value = 2*alpha
            if cur_node and cur_node.left_child_Node is not None:
                pre_prune_value += (0.0 if cur_node.left_child_Node.square_error is None
                                    else cur_node.left_child_Node.square_error)
            if cur_node and cur_node.right_child_Node is not None:
                pre_prune_value += (0.0 if cur_node.right_child_Node.square_error is None
                                    else cur_node.right_child_Node.square_error)

            # 计算剪枝前的损失值，当前节点即叶子节点
            after_prune_value = alpha + cur_node.square_error

            # 剪枝操作
            if after_prune_value <= pre_prune_value:
                cur_node.left_child_Node = None
                cur_node.right_child_Node = None
                cur_node.feature_idx, cur_node.feature_val = None, None
                cur_node.square_error = None
                    
                
    def prune(self, alpha=0.01):

        """
        决策树剪枝算法\n
        alpha: 剪枝阀值，权衡模型对训练数据的拟合程度与模型复杂度
        """        

        self.__prune_node__(self.root_node, alpha)
        return self.root_node