import numpy as np


from src.common.utilsDecisionTree.entropy import EntropyUtils
from src.common.utilsDecisionTree.treeNode_C import C_TreeNode
from src.common.utilsDecisionTree.dataBins import DataBinsWrapper



class DecisionTreeClassifier:

    """
    分类决策树\n
    1.划分标准：信息增益(率)，基尼指数增益，都按最大值选择特征属性\n
    2.创建决策树，递归算法实现\n
    3.预测，预测概率，预测类别\n
    4.数据的预处理操作\n
    5.剪枝处理\n
    """

    def __init__(self, criterion="CART", is_all_feature_R=False, dbw_feature_idx=None,
                 max_depth=None, min_samples_split=2, min_sample_leaf=1,
                 min_impurity_decrease=0, max_bins=10):
        
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
        
        self.utils = EntropyUtils()
        self.criterion = criterion
        if criterion.lower() == "cart":
            self.criterion_func = self.utils.gini_gain      # 基尼指数增益
        elif criterion.lower() == "c45":
            self.criterion_func = self.utils.info_gain_rate # 信息增益率
        elif criterion.lower() == "id3":
            self.criterion_func = self.utils.info_gain      # 信息增益
        else:
            raise ValueError("参数criterion仅限CART, C45或ID3...")
        self.is_all_feature_R = is_all_feature_R
        self.dbw_feature_idx = dbw_feature_idx
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_sample_leaf = min_sample_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.max_bins = max_bins
        self.root_node:C_TreeNode = None      # 决策树的根节点
        self.dbw = DataBinsWrapper(max_bins=max_bins)   # 连续数据离散化
        self.dbw_XrangeMap = {}   # 连续数据离散化后的分箱范围
        self.class_values = None  # 类别标签
    

    def __data_bin_wrapper__(self, x_samples):
        
        """
        针对特定的连续特征属性索引分别进行分箱
        x_samples：训练样本
        """
        
        self.dbw_feature_idx = np.asarray(self.dbw_feature_idx)
        x_samples_prop = []   # 分箱后的数据
        if not self.dbw_XrangeMap:
            # XrangeMap为空，创建决策树前的分箱操作
            for i in range(x_samples.shape[1]):
                if i in self.dbw_feature_idx:
                    # 连续数据
                    self.dbw.fit(x_samples[:,i])
                    self.dbw_XrangeMap[i] = self.dbw.XrangeMap
                    x_samples_prop.append(self.dbw.transform(x_samples[:, i]))
                else:
                    # 离散数据
                    x_samples_prop.append(x_samples[:, i])
        else:
            for i in range(x_samples.shape[1]):
                if i in self.dbw_feature_idx:
                    # 连续数据
                    x_samples_prop.append(self.dbw.transform(x_samples[:, i], self.dbw_XrangeMap[i]))
                else:
                    # 离散数据
                    x_samples_prop.append(x_samples[:, i])
        return np.asarray(x_samples_prop).T
            

    def fit(self, x_train, y_train, sample_weight=None):

        """
        决策树的创建，递归操作前的必要信息处理\n
        x_train：训练样本\n
        y_train：目标集\n
        sample_weight：各样本的权重\n
        """

        x_train, y_train = np.asarray(x_train), np.asarray(y_train)
        self.class_values = np.unique(y_train)
        n_samples, n_features = x_train.shape   # 训练样本的样本量和特征属性数目
        if sample_weight is None:
            sample_weight = np.asarray([1.0]*n_samples)
        self.root_node = C_TreeNode()       # 创建一个空树
        if self.is_all_feature_R:           # 全部是连续变量
            self.dbw.fit(x_train)
            x_train = self.dbw.transform(x_train)
        elif self.dbw_feature_idx:   # 混合类型数据
            x_train = self.__data_bin_wrapper__(x_train)
        self.__build_tree__(1, self.root_node, x_train, y_train, sample_weight)


    def __build_tree__(self, cur_depth, curr_node:C_TreeNode, x_train, y_train, sample_weight):

        """
        递归创建决策树，按照先序
        cur_depth：递归划分后的树的深度
        curr_node：递归后的当前根节点
        x_train：递归划分后的训练样本
        y_train：递归划分后的目标集合
        sample_weight：递归划分后的各样本的权重
        """

        n_samples, n_features = x_train.shape   # 当前样本子集中的样本量和特征属性数目
        target_dist, weight_dist = {}, {}       # 当前样本类别分布和权重分布
        class_labels = np.unique(y_train)       # 不同的类别值
        for label in class_labels:
            target_dist[label] = len(y_train[y_train==label])/n_samples
            weight_dist[label] = np.mean(sample_weight[y_train==label])  
        curr_node.n_samples = n_samples
        curr_node.target_dist = target_dist
        curr_node.weight_dist = weight_dist

        # 递归出口判断
        if len(target_dist) <= 1:    # 所有样本全属于同一个类别，递归出口1
            # 如果为0，则表示当前样本集合为空，递归出口2，3
            return
        if n_samples < self.min_samples_split:   # 样本数量小于最小样本数量
            return
        if self.max_depth is not None and cur_depth > self.max_depth:   # 树的深度超过最大深度
            return
        
        # 划分标准，选择最佳的划分特征及取值
        best_idx, best_val, best_criterion_val = None, None, 0.0
        for k in range(n_features):     # 对当前样本集合中每个特征计算划分标准
            for f_val in np.unique(x_train[:,k]):   # 当前特征的不同取值
                feat_k_values = (x_train[:,k] == f_val).astype(int)   # 是当前取值f_val则为1，否则为0
                criterion_val = self.criterion_func(feat_k_values, y_train, sample_weight)
                if criterion_val > best_criterion_val:
                    best_criterion_val = criterion_val      # 最佳的划分标准值
                    best_idx, best_val = k, f_val       # 当前最佳的特征索引和取值
        
        # 递归出口判断
        if best_idx is None:    # 当前属性为空，或者所有的样本在所有属性上取值相同，无法划分
            return
        if best_criterion_val <= self.min_impurity_decrease:   # 最佳划分标准值小于最小不纯度减少值
            return
        curr_node.feature_idx = best_idx
        curr_node.feature_val = best_val
        curr_node.criterion_val = best_criterion_val

        # print("当前节点的特征索引:",best_idx, "取值:",best_val,"最佳标准值:",best_criterion_val)
        # print("当前节点的类别分布:",target_dist)

        # 创建左子树，并递归创建以当前节点为子树根节点的左子树
        left_idx = np.where(x_train[:,best_idx] == best_val)      # 左子树包含的样本子集索引
        if len(left_idx) >= self.min_sample_leaf:
            left_child_Node = C_TreeNode()
            curr_node.left_child_Node = left_child_Node
            self.__build_tree__(cur_depth+1, left_child_Node, x_train[left_idx], 
                                y_train[left_idx], sample_weight[left_idx])
        # 创建右子树，并递归创建以当前节点为子树根节点的右子树
        right_idx = np.where(x_train[:,best_idx] != best_val)     # 右子树包含的样本子集索引
        if len(right_idx) >= self.min_sample_leaf:
            right_child_Node = C_TreeNode()
            curr_node.right_child_Node = right_child_Node
            self.__build_tree__(cur_depth+1, right_child_Node, x_train[right_idx], 
                                y_train[right_idx], sample_weight[right_idx])
    

    def __search_tree_predict__(self, cur_node:C_TreeNode, x_test):

        """
        根据测试样本从根节点到叶子节点搜索路径，判定类别
        按后序遍历搜索
        x_test：测试样本
        """

        if cur_node.left_child_Node and x_test[cur_node.feature_idx] == cur_node.feature_val:
            return self.__search_tree_predict__(cur_node.left_child_Node, x_test)
        elif cur_node.right_child_Node and x_test[cur_node.feature_idx] != cur_node.feature_val:
            return self.__search_tree_predict__(cur_node.right_child_Node, x_test)
        else:
            # 叶子节点：类别，包含有类别分布
            class_p = np.zeros(len(self.class_values))      # 测试样本的类别概率
            for i, c in enumerate(self.class_values):
                class_p[i] = cur_node.target_dist.get(c, 0)*cur_node.weight_dist.get(c, 1.0)
            class_p / np.sum(class_p)       # 归一化
        return class_p
            

    def predict_proba(self, x_test):

        """
        预测样本的概率\n
        x_test：测试样本集
        """

        x_test = np.asarray(x_test)
        if self.is_all_feature_R:
            if self.dbw_XrangeMap is not None:
                x_test = self.dbw.transform(x_test)
            else:
                raise ValueError("请先创建决策树")
        elif self.dbw_feature_idx is not None:
            x_test = self.__data_bin_wrapper__(x_test)
        prob_dist = []
        for i in range(x_test.shape[0]):
            prob_dist.append(self.__search_tree_predict__(self.root_node, x_test[i]))
        return np.asarray(prob_dist)
        


    def predict(self, x_test):

        """
        预测样本的类别\n
        x_test：测试样本集
        """

        x_test = np.asarray(x_test)
        return np.argmax(self.predict_proba(x_test), axis=1)
    

    def __prune_node__(self, cur_node:C_TreeNode, alpha):

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
            # 计算剪枝前的损失值，2表示当前节点包含两个叶子节点
            pre_prune_value = 2*alpha
            for child_node in [cur_node.left_child_Node, cur_node.right_child_Node]:
                # 计算左右子节点的经验熵
                if child_node is None:
                    continue
                for key, value in child_node.target_dist.items():   # 每个叶子节点的类别分布
                    pre_prune_value += -1*child_node.n_samples*value*np.log(value)*\
                                        child_node.weight_dist.get(key, 1.0)
            # 计算剪枝前的损失值，当前节点即叶子节点
            after_prune_value = alpha
            for key, value in cur_node.target_dist.items():     # 当前待剪枝的节点的类别分布
                after_prune_value += -1*cur_node.n_samples*value*np.log(value)*\
                                        cur_node.weight_dist.get(key, 1.0)
            # 剪枝操作
            if after_prune_value <= pre_prune_value:
                cur_node.left_child_Node = None
                cur_node.right_child_Node = None
                cur_node.feature_idx, cur_node.feature_val = None, None
                    
                
    def prune(self, alpha=0.01):

        """
        决策树剪枝算法\n
        alpha: 剪枝阀值，权衡模型对训练数据的拟合程度与模型复杂度
        """        

        self.__prune_node__(self.root_node, alpha)
        return self.root_node
