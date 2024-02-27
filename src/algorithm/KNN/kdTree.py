import heapq
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
from src.common.utilsKNN.distUtils import DistanceUtils
from src.common.utilsKNN.kdTreeNode import KDTreeNode



class KNearestNeighborKDTree:

    """
    KNN算法的实现，基于KD树结构
    1.fit：特征向量空间的划分，即构建KD树(建立KNN算法模型)
    2.predict: 预测，临近搜索
    3.可视化KD树
    """

    def __init__(self, k: int=5, p=2, view_kdt:bool=False):

        """
        KNN算法必要的参数初始化
        k: 近邻数
        p: 距离度量标准
        view_kdt: 是否可视化KD树
        """

        self.k = k      # 预测，近邻搜索时使用的参数，表示近邻数
        self.p = p      # 预测，近邻搜索时使用的参数，表示样本的近似度
        self.view_kdt = view_kdt

        self.dist_utils = DistanceUtils(p=self.p)   # 距离度量工具类
        self.kdt_root: KDTreeNode = None         # KD树根节点
        self.k_dimension = 0    # 特征空间维度，即样本的特征属性数
        self.k_neighbors = []    # 用于记录某个测试样本的近邻实例点
    

    def fit(self, x_train, y_train):

        """
        递归创建KD树，即对特征向量空间进行划分
        x_train: 训练样本集
        y_train: 训练样本集目标集合
        """

        if self.k < 1:
            raise ValueError("近邻数必须大于0")
        x_train, y_train = np.asarray(x_train), np.asarray(y_train)
        self.k_dimension = x_train.shape[1]     # 特征维度
        idx_array = np.arange(x_train.shape[0])  # 记录训练样本集的索引
        self.kdt_root = self.__build_kd_tree__(x_train, y_train, idx_array, 0)
        if self.view_kdt:
            self.draw_kd_tree()     # 可视化KD树
    

    def __build_kd_tree__(self, x_train, y_train, idx_array, kdt_depth):

        """
        递归创建KD树，KD树二叉树
        严格区分左右子树，表示对K维空间的一个划分
        x_train: 递归划分的训练样本集
        y_train: 递归划分的训练样本集目标集合
        idx_array: 递归划分的训练样本集索引
        depth: KD树的深度
        """

        if x_train.shape[0] == 0:   # 递归出口
            return 
        
        split_dimension = kdt_depth % self.k_dimension  # 数据的划分维度
        sorted(x_train, key=lambda x: x[split_dimension])  # 按照某个划分维度进行排序
        median_idx = x_train.shape[0] // 2  # 中位数所对应的数据的索引
        median_node = x_train[median_idx]  # 切分点作为当前子树的根节点

        # 划分左右子树区域
        left_instance, right_instance =\
              x_train[:median_idx], x_train[median_idx+1:]
        
        left_label, right_label =\
              y_train[:median_idx], y_train[median_idx+1:]
        
        left_idx, right_idx =\
              idx_array[:median_idx], idx_array[median_idx+1:]
        
        # 递归调用
        left_child = self.__build_kd_tree__(left_instance, left_label, 
                                            left_idx, kdt_depth+1)
        
        right_child = self.__build_kd_tree__(right_instance, right_label, 
                                             right_idx, kdt_depth+1)

        kdt_new_node = KDTreeNode(median_node, y_train[median_idx], 
                                  idx_array[median_idx], split_dimension, 
                                  left_child, right_child, kdt_depth)
        return kdt_new_node
    

    def __search_kd_tree__(self, kd_tree: KDTreeNode, x_test):

        """
        递归搜索KD树，后序遍历
        kd_tree: 以构建的KD树
        x_test: 测试样本
        """

        if kd_tree is None:     # 递归出口
            return
        # 测试样本与当前KD子树的根节点的距离
        distance = self.dist_utils.distance_func(kd_tree.instance_node, x_test)
        # 1.如果不够K个样本，继续递归
        # 2.如果搜索了K个样本，但是K个样本未必是最近邻的
        if (len(self.k_neighbors) < self.k) or (distance < self.k_neighbors[-1]["distance"]):
            self.__search_kd_tree__(kd_tree.left_child, x_test)     # 递归左子树
            self.__search_kd_tree__(kd_tree.right_child, x_test)    # 递归右子树
            # 整个搜索路径上的KD树节点存储在self.k_neighbors
            # 包含三个值，当前实例点，类别，距离
            self.k_neighbors.append({
                "node": kd_tree.instance_node,
                "label": kd_tree.instance_label,
                "distance": distance
            })
            # 按照距离排序，选择最小的K个近邻
            self.k_neighbors = heapq.nsmallest(self.k, self.k_neighbors,
                                                key=lambda x: x["distance"])
            



    def predict(self, x_test):

        """
        KD树的近邻搜索，即测试样本的预测
        x_test: 测试样本集
        """

        x_test = np.asarray(x_test)
        if self.kdt_root is None:
            raise ValueError("请先训练KD树")
        elif x_test.shape[1] != self.k_dimension:
            raise ValueError("测试样本集的特征维度与训练样本集不一致")
        else:
            y_test_hat = []     # 记录测试样本的预测结果
            for i in range(x_test.shape[0]):
                self.k_neighbors = []   # 调用递归搜索，则包含了k个最近邻的实例点
                self.__search_kd_tree__(self.kdt_root, x_test[i])
                y_test_labels = []
                # 取出每个近邻样本的类别标签
                for k in range(self.k):
                    y_test_labels.append(self.k_neighbors[k]["label"])  
                # 按分类规则(多数表决法)
                counter = Counter(y_test_labels)
                idx = int(np.argmax(list(counter.values())))
                y_test_hat.append(list(counter.keys())[idx])
        return np.asarray(y_test_hat)
    

    def __create_kd_tree__(self, graph, kdt_node: KDTreeNode, pos=None,
                           x=0, y=0, layer=1):
        
        """
        递归可视化KD树
        graph: 有向图对象
        kdt_node: 递归创建KD树的节点
        pos: 可视化中树的节点的位置，初始化时在(0,0)处绘制根节点
        x: 对应pos中的横坐标，随着递归更新
        y: 对应pos中的纵坐标，随着递归更新
        layer: KD树的层次
        """

        if pos is None:
            pos = {}
        pos[str(kdt_node.instance_idx)] = (x, y)
        if kdt_node.left_child:
            # 父节点指向左子树
            graph.add_edge(str(kdt_node.instance_idx), 
                           str(kdt_node.left_child.instance_idx))
            l_x, l_y = x - 1/2 ** layer, y - 1
            l_layer = layer + 1
            self.__create_kd_tree__(
                graph, kdt_node.left_child, pos, l_x, l_y, l_layer      # 递归
            )
        if kdt_node.right_child:
            # 父节点指向右子树
            graph.add_edge(str(kdt_node.instance_idx), 
                           str(kdt_node.right_child.instance_idx))
            r_x, r_y = x + 1/2 ** layer, y - 1
            r_layer = layer + 1
            self.__create_kd_tree__(
                graph, kdt_node.right_child, pos, r_x, r_y, r_layer      # 递归
            )
        return graph, pos


    def draw_kd_tree(self):

        """
        可视化KD树
        """

        directed_graph = nx.DiGraph()   # 初始化一个有向图
        graph, pos = self.__create_kd_tree__(directed_graph, self.kdt_root)
        fig, ax = plt.subplots(figsize=(20, 10))
        nx.draw_networkx(graph, pos, ax=ax, node_size=500, 
                         font_size=15,arrowsize=20)
        plt.show()
