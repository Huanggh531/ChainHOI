import numpy as np
import torch


def k_adjacency(A, k, with_self=False, self_factor=1):
    # A is a 2D square array
    if isinstance(A, torch.Tensor):
        A = A.data.cpu().numpy()
    assert isinstance(A, np.ndarray)
    Iden = np.eye(len(A), dtype=A.dtype)
    if k == 0:
        return Iden
    Ak = np.minimum(np.linalg.matrix_power(A + Iden, k), 1) - np.minimum(np.linalg.matrix_power(A + Iden, k - 1), 1)
    if with_self:
        Ak += (self_factor * Iden)
    return Ak


def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A


def normalize_digraph(A, dim=0):
    # A is a 2D square array
    Dl = np.sum(A, dim)
    h, w = A.shape
    Dn = np.zeros((w, w))

    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)

    AD = np.dot(A, Dn)
    return AD


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.eye(num_node)

    for i, j in edge:
        A[i, j] = 1
        A[j, i] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [
        np.linalg.matrix_power(A, d) for d in range(max_hop + 1)
    ]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


class Graph:
    """The Graph to model the skeletons.

    Args:
        layout (str): must be one of the following candidates: 'openpose', 'nturgb+d', 'coco'. Default: 'coco'.
        mode (str): must be one of the following candidates: 'stgcn_spatial', 'spatial'. Default: 'spatial'.
        max_hop (int): the maximal distance between two connected nodes.
            Default: 1
    """

    def __init__(self,
                 layout='coco',
                 mode='spatial',
                 max_hop=1,
                 nx_node=1,
                 num_filter=3,
                 init_std=0.02,
                 init_off=0.04,
                 extra_joint=False):

        self.max_hop = max_hop
        self.layout = layout
        self.mode = mode
        self.num_filter = num_filter
        self.init_std = init_std
        self.init_off = init_off
        self.nx_node = nx_node
        self.extra_joint = extra_joint

        assert nx_node == 1 or mode == 'random', "nx_node can be > 1 only if mode is 'random'"
        assert layout in ['hml3d_graph_1', 'hml3d_graph_2',
                          'kit_graph_1', 'kit_graph_2','behave_graph_1','behave_graph_2','behave_graph_3','behave_graph_4',
                          'behave_graph_5','behave_graph_6','behave_graph_7']

        self.get_layout(layout)
        self.hop_dis = get_hop_distance(self.num_node, self.inward, max_hop)

        assert hasattr(self, mode), f'Do Not Exist This Mode: {mode}'
        self.A = getattr(self, mode)()

    def __str__(self):
        return self.A

    def get_layout(self, layout):
        if layout=="behave_graph_1":
            self.num_node = 24
            self.inward = [
                (0, 1), (1, 4), (4, 7), (7, 10),
                (0, 2), (2, 5), (5, 8), (8, 11),
                (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
                (9, 13), (13, 16), (16, 18), (18, 20),
                (9, 14), (14, 17), (17, 19), (19, 21),
                # extra point
                (22, 7), (22, 8), (22, 10), (22, 11),
                #object line
                # (23,20),(23,18),(23,16),(23,13),
                # (23,21),(23,19),(23,17),(23,14),
                # (0,23),(12,23),(16,23),(17,23),
                # (20,23),(21,23),(10,23),(11,23),
                
            ]
            self.center = 0
        elif layout=="behave_graph_2":
            self.num_node = 24
            self.inward = [
                (0, 1), (1, 4), (4, 7), (7, 10),
                (0, 2), (2, 5), (5, 8), (8, 11),
                (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
                (9, 13), (13, 16), (16, 18), (18, 20),
                (9, 14), (14, 17), (17, 19), (19, 21),
                # extra point
                (22, 7), (22, 8), (22, 10), (22, 11),
                #object line
                (23,0),(23,1),(23,2),(23,3),(23,4),
                (23,5),(23,6),(23,7),(23,8),(23,9),
                (23,10),(23,11),(23,12),(23,13),(23,14),
                (23,15),(23,16),(23,17),(23,18),(23,19),
                (23,20),(23,21),  
            ]
            self.center = 0
        elif layout=="behave_graph_3":
            self.num_node = 24
            self.inward = [
                (0, 1), (1, 4), (4, 7), (7, 10),
                (0, 2), (2, 5), (5, 8), (8, 11),
                (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
                (9, 13), (13, 16), (16, 18), (18, 20),
                (9, 14), (14, 17), (17, 19), (19, 21),
                # extra point
                (22, 7), (22, 8), (22, 10), (22, 11),
                # #object line
                # (0,23),(12,23),(16,23),(17,23),
                # (20,23),(21,23),(10,23),(11,23),
            ]
            self.center = 0
        elif layout=="behave_graph_4":
            self.num_node = 23
            self.inward = [
                (0, 1), (1, 4), (4, 7), (7, 10),
                (0, 2), (2, 5), (5, 8), (8, 11),
                (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
                (9, 13), (13, 16), (16, 18), (18, 20),
                (9, 14), (14, 17), (17, 19), (19, 21),
                # # extra point
                # (22, 7), (22, 8), (22, 10), (22, 11),
                #object line
                (0, 22),(1, 22),(2, 22),(3, 22),(4, 22),
                (5, 22),(6, 22),(7, 22),(8, 22),(9, 22),
                (10, 22),(11, 22),(12, 22),(13, 22),(14, 22),
                (15, 22), (16, 22),(17, 22),(18, 22),(19, 22),(20, 22),(21, 22)
            ]
            self.center = 0
        elif layout=="behave_graph_5":
            self.num_node = 25
            self.inward = [
                (0, 1), (1, 4), (4, 7), (7, 10),
                (0, 2), (2, 5), (5, 8), (8, 11),
                (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
                (9, 13), (13, 16), (16, 18), (18, 20),
                (9, 14), (14, 17), (17, 19), (19, 21),
                # extra point
                (22, 7), (22, 8), (22, 10), (22, 11),
                # extra line
                (23,0),(23,1),(23,2),(23,3),(23,4),
                (23,5),(23,6),(23,7),(23,8),(23,9),
                (23,10),(23,11),(23,12),(23,13),(23,14),
                (23,15),(23,16),(23,17),(23,18),(23,19),
                (23,20),(23,21),  
                # object line
                (24,0),(24,1),(24,2),(24,3),(24,4),
                (24,5),(24,6),(24,7),(24,8),(24,9),
                (24,10),(24,11),(24,12),(24,13),(24,14),
                (24,15),(24,16),(24,17),(24,18),(24,19),
                (24,20),(24,21), 
            ]
            self.center = 0
        elif layout=="behave_graph_6":
            self.num_node = 24
            self.inward = [
                (0, 1), (1, 4), (4, 7), (7, 10),
                (0, 2), (2, 5), (5, 8), (8, 11),
                (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
                (9, 13), (13, 16), (16, 18), (18, 20),
                (9, 14), (14, 17), (17, 19), (19, 21),
                # extra point
                (22, 7), (22, 8), (22, 10), (22, 11),
                # extra contact
                (23,0),
                (23,10),(23,11),(23,12),(23,16),(23,17),
                (23,20),(23,21),  
            ]
            self.center = 0
        elif layout=="behave_graph_7":
            self.num_node = 24
            self.inward = [
                (0, 1), (1, 4), (4, 7), (7, 10),
                (0, 2), (2, 5), (5, 8), (8, 11),
                (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
                (9, 13), (13, 16), (16, 18), (18, 20),
                (9, 14), (14, 17), (17, 19), (19, 21),
                # extra point
                (22, 7), (22, 8), (22, 10), (22, 11),
                # extra contact
                (23,0),
                (23,20),(23,21),  
            ]
            self.center = 0
        else:
            raise ValueError(f'Do Not Exist This Layout: {layout}')

        if self.extra_joint:
            self.num_node += 1
            self.inward.extend([(self.num_node-1, i) for i in range(self.num_node-1)])

        self.self_link = [(i, i) for i in range(self.num_node)]
        self.outward = [(j, i) for (i, j) in self.inward]
        self.neighbor = self.inward + self.outward

    def stgcn_spatial(self):
        adj = np.zeros((self.num_node, self.num_node))
        adj[self.hop_dis <= self.max_hop] = 1
        normalize_adj = normalize_digraph(adj)
        hop_dis = self.hop_dis
        center = self.center

        A = []
        for hop in range(self.max_hop + 1):
            a_close = np.zeros((self.num_node, self.num_node))
            a_further = np.zeros((self.num_node, self.num_node))
            for i in range(self.num_node):
                for j in range(self.num_node):
                    if hop_dis[j, i] == hop:
                        if hop_dis[j, center] >= hop_dis[i, center]:
                            a_close[j, i] = normalize_adj[j, i]
                        else:
                            a_further[j, i] = normalize_adj[j, i]
            A.append(a_close)
            if hop > 0:
                A.append(a_further)
        return np.stack(A)

    def spatial(self):

        Iden = edge2mat(self.self_link, self.num_node)#自连接
        In = normalize_digraph(edge2mat(self.inward, self.num_node))
        Out = normalize_digraph(edge2mat(self.outward, self.num_node))
        A = np.stack((Iden, In, Out))
        return A

    def binary_adj(self):
        A = edge2mat(self.inward + self.outward, self.num_node)
        return A[None]

    def random(self):
        num_node = self.num_node * self.nx_node
        return np.random.randn(self.num_filter, num_node, num_node) * self.init_std + self.init_off