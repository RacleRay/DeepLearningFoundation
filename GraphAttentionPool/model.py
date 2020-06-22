import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        """图卷积：L*X*theta

        Args:
        ----------
            input_dim: int
                节点输入特征的维度
            output_dim: int
                输出特征维度
            use_bias : bool, optional
                是否使用偏置
        """
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, adjacency, input_feature):
        """邻接矩阵是稀疏矩阵，因此在计算时使用稀疏矩阵乘法"""
        support = torch.mm(input_feature, self.weight)
        output = torch.sparse.mm(adjacency, support)
        if self.use_bias:
            output += self.bias
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.input_dim) + ' -> ' \
            + str(self.output_dim) + ')'


def normalization(adjacency):
    """计算 L=D^-0.5 * (A+I) * D^-0.5,

    Args:
        adjacency: sp.csr_matrix.

    Returns:
        归一化后的邻接矩阵，类型为 torch.sparse.FloatTensor
    """
    adjacency += sp.eye(adjacency.shape[0])    # 增加自连接
    degree = np.array(adjacency.sum(1))
    d_hat = sp.diags(np.power(degree, -0.5).flatten())
    L = d_hat.dot(adjacency).dot(d_hat).tocoo()
    # 转换为 torch.sparse.FloatTensor
    indices = torch.from_numpy(np.asarray([L.row, L.col])).long()
    values = torch.from_numpy(L.data.astype(np.float32))
    tensor_adjacency = torch.sparse.FloatTensor(indices, values, L.shape)
    return tensor_adjacency


def filter_adjacency(adjacency, mask):
    """根据掩码mask对图结构进行更新

    Args:
        adjacency: torch.sparse.FloatTensor, 池化之前的邻接矩阵
        mask: torch.Tensor(dtype=torch.bool), 节点掩码向量

    Returns:
        torch.sparse.FloatTensor, 池化之后归一化邻接矩阵
    """
    device = adjacency.device
    mask = mask.cpu().numpy()

    indices = adjacency.coalesce().indices().cpu().numpy()
    num_nodes = adjacency.size(0)
    row, col = indices

    maskout_self_loop = row != col
    row = row[maskout_self_loop]
    col = col[maskout_self_loop]

    # - csr_matrix((data, indices, indptr), [shape=(M, N)])
    #       对于第i行，非0数据列是indices[indptr[i]:indptr[i+1]] 数据是data[indptr[i]:indptr[i+1]]
    # - csr_matrix((data, (row_ind, col_ind)), [shape=(M, N)])
    #       where data, row_ind and col_ind satisfy the relationship a[row_ind[k], col_ind[k]] = data[k].
    sparse_adjacency = sp.csr_matrix((np.ones(len(row)), (row, col)),
                                     shape=(num_nodes, num_nodes), dtype=np.float32)
    filtered_adjacency = sparse_adjacency[mask, :][:, mask]
    return normalization(filtered_adjacency).to(device)


def top_rank(attention_score, graph_indicator, keep_ratio):
    """基于给定的attention_score, 对每个图进行pooling操作.
    为了直观体现pooling过程，我们将每个图单独进行池化，最后再将它们级联起来进行下一步计算

    Arguments:
    ----------
        attention_score：torch.Tensor
            使用GCN计算出的注意力分数，Z = GCN(A, X)
        graph_indicator：torch.Tensor
            指示每个节点属于哪个图
        keep_ratio: float
            要保留的节点比例，保留的节点数量为int(N * keep_ratio)
    """
    # TODO: 确认是否是有序的, 必须是有序的
    graph_id_list = list(set(graph_indicator.cpu().numpy()))
    mask = attention_score.new_empty((0,), dtype=torch.bool)
    for graph_id in graph_id_list:
        graph_attn_score = attention_score[graph_indicator == graph_id]
        graph_node_num = len(graph_attn_score)
        graph_mask = attention_score.new_zeros((graph_node_num,),
                                                dtype=torch.bool)
        keep_graph_node_num = int(keep_ratio * graph_node_num)
        _, sorted_index = graph_attn_score.sort(descending=True)
        graph_mask[sorted_index[: keep_graph_node_num]] = True
        mask = torch.cat((mask, graph_mask))
    return mask


class SelfAttentionPooling(nn.Module):
    def __init__(self, input_dim, keep_ratio, activation=torch.tanh):
        super(SelfAttentionPooling, self).__init__()
        self.input_dim = input_dim
        self.keep_ratio = keep_ratio
        self.activation = activation
        self.attn_gcn = GraphConvolution(input_dim, 1)

    def forward(self, adjacency, input_feature, graph_indicator):
        """GCN计算节点得分。input_feature在此例中为节点的label信息"""
        attn_score = self.attn_gcn(adjacency, input_feature).squeeze()
        attn_score = self.activation(attn_score)

        mask = top_rank(attn_score, graph_indicator, self.keep_ratio)
        hidden = input_feature[mask] * attn_score[mask].view(-1, 1)
        mask_graph_indicator = graph_indicator[mask]
        mask_adjacency = filter_adjacency(adjacency, mask)
        return hidden, mask_graph_indicator, mask_adjacency


def global_max_pool(x, graph_indicator):
    """图模型全局信息提取，根据graph_indicator进行max_pooling操作。
    需要安装torch_scatter.
    https://github.com/rusty1s/pytorch_scatter
    """
    num = graph_indicator.max().item() + 1
    return torch_scatter.scatter_max(x, graph_indicator, dim=0, dim_size=num)[0]


def global_avg_pool(x, graph_indicator):
    num = graph_indicator.max().item() + 1
    return torch_scatter.scatter_mean(x, graph_indicator, dim=0, dim_size=num)


# =========================================================================
class ModelA(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes=2):
        """图分类模型结构A，适用于较小的数据集

        Args:
        ----
            input_dim: int, 输入特征的维度
            hidden_dim: int, 隐藏层单元数
            num_classes: 分类类别数 (default: 2)
        """
        super(ModelA, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.gcn1 = GraphConvolution(input_dim, hidden_dim)
        self.gcn2 = GraphConvolution(hidden_dim, hidden_dim)
        self.gcn3 = GraphConvolution(hidden_dim, hidden_dim)
        self.pool = SelfAttentionPooling(hidden_dim * 3, 0.5)
        self.fc1 = nn.Linear(hidden_dim * 3 * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, adjacency, input_feature, graph_indicator):
        gcn1 = F.relu(self.gcn1(adjacency, input_feature))
        gcn2 = F.relu(self.gcn2(adjacency, gcn1))
        gcn3 = F.relu(self.gcn3(adjacency, gcn2))

        gcn_feature = torch.cat((gcn1, gcn2, gcn3), dim=1)
        pool, pool_graph_indicator, pool_adjacency = self.pool(adjacency, gcn_feature,
                                                               graph_indicator)

        readout = torch.cat((global_avg_pool(pool, pool_graph_indicator),
                             global_max_pool(pool, pool_graph_indicator)), dim=1)

        fc1 = F.relu(self.fc1(readout))
        fc2 = F.relu(self.fc2(fc1))
        logits = self.fc3(fc2)

        return logits


class ModelB(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes=2):
        """图分类模型结构，适用于较大的数据集

        Args:
        -----
            input_dim: int, 输入特征的维度
            hidden_dim: int, 隐藏层单元数
            num_classes: int, 分类类别数 (default: 2)
        """
        super(ModelB, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.gcn1 = GraphConvolution(input_dim, hidden_dim)
        self.pool1 = SelfAttentionPooling(hidden_dim, 0.5)
        self.gcn2 = GraphConvolution(hidden_dim, hidden_dim)
        self.pool2 = SelfAttentionPooling(hidden_dim, 0.5)
        self.gcn3 = GraphConvolution(hidden_dim, hidden_dim)
        self.pool3 = SelfAttentionPooling(hidden_dim, 0.5)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes))

    def forward(self, adjacency, input_feature, graph_indicator):
        "每一层GCN分别进行pooling和输出"
        gcn1 = F.relu(self.gcn1(adjacency, input_feature))
        pool1, pool1_graph_indicator, pool1_adjacency = \
            self.pool1(adjacency, gcn1, graph_indicator)
        global_pool1 = torch.cat(
            [global_avg_pool(pool1, pool1_graph_indicator),
             global_max_pool(pool1, pool1_graph_indicator)],
            dim=1)

        gcn2 = F.relu(self.gcn2(pool1_adjacency, pool1))
        pool2, pool2_graph_indicator, pool2_adjacency = \
            self.pool2(pool1_adjacency, gcn2, pool1_graph_indicator)
        global_pool2 = torch.cat(
            [global_avg_pool(pool2, pool2_graph_indicator),
             global_max_pool(pool2, pool2_graph_indicator)],
            dim=1)

        gcn3 = F.relu(self.gcn3(pool2_adjacency, pool2))
        pool3, pool3_graph_indicator, pool3_adjacency = \
            self.pool3(pool2_adjacency, gcn3, pool2_graph_indicator)
        global_pool3 = torch.cat(
            [global_avg_pool(pool3, pool3_graph_indicator),
             global_max_pool(pool3, pool3_graph_indicator)],
            dim=1)

        readout = global_pool1 + global_pool2 + global_pool3

        logits = self.mlp(readout)
        return logits