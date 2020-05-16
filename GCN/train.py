import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from data_loader import CoraData, Data
from gc_layer import GraphConvolution


# 超参数定义
LEARNING_RATE = 0.01
WEIGHT_DACAY = 5e-4
L1_REGULAR = 5e-5
EPOCHS = 300
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def tensor_from_numpy(x, device):
    return torch.from_numpy(x).to(device)


# 加载数据，并转换为torch.Tensor
dataset = CoraData().data

node_feature = dataset.x / dataset.x.sum(1, keepdims=True)  # 归一化数据，使得每一行和为1
tensor_x = tensor_from_numpy(node_feature, DEVICE)
tensor_y = tensor_from_numpy(dataset.y, DEVICE)

tensor_train_mask = tensor_from_numpy(dataset.train_mask, DEVICE)
tensor_val_mask = tensor_from_numpy(dataset.val_mask, DEVICE)
tensor_test_mask = tensor_from_numpy(dataset.test_mask, DEVICE)

normalize_adjacency = CoraData.normalization(dataset.adjacency)   # 规范化邻接矩阵


num_nodes, input_dim = node_feature.shape
indices = torch.from_numpy(np.asarray([normalize_adjacency.row,
                                       normalize_adjacency.col]).astype('int64')).long()
values = torch.from_numpy(normalize_adjacency.data.astype(np.float32))
tensor_adjacency = torch.sparse.FloatTensor(indices, values,
                                            (num_nodes, num_nodes)).to(DEVICE)



# 模型定义：Model, Loss, Optimizer
class GcnNet(nn.Module):
    """
    定义一个包含两层GraphConvolution的模型
    """
    def __init__(self, input_dim=1433):
        super(GcnNet, self).__init__()
        self.gcn1 = GraphConvolution(input_dim, 32)
        self.gcn2 = GraphConvolution(32, 7)

    def forward(self, adjacency, feature, training=True):
        h = F.relu(self.gcn1(adjacency, feature))
        if training:
            h = F.dropout(h, 0.5, training)
        logits = self.gcn2(adjacency, h)
        return logits


model = GcnNet(input_dim).to(DEVICE)
criterion = nn.CrossEntropyLoss().to(DEVICE)
optimizer = optim.Adam(model.parameters(),
                       lr=LEARNING_RATE,
                       weight_decay=WEIGHT_DACAY)


# 训练主体函数
def train():
    loss_history = []
    val_acc_history = []
    model.train()
    train_y = tensor_y[tensor_train_mask]
    for epoch in range(EPOCHS):
        logits = model(tensor_adjacency, tensor_x, True)  # 前向传播
        train_mask_logits = logits[tensor_train_mask]   # 只选择训练节点进行监督

        l1_norm = torch.norm(model.gcn1.weight, p=1)

        loss = criterion(train_mask_logits, train_y) + L1_REGULAR * l1_norm   # 计算损失值
        optimizer.zero_grad()
        loss.backward()     # 反向传播计算参数的梯度
        optimizer.step()    # 使用优化方法进行梯度更新
        train_acc, _, _ = test(tensor_train_mask)     # 计算当前模型训练集上的准确率
        val_acc, _, _ = test(tensor_val_mask)     # 计算当前模型在验证集上的准确率
        # 记录训练过程中损失值和准确率的变化，用于画图
        loss_history.append(loss.item())
        val_acc_history.append(val_acc.item())
        print("Epoch {:03d}: Loss {:.4f}, TrainAcc {:.4}, ValAcc {:.4f}".format(
            epoch, loss.item(), train_acc.item(), val_acc.item()))

    return loss_history, val_acc_history


# 测试函数
def test(mask):
    model.eval()
    with torch.no_grad():
        logits = model(tensor_adjacency, tensor_x, False)
        test_mask_logits = logits[mask]
        predict_y = test_mask_logits.max(1)[1]
        accuarcy = torch.eq(predict_y, tensor_y[mask]).float().mean()
    return accuarcy, test_mask_logits.cpu().numpy(), tensor_y[mask].cpu().numpy()



def plot_loss_with_acc(loss_history, val_acc_history):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(range(len(loss_history)), loss_history,
             c=np.array([255, 71, 90]) / 255.)
    plt.ylabel('Loss')

    ax2 = fig.add_subplot(111, sharex=ax1, frameon=False)
    ax2.plot(range(len(val_acc_history)), val_acc_history,
             c=np.array([79, 179, 255]) / 255.)
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    plt.ylabel('ValAcc')

    plt.xlabel('Epoch')
    plt.title('Training Loss & Validation Accuracy')
    plt.show()


def tsne(test_logits, test_label):
    from sklearn.manifold import TSNE
    tsne = TSNE()
    out = tsne.fit_transform(test_logits)
    fig = plt.figure()
    for i in range(7):
        indices = test_label == i
        x, y = out[indices].T
        plt.scatter(x, y, label=str(i))
    plt.legend()
    plt.show()


if __name__ == "__main__":

    loss, val_acc = train()
    plot_loss_with_acc(loss, val_acc)

    test_acc, test_logits, test_label = test(tensor_test_mask)
    print("Test accuarcy: ", test_acc.item())

    tsne(test_logits, test_label)