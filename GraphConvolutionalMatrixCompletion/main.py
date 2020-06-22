import numpy as np
import torch
import torch.nn as nn
import scipy.sparse as sp
import torch.optim as optim
import torch.nn.functional as F

from data import MovielensDataset
from autoEncoder import GraphMatrixCompletion


# 预测目标为 user-item 打分matrix中空缺的部分

# hyper
DEVICE = torch.device('cuda:0')
LEARNING_RATE = 1e-2
EPOCHS = 2000
NODE_INPUT_DIM = 2625
SIDE_FEATURE_DIM = 41
GCN_HIDDEN_DIM = 500
SIDE_HIDDEN_DIM = 10
ENCODE_HIDDEN_DIM = 75
NUM_BASIS = 2
DROPOUT_RATIO = 0.7
WEIGHT_DACAY = 0.

# 5种打分
SCORES = torch.tensor([[1, 2, 3, 4, 5]]).to(DEVICE)


def to_torch_sparse_tensor(x, device='cpu'):
    if not sp.isspmatrix_coo(x):
        x = sp.coo_matrix(x)
    row, col = x.row, x.col

    indices = torch.from_numpy(np.asarray([row, col]).astype('int64')).long()
    values = torch.from_numpy(x.data.astype(np.float32))
    th_sparse_tensor = torch.sparse.FloatTensor(indices, values,
                                                x.shape).to(device)

    return th_sparse_tensor


def tensor_from_numpy(x, device='cpu'):
    return torch.from_numpy(x).to(device)


def expected_rmse(logits, label):
    "预测打分与实际打分的均方根误差"
    true_y = label + 1  # 原来的评分为1~5，作为label时为0~4
    prob = F.softmax(logits, dim=1)
    pred_y = torch.sum(prob * SCORES, dim=1)
    diff = torch.pow(true_y - pred_y, 2)
    return torch.sqrt(diff.mean())


def train():
    model.train()
    for e in range(EPOCHS):
        logits = model(*model_inputs)
        loss = criterion(logits[train_mask], labels[train_mask])
        rmse = expected_rmse(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("Epoch {:03d}: Loss: {:.4f}, RMSE: {:.4f}".format(e, loss.item(), rmse.item()))

        if (e + 1) % 100 == 0:
            test(e)
            model.train()


def test(e):
    model.eval()
    with torch.no_grad():
        logits = model(*model_inputs)
        test_mask = ~train_mask
        loss = criterion(logits[test_mask], labels[test_mask])
        rmse = expected_rmse(logits[test_mask], labels[test_mask])
        print('Test On Epoch {}: loss: {:.4f}, Test rmse: {:.4f}'.format(e, loss.item(), rmse.item()))
    return logits


if __name__ == '__main__':
    # 输入
    data = MovielensDataset()
    user2movie_adjacencies, movie2user_adjacencies, \
        user_side_feature, movie_side_feature, \
        user_identity_feature, movie_identity_feature, \
        user_indices, movie_indices, labels, train_mask = data.build_graph(
            *data.read_data())

    user2movie_adjacencies = [to_torch_sparse_tensor(adj, DEVICE) for adj in user2movie_adjacencies]
    movie2user_adjacencies = [to_torch_sparse_tensor(adj, DEVICE) for adj in movie2user_adjacencies]
    user_side_feature = tensor_from_numpy(user_side_feature, DEVICE).float()
    movie_side_feature = tensor_from_numpy(movie_side_feature, DEVICE).float()
    user_identity_feature = tensor_from_numpy(user_identity_feature, DEVICE).float()
    movie_identity_feature = tensor_from_numpy(movie_identity_feature, DEVICE).float()
    user_indices = tensor_from_numpy(user_indices, DEVICE).long()
    movie_indices = tensor_from_numpy(movie_indices, DEVICE).long()
    labels = tensor_from_numpy(labels, DEVICE)
    train_mask = tensor_from_numpy(train_mask, DEVICE)

    # 建模
    model = GraphMatrixCompletion(NODE_INPUT_DIM, SIDE_FEATURE_DIM, GCN_HIDDEN_DIM,
                            SIDE_HIDDEN_DIM, ENCODE_HIDDEN_DIM, dropout_ratio=DROPOUT_RATIO,
                            num_basis=NUM_BASIS).to(DEVICE)
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DACAY)

    model_inputs = (user2movie_adjacencies, movie2user_adjacencies,
                    user_identity_feature, movie_identity_feature,
                    user_side_feature, movie_side_feature, user_indices, movie_indices)

    train()