import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from data import DDDataset
from model import ModelA, ModelB, normalization


def tensor_from_numpy(x, device):
    return torch.from_numpy(x).to(device)

dataset = DDDataset()

# 模型输入数据准备
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

adjacency = dataset.sparse_adjacency
normalize_adjacency = normalization(adjacency).to(DEVICE)

node_labels = tensor_from_numpy(dataset.node_labels, DEVICE)
node_features = F.one_hot(node_labels, node_labels.max().item() + 1).float()

graph_indicator = tensor_from_numpy(dataset.graph_indicator, DEVICE)

graph_labels = tensor_from_numpy(dataset.graph_labels, DEVICE)
train_index = tensor_from_numpy(dataset.train_index, DEVICE)
test_index = tensor_from_numpy(dataset.test_index, DEVICE)
train_label = tensor_from_numpy(dataset.train_label, DEVICE)
test_label = tensor_from_numpy(dataset.test_label, DEVICE)


# 超参数设置
INPUT_DIM = node_features.size(1)
NUM_CLASSES = 2
EPOCHS = 200
HIDDEN_DIM = 32
LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.0001


# 模型初始化
model_g = ModelA(INPUT_DIM, HIDDEN_DIM, NUM_CLASSES).to(DEVICE)
# model_h = ModelB(INPUT_DIM, HIDDEN_DIM, NUM_CLASSES).to(DEVICE)

model = model_g


# Train
criterion = nn.CrossEntropyLoss().to(DEVICE)
optimizer = optim.Adam(model.parameters(), LEARNING_RATE, weight_decay=WEIGHT_DECAY)

model.train()
for epoch in range(EPOCHS):
    logits = model(normalize_adjacency, node_features, graph_indicator)
    loss = criterion(logits[train_index], train_label)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_acc = torch.eq(
        logits[train_index].max(1)[1], train_label).float().mean()
    # Best: 0.99
    print("Epoch {:03d}: Loss {:.4f}, TrainAcc {:.4}".format(
        epoch, loss.item(), train_acc.item()))


# Test
model.eval()
with torch.no_grad():
    logits = model(normalize_adjacency, node_features, graph_indicator)
    test_logits = logits[test_index]
    test_acc = torch.eq(
        test_logits.max(1)[1], test_label
    ).float().mean()

# Best: 0.77
print(test_acc.item())