import os
import urllib
import numpy as np
import scipy.sparse as sp
from zipfile import ZipFile
from sklearn.model_selection import train_test_split


class DDDataset(object):
    """D&D是一个包含1178个蛋白质结构的数据集(Dobson and Doig, 2003)。
    每个蛋白质都用一个图来表示，图中的节点是氨基酸，
    如果两个节点之间的距离小于6个原子，则用一条边连接。

    预测任务是将蛋白质结构分为酶和非酶。 -- 图分类任务"""
    url = "https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/DD.zip"

    def __init__(self, data_root="data", train_size=0.8):
        self.data_root = data_root
        self.maybe_download()
        sparse_adjacency, node_labels, graph_indicator, graph_labels = self.read_data()
        self.sparse_adjacency = sparse_adjacency.tocsr()
        self.node_labels = node_labels
        self.graph_indicator = graph_indicator
        self.graph_labels = graph_labels
        self.train_index, self.test_index = self.split_data(train_size)
        self.train_label = graph_labels[self.train_index]
        self.test_label = graph_labels[self.test_index]

    def split_data(self, train_size):
        unique_indicator = np.asarray(list(set(self.graph_indicator)))
        train_index, test_index = train_test_split(unique_indicator,
                                                   train_size=train_size,
                                                   random_state=1234)
        return train_index, test_index

    def __getitem__(self, index):
        mask = self.graph_indicator == index
        node_labels = self.node_labels[mask]
        graph_indicator = self.graph_indicator[mask]
        graph_labels = self.graph_labels[index]
        adjacency = self.sparse_adjacency[mask, :][:, mask]
        return adjacency, node_labels, graph_indicator, graph_labels

    def __len__(self):
        return len(self.graph_labels)

    def read_data(self):
        data_dir = os.path.join(self.data_root, "DD")
        print("Loading DD_A.txt")
        adjacency_list = np.genfromtxt(os.path.join(data_dir, "DD_A.txt"),
                                       dtype=np.int64, delimiter=',') - 1
        print("Loading DD_node_labels.txt")
        node_labels = np.genfromtxt(os.path.join(data_dir, "DD_node_labels.txt"),
                                    dtype=np.int64) - 1
        print("Loading DD_graph_indicator.txt")
        graph_indicator = np.genfromtxt(os.path.join(data_dir, "DD_graph_indicator.txt"),
                                        dtype=np.int64) - 1
        print("Loading DD_graph_labels.txt")
        graph_labels = np.genfromtxt(os.path.join(data_dir, "DD_graph_labels.txt"),
                                     dtype=np.int64) - 1
        num_nodes = len(node_labels)
        sparse_adjacency = sp.coo_matrix((np.ones(len(adjacency_list)),
                                          (adjacency_list[:, 0], adjacency_list[:, 1])),
                                         shape=(num_nodes, num_nodes), dtype=np.float32)
        print("Number of nodes: ", num_nodes)
        # node_infos = pd.DataFrame(columns=["node_labels", "graph_indicator"])
        # node_infos["node_labels"] = node_labels
        # node_infos["graph_indicator"] = graph_indicator
        return sparse_adjacency, node_labels, graph_indicator, graph_labels

    def maybe_download(self):
        save_path = os.path.join(self.data_root)
        if not os.path.exists(save_path):
            self.download_data(self.url, save_path)
        if not os.path.exists(os.path.join(self.data_root, "DD")):
            zipfilename = os.path.join(self.data_root, "DD.zip")
            with ZipFile(zipfilename, "r") as zipobj:
                zipobj.extractall(os.path.join(self.data_root))
                print("Extracting data from {}".format(zipfilename))

    @staticmethod
    def download_data(url, save_path):
        """数据下载工具，当原始数据不存在时将会进行下载"""
        print("Downloading data from {}".format(url))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        data = urllib.request.urlopen(url)
        filename = "DD.zip"
        with open(os.path.join(save_path, filename), 'wb') as f:
            f.write(data.read())
        return True
