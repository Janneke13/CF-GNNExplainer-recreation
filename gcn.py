import numpy as np
import torch
import torch.nn.functional as F
import scipy as sp
from scipy.sparse import coo_matrix
from scipy.sparse import diags
from sklearn.metrics import accuracy_score
import math

"""
All functions needed to create and train a GCN model (3-layer)
Note: the gcn_train.ipynb file is used to train, visualize and save the models (to be used in the experiments).
Note: "basic" in initialization stands for the initialization with kaiming for weights and zero for bias.
--- anything else in initialization would change the initialization to use their method.
"""


class GCNlayer(torch.nn.Module):
    """
    Implementation of a simple GCN layer, with kaiming uniform initialization for the weight matrix.
    This implementation includes a bias as well, which is initialized as zeroes.
    """

    def __init__(self, n_in, n_out, initialization='basic'):
        """
        Initialized a GCN layer
        :param n_in: number of inputs
        :param n_out: number of outputs
        """
        super().__init__()
        if initialization == "basic":
            matrix_to_fill = torch.empty(n_in, n_out)  # create a matrix of a certain shape (values don't matter)
            initialized_w = torch.nn.init.kaiming_uniform_(matrix_to_fill)  # use uniform kaiming to initialize
            self.W = torch.nn.Parameter(initialized_w)  # set W to be learnable
            bias = torch.zeros(n_out, dtype=torch.float)  # create a bias --> initialize as zeroes
            self.b = torch.nn.Parameter(bias)
        else:
            # similar to what they do in the paper:
            W = torch.empty(n_in, n_out, dtype=torch.float)
            b = torch.empty(n_out, dtype=torch.float)

            std = 1. / math.sqrt(W.size(1))
            W.data.uniform_(-std, std)
            b.data.uniform_(-std, std)

            self.W = torch.nn.Parameter(W)
            self.b = torch.nn.Parameter(b)

    def forward(self, X, A_hat):
        """
        Performs one forward pass of a GCN layer
        :param X: the feature matrix of the graph
        :param A_hat: the normalized adjacency matrix of the graph
        :return: the output of the forward pass
        """
        # perform one forward pass and add bias
        temp = torch.spmm(A_hat, X)
        Z = torch.spmm(temp, self.W) + self.b
        return Z


class GCN3LayerBasic(torch.nn.Module):
    """
    Implementation of a 3-layer GCN.
    Does not add a linear layer at the end.
    """

    def __init__(self, n_in, n_hid, n_cls, initialization="basic"):
        """
        Initialize basic gcn (without lin layer)
        :param n_in: nr of features
        :param n_hid: hidden size
        :param n_cls: number of classes
        """
        super().__init__()
        # three GCN layers (using the module that was used before)
        self.gcn_layer_1 = GCNlayer(n_in, n_hid, initialization=initialization)
        self.gcn_layer_2 = GCNlayer(n_hid, n_hid, initialization=initialization)
        self.gcn_layer_3 = GCNlayer(n_hid, n_cls, initialization=initialization)

    def forward(self, X, A_hat):
        """
        Performs one forward pass of the network.
        :param X: the feature matrix of the graph
        :param A_hat: the normalized adjacency matrix
        :return: the output of the forward pass
        """
        # perform three forward passes for the GCN layers:
        h_1 = self.gcn_layer_1.forward(X, A_hat)
        h_1_relu = F.relu(h_1)
        h_2 = self.gcn_layer_2.forward(h_1_relu, A_hat)
        h_2_relu = F.relu(h_2)
        h_3 = self.gcn_layer_3.forward(h_2_relu, A_hat)
        return h_3


class GCN3LayerLinear(torch.nn.Module):
    """
    Implementation of a 3-layer GCN.
    Adds a linear layer at the end, but does not do softmax (as cross-entropy is used).
    """

    def __init__(self, n_in, n_hid, n_cls, initialization="basic"):
        """
        Initializes the 3-layer GCN with a last linear layer
        :param n_in: number of input features
        :param n_hid: size of hidden layer
        :param n_cls: number of classes
        """
        super().__init__()
        # three GCN layers (using the module that was used before)
        self.gcn_layer_1 = GCNlayer(n_in, n_hid, initialization=initialization)
        self.gcn_layer_2 = GCNlayer(n_hid, n_hid, initialization=initialization)
        self.gcn_layer_3 = GCNlayer(n_hid, n_hid, initialization=initialization)
        self.linear_layer = torch.nn.Linear(n_hid * 3, n_cls)

    def forward(self, X, A_hat):
        """
        Performs one forward pass of the implemented GCN
        :param X: feature matrix
        :param A_hat: normalized adjacency matrix
        :return: the output of the forward pass
        """
        # perform three forward passes for the GCN layers:
        h_1 = self.gcn_layer_1.forward(X, A_hat)
        h_1_relu = F.relu(h_1)
        h_2 = self.gcn_layer_2.forward(h_1_relu, A_hat)
        h_2_relu = F.relu(h_2)
        h_3 = self.gcn_layer_3.forward(h_2_relu, A_hat)

        # create the input for the linear layer
        in_lin = torch.cat((h_1, h_2, h_3), dim=1)

        # perform the last linear layer
        output = self.linear_layer(in_lin)
        return output


def get_sparse_adjacency_normalized(number_nodes, adjacency_matrix):
    """
    Function to create a sparse adjacency matrix
    :param number_nodes: the number of nodes in the graph
    :param adjacency_matrix: the original adjacency matrix
    :return: the normalized adjacency matrix (self-loops added)
    """
    # create a coo_matrix of the adjacency matrix, and add the self_loops (while preserving the format)
    sparse_adjacency = coo_matrix(adjacency_matrix)
    sparse_A_tilde = coo_matrix(sparse_adjacency + sp.sparse.eye(number_nodes, format='coo'))

    # get the diagonal of the adjacency matrix with self-loops
    diag = np.array(sparse_A_tilde.sum(axis=1))
    diag = np.squeeze(diag)  # to get rid of the unnecessary dimension
    diag_tilde = diags(diag, offsets=0, shape=(number_nodes, number_nodes))

    # get the normalized adjacency matrix (in the scipy sparse coo format)
    A_hat = coo_matrix(diag_tilde.power(-0.5).dot(sparse_A_tilde).dot(diag_tilde.power(-0.5)))

    # put the normalized adjacency matrix in a torch tensor
    A_hat = torch.sparse_coo_tensor((A_hat.row, A_hat.col), A_hat.data, dtype=torch.float)

    return A_hat


def train_gcn(adjacency_matrix, features, labels, train_indices, test_indices, type_gcn="basic", initialization="basic"):
    """
    Train the defined GCN model and return it.
    Note that "test_indices" can also stand for the validation indices here.
    :param adjacency_matrix: the adjacency matrix of the graph
    :param features: the features of all vectors
    :param labels: the original labels of all vectors
    :param train_indices: the indices to train on
    :param test_indices: the indices to test on --> validation indices here if tuning
    :param type_gcn: "basic" or "linear" --> dependent on whether an additional linear layer was added to the gcn
    :return:
    """

    number_nodes = features.shape[0]
    feat_size = features.shape[1]
    hid_size = 20
    out_size = len(np.unique(labels))

    # create the model and optimizer (with the hyperparameters given by the paper)
    sparse_adj = get_sparse_adjacency_normalized(number_nodes, adjacency_matrix)

    # choose between without and with linear layer at the end
    if type_gcn == "basic":
        model = GCN3LayerBasic(feat_size, hid_size, out_size, initialization=initialization)
    elif type_gcn == "linear":
        model = GCN3LayerLinear(feat_size, hid_size, out_size, initialization=initialization)

    adam_opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
    loss_func = torch.nn.CrossEntropyLoss()

    # create lists for the data we need
    train_loss = []
    test_acc = []

    for epoch in range(1000):
        adam_opt.zero_grad()  # grads to zero

        output = model(features, sparse_adj)  # compute logits
        loss = loss_func(output[train_indices], labels[train_indices])  # calculate loss

        # calculate the gradients and clip them (with hyperparameter value given by original research)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        adam_opt.step()  # perform a step of the optimizer

        # add the training loss
        train_loss.append(loss.item())

        # find out what the testing accuracy is
        _, predictions = torch.max(output.data, 1)
        test_acc.append(accuracy_score(labels[test_indices], predictions[test_indices]))
        print("Epoch: ", epoch, " , accuracy = ", accuracy_score(labels[test_indices], predictions[test_indices]))

    return train_loss, test_acc, model






