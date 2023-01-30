import os
import numpy as np
import torch
import torch.nn.functional as F
import scipy as sp
from scipy.sparse import coo_matrix
from scipy.sparse import diags
from sklearn.metrics import accuracy_score
import pickle

"""
All functions needed to create and train a GCN model (3-layer)
The main of this file trains and saves the models for all datasets.
Note: the gcn_train.ipynb file is used to train, visualize and save the models (to be used in the experiments).
"""


class GCNlayer(torch.nn.Module):
    """
    Implementation of a simple GCN layer, with kaiming uniform initialization for the weight matrix.
    This implementation includes a bias as well, which is initialized as zeroes.
    """

    def __init__(self, n_in, n_out):
        """
        Initialized a GCN layer
        :param n_in: number of inputs
        :param n_out: number of outputs
        """
        super().__init__()
        matrix_to_fill = torch.empty(n_in, n_out)  # create a matrix of a certain shape (values don't matter)
        initialized_w = torch.nn.init.kaiming_uniform_(matrix_to_fill)  # use uniform kaiming to initialize
        self.W = torch.nn.Parameter(initialized_w)  # set W to be learnable
        bias = torch.zeros(n_out, dtype=torch.float)  # create a bias --> initialize as zeroes
        self.b = torch.nn.Parameter(bias)

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

    def __init__(self, n_in, n_hid, n_cls):
        """
        Initialize basic gcn (without lin layer)
        :param n_in: nr of features
        :param n_hid: hidden size
        :param n_cls: number of classes
        """
        super().__init__()
        # three GCN layers (using the module that was used before)
        self.gcn_layer_1 = GCNlayer(n_in, n_hid)
        self.gcn_layer_2 = GCNlayer(n_hid, n_hid)
        self.gcn_layer_3 = GCNlayer(n_hid, n_cls)

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

    def __init__(self, n_in, n_hid, n_cls):
        """
        Initializes the 3-layer GCN with a last linear layer
        :param n_in: number of input features
        :param n_hid: size of hidden layer
        :param n_cls: number of classes
        """
        super().__init__()
        # three GCN layers (using the module that was used before)
        self.gcn_layer_1 = GCNlayer(n_in, n_hid)
        self.gcn_layer_2 = GCNlayer(n_hid, n_hid)
        self.gcn_layer_3 = GCNlayer(n_hid, n_hid)
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


def train_gcn(adjacency_matrix, features, labels, train_indices, test_indices, type_gcn="basic"):
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
        model = GCN3LayerBasic(feat_size, hid_size, out_size)
    elif type_gcn == "linear":
        model = GCN3LayerLinear(feat_size, hid_size, out_size)

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


def train_and_save_models():
    """"
    Directly trains the tree models (linear) and saves them for further use.
    Does essentially the same as the gcn_train notebook, but this is a shortcut (in case needed).
    """

    # load data
    with open('data/syn1.pickle', 'rb') as pickle_file:
        data_syn1 = pickle.load(pickle_file)

    with open('data/syn4.pickle', 'rb') as pickle_file:
        data_syn4 = pickle.load(pickle_file)

    with open('data/syn5.pickle', 'rb') as pickle_file:
        data_syn5 = pickle.load(pickle_file)

    # squeeze the labels (as it has a singleton dim and then make it a tensor)
    labels_syn1 = np.squeeze(data_syn1['labels'])
    labels_syn1 = torch.tensor(labels_syn1)

    labels_syn4 = np.squeeze(data_syn4['labels'])
    labels_syn4 = torch.tensor(labels_syn4)

    labels_syn5 = np.squeeze(data_syn5['labels'])
    labels_syn5 = torch.tensor(labels_syn5)

    # same for features, but define the type of data here
    features_syn1 = np.squeeze(data_syn1['feat'])
    features_syn1 = torch.tensor(features_syn1, dtype=torch.float)

    features_syn4 = np.squeeze(data_syn4['feat'])
    features_syn4 = torch.tensor(features_syn4, dtype=torch.float)

    features_syn5 = np.squeeze(data_syn5['feat'])
    features_syn5 = torch.tensor(features_syn5, dtype=torch.float)

    # adjacency matrix will be turned into a tensor later on
    adjacency_matrix_syn1 = np.squeeze(data_syn1['adj'])
    adjacency_matrix_syn4 = np.squeeze(data_syn4['adj'])
    adjacency_matrix_syn5 = np.squeeze(data_syn5['adj'])

    # the indices are already a list --> turn into a tensor in case I want to go to the gpu later
    train_indices_full_syn1 = torch.tensor(data_syn1['train_idx'])
    train_indices_full_syn4 = torch.tensor(data_syn4['train_idx'])
    train_indices_full_syn5 = torch.tensor(data_syn5['train_idx'])

    test_indices_syn1 = torch.tensor(data_syn1['test_idx'])
    test_indices_syn4 = torch.tensor(data_syn4['test_idx'])
    test_indices_syn5 = torch.tensor(data_syn5['test_idx'])

    # for the first dataset
    np.random.seed(42)
    torch.manual_seed(42)
    train_loss_syn1, test_acc_syn1, syn_1_model = train_gcn(adjacency_matrix_syn1, features_syn1, labels_syn1,
                                                            train_indices_full_syn1, test_indices_syn1,
                                                            type_gcn="linear")

    # for the second dataset
    np.random.seed(42)
    torch.manual_seed(42)
    train_loss_syn4, test_acc_syn4, syn_4_model = train_gcn(adjacency_matrix_syn4, features_syn4, labels_syn4,
                                                            train_indices_full_syn4, test_indices_syn4,
                                                            type_gcn="linear")

    # for the third dataset
    np.random.seed(42)
    torch.manual_seed(42)
    train_loss_syn5, test_acc_syn5, syn_5_model = train_gcn(adjacency_matrix_syn5, features_syn5, labels_syn5,
                                                            train_indices_full_syn5, test_indices_syn5,
                                                            type_gcn="linear")

    # print accuracy on the models eventually:
    syn_1_model.eval()
    sparse_adj_1 = get_sparse_adjacency_normalized(features_syn1.shape[0], adjacency_matrix_syn1)
    eval_1 = syn_1_model(features_syn1, sparse_adj_1)

    # get predictions and print accuracy
    _, predictions_1 = torch.max(eval_1.data, 1)
    print("Test accuracy of Syn1 data: ",
          accuracy_score(labels_syn1[test_indices_syn1], predictions_1[test_indices_syn1]))

    syn_4_model.eval()
    sparse_adj_4 = get_sparse_adjacency_normalized(features_syn4.shape[0], adjacency_matrix_syn4)
    eval_4 = syn_4_model(features_syn4, sparse_adj_4)

    # get predictions and print accuracy
    _, predictions_4 = torch.max(eval_4.data, 1)
    print("Test accuracy of Syn4 data: ",
          accuracy_score(labels_syn4[test_indices_syn4], predictions_4[test_indices_syn4]))

    syn_5_model.eval()
    sparse_adj_5 = get_sparse_adjacency_normalized(features_syn5.shape[0], adjacency_matrix_syn5)
    eval_5 = syn_5_model(features_syn5, sparse_adj_5)

    # get predictions and print accuracy
    _, predictions_5 = torch.max(eval_5.data, 1)
    print("Test accuracy of Syn1 data: ",
          accuracy_score(labels_syn5[test_indices_syn5], predictions_5[test_indices_syn5]))

    # set the model settings to train again
    syn_1_model.train()
    syn_4_model.train()
    syn_5_model.train()

    # save the models in a folder
    if not os.path.exists("models"):
        os.makedirs("models")

    torch.save(syn_1_model, 'models/syn1model.pt')
    torch.save(syn_4_model, 'models/syn4model.pt')
    torch.save(syn_5_model, 'models/syn5model.pt')


if __name__ == '__main__':
    """
    Main performs the training and saving models
    """
    train_and_save_models()
