from gcn import *
import torch
from torch.nn.functional import nll_loss
from torch import sigmoid

"""
Contains all the classes and functions needed to create and run the GCN with the perturbation matrix.
Also contains a training function for the GCNPerturbed.
"""


class GCNLayerFixed(torch.nn.Module):
    """
    Implementation of a GCN layer, with already fixed weight matrices.
    """

    def __init__(self, weight_matrix, bias):
        """
        Creates a GCN layer with fixed weights.
        :param weight_matrix: the already trained weight matrix from f
        :param bias: the already trained bias vector from f
        """
        super().__init__()
        self.W = torch.nn.Parameter(weight_matrix, requires_grad=False)
        self.b = torch.nn.Parameter(bias, requires_grad=False)

    def forward(self, X, A_hat):
        """
        Performs the forward pass with fixed weights.
        :param X: the feature matrix
        :param A_hat: the normalized adjacency matrix
        :return: the output of one step of the message passing algorithm
        """
        # perform one forward pass and add bias
        temp = torch.spmm(A_hat, X)
        Z = torch.spmm(temp, self.W) + self.b
        return Z


class GCNPerturbed(torch.nn.Module):
    """
    Implementation of a GCN, but with a perturbation matrix.
    """

    def __init__(self, W1, b1, W2, b2, W3, b3, wl, bl, number_neighbour_nodes, initialization='ones'):
        """
        Initializes the GCN with fixed weight matrices.
        :param W1: pre-trained weight matrix for GCN layer 1
        :param b1: pre-trained bias vector for GCN layer 1
        :param W2: pre-trained weight matrix for GCN layer 2
        :param b2: pre-trained bias vector for GCN layer 2
        :param W3: pre-trained weight matrix for GCN layer 3
        :param b3: pre-trained bias vector for GCN layer 3
        :param wl: pre-trained weight matrix for linear layer
        :param bl: pre-trained bias vector for linear layer
        :param number_neighbour_nodes: the number of nodes in the neighbourhood as defined with (A_v)
        """""

        super().__init__()
        # three GCN layers (using the module that was used before)
        self.gcn_layer_1 = GCNLayerFixed(W1, b1)
        self.gcn_layer_2 = GCNLayerFixed(W2, b2)
        self.gcn_layer_3 = GCNLayerFixed(W3, b3)
        self.linear_layer = torch.nn.Linear(wl.shape[1], wl.shape[0])

        with torch.no_grad():
            self.linear_layer.weight.copy_(wl)
            self.linear_layer.bias.copy_(bl)

        self.linear_layer.weight.requires_grad = False
        self.linear_layer.bias.requires_grad = False

        # p_hat should be a vector --> have to use this to populate a symmetric matrix later on

        size_vector = int(number_neighbour_nodes * (number_neighbour_nodes + 1) / 2)

        if initialization == "ones":
            self.p_hat = torch.nn.Parameter(torch.ones(size_vector, requires_grad=True))
        elif initialization == "uniform":
            self.p_hat = torch.nn.Parameter(torch.FloatTensor(size_vector).uniform_(), requires_grad=True)


    # this is similar to the original GCN, but now uses the p_hat!!
    def forward(self, X, A):
        """
        Performs the forward pass including the perturbation.
        :param self: the GCN
        :param X: the feature matrix of this node neighbourhood
        :param A: the adjacency matrix of this node neighbourhood
        :return: the output of the gcn operation
        """

        # populate the matrix symmetrically!! --> diagonal should not matter, but keep it 1 for now
        index_row, index_col = torch.triu_indices(A.shape[0], A.shape[0])  # A.shape[0] stands for the nr of nodes

        # populate the matrix symmetrically --> with p_hat this time!!
        P = torch.zeros(A.shape[0], A.shape[0])
        P[index_row, index_col] = sigmoid(self.p_hat)
        P.T[index_row, index_col] = sigmoid(self.p_hat)

        a_pert = torch.ones(A.shape[0], A.shape[0], dtype=torch.float)
        a_pert.requires_grad = True  # set this to true, to calculate the grad later on

        # now, multiply this one (element-wise) with the adjacency matrix --> real-valued!!
        a_pert = torch.mul(P, A)

        # now, we can not use the function defined in the GCN, because it needs to be done with torch ops
        # as the grad of this needs to be found later --> needs to be fully differentiable
        a_pert = torch.eye(A.shape[0]) + a_pert

        # normalize the adjacency matrix
        d_tilde = torch.diag(torch.sum(a_pert, dim=1))
        d_tilde = torch.pow(d_tilde, -0.5)
        d_tilde[torch.isinf(d_tilde)] = 0
        A_hat = torch.mm(torch.mm(d_tilde, a_pert), d_tilde)

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

    # this is similar to the original GCN, but now uses the perturbation matrix as well --> also only uses slices
    # of the original adjacency matrix --> for both X and A these slices are defined before!
    def forward_binary(self, X, A):
        """
        Performs the forward pass including the perturbation.
        :param self: the GCN
        :param X: the feature matrix of this node neighbourhood
        :param A: the adjacency matrix of this node neighbourhood
        :return: the output of the gcn operation
        """
        # first, get the perturbation matrix (binary)
        pert = sigmoid(self.p_hat)
        pert = (pert > 0.5).float()

        # populate the matrix symmetrically!! --> diagonal should not matter, but keep it 1 for now
        index_row, index_col = torch.triu_indices(A.shape[0], A.shape[0])  # A.shape[0] stands for the nr of nodes

        # populate the matrix symmetrically
        P = torch.zeros(A.shape[0], A.shape[0])
        P[index_row, index_col] = pert
        P.T[index_row, index_col] = pert

        # now, multiply this one (element-wise) with the adjacency matrix
        a_pert = torch.mul(P, A)

        # now, normalize A (we can use the function defined for the basic GCN)
        A_hat = get_sparse_adjacency_normalized(X.shape[0], a_pert)

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
        return (output, P)


def loss_function(beta, output_g, prediction_original, prediction_perturbed, adjacency_matrix_original,
                  perturbation_matrix_binary):
    """
    Calculates the loss (both parts, then combines them), for the counterfactual explanations.
    :param beta: the importance of Ldist in comparison to Lpred
    :param output_g: the output of the GCN with perturbation matrix
    :param prediction_original: predictions as given by f
    :param prediction_perturbed: predictions as given after this perturbation
    :param adjacency_matrix_original: the original adjacency subgraph
    :param perturbation_matrix_binary: the perturbation matrix (as trained)
    :return: the loss and the perturbed adjacency matrix (the cf-example if they're different
    """
    # find nll loss using the output and the predictions
    nll_loss_part = nll_loss(F.log_softmax(output_g), target=prediction_original) 
    loss_pred = - (prediction_original == prediction_perturbed).float() * nll_loss_part

    # get the new adjacency matrix
    adjacency_matrix_perturbed = torch.mul(adjacency_matrix_original, perturbation_matrix_binary)
    adjacency_matrix_perturbed.requires_grad = True  # otherwise, we will only calculate the grad for loss pred

    # count the number of times the adjacency matrix is different --> divide by 2 (because symmetric)
    loss_dist = torch.sum((adjacency_matrix_perturbed != adjacency_matrix_original).float()).data / 2

    return loss_pred + beta * loss_dist, adjacency_matrix_perturbed, loss_dist, loss_pred


# old - i don't use this anymore, but i left it in for completeness
def create_subgraph_neighbourhood(start_vertex, k_hops, labels, features, original_adj):
    """
    Creates and returns the subgraph neighbourhood.
    :param start_vertex: the vertex you want to get the k-hop subgraph neighbourhood for.
    :param k_hops: the number of hops for the neighbourhood (e.g., the number of message passing steps)
    :param labels: the labels of all vertices
    :param features: the features of all vertices
    :param original_adj: the original adjacency matrix containing all vertices
    :return: the subgraph adjacency matrix, the vertex mapping, the perturbed labels and the perturbed features
    """

    # make a subgraph neighbourhood for the starting vertex
    vertices = set([start_vertex])
    edges = set([])

    for hop in range(k_hops):  # loop over the set amount of hops
        v_list = list(vertices)
        for vertex_1 in v_list:  # loop over all vertices already in the set --> this is the column we're in
            for vertex_2 in range(original_adj.shape[0]):  # loop over every incoming connection of a certain vertex - this is the row
                if original_adj[vertex_2, vertex_1] != 0:
                    vertices.add(vertex_2)  # add the vertex if there is a connection
                    edges.add((vertex_2, vertex_1))  # add the edge if it exists

    # create the adjacency matrix:
    adj = torch.zeros(len(vertices), len(vertices))

    # perform a certain mapping:
    vertex_list = list(vertices)
    vertex_list.sort()
    vertex_indices = range(len(vertices))
    vertex_mapping = {}

    for vertex in range(len(vertex_list)):
        vertex_mapping[vertex_list[vertex]] = vertex_indices[vertex]

    # fill new adjacency matrix
    for edge in edges:
        edge_head = vertex_mapping[edge[0]]
        edge_tail = vertex_mapping[edge[1]]
        adj[edge_head][edge_tail] = original_adj[edge[0]][edge[1]]

    vertex_list = torch.tensor(vertex_list)

    # get new labels (slice)
    labels_perturbed = labels[vertex_list]

    # get new features (slice)
    features_perturbed = features[vertex_list, :]

    return adj, vertex_mapping, labels_perturbed, features_perturbed


def create_subgraph_neighbourhood2(start_vertex, k_hops, labels, features, original_adj):
    """
    Creates and returns the subgraph neighbourhood.
    :param start_vertex: the vertex you want to get the k-hop subgraph neighbourhood for.
    :param k_hops: the number of hops for the neighbourhood (e.g., the number of message passing steps)
    :param labels: the labels of all vertices
    :param features: the features of all vertices
    :param original_adj: the original adjacency matrix containing all vertices
    :return: the subgraph adjacency matrix, the vertex mapping, the perturbed labels and the perturbed features
    """

    # make a subgraph neighbourhood for the starting vertex
    vertices = set([start_vertex])

    for hop in range(k_hops):  # loop over the set amount of hops
        for vertex_1 in list(vertices):  # loop over all vertices already in the set --> this is the column we're in
            for vertex_2 in range(
                    original_adj.shape[0]):  # loop over every incoming connection of a certain vertex - this is the row
                if original_adj[vertex_2, vertex_1] != 0:
                    vertices.add(vertex_2)  # add the vertex if there is a connection

    # perform a certain mapping:
    vertex_list = list(vertices)
    vertex_list.sort()
    vertex_indices = range(len(vertices))
    vertex_mapping = {}

    for vertex in range(len(vertex_list)):
        vertex_mapping[vertex_list[vertex]] = vertex_indices[vertex]

    vertex_list = torch.tensor(vertex_list)

    adj = torch.index_select(original_adj, 0, vertex_list)
    adj = torch.index_select(adj, 1, vertex_list)

    # get new labels (slice)
    labels_perturbed = labels[vertex_list]

    # get new features (slice)
    features_perturbed = features[vertex_list, :]

    return adj, vertex_mapping, labels_perturbed, features_perturbed
