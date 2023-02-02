import torch
from gcn import *
from gcn_perturbation_matrix import *


def get_cf_example(inx, pred_old, pert_model, optimizer, bet, k, adjacency_subgraph, labels_pert, features_pert):
    best_cf = []  # list of best counterfactual examples thus far
    train_loss = []  # list of all losses
    current_best = torch.inf  # best loss thus far

    for i in range(k):
        # need to add more here if needed:
        pert_matrix, prediction, loss = train_and_get_example(inx, pred_old, pert_model, optimizer, bet,
                                                              adjacency_subgraph, features_pert)

        if pred_old != prediction:  # if the prediction is different
            if not best_cf:  # if it is empty!!
                best_cf.append(pert_matrix)
                current_best = loss
            elif loss < current_best:
                best_cf.append(pert_matrix)
                current_best = loss
                print(loss)

        train_loss.append(loss)

    # return the list of all the best ones
    return best_cf, train_loss


def train_and_get_example(inx, pred_old, pert_model, optimizer, bet, adjacency_subgraph, features_pert):
    # set optimizer to zero grad!
    optimizer.zero_grad()

    # one forward step:
    output = pert_model(features_pert, adjacency_subgraph)
    output_perturbed, perturbation_matrix = pert_model.forward_binary(features_pert, adjacency_subgraph)

    # get the new prediction:
    new_prediction = torch.argmax(output_perturbed[inx])

    # calculate the loss:
    loss, perturbed_adj, loss_dist, loss_pred = loss_function(bet, output[inx], pred_old, new_prediction.item(),
                                                              adjacency_subgraph, perturbation_matrix)

    # calculate the grad
    loss.backward()

    # clip the gradients
    torch.nn.utils.clip_grad_norm_(pert_model.parameters(), 2.)

    # do a step with the optimizer
    optimizer.step()

    return perturbation_matrix, new_prediction, loss.item()


def get_examples_full():
    pass