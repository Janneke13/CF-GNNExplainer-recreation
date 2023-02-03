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
    return best_cf


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


# runs the counterfactual explanation method
def find_counterfactual_explanations(indices, model, predictions, labels, features, adjacency_matrix_orig, weights,
                                     alpha,
                                     beta, momentum, nr_hops, nr_iterations, extended=False):
    counterfactual_examples = []
    adjacency_neighbourhoods = []
    mapping_vertices = []
    number_counterfactual_examples = 0

    for index in indices:
        # get the old prediction
        old_prediction = predictions[index.item()]

        # get the subgraph neighbourhood
        adjacency_matrix, vertex_mapping, labels_perturbed, features_perturbed = create_subgraph_neighbourhood2(
            index.item(), nr_hops + 1, labels, features, adjacency_matrix_orig)

        new_index = vertex_mapping[index.item()]

        # test whether it gets the same outcome
        sparse_adj_test = get_sparse_adjacency_normalized(features_perturbed.shape[0], adjacency_matrix)
        with torch.no_grad():
            outputs_test = model(features_perturbed, sparse_adj_test)

        # get accuracy too (to check that it is the same as in the original)
        _, predictions_test = torch.max(outputs_test.data, 1)

        # as a small test:
        assert predictions_test[new_index].item() == old_prediction, "wrong prediction"

        # make a gcn model (to use for perturbation):
        if extended:
            model_pert = GCNPerturbed(weights["layer1_W"], weights["layer1_b"], weights["layer2_W"],
                                      weights["layer2_b"],
                                      weights["layer3_W"],
                                      weights["layer3_b"], weights["lin_weight"], weights["lin_b"],
                                      adjacency_matrix.shape[0], "uniform")
        else:
            model_pert = GCNPerturbed(weights["layer1_W"], weights["layer1_b"], weights["layer2_W"],
                                      weights["layer2_b"],
                                      weights["layer3_W"],
                                      weights["layer3_b"], weights["lin_weight"], weights["lin_b"],
                                      adjacency_matrix.shape[0])

        # from the model hyperparams:
        if momentum > 0:
            optim = torch.optim.SGD(model_pert.parameters(), lr=alpha, nesterov=True, momentum=momentum)
        else:
            optim = torch.optim.SGD(model_pert.parameters(), lr=alpha)

        # get the new cf example!
        examples_for_index = get_cf_example(new_index, old_prediction, model_pert, optim, beta, nr_iterations,
                                            adjacency_matrix, labels_perturbed, features_perturbed)

        # append to all examples!!
        counterfactual_examples.append(examples_for_index)
        adjacency_neighbourhoods.append(adjacency_matrix)
        mapping_vertices.append(vertex_mapping)

        # add one if a counterfactual example was found for this index
        if examples_for_index:
            number_counterfactual_examples += 1

    return counterfactual_examples, adjacency_neighbourhoods, mapping_vertices, number_counterfactual_examples
