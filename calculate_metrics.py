from statistics import mean
from statistics import stdev
import torch

"""
Contains all methods to calculate the metrics needed for evaluation.
"""


def explanation_size(examples_all, subgraphs):
    # make a list of all explanation sizes
    expl_size = []

    for i in range(len(examples_all)):
        if examples_all[i] != []:  # check whether there are counterfactual examples
            # take the "best" cf example and get the number of deleted edges
            # expl_size.append((sum(sum(i[-1] == 0)) / 2).item())
            new_matrix = torch.mul(subgraphs[i], examples_all[i][-1])
            explan_size = torch.sum((new_matrix != subgraphs[i]).float()).data / 2
            expl_size.append(explan_size.item())

    mean_expl_size = mean(expl_size)
    std_expl_size = stdev(expl_size)
    return mean_expl_size, std_expl_size, expl_size


def fidelity(total_nr, total_nr_examples):
    return 1.0 - float(total_nr_examples / total_nr)


def accuracy_explanation(test_indices, original_predictions, adjacency_matrices, perturbation_matrices, mapping_old):
    # accuracy:
    proportions = []

    for i in range(len(test_indices)):
        in_motif = 0
        total = 0

        mapping = dict([(value, key) for key, value in mapping_old[i].items()])

        if original_predictions[test_indices[i]] != 0 and perturbation_matrices[i] != []:  # only check the accuracy for nodes that are inside the motif
            adjusted = torch.mul(perturbation_matrices[i][-1], adjacency_matrices[i])
            indices = torch.nonzero(adjusted != adjacency_matrices[i])

            for j in indices:
                # if head and tail of edge is in the motif
                if original_predictions[mapping[j[0].item()]] != 0 and original_predictions[mapping[j[1].item()]] != 0:
                    in_motif = in_motif + 1
                total = total + 1

            # proportion is defined like how they did it in their code
            prop_correct = float(in_motif / total)
            proportions.append(prop_correct)

    if proportions != []:
        proportions_mean = mean(proportions)
        proportions_std = stdev(proportions)
    else:
        proportions_mean = None
        proportions_std = None
    return proportions_mean, proportions_std


def sparsity(adjacency_matrices, perturbation_matrices):
    spars = []

    for adj in range(len(adjacency_matrices)):
        if perturbation_matrices[adj] != []:
            perturbed = torch.mul(adjacency_matrices[adj], perturbation_matrices[adj][-1])
            distance = torch.sum((perturbed != adjacency_matrices[adj]).float()).data / 2
            nr_edges = torch.sum(adjacency_matrices[adj] != 0) / 2
            spars.append(1.0 - float(distance / nr_edges))

    mean_spars = mean(spars)
    std_spars = stdev(spars)

    return mean_spars, std_spars


