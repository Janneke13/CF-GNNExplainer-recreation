## CF-GNNExplainer, a recreation

_This repository aims at reproducing the results from paper:_

Lucic, Ana, et al. "Cf-gnnexplainer: Counterfactual explanations for graph neural networks." International Conference on Artificial Intelligence and Statistics. PMLR, 2022

### How to run the code:

First and foremost, one should install the packages as shown in the requirements.txt file in this Github.
Then, it is recommended to use the several notebooks given in this repository, which will perform the code step by step, and return metrics and plots.
The GCN models used in this repository, of which the predictions will be explained, are already given in the 'models' folder.

### About the structure of the code:
- In 'gcn.py', the regular GCN classes, as well as methods for normalization of the adjacency matrix and methods to train the GCN with the data, are given.
- In 'gcn_perturbation_matrix.py', the GCN with weight matrices constant and a perturbation matrix as parameter is given. A loss function and a function to get a neighbourhood graph are also given.
- In 'explainer_framework.py', functions to get explanations and train the GCN with the perturbation matrix are given.
- In 'calculate_metrics', functions to calculate the metrics as described in the paper/code are given.

**Additionally, notebooks are given to get several results and get baselines:**
- 'gcn_train_new.ipynb' trains the GCN model that was used in this recreation, and creates plots for it.
- 'baselines_notebook.ipynb' gets the baselines and calculates the metrics for them.
- 'CF-explainer-all.ipynb' trains the perturbation matrix and gets the metrics for this.
  - It consists of three different types: the pre-trained GCN model of the original paper added to my explanation framework, the GCN trained in this recreation with its explanation framework, and the GCN trained in this recreation, with my extended framework.
  
### Important notes:
- The models in folder models_original_paper, along with the gcn code in this folder, are used for evaluation and comparison purposes only.
These models were directly taken from the original codebase, which can be found in https://github.com/a-lucic/cf-gnnexplainer.  
- The data in folder 'data', is also directly taken from https://github.com/a-lucic/cf-gnnexplainer.
- The code in the gnnexplainer folder, is from the original GNNExplainer paper, which can be found in https://github.com/RexYing/gnn-model-explainer. 
- The rest of the code is written from scratch, while trying to consult the original code as little as possible.