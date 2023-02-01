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
- 'baselines_notebook.ipynb' gets the baselines and calculates the metrics for them.
- 'CF_explainer_train.ipynb' trains the perturbation matrix and gets the metrics for this.
- 'CF_explainer_train_extension.ipynb' is similar, but has a different initialization function (extension).
- 'CF_explainer_train_originalgcn.ipynb' is also similar, but uses the weight matrices and GCN of the original paper, to compare to my implemented and trained one.

### Important notes:
- The models in folder models_original_paper, along with the gcn code in this folder, are used for evaluation and comparison purposes only.
These models were directly taken from the original codebase, which can be found in https://github.com/a-lucic/cf-gnnexplainer.  
- The data in folder 'data', is also directly taken from https://github.com/a-lucic/cf-gnnexplainer.
- The rest of the code is written from scratch, while trying to consult the original code as little as possible.