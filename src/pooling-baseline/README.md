# Pooling baseline

[In this script](https://github.com/axelmarmet/protein_transformer/blob/main/src/pooling-baseline/pooling_baseline.py), you will find the code for our baseline. It is a basic MLP neural network with by default 4 layers of width 128. The input of the model is the precomputed MSA embedding (either the one from layer 2, or the one from layer 6, or the one from layer 11).

Results are shown [here](https://github.com/axelmarmet/protein_transformer/blob/main/src/pooling-baseline/pooling_baseline_results.txt). We are using Accuracy (TP + TN) / (TP + TN + FP + FN) as metric to compare our models. Different metrics should be considered in order to more accurately compare models (as discussed in this paper: https://arxiv.org/pdf/1511.02196.pdf).
