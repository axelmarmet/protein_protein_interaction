# MSA Transformer Example

[In this notebook](https://github.com/axelmarmet/protein_transformer/blob/main/msa_transformer_example/contact_prediction.ipynb), we explore contact prediction with the **ESM-1b** and **MSA Transformer** models, in which contact prediction is based on a logistic regression over the model's attention maps. This methodology is based on the ICLR 2021 paper, [Transformer protein language models are unsupervised structure learners](https://openreview.net/pdf?id=fylclEqgvgd).

## a3m files

The .a3m files are MSAs coming from the [trRosetta (v1)](https://yanglab.nankai.edu.cn/trRosetta/benchmark/) dataset, which are also used in the MSA Transformer paper.

## MSA?

A multiple sequence alignment consists of a set of evolutionarily related protein sequences. Since real protein sequences are likely to have insertions, deletions, and substitutions, the sequences are aligned by minimizing a Levenshtein distance-like metric over all the sequences. In practice heuristic alignment schemes are used.

## Contact prediction pipeline

We call "contact prediction" the ability to predict which pairs of amino acid residues in a protein are in contact with each other.

How does it work?

The Transformer is first pretrained (in an unsupervised fashion) on sequences from a large database (Uniref50) via Masked Language Modeling. It ultimately learns the tertiary structure of a protein sequence in its attention maps, which will be useful for predicting contact.

The idea now is to extract residue-residue contact information from the attention maps of the model.

To do so, they first pass the input sequence through the model to obtain the attention maps (one map for each head in each layer). They then symmetrize and apply APC to each attention map independently. The resulting maps are passed through an L1 regularized logistic regression (that has been trained in a supervised fashion on a small number (n ≤ 20) of proteins to determine which attention heads are informative). This logistic regression is applied independently at each amino acid pair (i, j) to extract the contact information.

## Our work

In our work (see [`..\src`](https://github.com/axelmarmet/protein_transformer/tree/main/src)) we will use similar architectures and ideas to predict protein-protein interaction (rather than residue-residue contact).
