# MSA to embeddings

[In this notebook](https://github.com/axelmarmet/protein_transformer/blob/main/src/msa_to_embeddings/MSA%20to%20embeddings.ipynb), you will find the code that we wrote to build the `MSA_transformer_embeddings.zip` file that can be found [here](https://drive.google.com/drive/folders/1LQsWhlHzIwj_lRylsT0WR001mFt7RyCc).

In it, we run the MSA transformer on our MSAs files and save the embeddings that the MSA transformer produces at different layer. For now, we saved only embeddings at layer 2, 6 and 11.

The idea is to use these MSA embeddings as input to our different downstream models to make protein-protein predictions. Pre-computing them before allows faster training (as we only do it once and for all).
