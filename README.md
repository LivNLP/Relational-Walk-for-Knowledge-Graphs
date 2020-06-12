# Relational-Walk-for-Knowledge-Graphs

This repository is for the Relational Walk model (RelWalk) that performs a random walk over a KG to explain what latent structure is being captured by knowledge graph embeddings (KGEs) of entities and relations. The RelWalk model is an extension of the random walk model of word embeddings proposed by [Arora et al., 2016](https://arxiv.org/abs/1502.03520) for KGEs to derive a scoring function that evaluates the strength of a relation r between two entities h (head) and t (tail) using their embeddings.

# Prerequisites 

The implementation of the RelWalk is built upon [OpenKE](https://github.com/thunlp/OpenKE/tree/OpenKE-Tensorflow1.0) that is implemented with TensorFlow. To implementation the code in this project requires:


     - python 
     - tensorflow

# Pre-trained RelWalk Embeddings
- FB15K-237 [download](https://www.dropbox.com/s/1whtpxmp4s9ol3f/Best_embeddings_SGD_lr0.001_d%3D100_epochs%3D1000_l1%3D10_l2%3D10_NegRate%3D20_unif_LogSoftmax.npy?dl=0)
- WN18RR [download]()
- FB13 [Download]()

To read the embeddings, use Read_Embeddings.py script in this repository. 
