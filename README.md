# Relational-Walk-for-Knowledge-Graphs

This repository is for the Relational Walk model (RelWalk) that performs a random walk over a KG to explain what latent structure is being captured by knowledge graph embeddings (KGEs) of entities and relations. The RelWalk model is an extension of the random walk model of word embeddings proposed by [Arora et al., 2016](https://arxiv.org/abs/1502.03520) for KGEs to derive a scoring function that evaluates the strength of a relation r between two entities h (head) and t (tail) using their embeddings.

# Prerequisites 

The implementation of the RelWalk is built upon [OpenKE](https://github.com/thunlp/OpenKE/tree/OpenKE-Tensorflow1.0) that is implemented with TensorFlow. To implementation the code in this project requires:
     - python 
     - tensorflow

# Code
