# cs274b final project
# Constructing Semantically Constrained Sentence Embeddings
# 
# Authors: Arseny Moskvichev, James Liu

The model implementation is provided in the model.py file (with comments)

A probit_factor_model.py provides an implementation of Newton-Rhapson optimization procedure, based on this implementation: https://github.com/cbernet/python-scripts/blob/master/SKLearn/linear_model/LogisticRegression.py

The main training script is provided in the file loadall.py

parse_concept_net.py, unbin.py, and relation_matrix.py are scripts that we used to preprocess data

Model settings (vocabulary size, regularization, embedding dimension) could be set in settings.py



