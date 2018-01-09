# Feedback Alignment in PyTorch

This is a simple implementation of [Direct Feedback Alignment Provides Learning in Deep Neural Networks](https://www.nature.com/articles/ncomms13276) in PyTorch.

Base codes are adapted from [official PyTorch tutorial](http://pytorch.org/docs/master/notes/extending.html).

It implements simple MLP with one hidden layer, without non-linear activation function.

Run train_fa_vs_bp_linear_model.py to compare performance between feedback alignment vs backpropgation.