################################################################################
# MIT License
#
# Copyright (c) 2024 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2024
# Date Created: 2024-10-28
################################################################################
"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from tqdm.auto import tqdm
from copy import deepcopy
from mlp_numpy import MLP
from modules import CrossEntropyModule
import cifar10_utils

import torch
import matplotlib.pyplot as plt

def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.

    Args:
      predictions: 2D float array of size [batch_size, n_classes], predictions of the model (logits)
      labels: 1D int array of size [batch_size]. Ground truth labels for
              each sample in the batch
    Returns:
      accuracy: scalar float, the accuracy of predictions between 0 and 1,
                i.e. the average correct predictions over the whole batch

    TODO:
    Implement accuracy computation.
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    preds = np.argmax(predictions, axis=1)
    correct = (preds == targets).astype(float)
    accuracy = np.mean(correct)
    #######################
    # END OF YOUR CODE    #
    #######################

    return accuracy


def evaluate_model(model, data_loader):
    """
    Performs the evaluation of the MLP model on a given dataset.

    Args:
      model: An instance of 'MLP', the model to evaluate.
      data_loader: The data loader of the dataset to evaluate.
    Returns:
      avg_accuracy: scalar float, the average accuracy of the model on the dataset.

    TODO:
    Implement evaluation of the MLP model on a given dataset.

    Hint: make sure to return the average accuracy of the whole dataset,
          independent of batch sizes (not all batches might be the same size).
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    model.clear_cache()
    correct = 0
    total = 0
    for batch in tqdm(data_loader, desc="Evaluating"):
        X, y = batch
        X = X.reshape(X.shape[0], -1).astype(np.float32)
        outputs = model.forward(X)
        acc = accuracy(outputs, y)
        correct += acc * X.shape[0]
        total += X.shape[0]
    avg_accuracy = correct / total
    #######################
    # END OF YOUR CODE    #
    #######################

    return avg_accuracy


def train(hidden_dims, lr, batch_size, epochs, seed, data_dir):
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that
                     performed best on the validation. Between 0.0 and 1.0
      logging_dict: An arbitrary object containing logging information. This is for you to
                    decide what to put in here.

    TODO:
    - Implement the training of the MLP model.
    - Evaluate your model on the whole validation set each epoch.
    - After finishing training, evaluate your model that performed best on the validation set,
      on the whole test dataset.
    - Integrate _all_ input arguments of this function in your training. You are allowed to add
      additional input argument if you assign it a default value that represents the plain training
      (e.g. '..., new_param=False')

    Hint: you can save your best model by deepcopy-ing it.
    """

    # Set the random seeds for reproducibility
    np.random.seed(seed)
    # Since the implementation is purely in NumPy, torch.manual_seed is not needed.

    ## Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(cifar10, batch_size=batch_size,
                                                  return_numpy=True)

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    train_loader = cifar10_loader['train']
    validation_loader = cifar10_loader['validation']
    test_loader = cifar10_loader['test']

    # TODO: Initialize model and loss module
    model = MLP(n_inputs=3072, n_hidden=hidden_dims, n_classes=10)
    loss_module = CrossEntropyModule()

    # TODO: Training loop including validation
    val_accuracies = []
    best_val_accuracy = 0.0
    best_model_state = deepcopy(model)
    logging_dict = {'epoch': [], 'val_accuracy': [], 'train_loss': []}

    for epoch in range(epochs):
      model.clear_cache()
      epoch_loss = 0.0
      for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}"):
        X, y = batch
        X = X.reshape(X.shape[0], -1).astype(np.float32)
        outputs = model.forward(X)
        loss = loss_module.forward(outputs, y)
        dout = loss_module.backward(outputs, y)
        model.backward(dout)

        for layer in model.layers:
          if isinstance(layer, LinearModule):
            layer.params['weight'] -= lr * layer.grads['weight']
            layer.params['bias'] -= lr * layer.grads['bias']

        epoch_loss += loss

      avg_epoch_loss = epoch_loss / len(train_loader)
      print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.4f}")
      logging_dict['epoch'].append(epoch+1)
      logging_dict['train_loss'].append(avg_epoch_loss)

      val_acc = evaluate_model(model, validation_loader)
      val_accuracies.append(val_acc)
      print(f"Validation Accuracy: {val_acc * 100:.2f}%")

      # TODO: Add any information you might want to save for plotting
      logging_dict['val_accuracy'].append(val_acc)

      # TODO: Test best model
      if val_acc > best_val_accuracy:
        best_val_accuracy = val_acc
        best_model_state = deepcopy(model)

    model = best_model_state

    test_accuracy = evaluate_model(model, test_loader)
    print(f"Test Accuracy of Best Model: {test_accuracy * 100:.2f}%")

    #######################
    # END OF YOUR CODE    #
    #######################

    return model, val_accuracies, test_accuracy, logging_dict


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()

    # Model hyperparameters
    parser.add_argument('--hidden_dims', default=[128], type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"')

    # Optimizer hyperparameters
    parser.add_argument('--lr', default=0.1, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--epochs', default=10, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR10 dataset.')

    args = parser.parse_args()
    kwargs = vars(args)

    model, val_accuracies, test_accuracy, logging_dict = train(**kwargs)
    # Feel free to add any additional functions, such as plotting of the loss curve here

    plt.figure(figsize=(10,5))
    plt.plot(logging_dict['epoch'], logging_dict['train_loss'], label='Training Loss')
    plt.plot(logging_dict['epoch'], [acc * 100 for acc in logging_dict['val_accuracy']], label='Validation Accuracy (%)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss / Accuracy')
    plt.title('Training Loss and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_loss_accuracy.png')
    plt.show()