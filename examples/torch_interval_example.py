""" Test case for Torch """
from __future__ import absolute_import

import torch
import numpy as np
from perceptron.models.classification.pytorch import PyTorchModel
from perceptron.benchmarks.interval_analysis import NaiveIntervalMetric
from perceptron.benchmarks.interval_analysis import SymbolicIntervalMetric
from perceptron.utils.criteria.classification import Misclassification
from perceptron.utils.image import load_mnist_image
from perceptron.utils.tools import bcolors

from perceptron.zoo.mnist.model import mnist_model

param = {
  # epsilon for testing
  "epsilon" : 0.3,
  # epsilon for training
  "train_epsilon" : 0.3,
  # wheter need to check for optimal verified bound
  "optimal_bound" : True,
  # whether to parallelize on multiple gpus for testing
  "parallel" : True,
  # size of the model
  "model_size" : "large",
  "model_method" : "clean",
  # dataset for testing
  "dataset" : "mnist",
  # number of classes of testing dataset
  "num_classes" : 10,
  # bounds for testing dataset
  "bounds" : (0,1),
  # data format
  "data_format" : "channels_first",
  # mnist example image
  "fname" : "mnist0.png",
  # shape of images
  "shape" : (28,28)
}

image = load_mnist_image(shape=(28, 28), data_format='channels_first', fname='mnist0.png')

epsilon = param["epsilon"]

if param["dataset"] == "mnist":

  net = mnist_model(model_size=param["model_size"],
                    method=param["model_method"],
                    train_epsilon=param["train_epsilon"]).eval()

  param["num_classes"] = 10

if torch.cuda.is_available():
    net = net.cuda()

fmodel = PyTorchModel(
            net, bounds=param["bounds"], 
            num_classes=param["num_classes"]
          )

label = np.argmax(fmodel.predictions(image))
metric1 = NaiveIntervalMetric(fmodel,\
          criterion=Misclassification(), threshold=epsilon)
metric2 = SymbolicIntervalMetric(fmodel,\
          criterion=Misclassification(), threshold=epsilon)

print(bcolors.BOLD + 'Process start' + bcolors.ENDC)

print("Analyze with naive interval analysis")
adversary1 = metric1(image, optimal_bound=param["optimal_bound"],\
                    epsilon=epsilon, parallel=param["parallel"],\
                    original_pred=label, unpack=False)

print("Analyze with symbolic interval analysis")
adversary2 = metric2(image, optimal_bound=param["optimal_bound"],\
                    epsilon=epsilon, parallel=param["parallel"],\
                    original_pred=label, unpack=False)

print(bcolors.BOLD + 'Process finished' + bcolors.ENDC)


