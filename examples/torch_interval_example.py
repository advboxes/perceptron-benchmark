""" Test case for Torch """
from __future__ import absolute_import

import torch
import numpy as np
from perceptron.models.classification.pytorch import PyTorchModel
from perceptron.benchmarks.interval_analysis import NaiveIntervalMetric
from perceptron.benchmarks.interval_analysis import SymbolicIntervalMetric
from perceptron.utils.criteria.classification import Misclassification
from perceptron.utils.image import load_mnist_image, load_cifar_image
from perceptron.utils.tools import bcolors

from perceptron.zoo.mnist.model import mnist_model
from perceptron.zoo.cifar10.model import cifar_model


param = {
	# wheter need to check for optimal verified bound
	"optimal_bound" : True,
	# whether to parallelize on multiple gpus for testing
	"parallel" : True,
	# size of the model
	"model_size" : "small",
	# the method used to train the existing models
	# including "clean", "madry", "mixtrain"
	"model_method" : "mixtrain",
	# dataset for testing
	# including ["mnist", "cifar10"]
	"dataset" : "cifar10",
	# number of classes of testing dataset
	"num_classes" : 10,
	# bounds for testing dataset
	"bounds" : (0,1),
	# whether the example is needed to be normalized
	"normalize" : False,
	# data format
	"data_format" : "channels_first",
}

assert param["dataset"] in ["mnist", "cifar10"],\
        "Only support mnist and cifar10 now"

# first set the testing Linf epsilon and the Linf
# bound used for training
if param["dataset"]=="mnist":
	# Linf range out of 1
	# epsilon for testing,
	param["epsilon"] = 0.3
	# epsilon for training
	param["train_epsilon"] = 0.3

if param["dataset"]=="cifar10":
	# Linf range out of 255
	# epsilon for testing,
	param["epsilon"] = 2
	# epsilon for training
	param["train_epsilon"] = 2


if param["dataset"]=="mnist":

	param["shape"] = (28,28)
	param["num_classes"] = 10

	# mnist example image
	param["fname"] = "mnist0.png"

	# the threshold for finding the optimal bound
	param["threshold"] = 0.001

	image = load_mnist_image(shape=param["shape"],
	                data_format='channels_first',
	                fname=param["fname"]
	            )

	net = mnist_model(model_size=param["model_size"],
	                method=param["model_method"],
	                train_epsilon=param["train_epsilon"]
	            ).eval()


if param["dataset"]=="cifar10":

	param["shape"] = (32,32)
	param["num_classes"] = 10

	# mnist example image
	param["fname"] = "cifar0.png"

	# whether the example is needed to be normalized
	param["normalize"] = True

	# the threshold for finding the optimal bound
	# should be set to be smaller than mnist
	param["threshold"] = 0.00001

	epsilon = param["epsilon"] /255.

	# Note: we do normalize for cifar by default.
	image = load_cifar_image(shape=param["shape"],
	            data_format='channels_first',
	            fname=param["fname"],
	            normalize=param["normalize"]
	        )

	param["bounds"] = (np.min(image), np.max(image))

	if param["normalize"]:
		epsilon = epsilon / 0.225

	net = cifar_model(model_size=param["model_size"],
	                method=param["model_method"],
	                train_epsilon=param["train_epsilon"]
	              ).eval()

  
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
                    original_pred=label, unpack=False,\
                    threshold=param["threshold"],\
                    normalize=param["normalize"]
                )

print("Analyze with symbolic interval analysis")
adversary2 = metric2(image, optimal_bound=param["optimal_bound"],\
                    epsilon=epsilon, parallel=param["parallel"],\
                    original_pred=label, unpack=False,\
                    threshold=param["threshold"],\
                    normalize=param["normalize"]
                )

print(bcolors.BOLD + 'Process finished' + bcolors.ENDC)


