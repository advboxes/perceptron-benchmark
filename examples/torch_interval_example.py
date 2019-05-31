""" Test case for Torch """
from __future__ import absolute_import

import torch
import torchvision.models as models
import numpy as np
from perceptron.models.classification.pytorch import PyTorchModel
from perceptron.utils.image import imagenet_example
from perceptron.benchmarks.interval_analysis import SymbolicIntervalMetric, NaiveIntervalMetric
from perceptron.utils.criteria.classification import Misclassification
from perceptron.utils.tools import plot_image
from perceptron.utils.tools import bcolors

from perceptron.zoo.mnist.model import mnist_model


def load_mnist_image(fname, shape=(28, 28), dtype=np.float32,
            bounds=(0,1), data_format="channels_first"):
    from PIL import Image

    path = './perceptron/utils/images/' + fname
    #image = load_image(dtype=np.uint8, fname="mnist0.")
    image = Image.open(path)
    image = np.asarray(image, dtype=dtype)
    if(data_format=='channels_first'):
        image = image.reshape([1]+list(shape))
    else:
        image = image.reshape(list(shape)+[1])

    if bounds != (0, 255):
        image /= 255.

    return image

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
  "model_size" : "small",
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


image = load_mnist_image(param["fname"],\
              shape=param["shape"],\
              bounds=param["bounds"],\
              data_format=param["data_format"])

epsilon = param["epsilon"]

if param["dataset"] == "mnist":

  net = mnist_model(model_size=param["model_size"],\
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


