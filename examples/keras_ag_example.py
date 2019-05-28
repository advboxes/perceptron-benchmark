""" Test case for Keras """
from __future__ import absolute_import

import numpy as np
import keras.applications as models
from perceptron.models.classification.keras import KerasModel
from perceptron.utils.image import imagenet_example
from perceptron.benchmarks.additive_noise import AdditiveGaussianNoiseMetric
from perceptron.utils.criteria.classification import TopKMisclassification
from perceptron.utils.tools import plot_image
from perceptron.utils.tools import bcolors

# instantiate the model from keras applications
xception = models.Xception(weights='imagenet')

# initialize the KerasModel
# keras xception has input bound (0, 1)
mean = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
std = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))
kmodel = KerasModel(xception, bounds=(0, 1), preprocessing=(mean, std))

# get source image and label
# the model Xception expects values in [0, 1] with shape (299, 299), and channles_last
image, _ = imagenet_example(shape=(299, 299), data_format='channels_last')
image /= 255.0
label = np.argmax(kmodel.predictions(image))

metric = AdditiveGaussianNoiseMetric(kmodel, criterion=TopKMisclassification(10))

print(bcolors.BOLD + 'Process start' + bcolors.ENDC)
adversary = metric(image, label, unpack=False, epsilons=1000)  # choose 1000 different epsilon values in [0, 1]
print(bcolors.BOLD + 'Process finished' + bcolors.ENDC)

if adversary.image is None:
    print(bcolors.WARNING + 'Warning: Cannot find an adversary!' + bcolors.ENDC)
    exit(-1)

###################  print summary info  #####################################
keywords = ['Keras', 'Xception', 'Top10Misclassification', 'AdditiveGaussian']

true_label = np.argmax(kmodel.predictions(image))
fake_label = np.argmax(kmodel.predictions(adversary.image))

# interpret the label as human language
with open('perceptron/utils/labels.txt') as info:
    imagenet_dict = eval(info.read())

print(bcolors.HEADER + bcolors.UNDERLINE + 'Summary:' + bcolors.ENDC)
print('Configuration:' + bcolors.CYAN + ' --framework %s '
                                         '--model %s --criterion %s '
                                         '--metric %s' % tuple(keywords) + bcolors.ENDC)
print('The predicted label of original image is '
      + bcolors.GREEN + imagenet_dict[true_label] + bcolors.ENDC)
print('The predicted label of adversary image is '
      + bcolors.RED + imagenet_dict[fake_label] + bcolors.ENDC)
print('Minimum perturbation required: %s' % bcolors.BLUE
      + str(adversary.distance) + bcolors.ENDC)
print('\n')

plot_image(adversary,
           title=', '.join(keywords),
           figname='examples/images/%s.png' % '_'.join(keywords))

