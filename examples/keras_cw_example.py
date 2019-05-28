""" Test case for Keras """
from __future__ import absolute_import

import numpy as np
import keras.applications as models
from perceptron.models.classification.keras import KerasModel
from perceptron.utils.image import imagenet_example
from perceptron.benchmarks.carlini_wagner import CarliniWagnerL2Metric
from perceptron.utils.criteria.classification import Misclassification
from perceptron.utils.tools import plot_image
from perceptron.utils.tools import bcolors

# instantiate the model from keras applications
resnet50 = models.ResNet50(weights='imagenet')

# initialize the KerasModel
# keras resnet50 has input bound (0, 255)
preprocessing = (np.array([104, 116, 123]), 1)  # the mean and stv of the whole dataset
kmodel = KerasModel(resnet50, bounds=(0, 255), preprocessing=preprocessing)

# get source image and label
# the model expects values in [0, 255], and channles_last
image, _ = imagenet_example(data_format='channels_last')
label = np.argmax(kmodel.predictions(image))

metric = CarliniWagnerL2Metric(kmodel, criterion=Misclassification())


print(bcolors.BOLD + 'Process start' + bcolors.ENDC)
adversary = metric(image, label, unpack=False)
print(bcolors.BOLD + 'Process finished' + bcolors.ENDC)

if adversary.image is None:
    print(bcolors.WARNING + 'Warning: Cannot find an adversary!' + bcolors.ENDC)
    exit(-1)

###################  print summary info  #####################################
keywords = ['Keras', 'ResNet50', 'Misclassification', 'CarliniWagnerL2']

true_label = np.argmax(kmodel.predictions(image))
fake_label = np.argmax(kmodel.predictions(adversary.image))

# interpret the label as human language
with open('perceptron/utils/labels.txt') as info:
    imagenet_dict = eval(info.read())

print(bcolors.HEADER + bcolors.UNDERLINE + 'Summary:' + bcolors.ENDC)
print('Configuration:'  + bcolors.CYAN + ' --framework %s '
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
