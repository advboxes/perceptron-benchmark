""" Test case for Keras """

import numpy as np
import keras.applications as models
from perceptron.models.classification.keras import KerasModel
from perceptron.utils.image import imagenet_example
from perceptron.benchmarks.blended_noise import BlendedUniformNoiseMetric
from perceptron.utils.criteria.classification import TopKMisclassification
from perceptron.utils.tools import plot_image
from perceptron.utils.tools import bcolors


# instantiate the model from keras applications
vgg16 = models.VGG16(weights='imagenet')

# initialize the KerasModel
# keras vgg16 has input bound (0, 255)
preprocessing = (np.array([104, 116, 123]), 1)  # the mean and stv of the whole dataset
kmodel = KerasModel(vgg16, bounds=(0, 255), preprocessing=preprocessing)

# get source image and label
# the model expects values in [0, 255], and channles_last
image, _ = imagenet_example(data_format='channels_last')
label = np.argmax(kmodel.predictions(image))

metric = BlendedUniformNoiseMetric(kmodel, criterion=TopKMisclassification(10))

print(bcolors.BOLD + 'Process start' + bcolors.ENDC)
adversary = metric(image, label, unpack=False, epsilons=100)  # choose 100 different epsilon values in [0, 1]
print(bcolors.BOLD + 'Process finished' + bcolors.ENDC)

if adversary.image is None:
    print(bcolors.WARNING + 'Warning: Cannot find an adversary!' + bcolors.ENDC)
    exit(-1)

###################  print summary info  #####################################
keywords = ['Keras', 'VGG16', 'Top10Misclassification', 'BlendedUniform']

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
