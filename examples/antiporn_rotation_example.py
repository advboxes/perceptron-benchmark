# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from perceptron.utils.image import imagenet_example, load_image
from perceptron.models.classification.cloud import AipAntiPornModel
from perceptron.benchmarks.rotation import RotationMetric
from perceptron.utils.criteria.classification import MisclassificationAntiPorn
import numpy as np
from perceptron.utils.tools import plot_image
from perceptron.utils.tools import bcolors


# fill in your baidu AIP credentials
appId = '#'
apiKey = '#'
secretKey = "#"

credential = (appId, apiKey, secretKey)
model = AipAntiPornModel(credential)

# get source image and label
image = load_image(dtype=np.uint8, fname='porn.jpeg')

metric = RotationMetric(model, criterion=MisclassificationAntiPorn())

print(bcolors.BOLD + 'Process start' + bcolors.ENDC)
adversary = metric(image, ang_range=(-180., 180.), epsilons=1000, unpack=False)
print(bcolors.BOLD + 'Process finished' + bcolors.ENDC)

if adversary.image is None:
    print(bcolors.WARNING + 'Warning: Cannot find an adversary!' + bcolors.ENDC)
    exit(-1)

###################  print summary info  #####################################
keywords = ['Cloud', 'AipAntiPorn', 'Misclassification', 'Rotation']

true_label = str(model.predictions(image))
fake_label = str(model.predictions(adversary.image))

# interpret the label as human language
print(bcolors.HEADER + bcolors.UNDERLINE + 'Summary:' + bcolors.ENDC)
print('Configuration:'  + bcolors.CYAN + ' --framework %s '
                                         '--model %s --criterion %s '
                                         '--metric %s' % tuple(keywords) + bcolors.ENDC)
print('The predicted label of original image is '
      + bcolors.GREEN + true_label + bcolors.ENDC)
print('The predicted label of adversary image is '
      + bcolors.RED + fake_label + bcolors.ENDC)
print('Minimum perturbation required: %s' % bcolors.BLUE
      + str(adversary.distance) + bcolors.ENDC)
print('Verifiable bound: %s' % bcolors.BLUE
      + str(adversary.verifiable_bounds) + bcolors.ENDC)
print('\n')

plot_image(adversary,
           title=', '.join(keywords),
           figname='examples/images/%s.png' % '_'.join(keywords))

