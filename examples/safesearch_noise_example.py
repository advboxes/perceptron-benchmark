# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from perceptron.utils.image import imagenet_example, load_image
from perceptron.models.classification.cloud import GoogleSafeSearchModel
from perceptron.benchmarks.contrast_reduction import ContrastReductionMetric
from perceptron.utils.criteria.classification import MisclassificationSafeSearch
import numpy as np
from perceptron.utils.tools import plot_image
from perceptron.utils.tools import bcolors


# before running the example, please set the variable GOOGLE_APPLICATION_CREDENTIALS as follows
# export GOOGLE_APPLICATION_CREDENTIALS="[PATH]"
# replace [PATH] with the file path of the JSON file that contains your service account key
# For more details, please refer to https://cloud.google.com/docs/authentication/getting-started#auth-cloud-implicit-python

model = GoogleSafeSearchModel()

# get source image and label
image = load_image(dtype=np.uint8, fname='porn.jpeg')

metric = ContrastReductionMetric(model, criterion=MisclassificationSafeSearch())


print(bcolors.BOLD + 'Process start' + bcolors.ENDC)
adversary = metric(image, epsilons=10, unpack=False)
print(bcolors.BOLD + 'Process finished' + bcolors.ENDC)

if adversary.image is None:
    print(bcolors.WARNING + 'Warning: Cannot find an adversary!' + bcolors.ENDC)
    exit(-1)

###################  print summary info  #####################################
keywords = ['Cloud', 'GoogleSafeSearch', 'Misclassification', 'ContrastReduction']

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
print('\n')

plot_image(adversary,
           title=', '.join(keywords),
           figname='examples/images/%s.png' % '_'.join(keywords))

