""" Test case for Torch """

from __future__ import absolute_import

import torch
import torchvision.models as models
import numpy as np
from perceptron.models.classification.pytorch import PyTorchModel
from perceptron.utils.image import load_imagenet_image
from perceptron.benchmarks.brightness import BrightnessMetric
from perceptron.utils.criteria.classification import Misclassification
from perceptron.utils.tools import plot_image
from perceptron.utils.tools import bcolors
from perceptron.utils.tools import load_pytorch_model
from PIL import Image
import perceptron
import os
import csv

# instantiate the model
resnet18 = models.resnet18(pretrained=True).eval()
if torch.cuda.is_available():
    resnet18 = resnet18.cuda()

# initialize the PyTorchModel
mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
models = ['alexnet', 'vgg16', 'vgg19', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
          'resnet152', 'inception_v3', 'squeezenet1_0', 'densenet121']
mse_dict = {}
output_folder = './out/cw2/'
for i in range(10):
    image_name = '0%s.jpg' % i
    image = load_imagenet_image(image_name, data_format='channels_first')
    for model_name in models:
        pmodel = load_pytorch_model(model_name)
        fmodel = PyTorchModel(
            pmodel, bounds=(0, 1), num_classes=1000, preprocessing=(mean, std))
        label = np.argmax(fmodel.predictions(image))
        ################
        metric = perceptron.benchmarks.CarliniWagnerL2Metric(fmodel, criterion=Misclassification())
        ################
        adversary = metric(image, label, unpack=False)  # set 'unpack' as false so we can access the detailed info of adversary
        if adversary.image is None:
            print(bcolors.WARNING + 'Warning: Cannot find an adversary!' + bcolors.ENDC)
            continue
        if model_name in mse_dict:
            mse_dict[model_name] += adversary.distance.value
        else:
            mse_dict[model_name] = adversary.distance.value
        adv_image = adversary.image
        output_adv_folder = output_folder + str(i) + '/'
        if not os.path.exists(output_adv_folder):
            os.makedirs(output_adv_folder)
        adv_file_name = os.path.join(output_adv_folder, '%s.jpg' % model_name)
        if adv_image.shape[0] == 3 :
            adv_image = np.transpose(adv_image, (1, 2, 0))
        adv_image = (adv_image * 255.).astype(np.uint8)
        Image.fromarray(adv_image).save(adv_file_name)
print(mse_dict)
w = csv.writer(open(os.path.join(output_folder, 'mse.csv'), 'w'))
for key, val in mse_dict.items():
    w.writerow([key, val / 10])
# set the type of noise which will used to generate the adversarial examples