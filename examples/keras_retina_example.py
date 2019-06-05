""" Test case for Keras """

from perceptron.zoo.retinanet_resnet_50.retina_resnet50 import Retina_Resnet50
from perceptron.models.detection.keras_retina_resnet50 import KerasResNet50RetinaNetModel
from perceptron.utils.image import load_image
from perceptron.benchmarks.brightness import BrightnessMetric
from perceptron.utils.criteria.detection import TargetClassMiss
from perceptron.utils.tools import bcolors
from perceptron.utils.tools import plot_image_objectdetection


# instantiate the model from keras applications
# retina_resnet50 = Retina_Resnet50()
# initialize the KerasResNet50RetinaNetModel
kmodel = KerasResNet50RetinaNetModel(None, bounds=(0, 255))

# get source image and label
# the model expects values in [0, 1], and channles_last
image = load_image(shape=(416, 416), bounds=(0, 255), fname='car.png')

metric = BrightnessMetric(kmodel, criterion=TargetClassMiss(2))


print(bcolors.BOLD + 'Process start' + bcolors.ENDC)
adversary = metric(image, unpack=False)
print(bcolors.BOLD + 'Process finished' + bcolors.ENDC)

if adversary.image is None:
    print(bcolors.WARNING + 'Warning: Cannot find an adversary!' + bcolors.ENDC)
    exit(-1)

###################  print summary info  #####################################

keywords = ['Keras', 'RetinaResnet50', 'TargetClassMiss', 'BrightnessMetric']

print(bcolors.HEADER + bcolors.UNDERLINE + 'Summary:' + bcolors.ENDC)
print('Configuration:'  + bcolors.CYAN + ' --framework %s '
                                         '--model %s --criterion %s '
                                         '--metric %s' % tuple(keywords) + bcolors.ENDC)
                                         
print('Minimum perturbation required: %s' % bcolors.BLUE
      + str(adversary.distance) + bcolors.ENDC)
print('\n')

# print the original image and the adversary
plot_image_objectdetection(adversary, kmodel, bounds=(0, 255), title=", ".join(keywords), figname='examples/images/%s.png' % '_'.join(keywords))
