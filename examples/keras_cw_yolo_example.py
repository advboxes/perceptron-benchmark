""" Test case for Keras """

from perceptron.zoo.yolov3.model import YOLOv3
from perceptron.models.detection.keras_yolov3 import KerasYOLOv3Model
from perceptron.utils.image import load_image
from perceptron.benchmarks.carlini_wagner import CarliniWagnerLinfMetric
from perceptron.utils.criteria.detection import TargetClassMiss
from perceptron.utils.tools import bcolors
from perceptron.utils.tools import plot_image_objectdetection

# instantiate the model from keras applications
yolov3 = YOLOv3()

# initialize the KerasYOLOv3Model
kmodel = KerasYOLOv3Model(yolov3, bounds=(0, 1))

# get source image and label
# the model expects values in [0, 1], and channles_last
image = load_image(shape=(416, 416), bounds=(0, 1), fname='car.png')

metric = CarliniWagnerLinfMetric(kmodel, criterion=TargetClassMiss(2))


print(bcolors.BOLD + 'Process start' + bcolors.ENDC)
adversary = metric(image, binary_search_steps=1, unpack=False)
print(bcolors.BOLD + 'Process finished' + bcolors.ENDC)

if adversary.image is None:
    print(bcolors.WARNING + 'Warning: Cannot find an adversary!' + bcolors.ENDC)
    exit(-1)

###################  print summary info  #####################################

keywords = ['Keras', 'YoLoV3', 'TargetClassMiss', 'CarliniWagnerLINF']

print(bcolors.HEADER + bcolors.UNDERLINE + 'Summary:' + bcolors.ENDC)
print('Configuration:'  + bcolors.CYAN + ' --framework %s '
                                         '--model %s --criterion %s '
                                         '--metric %s' % tuple(keywords) + bcolors.ENDC)
                                         
print('Minimum perturbation required: %s' % bcolors.BLUE
      + str(adversary.distance) + bcolors.ENDC)
print('\n')

# print the original image and the adversary
plot_image_objectdetection(adversary, kmodel, title=", ".join(keywords), figname='examples/images/%s.png' % '_'.join(keywords))
