"""Test for carlini & wagner attack."""
from __future__ import absolute_import

# To be removed later
import sys
sys.path.append('/workspace/projects/baidu/aisec/perceptron')

from perceptron.utils.criteria.classification import Misclassification
from perceptron.utils.image import imagenet_example
from perceptron.utils.image import load_image
from perceptron.attacks import CarliniWagnerL2Attack as Attack
import numpy as np



def test_untargeted_vgg16(image, label=None):
    import torch
    import torchvision.models as models
    from perceptron.models.classification import PyTorchModel
    mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    vgg16 = models.vgg16(pretrained=True).eval()
    if torch.cuda.is_available():
        vgg16 = vgg16.cuda()
    model = PyTorchModel(
        vgg16, bounds=(0, 1), num_classes=1000, preprocessing=(mean, std))
    print(np.argmax(model.predictions(image)))
    attack = Attack(model, criterion=Misclassification())
    print(image.shape)
    adversarial = attack(image, label, unpack=True)


def test_untargeted_resnet18(image, label=None):
    import torch
    import torchvision.models as models
    from perceptron.models.classification import PyTorchModel
    mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    resnet18 = models.resnet18(pretrained=True).eval()
    if torch.cuda.is_available():
        resnet18 = resnet18.cuda()
    model = PyTorchModel(
        resnet18, bounds=(0, 1), num_classes=1000, preprocessing=(mean, std))
    print(np.argmax(model.predictions(image)))
    attack = Attack(model, criterion=Misclassification())
    adversarial = attack(image, label, unpack=True)


if __name__ == "__main__":
    image = load_image(
        shape=(224, 224), data_format='channels_first', fname='car.png')
    image = image / 255.
    label = 644
    test_untargeted_vgg16(image, label)
