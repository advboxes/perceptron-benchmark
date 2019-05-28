"""Test robustness of cloud model."""
from __future__ import absolute_import
# To be removed later
import sys
sys.path.append('/workspace/projects/baidu/aisec/perceptron')
import numpy as np
from perceptron.attacks import GaussianBlurAttack as Attack
from perceptron.utils.image import load_image
from perceptron.utils.image import imagenet_example
from perceptron.utils.criteria.classification import Misclassification
from perceptron.utils.image import load_image
from perceptron.models.classification.cloud import AipAntiPornModel, GoogleSafeSearchModel, GoogleObjectDetectionModel
from perceptron.utils.criteria.classification import MisclassificationSafeSearch
from perceptron.utils.criteria.detection import TargetClassMissGoogle

from perceptron.attacks import GaussianBlurAttack

def test_untargeted_SafeSearch(image, label=None):
    model = GoogleSafeSearchModel()
    pred = model.predictions(image)
    attack = Attack(model, criterion=MisclassificationSafeSearch())
    adversarial_obj = attack(image, label, unpack=False, epsilons=100)
    print(adversarial_obj.distance)
    return adversarial_obj.distance, adversarial_obj.image

if __name__ == "__main__":
    path = 'mia.png'
    img = load_image(dtype=np.uint8, fname=path)
    dist, image = test_untargeted_SafeSearch(img)
    from PIL import Image
    print(image.shape)
    print(image[0][0])
    adversarial_show = (image).astype(np.uint8)
    img = Image.fromarray(adversarial_show).save('./perceptron/utils/images/test_out_safe_search.png')
