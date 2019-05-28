"""Test for Gaussian blur attack."""
from __future__ import absolute_import
# To be removed later
import sys
sys.path.append('/home/yantao/workspace/projects/baidu/aisec/perceptron')
import numpy as np
from perceptron.attacks import GaussianBlurAttack as Attack
from perceptron.utils.image import load_image
from perceptron.utils.image import imagenet_example
from perceptron.utils.criteria.classification import Misclassification


def test_untargeted_AlexNet(image, label=None):
    import torch
    import torchvision.models as models
    from perceptron.models.classification import PyTorchModel
    mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    model_pyt = models.alexnet(pretrained=True).eval()
    if torch.cuda.is_available():
        model_pyt = model_pyt.cuda()
    model = PyTorchModel(
        model_pyt, bounds=(0, 1), num_classes=1000, preprocessing=(mean, std))
    print(np.argmax(model.predictions(image)))
    attack = Attack(model, criterion=Misclassification())
    adversarial_obj = attack(image, label, unpack=False, epsilons=10000)
    distance = adversarial_obj.distance
    adversarial = adversarial_obj.image
    return distance, adversarial


def test_untargeted_vgg16(image, label=None):
    import torch
    import torchvision.models as models
    from perceptron.models.classification import PyTorchModel
    mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    model_pyt = models.vgg16(pretrained=True).eval()
    if torch.cuda.is_available():
        model_pyt = model_pyt.cuda()
    model = PyTorchModel(
        model_pyt, bounds=(0, 1), num_classes=1000, preprocessing=(mean, std))
    print(np.argmax(model.predictions(image)))
    attack = Attack(model, criterion=Misclassification())
    adversarial_obj = attack(image, label, unpack=False, epsilons=10000)
    distance = adversarial_obj.distance
    adversarial = adversarial_obj.image
    return distance, adversarial


def test_untargeted_resnet18(image, label=None):
    import torch
    import torchvision.models as models
    from perceptron.models.classification import PyTorchModel
    mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    model_pyt = models.resnet18(pretrained=True).eval()
    if torch.cuda.is_available():
        model_pyt = model_pyt.cuda()
    model = PyTorchModel(
        model_pyt, bounds=(0, 1), num_classes=1000, preprocessing=(mean, std))
    print(np.argmax(model.predictions(image)))
    attack = Attack(model, criterion=Misclassification())
    adversarial_obj = attack(image, label, unpack=False, epsilons=10000)
    distance = adversarial_obj.distance
    adversarial = adversarial_obj.image
    return distance, adversarial


def test_untargeted_resnet50(image, label=None):
    import torch
    import torchvision.models as models
    from perceptron.models.classification import PyTorchModel
    mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    model_pyt = models.resnet50(pretrained=True).eval()
    if torch.cuda.is_available():
        model_pyt = model_pyt.cuda()
    model = PyTorchModel(
        model_pyt, bounds=(0, 1), num_classes=1000, preprocessing=(mean, std))
    print(np.argmax(model.predictions(image)))
    attack = Attack(model, criterion=Misclassification())
    adversarial_obj = attack(image, label, unpack=False, epsilons=10000)
    distance = adversarial_obj.distance
    adversarial = adversarial_obj.image
    return distance, adversarial


def test_untargeted_resnet152(image, label=None):
    import torch
    import torchvision.models as models
    from perceptron.models.classification import PyTorchModel
    mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    model_pyt = models.resnet152(pretrained=True).eval()
    if torch.cuda.is_available():
        model_pyt = model_pyt.cuda()
    model = PyTorchModel(
        model_pyt, bounds=(0, 1), num_classes=1000, preprocessing=(mean, std))
    print(np.argmax(model.predictions(image)))
    attack = Attack(model, criterion=Misclassification())
    adversarial_obj = attack(image, label, unpack=False, epsilons=10000)
    distance = adversarial_obj.distance
    adversarial = adversarial_obj.image
    return distance, adversarial


def test_untargeted_squeezenet1_1(image, label=None):
    import torch
    import torchvision.models as models
    from perceptron.models.classification import PyTorchModel
    mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    model_pyt = models.squeezenet1_1(pretrained=True).eval()
    if torch.cuda.is_available():
        model_pyt = model_pyt.cuda()
    model = PyTorchModel(
        model_pyt, bounds=(0, 1), num_classes=1000, preprocessing=(mean, std))
    print(np.argmax(model.predictions(image)))
    attack = Attack(model, criterion=Misclassification())
    adversarial_obj = attack(image, label, unpack=False, epsilons=10000)
    distance = adversarial_obj.distance
    adversarial = adversarial_obj.image
    return distance, adversarial


def test_untargeted_densenet121(image, label=None):
    import torch
    import torchvision.models as models
    from perceptron.models.classification import PyTorchModel
    mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    model_pyt = models.densenet121(pretrained=True).eval()
    if torch.cuda.is_available():
        model_pyt = model_pyt.cuda()
    model = PyTorchModel(
        model_pyt, bounds=(0, 1), num_classes=1000, preprocessing=(mean, std))
    print(np.argmax(model.predictions(image)))
    attack = Attack(model, criterion=Misclassification())
    adversarial_obj = attack(image, label, unpack=False, epsilons=10000)
    distance = adversarial_obj.distance
    adversarial = adversarial_obj.image
    return distance, adversarial


def test_untargeted_densenet201(image, label=None):
    import torch
    import torchvision.models as models
    from perceptron.models.classification import PyTorchModel
    mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    model_pyt = models.densenet201(pretrained=True).eval()
    if torch.cuda.is_available():
        model_pyt = model_pyt.cuda()
    model = PyTorchModel(
        model_pyt, bounds=(0, 1), num_classes=1000, preprocessing=(mean, std))
    print(np.argmax(model.predictions(image)))
    attack = Attack(model, criterion=Misclassification())
    adversarial_obj = attack(image, label, unpack=False, epsilons=10000)
    distance = adversarial_obj.distance
    adversarial = adversarial_obj.image
    return distance, adversarial


def test_untargeted_inception_v3(image, label=None):
    import torch
    import torchvision.models as models
    from perceptron.models.classification import PyTorchModel
    mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    model_pyt = models.inception_v3(pretrained=True).eval()
    if torch.cuda.is_available():
        model_pyt = model_pyt.cuda()
    model = PyTorchModel(
        model_pyt, bounds=(0, 1), num_classes=1000, preprocessing=(mean, std))
    print(np.argmax(model.predictions(image)))
    attack = Attack(model, criterion=Misclassification())
    adversarial_obj = attack(image, label, unpack=False, epsilons=10000)
    distance = adversarial_obj.distance
    adversarial = adversarial_obj.image
    return distance, adversarial


def test_untargeted_Xception(image, label=None):
    import keras
    from perceptron.models.classification.keras import KerasModel
    mean = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
    std = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))
    model_keras = keras.applications.xception.Xception(weights='imagenet')
    model = KerasModel(model_keras, bounds=(0, 1), preprocessing=(mean, std))
    print(np.argmax(model.predictions(image)))
    attack = Attack(model, criterion=Misclassification())
    adversarial_obj = attack(image, label, unpack=False, epsilons=10000)
    distance = adversarial_obj.distance
    adversarial = adversarial_obj.image
    return distance, adversarial


def test_untargeted_mobilenet_v2(image, label=None):
    import keras
    from perceptron.models.classification.keras import KerasModel
    mean = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
    std = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))
    preprocessing = (mean, std)
    model_keras = keras.applications.mobilenet_v2.MobileNetV2(
        weights='imagenet')
    model = KerasModel(model_keras, bounds=(0, 1), preprocessing=preprocessing)
    print(np.argmax(model.predictions(image)))
    attack = Attack(model, criterion=Misclassification())
    adversarial_obj = attack(image, label, unpack=False, epsilons=10000)
    distance = adversarial_obj.distance
    adversarial = adversarial_obj.image
    return distance, adversarial


def test_untargeted_NASNetMobile(image, label=None):
    import keras
    from perceptron.models.classification.keras import KerasModel
    mean = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
    std = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))
    preprocessing = (mean, std)
    model_keras = keras.applications.nasnet.NASNetMobile(weights='imagenet')
    model = KerasModel(model_keras, bounds=(0, 1), preprocessing=preprocessing)
    print(np.argmax(model.predictions(image)))
    attack = Attack(model, criterion=Misclassification())
    adversarial_obj = attack(image, label, unpack=False, epsilons=10000)
    distance = adversarial_obj.distance
    adversarial = adversarial_obj.image
    return distance, adversarial


def test_untargeted_NASNetLarge(image, label=None):
    import keras
    from perceptron.models.classification.keras import KerasModel
    mean = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
    std = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))
    preprocessing = (mean, std)
    model_keras = keras.applications.nasnet.NASNetLarge(weights='imagenet')
    model = KerasModel(model_keras, bounds=(0, 1), preprocessing=preprocessing)
    print(np.argmax(model.predictions(image)))
    attack = Attack(model, criterion=Misclassification())
    import pdb
    pdb.set_trace()
    adversarial_obj = attack(image, label, unpack=False, epsilons=10000)
    distance = adversarial_obj.distance
    adversarial = adversarial_obj.image
    return distance, adversarial


if __name__ == "__main__":
    import csv
    import pdb
    file_name = '../temp_test_result.csv'
    img_dir = '../temp_images'
    with open(file_name, 'w') as csvfile:
        writer = csv.writer(
            csvfile,
            delimiter=',',
            quotechar='"',
            quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["model", "result"])

    from PIL import Image
    # alexnet
    image = load_image(
        shape=(
            224,
            224),
        data_format='channels_first',
        fname='test_car.png')
    image = image / 255.
    label = 656
    distance, adversarial = test_untargeted_AlexNet(image, label)
    adversarial_show = np.transpose(adversarial, (1, 2, 0))
    adversarial_show = (adversarial_show * 255).astype(np.uint8)
    img = Image.fromarray(adversarial_show).save(img_dir + '/test_out_alex.png')
    with open(file_name, 'a') as csvfile:
        writer = csv.writer(
            csvfile,
            delimiter=',',
            quotechar='"',
            quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["alexnet", "{0:.4e}".format(distance.value)])

    # vgg16
    image = load_image(
        shape=(
            224,
            224),
        data_format='channels_first',
        fname='test_car.png')
    image = image / 255.
    label = 656
    distance, adversarial = test_untargeted_vgg16(image, label)
    adversarial_show = np.transpose(adversarial, (1, 2, 0))
    adversarial_show = (adversarial_show * 255).astype(np.uint8)
    img = Image.fromarray(adversarial_show).save(img_dir + '/test_out_vgg16.png')
    with open(file_name, 'a') as csvfile:
        writer = csv.writer(
            csvfile,
            delimiter=',',
            quotechar='"',
            quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["vgg16", "{0:.4e}".format(distance.value)])

    # resnet18
    image = load_image(
        shape=(
            224,
            224),
        data_format='channels_first',
        fname='test_car.png')
    image = image / 255.
    label = 656
    distance, adversarial = test_untargeted_resnet18(image, label)
    adversarial_show = np.transpose(adversarial, (1, 2, 0))
    adversarial_show = (adversarial_show * 255).astype(np.uint8)
    img = Image.fromarray(adversarial_show).save(
        img_dir + '/test_out_resnet18.png')
    with open(file_name, 'a') as csvfile:
        writer = csv.writer(
            csvfile,
            delimiter=',',
            quotechar='"',
            quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["resnet18", "{0:.4e}".format(distance.value)])

    # resnet50
    image = load_image(
        shape=(
            224,
            224),
        data_format='channels_first',
        fname='test_car.png')
    image = image / 255.
    label = 656
    distance, adversarial = test_untargeted_resnet50(image, label)
    adversarial_show = np.transpose(adversarial, (1, 2, 0))
    adversarial_show = (adversarial_show * 255).astype(np.uint8)
    img = Image.fromarray(adversarial_show).save(
        img_dir + '/test_out_resnet50.png')
    with open(file_name, 'a') as csvfile:
        writer = csv.writer(
            csvfile,
            delimiter=',',
            quotechar='"',
            quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["resnet50", "{0:.4e}".format(distance.value)])

    # resnet152
    image = load_image(
        shape=(
            224,
            224),
        data_format='channels_first',
        fname='test_car.png')
    image = image / 255.
    label = 656
    distance, adversarial = test_untargeted_resnet152(image, label)
    adversarial_show = np.transpose(adversarial, (1, 2, 0))
    adversarial_show = (adversarial_show * 255).astype(np.uint8)
    img = Image.fromarray(adversarial_show).save(
        img_dir + '/test_out_resnet152.png')
    with open(file_name, 'a') as csvfile:
        writer = csv.writer(
            csvfile,
            delimiter=',',
            quotechar='"',
            quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["resnet152", "{0:.4e}".format(distance.value)])

    # squeezenet1_1
    image = load_image(
        shape=(
            224,
            224),
        data_format='channels_first',
        fname='test_car.png')
    image = image / 255.
    label = 656
    distance, adversarial = test_untargeted_squeezenet1_1(image, label)
    adversarial_show = np.transpose(adversarial, (1, 2, 0))
    adversarial_show = (adversarial_show * 255).astype(np.uint8)
    img = Image.fromarray(adversarial_show).save(
        img_dir + '/test_out_squeezenet1_1.png')
    with open(file_name, 'a') as csvfile:
        writer = csv.writer(
            csvfile,
            delimiter=',',
            quotechar='"',
            quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["squeezenet1_1", "{0:.4e}".format(distance.value)])

    # densenet121
    image = load_image(
        shape=(
            224,
            224),
        data_format='channels_first',
        fname='test_car.png')
    image = image / 255.
    label = 656
    distance, adversarial = test_untargeted_densenet121(image, label)
    adversarial_show = np.transpose(adversarial, (1, 2, 0))
    adversarial_show = (adversarial_show * 255).astype(np.uint8)
    img = Image.fromarray(adversarial_show).save(
        img_dir + '/test_out_densenet121.png')
    with open(file_name, 'a') as csvfile:
        writer = csv.writer(
            csvfile,
            delimiter=',',
            quotechar='"',
            quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["densenet121", "{0:.4e}".format(distance.value)])

    # densenet201
    image = load_image(
        shape=(
            224,
            224),
        data_format='channels_first',
        fname='test_car.png')
    image = image / 255.
    label = 656
    distance, adversarial = test_untargeted_densenet201(image, label)
    adversarial_show = np.transpose(adversarial, (1, 2, 0))
    adversarial_show = (adversarial_show * 255).astype(np.uint8)
    img = Image.fromarray(adversarial_show).save(
        img_dir + '/test_out_densenet201.png')
    with open(file_name, 'a') as csvfile:
        writer = csv.writer(
            csvfile,
            delimiter=',',
            quotechar='"',
            quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["densenet201", "{0:.4e}".format(distance.value)])

    # inception_v3
    image = load_image(
        shape=(
            299,
            299),
        data_format='channels_first',
        fname='test_car.png')
    image = image / 255.
    label = 656
    distance, adversarial = test_untargeted_inception_v3(image, label)
    adversarial_show = np.transpose(adversarial, (1, 2, 0))
    adversarial_show = (adversarial_show * 255).astype(np.uint8)
    img = Image.fromarray(adversarial_show).save(
        img_dir + '/test_out_inception_v3.png')
    with open(file_name, 'a') as csvfile:
        writer = csv.writer(
            csvfile,
            delimiter=',',
            quotechar='"',
            quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["inception_v3", "{0:.4e}".format(distance.value)])

    # Xception
    image = load_image(
        shape=(
            299,
            299),
        data_format='channels_last',
        fname='test_car.png')
    image = image / 255.
    label = 656
    distance, adversarial = test_untargeted_Xception(image, label)
    adversarial_show = adversarial
    adversarial_show = (adversarial_show * 255).astype(np.uint8)
    img = Image.fromarray(adversarial_show).save(
        img_dir + '/test_out_Xception.png')
    with open(file_name, 'a') as csvfile:
        writer = csv.writer(
            csvfile,
            delimiter=',',
            quotechar='"',
            quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["Xception", "{0:.4e}".format(distance.value)])

    # mobilenet_v2
    image = load_image(
        shape=(
            224,
            224),
        data_format='channels_last',
        fname='test_car.png')
    image = image / 255.
    label = 436
    distance, adversarial = test_untargeted_mobilenet_v2(image, label)
    adversarial_show = adversarial
    adversarial_show = (adversarial_show * 255).astype(np.uint8)
    img = Image.fromarray(adversarial_show).save(
        img_dir + '/test_out_mobilenet_v2.png')
    with open(file_name, 'a') as csvfile:
        writer = csv.writer(
            csvfile,
            delimiter=',',
            quotechar='"',
            quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["mobilenet_v2", "{0:.4e}".format(distance.value)])

    # NASNetMobile
    image = load_image(
        shape=(
            224,
            224),
        data_format='channels_last',
        fname='test_car.png')
    image = image / 255.
    label = 436
    distance, adversarial = test_untargeted_NASNetMobile(image, label)
    adversarial_show = adversarial
    adversarial_show = (adversarial_show * 255).astype(np.uint8)
    img = Image.fromarray(adversarial_show).save(
        img_dir + '/test_out_NASNetMobile.png')
    with open(file_name, 'a') as csvfile:
        writer = csv.writer(
            csvfile,
            delimiter=',',
            quotechar='"',
            quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["NASNetMobile", "{0:.4e}".format(distance.value)])

    # NASNetLarge
    image = load_image(
        shape=(
            331,
            331),
        data_format='channels_last',
        fname='test_car.png')
    image = image / 255.
    label = 656
    distance, adversarial = test_untargeted_NASNetLarge(image, label)
    adversarial_show = adversarial
    adversarial_show = (adversarial_show * 255).astype(np.uint8)
    img = Image.fromarray(adversarial_show).save(
        img_dir + '/test_out_NASNetLarge.png')
    with open(file_name, 'a') as csvfile:
        writer = csv.writer(
            csvfile,
            delimiter=',',
            quotechar='"',
            quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["NASNetLarge", "{0:.4e}".format(distance.value)])
