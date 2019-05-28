#!/bin/bash
set -x #echo on

#keywords = ['Keras', 'Xception', 'Top10Misclassification', 'AdditiveGaussian']
python perceptron/launcher.py \
    --framework keras \
    --model xception \
    --criteria topk_misclassification\
    --metric additive_gaussian_noise \
    --image example.png


#keywords = ['Keras', 'VGG16', 'Top10Misclassification', 'BlendedUniform']
python perceptron/launcher.py \
    --framework keras \
    --model vgg16 \
    --criteria topk_misclassification\
    --metric blend_uniform_noise \
    --image example.png


#keywords = ['Keras', 'ResNet50', 'Misclassification', 'CarliniWagnerL2']
python perceptron/launcher.py \
    --framework keras \
    --model resnet50 \
    --criteria misclassification\
    --metric carlini_wagner_l2 \
    --image example.png


#keywords = ['PyTorch', 'Resent18', 'Misclassification', 'CarliniWagnerL2']
python perceptron/launcher.py \
    --framework pytorch \
    --model resnet18 \
    --criteria misclassification\
    --metric carlini_wagner_l2 \
    --image example.png


#keywords = ['PyTorch', 'Densenet201', 'TargetClass', 'GaussianBlur']
python perceptron/launcher.py \
    --framework pytorch \
    --model densenet201 \
    --criteria target_class --target_class 300\
    --metric gaussian_blur \
    --image example.png



#keywords = ['PyTorch', 'Vgg11', 'Misclassification', 'SaltAndPepper']
python perceptron/launcher.py \
    --framework pytorch \
    --model vgg11 \
    --criteria misclassification \
    --metric salt_and_pepper_noise \
    --image example.png

#keywords = ['Cloud', 'AipAntiPorn', 'Misclassification', 'Rotation']
python perceptron/launcher.py \
    --framework cloud \
    --model aip_antiporn \
    --criteria misclassification_antiporn \
    --metric rotation \
    --image porn.jpeg


#keywords = ['Cloud', 'GoogleSafeSearch', 'Misclassification', 'ContrastReduction']
python perceptron/launcher.py \
    --framework cloud \
    --model google_safesearch \
    --criteria misclassification_safesearch \
    --metric contrast_reduction \
    --image porn.jpeg


