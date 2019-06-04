"""retina_resnet50"""
import keras
from perceptron.zoo.retinanet_resnet_50 import models
import tensorflow as tf


def retina_resnet50(weights_file="resnet50_coco_best_v2.1.0.h5"):
    from perceptron.utils.func import maybe_download_model_data
    keras.backend.tensorflow_backend.set_session(get_session())
    weight_fpath = maybe_download_model_data(weights_file,
        'https://perceptron-benchmark.s3-us-west-1.amazonaws.com/models/coco/resnet50_coco_best_v2.1.0.h5')
    model = models.load_model(weight_fpath, backbone_name='resnet50')
    return model


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)
