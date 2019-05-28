"""retina_resnet50"""
import keras
from perceptron.zoo.retinanet_resnet_50 import models
import tensorflow as tf


def retina_resnet50(
        weights_path="/home/yantao/workspace/projects/baidu/aisec/perc" +
        "eptron/perceptron/zoo/retinanet_resnet_50/model_data/resnet50_coco_best_v2" +
        ".1.0.h5"):
    keras.backend.tensorflow_backend.set_session(get_session())
    model = models.load_model(weights_path, backbone_name='resnet50')
    return model


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)
