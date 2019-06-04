# Copyright 2019 Baidu Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Keras model wrapper for YOLOv3 object detection."""

from __future__ import absolute_import
import numpy as np
import logging
import os
from perceptron.models.base import DifferentiableModel
from perceptron.utils.criteria.detection import TargetClassMiss, RegionalTargetClassMiss
from keras import backend as K
import keras


class KerasYOLOv3Model(DifferentiableModel):
    """Create a :class:`Model` instance from a `Keras` model.

    Parameters
    ----------
    model : `keras.model.Model`
        The `Keras` model that are loaded.
    bounds : tuple
        Tuple of lower and upper bound for the pixel values, usually
        (0, 1) or (0, 255).
    model_image_shape : tuple
        Tuple of the model input shape in format (height, width).
    channel_axis : int
        The index of the axis that represents color channels.
    num_scales : int
        Number of scales, if the model detects object at
        different distances.
    num_anchors : int
        Number of anchor boxes for each scale.
    num_classes : int
        Number of classes for which the model will output predictions.
    max_boxes : int
        The maximum number of boxes allowed in the prediction output.
    anchors_path : str
        The path to the file containing anchor box coordinates.
    classes_path : str
        The path to the file containing class names.
    score : float
        The score threshold for considering a box as containing objects.
    iou : float
        The intersection over union (IoU) threshold.
    preprocessing: 2-element tuple with floats or numpy arrays
        Elementwises preprocessing of input; we first substract the first
        element of preprocessing from the input and then divide the input
        by the second element.

    """

    def __init__(
            self,
            model,
            bounds,
            model_image_shape=(416, 416),
            channel_axis=3,
            num_scales=3,
            num_anchors=3,
            num_classes=80,
            max_boxes=20,
            anchors_path='yolov3_anchors.txt',
            classes_path='coco_classes.txt',
            score=0.3,
            iou=0.45,
            preprocessing=(0, 1)):

        super(KerasYOLOv3Model, self).__init__(bounds=bounds,
                                               channel_axis=channel_axis,
                                               preprocessing=preprocessing)

        #  Check if model data files exist.
        model_data_path = os.path.join(
            os.path.dirname(__file__),
            '../../zoo/yolov3/model_data/')

        model_input = model.input
        model_output = model.output
        #  model_output should be list of ndarrays.

        assert len(model_output) == num_scales, \
            "number of scales doesn't match model output"

        logits_per_grid = model_output[0].shape[-1]

        assert (num_classes + 5) * num_anchors == logits_per_grid, \
            "number of logits per grid cell doesn't match model output"

        self._task = 'det'
        self._model_image_shape = model_image_shape
        self._num_classes = num_classes
        self._num_scales = num_scales
        self._num_anchors = num_anchors
        self._classes_path = os.path.join(model_data_path, classes_path)
        self._class_names = self.get_class()
        self._anchors_path = os.path.join(model_data_path, anchors_path)
        self._anchors = self._get_anchors()
        self._score = score
        self._iou = iou
        self._max_boxes = max_boxes

        _boxes, _box_scores, _box_confidence_logits, \
            _box_class_probs_logits, _box_coord_logits = self._gather_feats(
                model_output, self._anchors,
                self._num_classes, self._model_image_shape)

        boxes, scores, classes = self._eval_pred(
            _boxes, _box_scores, self._num_classes,
            self._max_boxes, self._score, self._iou)

        # For attack use only.
        target_class = K.placeholder(dtype='int32')
        tgt_cls_loss = self._target_class_loss(
            target_class, _box_scores, _box_class_probs_logits)

        tgt_cls_gradient = K.gradients(tgt_cls_loss, model_input)
        tgt_cls_gradient = tgt_cls_gradient[0]
        tgt_cls_gradient = K.squeeze(tgt_cls_gradient, axis=0)

        self._batch_gather_feats_fn = K.function(
            [model_input],
            [_boxes, _box_scores, _box_confidence_logits,
             _box_class_probs_logits, _box_coord_logits])
        self._batch_pred_fn = K.function(
            [_boxes, _box_scores],
            [boxes, scores, classes]
        )
        self._tgt_cls_bw_grad_fn = K.function(
            [target_class, model_input],
            [tgt_cls_loss, tgt_cls_gradient]
        )
        self._tgt_cls_pred_and_grad_fn = K.function(
            [model_input, target_class],
            [boxes, scores, classes, tgt_cls_loss, tgt_cls_gradient]
        )

    def get_class(self):
        classes_path = os.path.expanduser(self._classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self._anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def num_classes(self):
        """Return the number of classes."""
        return self._num_classes

    def class_names(self):
        """Return the class names as list."""
        return self._class_names

    def model_task(self):
        """Return the task of the model: classification of detection."""
        return self._task

    def batch_predictions(self, images):
        """Batch prediction of images.

        Parameters
        ----------
        images : `numpy.ndarray`
            The input image in [b, h, n, c] ndarry format.

        Returns
        -------
        list
            List of batch prediction resutls.
            Each element is a dictionary containing:
            {'boxes', 'scores', 'classes}
        """
        
        px, _ = self._process_input(images)

        _boxes, _box_scores, _box_confidence_logits, \
            _box_class_probs_logits, _box_coord_logits = \
            self._batch_gather_feats_fn([px])

        boxes, scores, classes = self._batch_pred_fn(
            [_boxes, _box_scores])

        predictions = []
        for i in range(len(boxes)):
            num = (scores[i] > 0.).sum()
            pred = {}
            pred['boxes'] = boxes[i][:num].tolist()
            pred['scores'] = scores[i][:num].tolist()
            pred['classes'] = classes[i][:num].tolist()
            predictions.append(pred)

        assert len(predictions) == images.shape[0], "batch size doesn't match."

        return predictions

    def predictions_and_gradient(self, image, criterion):
        """ Returns both predictions and gradients, and
        potentially loss w.r.t. to certain criterion.
        """

        input_shape = image.shape
        px, dpdx = self._process_input(image)

        if isinstance(criterion, TargetClassMiss) or \
                isinstance(criterion, RegionalTargetClassMiss):
            boxes, scores, classes, loss, gradient =\
                self._tgt_cls_pred_and_grad_fn(
                    [px[np.newaxis], criterion.target_class()])
        else:
            raise NotImplementedError

        prediction = {}
        num = (scores[0] > 0.).sum()
        prediction['boxes'] = boxes[0][:num].tolist()
        prediction['scores'] = scores[0][:num].tolist()
        prediction['classes'] = classes[0][:num].tolist()

        gradient = self._process_gradient(dpdx, gradient)
        assert gradient.shape == input_shape
        return prediction, loss, gradient,

    def backward(self, target_class, image):
        """Get gradient with respect to the image."""
        px, dpdx = self._process_input(image)
        loss, gradient = self._tgt_cls_bw_grad_fn([
            target_class,
            px[np.newaxis],
        ])
        gradient = self._process_gradient(dpdx, gradient)
        return loss, gradient

    def _gather_feats(
            self,
            yolo_outputs,
            anchors,
            num_classes,
            image_shape):
        """Evaluate model output to get _boxes and _boxes_scores logits.

        """
        num_layers = len(yolo_outputs)
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] \
            if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]  # default setting
        input_shape = K.shape(yolo_outputs[0])[1:3] * 32
        batch_size = K.shape(yolo_outputs[0])[0]

        boxes = []
        box_scores = []
        box_coord_logits = []
        box_confidence_logits = []
        box_class_probs_logits = []

        for l in range(num_layers):
            _boxes, _box_scores, _box_coord_logits, \
                _box_confidence_logits, _box_class_probs_logits =\
                self._boxes_and_scores(
                    yolo_outputs[l], anchors[anchor_mask[l]],
                    num_classes, input_shape, image_shape, batch_size,
                    verbose=True)

            boxes.append(_boxes)
            box_scores.append(_box_scores)
            box_coord_logits.append(_box_coord_logits)
            box_confidence_logits.append(_box_confidence_logits)
            box_class_probs_logits.append(_box_class_probs_logits)

        boxes = K.concatenate(boxes, axis=1)  # [batch_size, num_boxes, 4]
        box_scores = K.concatenate(box_scores, axis=1)
        box_coord_logits = K.concatenate(box_coord_logits, axis=1)
        box_confidence_logits = K.concatenate(box_confidence_logits, axis=1)
        box_class_probs_logits = K.concatenate(box_class_probs_logits, axis=1)

        return boxes, box_scores, box_confidence_logits, \
            box_class_probs_logits, box_coord_logits

    def _target_class_loss(
            self,
            target_class,
            box_scores,
            box_class_probs_logits):
        """ Evaluate target_class_loss w.r.t. the input.

        """
        box_scores = K.squeeze(box_scores, axis=0)
        box_class_probs_logits = K.squeeze(box_class_probs_logits, axis=0)
        import tensorflow as tf
        boi_idx = tf.where(box_scores[:, target_class] > self._score)
        loss_box_class_conf = tf.reduce_mean(
            tf.gather(box_class_probs_logits[:, target_class], boi_idx))

        # Avoid the propagation of nan
        return tf.cond(
            tf.is_nan(loss_box_class_conf),
            lambda: tf.constant(0.),
            lambda: loss_box_class_conf)

    def _eval_pred(
            self,
            boxes,
            box_scores,
            num_classes,
            max_boxes=20,
            score_threshold=.6,
            iou_threshold=.5):
        """ Evaluate logits for boxes and scores to final boxes, class, scores
            results

        """
        import tensorflow as tf

        def process_batch(params):
            boxes, box_scores = params
            mask = box_scores >= score_threshold
            max_boxes_tensor = K.constant(max_boxes, dtype='int32')
            boxes_ = []
            scores_ = []
            classes_ = []
            for c in range(num_classes):
                # TODO: use keras backend instead of tf.
                class_boxes = tf.boolean_mask(boxes, mask[:, c])
                class_box_scores = tf.boolean_mask(
                    box_scores[:, c], mask[:, c])
                nms_index = tf.image.non_max_suppression(
                    class_boxes, class_box_scores,
                    max_boxes_tensor, iou_threshold=iou_threshold)
                class_boxes = K.gather(class_boxes, nms_index)
                class_box_scores = K.gather(class_box_scores, nms_index)
                classes = K.ones_like(class_box_scores, 'int32') * c
                boxes_.append(class_boxes)
                scores_.append(class_box_scores)
                classes_.append(classes)
            boxes_ = K.concatenate(boxes_, axis=0)
            scores_ = K.concatenate(scores_, axis=0)
            classes_ = K.concatenate(classes_, axis=0)

            pad_len = max_boxes - tf.shape(boxes_)[0]
            pad_boxes = tf.zeros([pad_len, 4], dtype=tf.float32)
            pad_scores = tf.zeros(pad_len, dtype=tf.float32)
            pad_classes = tf.zeros(pad_len, dtype=tf.int32)
            boxes_ = tf.concat([boxes_, pad_boxes], axis=0)
            scores_ = tf.concat([scores_, pad_scores], axis=0)
            classes_ = tf.concat([classes_, pad_classes], axis=0)

            return boxes_, scores_, classes_

        boxes_, scores_, classes_ = tf.map_fn(
            process_batch,
            (boxes, box_scores),
            dtype=(tf.float32, tf.float32, tf.int32))

        return boxes_, scores_, classes_

    def _boxes_and_scores(
            self, feats, anchors, num_classes,
            input_shape, image_shape, batch_size, verbose=False):
        """ Convert Conv layer output to boxes.
        Multiply box_confidence with class_confidence to get real
        box_scores for each class.

        Parameters
        ----------
        feats : `Tensor`
            Elements in the output list from `K.model.output`,
            shape = (N, 13, 13, 255).
        anchors : list
            anchors.
        num_classes : int
            num of classes.
        input_shape: tuple
            input shape obtained from model output grid information.
        image_shape: tuple
            placeholder for ORIGINAL image data shape.
        """

        if verbose is True:
            box_xy, box_wh, box_confidence, box_class_probs, \
                box_coord_logits, box_confidence_logits, \
                box_class_probs_logits = self._model_head(
                    feats, anchors, num_classes, input_shape,
                    batch_size, verbose=verbose)
        else:
            box_xy, box_wh, box_confidence, box_class_probs = self._model_head(
                feats, anchors, num_classes, input_shape, batch_size)

        boxes = self._correct_boxes(
            box_xy, box_wh, input_shape, image_shape)
        boxes = K.reshape(boxes, [batch_size, -1, 4])
        box_scores = box_confidence * box_class_probs
        box_scores = K.reshape(box_scores, [batch_size, -1, num_classes])

        if verbose is True:
            box_coord_logits = K.reshape(
                box_coord_logits, [batch_size, -1, 4])
            box_confidence_logits = K.reshape(
                box_confidence_logits, [batch_size, -1])
            box_class_probs_logits = K.reshape(
                box_class_probs_logits, [batch_size, -1, num_classes])
            return boxes, box_scores, box_coord_logits,\
                box_confidence_logits, box_class_probs_logits

        return boxes, box_scores

    def _model_head(
            self, feats, anchors, num_classes,
            input_shape, batch_size, calc_loss=False, verbose=False):
        """Convert final layer features to bounding box parameters.
        No threshold or nms applied yet.

        Args:
            feats : `Tensor`
                Elements in the output list from K.model.output:
                shape = (N, 13, 13, 255)
            anchors : list
                anchors.
            num_classes : int
                num of classes.
            input_shape : tuple
                input shape obtained from model output grid information.

        Returns:
            Breaking the (num_class + 5) output logits into box_xy,
            box_wh, box_confidence, and box_class_probs.
        """

        num_anchors = len(anchors)
        # Reshape to batch, height, width, num_anchors, box_params.
        anchors_tensor = K.reshape(
            K.constant(anchors), [1, 1, 1, num_anchors, 2])

        grid_shape = K.shape(feats)[1:3]  # height, width
        grid_y = K.tile(
            K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
            [1, grid_shape[1], 1, 1])
        grid_x = K.tile(
            K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
            [grid_shape[0], 1, 1, 1])
        grid = K.concatenate([grid_x, grid_y])
        grid = K.cast(grid, K.dtype(feats))

        feats = K.reshape(
            feats, [-1, batch_size, grid_shape[0],
                    grid_shape[1], num_anchors, num_classes + 5])

        # Adjust preditions to each spatial grid point and anchor size.
        box_xy = (K.sigmoid(feats[..., :2]) + grid) /\
            K.cast(grid_shape[::-1], K.dtype(feats))
        box_wh = K.exp(feats[..., 2:4]) * anchors_tensor /\
            K.cast(input_shape[::-1], K.dtype(feats))
        box_confidence = K.sigmoid(feats[..., 4:5])
        box_class_probs = K.sigmoid(feats[..., 5:])

        if calc_loss is True:
            return grid, feats, box_xy, box_wh

        if verbose is True:
            # In verbose mode, return logits BEFORE sigmoid activation
            box_coord_logits = feats[..., :4]
            box_confidence_logits = feats[..., 4: 5]
            box_class_probs_logits = feats[..., 5:]
            return box_xy, box_wh, box_confidence, box_class_probs, \
                box_coord_logits, box_confidence_logits, \
                box_class_probs_logits

        return box_xy, box_wh, box_confidence, box_class_probs

    def _correct_boxes(
            self, box_xy, box_wh, input_shape, image_shape):
        """Get corrected boxes, which are scaled to original shape."""
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        input_shape = K.cast(input_shape, K.dtype(box_yx))
        image_shape = K.cast(image_shape, K.dtype(box_yx))
        new_shape = K.round(image_shape * K.min(input_shape / image_shape))
        offset = (input_shape - new_shape) / 2. / input_shape
        scale = input_shape / new_shape
        box_yx = (box_yx - offset) * scale
        box_hw *= scale

        box_mins = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)
        boxes = K.concatenate([
            box_mins[..., 0:1],  # y_min
            box_mins[..., 1:2],  # x_min
            box_maxes[..., 0:1],  # y_max
            box_maxes[..., 1:2]  # x_max
        ])

        # Scale boxes back to original image shape.
        boxes *= K.concatenate([image_shape, image_shape])
        return boxes
