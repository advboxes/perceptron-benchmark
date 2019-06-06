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

from __future__ import absolute_import

import numpy as np
from perceptron.models.base import DifferentiableModel


class GaussianModel(DifferentiableModel):
    """Creates a :class:`Model` instance from a `Gaussian` module.

    Parameters
    ----------
    model : `torch.nn.Module`
        The PyTorch model that are loaded.
    bounds : tuple
        Tuple of lower and upper bound for the pixel values, usually
        (0, 1) or (0, 255).
    num_classes : int
        Number of classes for which the model will output predictions.
    channel_axis : int
        The index of the axis that represents color channels.
    device : string
        A string specifying the device to do computation on.
        If None, will default to "cuda:0" if torch.cuda.is_available()
        or "cpu" if not.
    preprocessing: 2-element tuple with floats or numpy arrays
        Elementwises preprocessing of input; we first subtract the first
        element of preprocessing from the input and then divide the input by
        the second element.
    """

    def __init__(
            self,
            model,
            bounds=(0, 1),
            channel_axis=1,
            std=0.3,
            iterations=100,
            preprocessing=(0, 1)):

        super(GaussianModel, self).__init__(bounds=bounds,
                                            channel_axis=channel_axis,
                                            preprocessing=preprocessing)
        self._model = model
        self._num_classes = self._model.num_classes()
        self._iterations = iterations
        self._std = std

    def batch_predictions(self, images):
        """Batch prediction of images."""
        # lazy import
        import torch

        images, _ = self._process_input(images)
        n = len(images)
        labels = np.empty(shape=(n,), dtype=np.int32)
        bounds = np.empty(shape=(n,), dtype=np.float32)
        for i in range(n):
            labels[i], bounds[i] = self.predictions(images[i])

        return labels, bounds

    def predictions(self, image, forward_batch_size=32):
        from scipy.stats import norm
        image, _ = self._process_input(image)
        image_batch = np.vstack([[image]] * self._iterations)
        noise = np.random.normal(scale=self._std, size=image_batch.shape).astype(np.float32)
        image_batch += noise
        predictions = self._model.batch_predictions(image_batch)
        logits = np.argmax(predictions, axis=1)
        one_hot = np.zeros([self._iterations, self._num_classes])
        logits_one_hot = np.eye(self._num_classes)[logits]
        one_hot += logits_one_hot
        one_hot = np.sum(one_hot, axis=0)
        ranks = sorted(one_hot / np.sum(one_hot))[::-1]
        qi = ranks[0] - 1e-9
        qj = ranks[1] + 1e-9
        bound = self._std / 2. * (norm.ppf(qi) - norm.ppf(qj))
        return np.argmax(one_hot), bound

    def num_classes(self):
        """Return number of classes."""
        return self._num_classes

    def model_task(self):
        """Get the task that the model is used for."""
        return self._model.model_task()

    def predictions_and_gradient(self, image, label):
        """Returns both predictions and gradients."""
        return self._model.predictions_and_gradient(image, label)

    def _loss_fn(self, image, label):
        return self._loss_fn(image, label)

    def backward(self, gradient, image):
        """Get gradients w.r.t. the original image."""
        # lazy import
        return self.backward(gradient, image)
