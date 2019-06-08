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

"""Image classification model wrapper for cloud API models."""

from __future__ import absolute_import
from perceptron.models.base import Model
from perceptron.utils.image import ndarray_to_bytes


class AipModel(Model):
    """Base class for models hosted on Baidu AIP platform.

    Parameters
    ----------
    credential : tuple
        Tuple of (appId, apiKey, secretKey) for using AIP API.
    bounds : tuple
        Tuple of lower and upper bound for the pixel values, usually
        (0, 1) or (0, 255).
    channel_axis : int
        The index of the axis that represents color channels.
    preprocessing: 2-element tuple with floats or numpy arrays
        Elementwises preprocessing of input; we first substract the first
        element of preprocessing from the input and then divide the input
        by the second element.

    """

    def __init__(
            self,
            credential,
            bounds=(0, 255),
            channel_axis=3,
            preprocessing=(0, 1)):

        # lazy import
        super(AipModel, self).__init__(bounds=bounds,
                                       channel_axis=channel_axis,
                                       preprocessing=preprocessing)
        self._appId, self._apiKey, self._secretKey = credential


class AipAntiPornModel(AipModel):
    """Create a :class:`Model` instance from an `AipAntiPorn` model.

    Parameters
    ----------
    credential : tuple
        Tuple of (appId, apiKey, secretKey) for using AIP API.
    bounds : tuple
        Tuple of lower and upper bound for the pixel values, usually
        (0, 1) or (0, 255).
    channel_axis : int
        The index of the axis that represents color channels.
    preprocessing: 2-element tuple with floats or numpy arrays
        Elementwises preprocessing of input; we first substract the first
        element of preprocessing from the input and then divide the input
        by the second element.

    """

    def __init__(
            self,
            credential,
            bounds=(0, 255),
            channel_axis=3,
            preprocessing=(0, 1)):

        from aip import AipImageCensor
        super(AipAntiPornModel, self).__init__(
            credential=credential,
            bounds=bounds,
            channel_axis=channel_axis,
            preprocessing=preprocessing)

        self._task = 'cls'
        self.model = AipImageCensor(
            self._appId,
            self._apiKey,
            self._secretKey)

    def predictions(self, image):
        """Get prediction for input image

        Parameters
        ----------
        image : `numpy.ndarray`
            The input image in [h, n, c] ndarry format.

        Returns
        -------
        list
            List of anitporn prediction resutls.
            Each element is a dictionary containing:
            {'class_name', 'probability'}

        """

        image_bytes = ndarray_to_bytes(image)
        predictions = self.model.antiPorn(image_bytes)
        if 'result' in predictions:
            return predictions['result']
        return predictions

    def model_task(self):
        """Get the task that the model is used for."""
        return self._task


class GoogleCloudModel(Model):
    """Base class for models in Google Cloud.

    Parameters
    ----------
    bounds : tuple
        Tuple of lower and upper bound for the pixel values, usually
        (0, 1) or (0, 255).
    channel_axis : int
        The index of the axis that represents color channels.
    preprocessing: 2-element tuple with floats or numpy arrays
        Elementwises preprocessing of input; we first substract the first
        element of preprocessing from the input and then divide the input
        by the second element.

    Notes
    -----
    To use google cloud vision models, you need to install its package
    `pip instlal --upgrade google-cloud-vision`.

    """

    def __init__(
            self,
            bounds=(0, 255),
            channel_axis=3,
            preprocessing=(0, 1)):

        # lazy import
        import os
        super(GoogleCloudModel, self).__init__(bounds=bounds,
                                               channel_axis=channel_axis,
                                               preprocessing=preprocessing)

        assert os.environ.get('GOOGLE_APPLICATION_CREDENTIALS') is not None,\
            ' Credential GOOGLE_APPLICATION_CREDENTIALS needs to be set.'


class GoogleSafeSearchModel(GoogleCloudModel):
    """Create a :class: `Model` instance from a
    `GoogleSafeSearchModel` model.

    Parameters
    ----------
    bounds : tuple
        Tuple of lower and upper bound for the pixel values, usually
        (0, 1) or (0, 255).
    channel_axis : int
        The index of the axis that represents color channels.
    preprocessing: 2-element tuple with floats or numpy arrays
        Elementwises preprocessing of input; we first substract the first
        element of preprocessing from the input and then divide the input
        by the second element.

    """

    def __init__(
            self,
            bounds=(0, 255),
            channel_axis=3,
            preprocessing=(0, 1)):

        from google.cloud import vision
        super(GoogleSafeSearchModel, self).__init__(
            bounds=bounds,
            channel_axis=channel_axis,
            preprocessing=preprocessing)

        self._task = 'cls'
        self.model = vision.ImageAnnotatorClient()

    def predictions(self, image):
        """Get prediction for input image.

        Parameters
        ----------
        image : `numpy.ndarray`
            The input image in [h, n, c] ndarry format.

        Returns
        -------
        list
            List of prediction resutls.
            Each element is a dictionary containing:
            {'adult', 'medical', 'racy', 'spoof', 'violence'}.

        """
        from google.cloud import vision
        from protobuf_to_dict import protobuf_to_dict
        image_bytes = ndarray_to_bytes(image)
        image = vision.types.Image(content=image_bytes)
        response = self.model.safe_search_detection(image=image)
        predictions = protobuf_to_dict(response)['safe_search_annotation']
        return predictions

    def model_task(self):
        """Get the task that the model is used for."""
        return self._task


class GoogleObjectDetectionModel(GoogleCloudModel):
    """Create a :class: `Model` instance from a
    `GoogleObjectDetectionModel` model.

    Parameters
    ----------
    bounds : tuple
        Tuple of lower and upper bound for the pixel values, usually
        (0, 1) or (0, 255).
    channel_axis : int
        The index of the axis that represents color channels.
    preprocessing: 2-element tuple with floats or numpy arrays
        Elementwises preprocessing of input; we first substract the first
        element of preprocessing from the input and then divide the input
        by the second element.

    """

    def __init__(
            self,
            bounds=(0, 255),
            channel_axis=3,
            preprocessing=(0, 1)):

        from google.cloud import vision
        super(GoogleObjectDetectionModel, self).__init__(
            bounds=bounds,
            channel_axis=channel_axis,
            preprocessing=preprocessing)

        self.model = vision.ImageAnnotatorClient()
        self._task = 'det'

    def predictions(self, image):
        """Get detection result for input image.

        Parameters
        ----------
        image : `numpy.ndarray`
            The input image in [h, n, c] ndarry format.

        Returns
        -------
        list
            List of batch prediction resutls.
            Each element is a dictionary containing:
            {'name', 'score', 'mid', 'bounding_poly'}.

        """
        from google.cloud import vision
        from google.protobuf.json_format import MessageToJson
        from protobuf_to_dict import protobuf_to_dict
        image_bytes = ndarray_to_bytes(image)
        image = vision.types.Image(content=image_bytes)
        response = self.model.object_localization(
            image=image).localized_object_annotations
        predictions = []
        for object in response:
            predictions.append(protobuf_to_dict(object))

        return predictions

    def model_task(self):
        """Get the task that the model is used for."""
        return self._task


class GoogleOCRModel(GoogleCloudModel):
    """Create a :class: `Model` instance from a
    `GoogleOCR` model.

    Parameters
    ----------
    bounds : tuple
        Tuple of lower and upper bound for the pixel values, usually
        (0, 1) or (0, 255).
    channel_axis : int
        The index of the axis that represents color channels.
    preprocessing: 2-element tuple with floats or numpy arrays
        Elementwises preprocessing of input; we first substract the first
        element of preprocessing from the input and then divide the input
        by the second element.

    """

    def __init__(
            self,
            bounds=(0, 255),
            channel_axis=3,
            preprocessing=(0, 1)):

        from google.cloud import vision
        super(GoogleOCRModel, self).__init__(
            bounds=bounds,
            channel_axis=channel_axis,
            preprocessing=preprocessing)

        self.model = vision.ImageAnnotatorClient()

    def predictions(self, image):
        """Get detection result for input image.

        Parameters
        ----------
        image : `numpy.ndarray`
            The input image in [h, n, c] ndarry format.

        Returns
        -------
        list
            List of batch prediction resutls.
            Each element is a dictionary containing:
            {'name', 'score', 'mid', 'bounding_poly'}.

        """
        from google.cloud import vision
        from google.protobuf.json_format import MessageToJson
        from protobuf_to_dict import protobuf_to_dict
        image_bytes = ndarray_to_bytes(image)
        image = vision.types.Image(content=image_bytes)
        response = self.model.object_localization(
            image=image).localized_object_annotations
        predictions = []
        for object in response:
            predictions.append(protobuf_to_dict(object))

        return predictions
