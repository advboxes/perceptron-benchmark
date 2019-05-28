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

"""Different filters as feature squeezer."""

import numpy as np
from scipy import ndimage
import cv2


class BinaryFilter():
    """Binary filter as feature squeezer as described in [1]_.

    References
    ----------
    .. [1] Weilin et: "Feature Squeezing: Detecting Adversarial
           Examples in Deep Neural Networks.

    """

    def __init__(self):
        pass

    def __call__(self, img_batch_np, threshold):
        """Squeeze image by binary filter

        Parameters
        ----------
        img_batch_np : array
            Input image batch or image
        threshold : float
            Threshold for binarlize
        """
        x_bin = np.maximum(np.sign(img_batch_np - threshold), 0)
        return x_bin


class BinaryRandomFilter():
    """Binary filter with randomness."""

    def __init__(self):
        pass

    def __call__(self, img_batch_np, threshold, stddev=0.125):
        """Squeeze noise added image by binary filter.

        Parameters
        ----------
        img_batch_np : array
            Input image batch or image
        threshold : float
            Threshold for binarlize
        stddev : float
            Standard deviation for gaussian nosie
        """
        if stddev == 0.:
            rand_array = np.zeros(img_batch_np.shape)
        else:
            rand_array = np.random.normal(loc=0.,
                                          scale=stddev,
                                          size=img_batch_np.shape)
        x_bin = np.maximum(np.sign(np.add(img_batch_np,
                                          rand_array) - threshold), 0)
        return x_bin


class MedianFilter():
    """Median filter as feature squeezer as described in [1]_.

    References
    ----------
    .. [1] Weilin et: "Feature Squeezing: Detecting Adversarial
           Examples in Deep Neural Networks.

    """

    def __init__(self):
        pass

    def __call__(self, img_batch_np, width, height=-1):
        """Squeeze image by meadia filter

        Parameters
        ----------
        img_batch_np : array
            Input image batch or image
        width : int
            The width of the sliding window (number of pixels)
        height : int
            The height of the window. The same as width by default.
        """

        if height == -1:
            height = width
        x_mid = ndimage.filters.median_filter(img_batch_np,
                                              size=(1, width, height, 1),
                                              mode='reflect'
                                              )
        return x_mid
