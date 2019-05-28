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

"""Bit depth reduction as feature squeezer."""

import numpy as np


class BitDepth():
    """Bit-depth reduction as feature squeezer as described in [1]_.

    References
    ----------
    .. [1] Weilin et: "Feature Squeezing: Detecting Adversarial
           Examples in Deep Neural Networks.

    """

    def __init__(self, bits, bound=(0.0, 1.0)):
        self._bound = bound
        self.bits = bits

    def __call__(self, img_batch_np):
        """Squeeze image by bits.

        Parameters
        ----------
        img_batch_np : array
            input image batch or image.
        """
        precisions = 2 ** self.bits
        img_batch_norm = img_batch_np / (self._bound[1] -
                                         self._bound[0])
        npp_int = precisions - 1
        img_batch_int = np.rint(img_batch_norm * npp_int)
        img_batch_out = img_batch_int / npp_int
        return img_batch_out


class BitDepthRandom():
    """Bit-depth reduction with random bits."""

    def __init__(self, bound=(0.0, 1.0)):
        self._bound = bound

    def __call__(self, img_batch_np, bits, stddev=0.125):
        """Squeeze image by bits with random target depth.

        Parameters
        ----------
        img_batch_np : array
            Input image batch or image.
        bits : int
            Bit depth for squeezing.
        stddev : float
            Standard deviation for gaussian nosie.
        """
        if stddev == 0.:
            rand_array = np.zeros(img_batch_np.shape)
        else:
            rand_array = np.random.normal(loc=0.,
                                          scale=stddev,
                                          size=img_batch_np.shape)
        x_random = np.add(img_batch_np, rand_array)
        _bit_depth = BitDepth(x_random, bound=self._bound)
        return _bit_depth