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

"""Launcher tools."""

import torch
from .symbolic_interval.symbolic_network import sym_interval_analyze
from .symbolic_interval.symbolic_network import naive_interval_analyze


use_cuda = torch.cuda.is_available()

def symbolic(net, epsilon, X, y, parallel=False, proj=None):
    """ Return the results analyzed by symbolic interval
    analyze including (1) cross-entropy of tested images on the
    the targeted network; (2) whether the image can be verified

    Parameters
    ----------
    net : pytorch nn.Sequential model
        Targeted testing model
    epsilon : float
        The testing Linf epsilon
    X : numpy image
    y : the label we assume to be correct
    parallel : bool
        Whether we need to parallelize the testing
    proj : int 
        project input dimension for controlling the
        tightness of the estimation
    """
    X = torch.FloatTensor(X)
    y = torch.LongTensor([y])

    if use_cuda:
        X = X.cuda()
        y = y.cuda().long()

    if len(X.shape) == 3:
        X = X.unsqueeze(0)

    assert len(X.shape) == 4 and len(y.shape) == 1,\
                "have to expand to a batch"
    
    # print("parallel", parallel)
    
    iloss, ierr = sym_interval_analyze(net, epsilon, X, y,\
                        use_cuda, parallel, proj)

    return iloss, ierr


def naive(net, epsilon, X, y, parallel=False, proj=None):
    """ Return the results analyzed by naive interval
    analyze including (1) cross-entropy of tested images on the
    the targeted network; (2) whether the image can be verified

    Parameters
    ----------
    net : pytorch nn.Sequential model
        Targeted testing model
    epsilon : float
        The testing Linf epsilon
    X : numpy image
    y : the label we assume to be correct
    parallel : bool
        Whether we need to parallelize the testing
    proj : int 
        project input dimension for controlling the
        tightness of the estimation
    """

    X = torch.FloatTensor(X)
    y = torch.LongTensor([y])

    if use_cuda:
        X = X.cuda()
        y = y.cuda().long()

    if len(X.shape) == 3:
        X = X.unsqueeze(0)

    assert len(X.shape) == 4 and len(y.shape) == 1,\
                "have to expand to a batch"

    # print("parallel", parallel)
    iloss, ierr = naive_interval_analyze(net, epsilon, X, y,\
                        use_cuda, parallel)

    return iloss, ierr


