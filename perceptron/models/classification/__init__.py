"""Provides class to wrap existing models in different frameworks
so that they provide a unified API to the benchmarks.
"""

from .keras import KerasModel
from .pytorch import PyTorchModel
from .cloud import AipModel
from .cloud import AipAntiPornModel
from .cloud import GoogleCloudModel
from .cloud import GoogleSafeSearchModel
from .cloud import GoogleObjectDetectionModel
