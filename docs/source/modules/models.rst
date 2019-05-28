:mod:`perceptron.models`
========================

.. automodule:: perceptron.models

.. autosummary::
   :nosignatures:

   Model
   DifferentiableModel


.. automodule:: perceptron.models.classification

.. autosummary::
   :nosignatures:

   KerasModel
   PyTorchModel
   AipModel
   AipAntiPornModel
   GoogleCloudModel
   GoogleSafeSearchModel
   GoogleObjectDetectionModel


.. automodule:: perceptron.models.detection

.. autosummary::
   :nosignatures:

   KerasYOLOv3Model
   KerasSSD300Model


.. automodule:: perceptron.models

.. autoclass:: Model
   :members:

.. autoclass:: DifferentiableModel
   :members:
.. toctree::

    models/classification
    models/detection
    models/cloud
