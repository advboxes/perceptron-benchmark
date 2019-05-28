Summary
=======

Supported attacks
-----------------

.. automodule:: perceptron.benchmarks

.. autosummary::
   :nosignatures:

   CarliniWagnerL2Metric
   CarliniWagnerLinfMetric
   AdditiveNoiseMetric
   AdditiveGaussianNoiseMetric
   AdditiveUniformNoiseMetric
   BlendedUniformNoiseMetric
   GaussianBlurMetric
   BrightnessMetric
   ContrastReductionMetric
   MotionBlurMetric
   RotationMetric
   SaltAndPepperNoiseMetric
   SpatialMetric

Supported models
----------------

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
   KerasResNet50RetinaNetModel

Supported adversarial criterions
--------------------------------

.. automodule:: perceptron.utils.criteria

.. autosummary::
   :nosignatures:

   Misclassification
   ConfidentMisclassification
   TopKMisclassification
   TargetClass
   OriginalClassProbability
   TargetClassProbability
   MisclassificationAntiPorn
   MisclassificationSafeSearch
   TargetClassMiss
   TargetClassMissGoogle
   WeightedAP

Supported distance metrics
--------------------------

.. automodule:: perceptron.utils.distances

.. autosummary::
   :nosignatures:

   MeanSquaredDistance
   MeanAbsoluteDistance
   Linfinity
   L0
   MSE
   MAE
   Linf
