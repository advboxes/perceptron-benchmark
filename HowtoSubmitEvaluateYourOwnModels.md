# How to Submit/Evaluate Your Own Models
To ease the trouble evaluating a customized deep learning models, we provide guidelines, and examples to port user's models and make them work with our perceptron robustness benchmark.

## Model Format We Support

 - Keras
 - PyTorch
 
Currently we support models built from two popluar deep learning frameworks: **Keras** and **PyTorch**.  If your models are built from other deep learning frameworks, please try to convert your models to the format we support. At the same time, we are still working on supporting models from more frameworks. 


## ImageNet Classification Model
Before evaluating your classification model, please make sure both the model implementation and weights files are ready.  In addition to these, we request users to create a sub-class of either **KerasModel** or **PyTorchModel** and complete the *load_model()* method so that the model can be successfully loaded for evaluation.  Here we provide 2 examples for users to get some ideas.

### ***KerasModel Example***
```python
    from perceptron.models.classification.keras import KerasModel
    
    class KerasModelUpload(KerasModel):
        def __init__(self,
                 bounds,
                 channel_axis=3,
                 preprocessing=(0,1),
                 predicts='logits'):
            #load model
            model = self.load_model()

            super(KerasModelUpload, self).__init__(model = model,
                                            bounds=bounds,
                                            channel_axis = channel_axis,
                                            preprocessing=preprocessing,
                                            predicts=predicts) 
                                           
        def load_model(self):
            '''
            To be implemented...
            model evaluation participants need to implement this and make sure a keras model can be loaded and fully-functional
            '''
            pass
```

### ***PyTorchModel Example***

```python 

    import torch
    from perceptron.models.classification.pytorch import PyTorchModel


    class PyModelUpload(PyTorchModel):
        def __init__(self,
                   bounds,
                   num_classes,
                   channel_axis=1,
                   device=None,
                   preprocessing=(0,1)):
            #load model
            model = self.load_model()
            if torch.cuda.is_available():
                 model = model.cuda()

            model = model.eval()

            super(PyModelUpload, self).__init__(model = model,
                                              bounds=bounds,
                                              num_classes = num_classes,
                                              channel_axis = channel_axis,
                                              device=device,
                                              preprocessing=preprocessing)

        def load_model(self):
            '''
            To be implemented...
            model evaluation participants need to implement this and make sure a pytorch model can be loaded and fully-functional
            '''
            pass
```

### UseCases

Let us walk through 2 use cases step by step on how to prepare user customized models for robustness evaluation.



- [Use case 1:  Prepare InceptionResNetV2 model in Keras Format. ](usecase_kerasmodel.md)
- [Use case 2: Prepare ResNext101_32x4d model in PyTorch Format.](usecase_pytorchmodel.md)

We are still working on object detection model use cases. Please stay tuned.  

## Model Submission
To submit your own model for robustness evaluation, you need to put all the following information into a tarball.

- model weight file
- model class and its dependencies
- Dockfile that configures the image with the required libraries, and paths of model weight file and model class, config files etc.,

User can find out how to create a tarball by reading through our [Use case 1](usecase_kerasmodel.md). Once we have your docker file, we will run some test cases to make sure your model can work with our perceptron benchmark tool. If you upload an imagenet classification model, you have to achieve a state-of-the-art accuracy, otherwise your model won't be forwarded for robustness evaluation.


Note -- For imagenet classification model, we assume the output dimension is 1000 for 1000 classes for the uploaded models. If the output of user's model has less number of classes, user needs to include the class label file in the tarball, and we will not test the model with images whose class is not in the label file. The class label shall be in the format of [synset](https://github.com/tensorflow/models/blob/master/research/slim/datasets/imagenet_lsvrc_2015_synsets.tx) for example 
```bash
n01694178
n01843065
n02037110 
```

For object detection models, we assume the user submitted models follow the same labels as COCO dataset. Here is the [label file](./perceptron/zoo/yolov3/model_data/coco_classes.txt) for your reference. By default, the line number corresponds to the category id for each object class. For example, object 'bus' corresponds to category id 6. If your model predicted object class is different from COCO dataset label, please include your class label/category id  for 'bus' and save it to a file `bus.label` in your tarball. 

