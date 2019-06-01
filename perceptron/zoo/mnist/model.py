import numpy as np
import torch 
import torch.nn as nn



class Flatten(nn.Module):
    '''Return flatten layers.
    Please use this Flatten layer to flat the convolutional layers when 
    design your own models.
    '''
    def forward(self, x):
        return x.view(x.size(0), -1)


def mnist_model(model_size="small", method="mixtrain", pretrained=True, train_epsilon=0.1):
    """Returns a mnist model.

    Parameters
    ----------
    model : str
        The predefined structure of the models
    method : str
        Methods that is used for training the models
    pretrained : bool
        Whether need to load the pretrained model
    train_epsilon : float
        The Linf epsilon used for training
    
    A list of trained models
    ---------
    mixtrain:
    mnist_small_mixtrain_0.1 : acc 99, vra 96
    mnist_small_mixtrain_0.3 : acc 94, vra 60
    mnist_small_mixtrain_0.1 : acc 99, vra 95
    mnist_small_mixtrain_0.1 : acc 97, vra 59
    clean:
    mnist_small_clean : acc 99, vra 0
    mnist_large_clean : acc 99, vra 0

    Returns
    -------
    model : torch nn.Sequential
        The loaded model

    """

    if model_size == "small":
        net = mnist_small()
    if model_size == "large":
        net = mnist_large()

    if method != "clean":
        MODEL_NAME = "mnist_" +\
                    model_size + "_" + method + "_" + str(train_epsilon)
    else:
        MODEL_NAME = "mnist_" +\
                    model_size + "_" + method

    if pretrained:
        from perceptron.utils.func import maybe_download_model_data
        weight_file = MODEL_NAME+".pth"
        weight_fpath = maybe_download_model_data(
                    weight_file,
                    "https://github.com/baidu-advbox/perceptron-benchmark/releases/download/v1.0/" + MODEL_NAME+".pth")
        print("====",weight_fpath)
        net.load_state_dict(torch.load(weight_fpath))
        print("Load model from", MODEL_NAME + ".pth")
        return net


def mnist_small(): 
    model = nn.Sequential(
        nn.Conv2d(1, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(32*7*7,100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model


def mnist_large(): 
    model = nn.Sequential(
        nn.Conv2d(1, 32, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(64*7*7,512),
        nn.ReLU(),
        nn.Linear(512,512),
        nn.ReLU(),
        nn.Linear(512,10)
    )
    return model


