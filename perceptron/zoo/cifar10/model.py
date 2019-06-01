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


def cifar_model(model_size="small", method="mixtrain",
                pretrained=True, train_epsilon=0.1):
    """Returns a cifar model.

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
    small:
    cifar_small_clean_2 : acc 78, vra 0
    cifar_small_madry_2 : acc 71, vra 60
    cifar_small_mixtrain_2 : acc 70, vra 48

    large:
    cifar_large_clean : acc 87, vra 0
    cifar_large_mixtrain : acc 74, vra 51

    Returns
    -------
    model : torch nn.Sequential
        The loaded model

    """

    if model_size == "small":
        net = cifar_small()
    if model_size == "large":
        net = cifar_large()

    if method != "clean":
        MODEL_NAME = "cifar_" +\
                    model_size + "_" + method + "_" + str(train_epsilon)
    else:
        MODEL_NAME = "cifar_" +\
                    model_size + "_" + method

    if pretrained:
        from perceptron.utils.func import maybe_download_model_data
        weight_file = MODEL_NAME+".pth"
        weight_fpath = maybe_download_model_data(
                    weight_file,
                    "https://perceptron-benchmark.s3-us-west-1.amazonaws.com/models/cifar/" + MODEL_NAME+".pth")
        # print("====",weight_fpath)
        net.load_state_dict(torch.load(weight_fpath))
        print("Load model from", MODEL_NAME + ".pth")
        return net


def cifar_small(): 
    model = nn.Sequential(
        nn.Conv2d(3, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(32*8*8,100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model



def cifar_large(): 
    model = nn.Sequential(
        nn.Conv2d(3, 32, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(64*8*8,512),
        nn.ReLU(),
        nn.Linear(512,512),
        nn.ReLU(),
        nn.Linear(512,10)
    )
    return model


