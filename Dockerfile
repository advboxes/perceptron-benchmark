FROM nvcr.io/nvidia/tensorflow:19.08-py3

RUN apt-get update && apt-get install -y sudo libsm6 libxext6 libxrender-dev
RUN sudo pip3 install https://download.pytorch.org/whl/cu102/torch-1.5.1-cp36-cp36m-linux_x86_64.whl
RUN sudo pip3 install torchvision
RUN sudo pip3 install keras==2.2.5
RUN sudo pip3 install tqdm
RUN sudo pip3 install opencv-python


RUN mkdir /perceptron
WORKDIR /perceptron
ADD . /perceptron
RUN sudo pip3 install -e .
