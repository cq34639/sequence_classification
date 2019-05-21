FROM tensorflow/tensorflow:2.0.0a0-gpu-py3

RUN apt-get update

WORKDIR /
ADD ./docker/requirements.txt /dependencies/requirements.txt
ADD ./data /data
ADD ./src /src
ADD ./default_parameters.ini /default_parameters.ini

RUN pip install pip-tools
RUN pip install -r /dependencies/requirements.txt
