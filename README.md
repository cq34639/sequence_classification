# Textual Sequence Classification

This repository serves as a quick turnaround example case of sequential text (sentence) classification, trained on the works of two authors: Jane Austin and Charlotte Bronte.

# Prerequisites

Note that this code was built assuming that it would be built on Linux (I specifically used Ubuntu 16.04).

Running this code requires docker (for cpu-based machines) or nvidia-docker (for gpu-enabled machines). These packages can be found here
* [docker](https://docs.docker.com/install/linux/docker-ce/ubuntu/)
* [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

# Execution

All relevant processes (container building and running, plus model training) can be run in a single command by calling 
```
cd <project_checkout_root> (e.g., ./sequence_classification/)
sh build_and_run.sh 
```
This aggregates the following steps, which can also be run manually if desired:
* `docker build -f docker/Dockerfile . -t sequence_classification:latest`
* On GPU: `nvidia-docker run -it --rm sequence_classification:latest /bin/bash`
* OR on CPU: `docker run -it --rm sequence_classification:latest /bin/bash`
* (Inside container): `python /src/sequence_classifier.py -d [training data dir] -m [model type: fcn or rnn] -c [network config file]`

# Procedures
Due to limited time availability, two network architectures were explored: a fully connected dense network, and a recurrent neural network. These were implemented in TensorFlow 2.0-Alpha, with hyperparameter optimization via Bayesian grid search. In both networks currently attempted, I set this search space across:
 * learning rate, 
 * dropout rate,
 * batch size,
 * number of epochs,
 * number of neurons per layer / number of layers
 
 The dense network was selected as a baseline architecture to measure the recurrent network against. Given that sentence context and word order are important for NLP, one would expect that the recurrent network and its LSTM memory would better able to handle the time series sequences, and the results were very similar across both methods. Certainly, the approaches can be further optimized, and additional exploration could involve a more diverse cross-validation search space, more layers, a convolutional based architecture, or an attention-based RNN.   

# Results
Currently, the optimal metrics found are:
* Fully connected (dense) network:
    * Final Training Loss: 0.0914
    * Final Training Accuracy: 0.9645
    * Test Precision: 0.90
    * Test Recall: 0.92
    * Test F1 Score: 0.91

* Recurrent neural network:
    * Final Training Loss: 0.0586
    * Final Training Accuracy: 0.9786
    * Test Precision: 0.91
    * Test Recall: 0.87
    * Test F1 Score: 0.89