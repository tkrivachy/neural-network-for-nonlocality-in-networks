# Neural Network for Nonlocality in Networks
Code for the work in: https://arxiv.org/abs/1907.10552

If you use this code please cite
T. Kriváchy, Y. Cai, D. Cavalcanti, A. Tavakoli, N. Gisin, N. Brunner, A neural network oracle for quantum nonlocality problems in networks, arXiv:1907.10552

## Intro
In the paper above, we describe a generative neural network to tackle the classical causal inference problem encountered in quantum network nonlocality. The neural network basically tries to learn classical/local models for a given distribution and causal structure.

## Usage
In sample_code, set your parameters in config.py. Then run train.py. For a first test, **just run train.py** to see training for the Fritz distribution and its noisy versions (visibility added to the singlet).

In case you'd like to modify the sample target distribution, you can choose between the elegant distribution, Fritz distribution, or Renou et al. distribution. See detailed namings in targets.py and update target distribution in config.py.

In case you'd like to modify the causal structure, check out utils.py. It is currently set to be the triangle structure.

## Code structure
In sample_code you will find sample code for running the algorithm (including adding noise), for several possible distributions.

* `config.py`: Contains the Config class. A joint, global config file (named pnn, for Parameter of Neural Network) is used among all files. This contains (almost) all meta-data related to the neural network, target distribution, and training. Note that the causal structure is defined in utils.py.
* `utils.py`: Contains utilities. Most notably the build_model() function is located here, which defines the causal structure.
* `targets.py`: Auxiliary file used by me to generate target distributions. If you want your own target distributions either add them to the list in this file, or just load them directly into Config class and disregard the targets.py file.
* `train.py`: **Run this python script to train the network for a set of target distributions.** Target distributions and neural network parameters should be defined in config.py.

## Output
In the sample code we look at the Fritz distribution for 10 different singlet visibility levels between 0.5 and 1. 
* figs_training_loop: Contains the distances as a function of visibility.
* figs_distributions: Contains the target distributions (red dots) and the learned distributions (green squares), for each of the 10 target distributions.
* saved_models: Contains final models for each of the 10 target distribution examined.
* saved_results: Contains final distances for each of the 10 target distributions.
