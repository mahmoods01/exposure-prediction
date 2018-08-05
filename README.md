# README

This repo contains the code used for predicting impending exposure
to malicious content within-session, as proposed in the paper 
_"Predicting Impending Exposure to Malicious Content from User 
Behavior."_ Please note that this code requires you to provide your
own data for training and testing, as the original data used in the
paper remains confidential, to maintain the users' privacy.

## Description

The three main scripts that one needs to run to evaluate the system
are:

* `compute-features.py`: Used to compute the features used by the neural networks to perform predictions.
* `train-nn.py`: Used to train the neural network.
* `test-nn.py`: Used to evaluate the neural network.

Each one of the scripts receives several arguements. The descriptions 
of the arguements can be seen by running the scripts with the `--help`
option.

## Dependencies

The main dependencies are:

* `keras`
* `tensorflow`
* `sklearn`
* `pandas`
* `annoy`

The last dependency, `annoy`, is only needed if the `SMOTE`/`ADASYN` 
algorithms are used during training. As mentioned in the paper, we 
didn't find them to be useful.

## Citation

If you use our code, please cite our paper:

~~~~
@inproceedings{Sharif18Prediction,
  author = {Mahmood Sharif and Jumpei Urakawa and Nicolas Christin
			  and Ayumu Kubota and Akira Yamada},
  title = {Predicting Impending Exposure to Malicious Content from 
  			User Behavior},
  booktitle = {Proceedings of the 25th ACM SIGSAC Conference on 
  				Computer and Communications Security},
  year = 2018
} 
~~~~