# Bachelor's Thesis: Rank-1-Bayesian-Neural-Networks
Code used in my bachelor project studying the use of rank-1 factorization in regard to Bayesian neural networks (BNNs). This bachelor's thesis has been prepared at the Technical University of Denmark. It aims to reproduce the findings by Dusenberry et al. [[1]](#1), who originally introduced rank-1 Bayesian neural networks in 2020. Furthermore, various experiments are conducted to validate the robustness and effiency of this framework.
The rank-1 BNN as well as the baseline frameworks (a deterministic Wide ResNet, Monte Carlo dropout and Deep Ensembles) have been implemented in PyTorch. 

## Environment Setup
The file `requirements.txt` contained in the `environment_setup`folder contains the packages used for this project. 

## Code
The folder `BNN` contains all files related to the BNN implementation, such as the custom loss function, scheduler and layers (convolutional and fully connected layer). 
- The files `bnn_conv_layer_mix.py` and `bnn_linear_layer_mix.py` contain the custom layers to incorporate rank-1 Bayesian principles. 
- `rank1_bnn_main.py`contains the main training and evaluation script used for the rank-1 BNN. Slight variations of this file used for the sake of convenience during the implementation process can also be found there. 
- `bnn_utils.py` contains the loss function, custom scheduler, the function to compute the KL divergence used in the loss and some initialization functions. 
- `corrupted_data_utils.py` contains the functionality to a simplistic corrupted version of the CIFAR-10 test dataset. 


The folder `baselines` contains the different baseline models (i.e. deterministic Wide Resnet 28-10, MC dropout and deep ensembles). 
- The file `wide_resnet.py` contains the Wide ResNet architecture. Not that a slightly separate file was created for the MC dropout network structure with a different placement of the dropout layer. 
- The files `deterministic.py`, `mc_dropout.py` and `ensembles.py` contain the training/evaluation scripts for the different baselines. 

The folder `results_processing` contains various functionality used in process of working with the results. 

Furthermore, the script `split_data.py`was used to create the validation set used in this project. 

## References 
<a id="1">[1]</a> 
Dusenberry, Michael, et al. "Efficient and scalable bayesian neural nets with rank-1 factors." International conference on machine learning. PMLR, 2020. https://arxiv.org/abs/2005.07186 


