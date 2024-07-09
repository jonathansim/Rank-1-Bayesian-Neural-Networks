# Bachelor's Thesis: Rank-1-Bayesian-Neural-Networks
Code used in my bachelor project studying the use of rank-1 factorization in regard to Bayesian neural networks (BNN). This bachelor's thesis has been prepared at the Technical University of Denmark. It aims to reproduce the findings by Dusenberry et al., who originally introduced rank-1 Bayesian neural networks in 2020. Furthermore, various experiments are conducted to validate the robustness and effiency of this framework.
The rank-1 BNN as well as the baseline frameworks (a deterministic Wide ResNet, Monte Carlo dropout and Deep Ensembles) have been implemented in PyTorch. 

The folder *baselines* contains the different baseline models (i.e. deterministic Wide Resnet 28-10, MC dropout and deep ensembles). 
The folder *BNN* contains all files related to the BNN implementation, such as the custom loss function, scheduler and layers (convolutional and fully connected layer). 
