# Bipartite_Graphs_Manual

## Bipartite Network Simulation from et-BFRY
This repository simulates a bipartite network based on the et-BFRY model.

**Objective**
The main objective of this repository is to simulate a bipartite network from the et-BFRY, leveraging various input parameters. It calculates the necessary scores and adjacency matrix based on a Bernoulli distribution.

**Variables**
The following variables are given as input parameters for the simulation:\

* L: Number of nodes in one set of the bipartite graph.
* L_hat: Number of nodes in the other set of the bipartite graph.
* p: Number of communities.
* alpha: Parameter influencing the community structure in the graph.
* alpha_hat: Modified parameter for community structure.
* sigma: A parameter related to the edge probabilities.
* sigma_hat: Modified parameter for edge probabilities.
* a: A vector of length p, associated with the community structure.
* b: A vector of length p, associated with the edge probabilities.
* a_hat: A vector of length p, modified vector associated with the community structure.
* b_hat: A vector of length p, modified vector associated with the edge probabilities.
## Calculations
The repository implements the following calculations based on the provided inputs:

* Scores: Calculated based on the community structure and edge probabilities.
* Scores_hat: Modified scores based on adjusted parameters.
* Wio: Weight matrix for the first set of nodes.
* Wj0: Weight matrix for the second set of nodes.
*Adjacency Matrix: The adjacency matrix is calculated based on a given Bernoulli distribution, representing the connection probabilities between the two sets of nodes.
## Istalations 
To run the simulations, you need to install the following dependencies:
```
pip install numpy scipy
pip install numpyro
pip install funsor
```
## MCMC Inference for Bipartite Network
This section of the repository is focused on inferring a bipartite network using Markov Chain Monte Carlo (MCMC) methods. The goal is to estimate the parameters of the bipartite network model based on the provided input data.

### Objective
The objective of this repository is to infer the parameters of a bipartite network using MCMC sampling. The given input variables, including the adjacency matrix and model parameters, are used to perform Bayesian inference to estimate the model parameters.

### Given Variables
The following variables are provided as input for the MCMC inference:

* L: Number of nodes in the first set of the bipartite graph.
* L_hat: Number of nodes in the second set of the bipartite graph.
* P: Number of communities.
* Number of chains: Number of MCMC chains to run.
* Number of samples: Number of samples to generate from each chain.
* Warmup: Number of warmup steps for MCMC.
* Adjacency Matrix: The observed adjacency matrix representing the bipartite network.
* t: A model parameter related to the structure of the graph.
* t_hat: A modified model parameter.


**Parameters to be Inferred**
Using MCMC sampling, the following parameters will be inferred:

* Alpha: Parameter influencing the community structure in the graph.
* Alpha_hat: Modified parameter for community structure.
* Sigma: A parameter related to the edge probabilities in the graph.
* Sigma_hat: Modified parameter for edge probabilities.
* a: Vector of length P representing community structure.
* b: Vector of length P representing edge probabilities.
* a_hat: Modified vector of length P for community structure.
* b_hat: Modified vector of length P for edge probabilities.
* Scores: Calculated based on the community structure and edge probabilities.
* Scores_hat: Modified scores based on adjusted parameters.
* Wio: Weight matrix for the first set of nodes.
* Wj0: Weight matrix for the second set of nodes.

# File Guides
* ```Bipartite_Network.ipynb```: The first trial of simulate and inference bipartite networks
* ```Simulation Functions.py```, ```Simulations Main.py```, ```Inference Network Function.py``` and ```inference netrok main.py``` are thi main .py files that create the classes for automation of inference and simulation of a bipartite network
* ```Simulate_Inference_Bipartite_Network_Classes.ipynb``` Uses Classes ONLY for simulating and Infering Bipartite Networks
  
