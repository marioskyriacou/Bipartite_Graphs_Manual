# Bipartite_Graphs_Manual

## Bipartite Network Simulation from et-BFRY
This repository simulates a bipartite network based on the et-BFRY model.

**Objective**
The main objective of this repository is to simulate a bipartite network from the et-BFRY, leveraging various input parameters. It calculates the necessary scores and adjacency matrix based on a Bernoulli distribution.

**Variables**
The following variables are given as input parameters for the simulation:\

* L: Number of nodes in one set of the bipartite graph.
* L_hat: Number of nodes in the other set of the bipartite graph.
-p: Number of communities.
-alpha: Parameter influencing the community structure in the graph.
alpha_hat: Modified parameter for community structure.
sigma: A parameter related to the edge probabilities.
sigma_hat: Modified parameter for edge probabilities.
a: A vector of length p, associated with the community structure.
b: A vector of length p, associated with the edge probabilities.
a_hat: A vector of length p, modified vector associated with the community structure.
b_hat: A vector of length p, modified vector associated with the edge probabilities.
Calculations
The repository implements the following calculations based on the provided inputs:

Scores: Calculated based on the community structure and edge probabilities.
Scores_hat: Modified scores based on adjusted parameters.
Wio: Weight matrix for the first set of nodes.
Wj0: Weight matrix for the second set of nodes.
Adjacency Matrix: The adjacency matrix is calculated based on a given Bernoulli distribution, representing the connection probabilities between the two sets of nodes.
