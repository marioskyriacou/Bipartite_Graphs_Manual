############################## Installations ##############################
!pip install numpyro
!pip install funsor
######################### Libraries ##################################
# Main Libraries
import pandas as pd
import numpy as np
import numpyro
import os
import jax
import time
import pickle
import networkx as nx
import matplotlib
import seaborn as sns

print(f'Pandas: {pd.__version__}')
print(f'Numpy: {np.__version__}')
print(f'Numpyro: {numpyro.__version__}')
print(f'Networkx: {nx.__version__}')
print(f'Jax: {jax.__version__} // Jax Devices:{jax.devices()[0]}')
print(f'Pickle: {pickle.format_version}')
print(f'Matplotlib: {matplotlib.__version__}')
print(f'Seaborn: {sns.__version__}')

# Visualization
import matplotlib.pyplot as plt
# Jax
from jax import numpy as jnp
from jax import random as jrandom
from jax import vmap
import jax.scipy.special as special
import jax.random as random
# NumpyRO
import numpyro.distributions as dist
from numpyro.diagnostics import hpdi
from numpyro.infer import (MCMC, HMC, MixedHMC, init_to_value, NUTS, DiscreteHMCGibbs)
from numpyro.infer import Predictive
from numpyro.distributions import constraints
## Generate a graph from the prior model
from jax import random as random_jx
from jax import lax, jit, ops
from numpyro.infer import Predictive
# table creation
from tabulate import tabulate

######################### Helper Functions #########################
# I create the etBFRY function by making a class
class etBFRY(numpyro.distributions.Distribution):
    arg_constraints = {
        # "sigma": constraints.unit_interval,
        # "alpha": constraints.positive,
        "tau": constraints.positive}
    support = constraints.positive
    reparametrized_params = ["sigma", "alpha", "tau"]

    def __init__(self, L, alpha, sigma, tau):
        #read parameters
        self.L = L
        self.sigma = sigma
        self.tau = tau
        self.alpha = alpha
        self.t_as = (self.sigma*self.L/self.alpha)**(1.0/self.sigma)
        super().__init__(batch_shape = (1, self.L), event_shape=())

    def sample(self, key, sample_shape=()):
        #generate from the distribution
        shape = sample_shape + self.batch_shape
        G_dist = dist.Gamma(1.0-self.sigma, 1.0)
        G = G_dist.sample(key,shape) if isinstance(self.sigma, float) else G_dist.sample(key,(1,))
        U_dist = dist.Uniform()
        U = U_dist.sample(key,shape)
        R = G*((1.0-U)*(self.t_as+self.tau)**self.sigma + U*self.tau**self.sigma)**(-1.0/self.sigma)
        return R

    def log_prob(self, value):
      #evaluate the log probability density
      # return jnp.log(self.sigma)+(-1-self.sigma)*jnp.log(value)-self.tau*value+jnp.log(1-jnp.exp(-value*self.t_as))-special.gammaln(1-self.sigma)-jnp.log((self.tau+self.t_as)**self.sigma-self.tau**self.sigma)
      return jnp.log(self.sigma / ((self.tau+self.t_as)**self.sigma-self.tau**self.sigma))+(-1-self.sigma)*jnp.log(value)-self.tau*value+jnp.log(1-jnp.exp(-value*self.t_as))-special.gammaln(1-self.sigma)

def check_alpha(alpha):
    if alpha is None:
      #create a smaple from the below distribution , if a value is not given
      # Distributions: dist.HalfNormal(20) (mean=talpha*1.25=30*1.25 ~37) (var = (1-2pi)scale*2) //  dist.HalfCauchy(20) // dist.Gamma(90,3)
      alpha = numpyro.sample("alpha", dist.HalfNormal(20))
    else:
      #if you generate a graph
      alpha = numpyro.deterministic("alpha",alpha) # if given a value

    return alpha

def check_alpha_prime(alpha_prime):
    #create a smaple from the below distribution , if a value is not given
    #Distributions:  dist.HalfNormal(20) (mean=talpha*1.25=30*1.25 ~37) (var = (1-2pi)scale*2) //  dist.HalfCauchy(20) // dist.Gamma(90,3)
    if alpha_prime is None:
      alpha_prime = numpyro.sample("alpha_prime", dist.HalfNormal(20))
    else:
      #if you generate a graph
      alpha_prime = numpyro.deterministic("alpha_prime",alpha_prime) # if given a value

    return alpha_prime


def check_sigma(sigma):
  if sigma is None:
    #create a smaple from the below distribution , if a value is not given
    # Distributions:  dist.HalfNormal(1) 1 - sigma ~ Improper Unif //  dist.Uniform(0,1) //  TruncatedDistribution(dist.Normal(1,1), low=0, high=1) //
    # truncated_normal_model(num_observations, high=1, x=None) //  TruncatedNormal(loc=1, scale=1,low=0, high=1) //  dist.TruncatedDistribution(dist.Normal(0.5, 1), low=0.01, high=2)
    sigma = numpyro.sample("sigma", dist.Beta(2,10)) # (1/9, 1) (1,4)
  else:
    sigma = numpyro.deterministic("sigma", sigma)
  return sigma

def check_sigma_prime(sigma_prime):
   #create a smaple from the below distribution , if a value is not given
   # Distributions: dist.HalfNormal(1) 1 - sigma ~ Improper Unif // dist.Uniform(0,1) // TruncatedDistribution(dist.Normal(1,1), low=0, high=1)
   # truncated_normal_model(num_observations, high=1, x=None) // TruncatedNormal(loc=1, scale=1,low=0, high=1) //
   # dist.TruncatedDistribution(dist.Normal(0.5, 1), low=0.01, high=2)
  if sigma_prime is None:
    sigma_prime = numpyro.sample("sigma_prime", dist.Beta(2,10)) # 1/9, 1
  else:
    sigma_prime = numpyro.deterministic("sigma_prime",sigma_prime)
  return sigma_prime

def check_tau(tau):
    if tau is None:
      #create a smaple from the below distribution , if a value is not given
      tau = numpyro.sample("tau", dist.HalfNormal(1))
    else:
      tau = numpyro.deterministic("tau",tau)
    return tau

def check_tau_prime(tau_prime):
   #create a smaple from the below distribution , if a value is not given
    if tau_prime is None:
      tau_prime = numpyro.sample("tau_prime", dist.HalfNormal(1))
    else:
      tau_prime = numpyro.deterministic("tau_prime",tau_prime)
    return tau_prime

def check_a(a, p):
  if a is None:
    a = numpyro.sample("a", dist.HalfNormal(2), sample_shape=(p, ))
  else:
    a = numpyro.deterministic("a", a)
  return a


def check_a_prime(a_prime, p):
  if a_prime is None:
    a_prime = numpyro.sample("a_prime", dist.HalfNormal(2), sample_shape=(p, ))
  else:
    a_prime = numpyro.deterministic("a_prime", a_prime)
  return a_prime

def check_b(b, p):
  if b is None:
    b = numpyro.sample("b", dist.HalfNormal(2), sample_shape=(p, ))
    # b = numpyro.sample("b", dist.Gamma(0.01, 0.01), sample_shape=(p,))
  else:
    b = numpyro.deterministic("b", b)
  return b

def check_b_prime(b_prime, p):
  if b_prime is None:
    b_prime = numpyro.sample("b_prime", dist.HalfNormal(2), sample_shape=(p, ))
    # b = numpyro.sample("b", dist.Gamma(0.01, 0.01), sample_shape=(p,))
  else:
    b_prime = numpyro.deterministic("b_prime", b_prime)
  return b_prime

############################ Main Function - Bipartite Network #############################################
def Bibartite_netwokr(args):
  L, L_prime = args['L'], args['L_prime']
  p=args['p'] # number of communities

  #hyperparameters of the BFRY upper & bottom
  tau, tau_prime = args['tau'], args['tau_prime']
  sigma, sigma_prime = args['sigma'],  args['sigma_prime']
  alpha, alpha_prime = args['alpha'], args['alpha_prime']
  wi0, wj0 = args['wi0'], args['wj0']

  # Upper & bottom hyperparameters of the scores (levels of affiliations to communities)
  a, a_prime= args['a'], args['a_prime']
  b, b_prime = args['b'], args['b_prime']
  scores, scores_prime = args['scores'], args['scores_prime']

  #Check given hyperparameters
  tau, tau_prime = check_tau(tau), check_tau_prime(tau_prime)
  sigma, sigma_prime = check_sigma(sigma), check_sigma_prime(sigma_prime)
  alpha, alpha_prime = check_alpha(alpha), check_alpha_prime(alpha_prime)
  a, a_prime = check_a(a, p), check_a_prime(a_prime, p)
  b, b_prime = check_b(b, p), check_b_prime(b_prime, p)

  print(f'Upper hyperparameters:\n tau:{tau}, sigma:{sigma}, alpha:{alpha}, a:{a}, b:{b}')
  print(f'Bottom hyperparameters:\n tau:{tau_prime}, sigma:{sigma_prime}, alpha:{alpha_prime}, a:{a_prime}, b:{b_prime}')

  # adjacency matrix
  Z_obs=args['Z_obs']

  ####### Affiliation Scorres for Upper Nodes & Wi0 ###########
  if scores is None:
      # Ditributions: dist.Beta(a, b), sample_shape=(L,p) // dist.Gamma(a, b), sample_shape=(L, p) // dist.HalfNormal(4), sample_shape=(L,)
      scores = numpyro.sample("scores", dist.Gamma(a, b), sample_shape=(L, )) # if a, b vectors
  else:
      scores = numpyro.deterministic("scores", scores)

  if wi0 is None:
    wi0 = numpyro.sample("wi0", etBFRY(L, alpha, sigma, tau))
    wi0 = wi0.reshape(L,)
  else:
    wi0 = numpyro.deterministic("wi0", wi0)

  ####### Affiliation Scorres for Bottom Nodes & Wj0 ###########
  if scores_prime is None:
      #Distributions: dist.Beta(a, b), sample_shape=(L,p) // dist.Gamma(a_prime, b_prime), sample_shape=(L_prime, p) //  dist.HalfNormal(4), sample_shape=(L,)
      scores_prime = numpyro.sample("scores_prime", dist.Gamma(a, b), sample_shape=(L_prime, )) # if a, b vectors
  else:
      scores_prime = numpyro.deterministic("scores_prime", scores_prime)

  if wj0 is None:
    wj0 = numpyro.sample("wj0", etBFRY(L_prime, alpha_prime, sigma_prime, tau_prime))
    wj0=wj0.reshape(L_prime,)
  else:
    wj0 = numpyro.deterministic("wj0", wj0)

  ########## Adjanceny Matrix ############
  Wi = jnp.transpose(jnp.multiply(jnp.transpose(scores), wi0))
  Wj = jnp.transpose(jnp.multiply(jnp.transpose(scores_prime), wj0))
  W_adj = jnp.matmul(Wi, jnp.transpose(Wj))## (L,T,p) x (L_prime,T,p)

  numpyro.sample("Z_obs", dist.Bernoulli(1-jnp.exp(-2*W_adj)), obs=Z_obs)
  #numpyro.sample("N_obs", dist.Poisson(ww), obs=N_obs)


############################ Plot degree Distribution #############################################
def plt_deg_distr(deg, sigma='NA', binned=False):
    # Plot Degree Distributions:
    # Count the degrees of each nodes for one type of nodes (A or B) -> eliminate the 0 connections
    # plot the degree distribution 
    #
    deg = deg[deg > 0]
    num_nodes = len(deg)
    freq = pd.Series(deg).value_counts().to_dict()  # Count the occurrences of each degree

    if binned == True:
        freq = [x / num_nodes for x in list(freq.values())] ## Normalize frequency values
        bins = np.exp(np.linspace(np.log(min(freq)), np.log(max(freq)), 20)) #logarithmic bins for frequency values
        sizebins = (bins[1:] - bins[:-1]) #width of each bin 
        # sizebins = np.append(sizebins, 1)
        counts = np.histogram(freq, bins=bins)[0]
        freq = counts/sizebins #Normalize counts by bin size
        freq = freq/sum(freq) #Further normalize frequency so the total sums to 1
        plt.figure()
        plt.plot(bins[:-1], freq, 'bo', label='empirical')
        plt.legend()
    else:
        plt.figure()
        plt.plot(list(freq.keys()), [np.exp(np.log(x) - np.log(num_nodes)) for x in list(freq.values())], 'bo')
        plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('deg')
    plt.ylabel('frequency')

    return freq
########################### Add zeors to the adj MAtrix ##########################
def add_zero_adj_matrix(adj_matrix, ratio = 1.1):
  # add zeros to the adj matrix based on a given ratio
  print(f'Adj matrix: {adj_matrix.shape} ')

  L_upper, L_lower = adj_matrix.shape[0], adj_matrix.shape[1] # Dimenshion Shapes
  L_upper_new, L_lower_new = int(L_upper * 1.1), int(L_lower * ratio) # ratio of update shape 1.5
  # Create new matrix
  matrix_new = np.zeros((L_upper_new, L_lower_new))
  matrix_new[0:L_upper, 0:L_lower]  = np.array(adj_matrix)
  print(f'New Adj matrix: {matrix_new.shape} ')
  return matrix_new
########################### Bipartite Statistics Function  ###########################
def bipartite_stats(bottom_nodes, upper_nodes, B):
  # main Vairbles
  num_edges = len(B.edges)

  #Statistics
  top_average = num_edges / upper_nodes
  bottom_average = num_edges / bottom_nodes
  graph_average  = (2*num_edges) / (top_average + bottom_average)
  density = num_edges / (bottom_nodes * upper_nodes)

  sparsity = num_edges / (bottom_nodes + upper_nodes)**2

  # Prepare data for tabulation
  table_data = [
      ['# Top Nodes', bottom_nodes],
      ['# Bottom Nodes', upper_nodes],
      ['# Edges', num_edges],
      ['Top average degree ', round(top_average, 4)],
      ['Bottom Average degree', round(bottom_average, 4)],
      ['Graph Average degree', round(graph_average, 4)],
      ['Density', round(density, 4)],
      ['Sparsity', round(sparsity, 4)]
  ]
  print(tabulate(table_data, headers=['Metric', 'Value'], tablefmt='grid'))
########################### Main Function  ###########################
def simulations(args, add_zeros = False, simulation_plots = False ):
  rng_key, rng_key_predict = random_jx.split(random_jx.PRNGKey(22)) # ???
  Bipartite_graph_predictive = Predictive(Bibartite_netwokr, num_samples = args["batch_size"])
  Bipartite_graph_draws = Bipartite_graph_predictive(rng_key_predict, args)
  # get adj matrix
  adj_matrix = Bipartite_graph_draws['Z_obs'][0]
  if add_zeros:
    adj_matrix = add_zero_adj_matrix(adj_matrix = adj_matrix, ratio = 1.1)

  # if simulation_plots:
  #     plot_bipartite(matrix = adj_matrix,
  #                    plt_adj_matrix = False,
  #                    plot_graph = False,
  #                    print_stats = True)

  return adj_matrix, Bipartite_graph_draws
