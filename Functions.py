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
from numpyro.infer import (
    MCMC,
    HMC,
    MixedHMC,
    init_to_value,
    NUTS,
    DiscreteHMCGibbs
)
from numpyro.infer import Predictive
from numpyro.distributions import constraints

## Generate a graph from the prior model
from jax import random as random_jx
from jax import lax, jit, ops
from numpyro.infer import Predictive

# table creation
from tabulate import tabulate

