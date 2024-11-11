######################## Simulate a bipartite Network ########################

# Varibales 
p = 3 # number of comunities (not inferred)
L, L_prime = 180, 120 # number of upper & bottom nodes (not inferred)
# BFRY parameters
talpha, talpha_prime = 15, 10
tsigma, tsigma_prime = 0.2, 0.2
ttau, ttau_prime = 1.0, 1.0
# Gamma parameters (affiliation)
ta, ta_prime = jnp.array([1.5, 1.2, 1.0]), jnp.array( [1.1, 1.0, 1.0]) # vector len == community len
tb, tb_prime = jnp.array([2.0, 1.9, 2.0]), jnp.array([1.5, 1.9, 2.0])

## Set the parameter values you want to use
args={}
args['p'] = p

# Upper Nodes
args['L'] =  L
args['alpha'] = talpha;
args['sigma'] = tsigma;
args['tau'] = ttau;
args['a'] = ta #*np.ones(args['p']) ## 1xp;
print('True a for all the upper communities is', args['a'])
args['b'] = tb #*np.ones(args['p']) ## 1xp
args['scores']=None
args['wi0']=None

# Bottom Nodes
args['L_prime'] = L_prime
args['alpha_prime'] = talpha_prime;
args['sigma_prime'] = tsigma_prime; # None
args['tau_prime'] = ttau_prime;
args['a_prime'] = ta_prime #*np.ones(args['p']) ## 1xp;
print('True a_prime for all the bottom communities is', args['a_prime'])
args['b_prime'] = tb_prime #*np.ones(args['p']) ## 1xp
args['scores_prime']=None
args['wj0']=None

args['batch_size'] = 1 # number of graphs to simulate
args['Z_obs']=None # given


adj_matrix, Bipartite_graph_draws = simulations(args = args, add_zeros = True, simulation_plots = False)
