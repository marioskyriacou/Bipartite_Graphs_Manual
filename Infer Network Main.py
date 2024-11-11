
p = 3 # number of comunities (not inferred)
L_mcmc, L_prime_mcmc = 198, 132 # number of upper & bottom nodes (not inferred)

num_chains = 2 # 1 or 2 chain
num_samples = 7000
num_warmup = 200
thinning = 2 # stable at 2

# set the arguments
args_mcmc={}

args_mcmc['p'] = p
args_mcmc['Z_obs'] = adj_matrix

# Upper Nodes
args_mcmc['L'] = L_mcmc
args_mcmc['alpha'] = None
args_mcmc['sigma'] = None
args_mcmc['tau'] = ttau
args_mcmc['scores'] = None
args_mcmc['a'] = None
args_mcmc['b'] = None
args_mcmc['wi0'] = None

# Bottom Nodes
args_mcmc['L_prime'] = L_prime_mcmc
args_mcmc['alpha_prime'] = None
args_mcmc['sigma_prime'] = None
args_mcmc['tau_prime'] = ttau_prime
args_mcmc['scores_prime'] = None
args_mcmc['a_prime'] = None
args_mcmc['b_prime'] = None
args_mcmc['wj0'] = None

args_mcmc['samples']={}

args_mcmc['num_chains'] = num_chains
args_mcmc['num_samples'] = num_samples
args_mcmc['thinning'] = thinning


 #, max_tree_depth=12, init_strategy=init_to_value(values={'a':ta, 'a_prime':ta_prime,  'b':tb, 'b_prime':tb_prime,'sigma':tsigma, 'sigma_prime':tsigma_prime,  'alpha': talpha, 'alpha_prime': talpha_prime}))
 #
 #init_strategy = ,
kernel = NUTS(Bibartite_netwokr,
              init_strategy = init_to_value(values={'a':ta, 'a_prime':ta_prime,
                                                    'b':tb, 'b_prime':tb_prime,
                                                    'sigma':tsigma, 'sigma_prime':tsigma_prime,
                                                    'alpha': talpha, 'alpha_prime': talpha_prime }),
              step_size=0.05)

samples = inference(args_mcmc, kernel)

plot_traces(args_mcmc, samples, args)
plot_MCMC_histograms(args_mcmc, samples, args)
Posterior_distribution(graph_draws = Bipartite_graph_draws,
                       num_samples = num_samples,
                       thinning = thinning,
                       num_chains = num_chains,
                       L = L_mcmc,
                       L_prime = L_prime_mcmc)
Posterior_distribution_Log(graph_draws = Bipartite_graph_draws,
                       num_samples = num_samples,
                       thinning = thinning,
                       num_chains = num_chains,
                       L = L_mcmc,
                       L_prime = L_prime_mcmc)
num_draws=500 # change to 500
_, _, _, _, _, _, _, _, _, _ = plot_post_degree_distribution(num_draws = num_draws, 
                                                             samples = samples,  
                                                             args_dict=args_mcmc,
                                                             original_adj_matrix = adj_matrix, 
                                                             use_weights=0)
