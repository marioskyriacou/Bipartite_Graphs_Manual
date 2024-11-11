##################### Main Function to Infere a Bipartite Netwrok #####################
def inference(args_mcmc, kernel):
  # This function uses the given arguments and uses MCMC algorithm to infer a bipartite network 
  rng_key = random.PRNGKey(2)
  rng_key, rng_key_ = random.split(rng_key)

  args_mcmc['num_chains'] = num_chains
  args_mcmc['num_samples'] = num_samples
  args_mcmc['thinning'] = thinning

  mcmc = MCMC(kernel,
            num_warmup = num_warmup,
            num_samples = num_samples,
            num_chains = num_chains,
            thinning = thinning)

  mcmc.run(rng_key_,  args = args_mcmc)
  mcmc.print_summary()
  samples = mcmc.get_samples()
  return samples


################################ Functions-Results of Infer Network #################################

################################ Traces and Histograms Plots  #################################
def plot_traces(args_mcmc, samples, args):
    # Plot the trace plots for MCMC samples of specified parameters
    # Variables needed/ # Calculate the number of iterations for each parameter
    num_chains=args_mcmc['num_chains']
    iters = int(samples['wi0'].shape[0] / num_chains)
    iters_prime = int(samples['wj0'].shape[0] / num_chains)

    # Identify the parameters that need to be plot
    parameters_list = ['alpha', 'sigma', 'tau', 'a', 'b', 'alpha_prime', 'sigma_prime', 'tau_prime', 'a_prime', 'b_prime']
    plt_params = [key for key, value in args_mcmc.items() if args_mcmc[key] is None and key in parameters_list]

    # Plot each parameter in a graph
    for param in plt_params:
      ## Check if the parameter has multiple subplots
      if len(samples[param].shape) > 1:
        num_subplots = samples[param].shape[1]
        fig, axes = plt.subplots(1, num_subplots, figsize=(15, 5))
        axes = np.array(axes).flatten()

        for i in range(num_subplots):
            for j in range(num_chains):
              #Plot the MCMC trace for each chain
              axes[i].plot(np.arange(iters), samples[param][:, i][j*iters_prime:(j+1)*iters_prime], 'o-', alpha = 0.8,  label=f'{param}[{i}]')
              #plt.plot(np.arange(iters), samples[param][i*iters:(i+1)*iters], 'o-')
            axes[i].axhline(y=args[param][i], color='red')
            axes[i].set_title(f'{param}[{i}]')
      else:
        #For single-dimensional parameters, create a new figure
        plt.figure()
        for i in range(num_chains):
          plt.plot(np.arange(iters_prime), samples[param][i*iters_prime:(i+1)*iters_prime], 'o-', alpha = 0.8)
        plt.axhline(y=np.mean(args[param]), color='red')
        plt.title(param)

def plot_MCMC_histograms(args_mcmc, samples, args):
    # Plot the trace plots for MCMC samples of specified parameters
    # Variables needed/ # Calculate the number of iterations for each parameter
    
    # Varibles
    num_bins=20
    num_chains=args_mcmc['num_chains']
    iters = int(samples['wi0'].shape[0] / num_chains)
    iters_prime = int(samples['wj0'].shape[0] / num_chains)

    # Identify the parameters that need to be plot
    parameters_list = ['alpha', 'sigma', 'tau', 'a', 'b', 'alpha_prime', 'sigma_prime', 'tau_prime', 'a_prime', 'b_prime']
    plt_params = [key for key, value in args_mcmc.items() if args_mcmc[key] is None and key in parameters_list]

    # Plot each parameter in a graph
    for param in plt_params:
      if len(samples[param].shape) > 1:
        num_subplots = samples[param].shape[1]
        fig, axes = plt.subplots(1, num_subplots, figsize=(15, 5))
        axes = np.array(axes).flatten()

        for i in range(num_subplots):
            for j in range(num_chains):
               #Plot the MCMC histogram  for each chain
              axes[i].hist(samples[param][:, i][j*iters_prime:(j+1)*iters_prime], num_bins, alpha = 0.5, label=f'{param}[{i}]')
            axes[i].axvline(x=args[param][i], color='red')
            axes[i].set_title(f'{param}[{i}]')
      else:

        plt.figure()
        for i in range(num_chains):
          #For single-dimensional parameters, create a new figure
          plt.hist(samples[param][i*iters:(i+1)*iters], num_bins, alpha = 0.8, label = str(param) )
        plt.axvline(x=np.mean(args[param]), color='red')
        plt.title(param)
        
######################################### Posterior Distribution Plots (Log Posterior) ############################################
def Posterior_distribution(graph_draws, num_samples, thinning, num_chains, L, L_prime):
  # Plot the posterior distribution for the top m nodes based on estimated weights wi0 and wj0.
  tgraph_index = 1
  m = 50 ## Number of top nodes to plot
  # Extract true and estimated values for Wi0 
  Wi0 = graph_draws['wi0']
  wi0_est = samples['wi0']
  # Reshape and calculate the 95% highest posterior density interval (HPDI) for Wi0
  wi0_est = wi0_est.reshape(int(num_samples/thinning*num_chains), L)
  wi0_est = hpdi(wi0_est, prob=0.95)
 # Extract true and estimated values for Wj0 
  Wj0 = graph_draws['wj0']
  wj0_est = samples['wj0']
  # Reshape and calculate the 95% highest posterior density interval (HPDI) for Wi0
  wj0_est = wj0_est.reshape(int(num_samples/thinning*num_chains), L_prime)
  wj0_est = hpdi(wj0_est, prob=0.95)
  print(f'Wi0 Shape:{wi0_est.shape}\nWj0 Shape:{wj0_est.shape}')
  #indices of the top m nodes by sorting Wi0 in decreasing order
  ind_max_wi0 = np.argsort(-Wi0[tgraph_index, 0, :])[:m]
  w0i_ci_max = wi0_est[:, ind_max_wi0]

  #indices of the top m nodes by sorting Wj0 in decreasing order 
  ind_max_wj0 = np.argsort(-Wj0[tgraph_index, 0, :])[:m]
  w0j_ci_max = wj0_est[:, ind_max_wj0]

  #x-axis for each node index and y-axis for confidence intervals
  x = [(i, i) for i in range(m)]
  y = [(w0i_ci_max[0, i], w0i_ci_max[1, i]) for i in range(m)]
  y_prime = [(w0j_ci_max[0, i], w0j_ci_max[1, i]) for i in range(m)]

  #Plots
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
  
  # Upper Nodes
  for i, j in zip(x, y):
      ax1.plot((i[0], i[1]), (j[0], j[1]), color='blue')
      ax1.scatter(i[0], Wi0[tgraph_index, 0, ind_max_wi0[i[0]]], color='red')

  ax1.plot((i[0], i[1]), (j[0], j[1]), color='blue', label='95% CI')
  ax1.scatter(i[0], Wi0[tgraph_index, 0, ind_max_wi0[i[0]]], color='red', label='true')
  ax1.title.set_text('Upper Nodes')
  ax1.set_xlabel('index of nodes sorted by decreasing wi0')
  ax1.set_ylabel('wi0')

  # Lower Nodes
  for i, j in zip(x, y_prime):
      ax2.plot((i[0], i[1]), (j[0], j[1]), color='blue')
      ax2.scatter(i[0], Wj0[tgraph_index, 0, ind_max_wj0[i[0]]], color='red')

  ax2.plot((i[0], i[1]), (j[0], j[1]), color='blue', label='95% CI')
  ax2.scatter(i[0], Wj0[tgraph_index, 0, ind_max_wj0[i[0]]], color='red', label='true')
  ax2.title.set_text('Lower Nodes')
  ax2.set_xlabel('index of nodes sorted by decreasing wj0')
  ax2.set_ylabel('wj0')

  fig.tight_layout()
  plt.show()

def Posterior_distribution_Log(graph_draws, num_samples, thinning, num_chains, L, L_prime):
  tgraph_index = 1
  m = 50
  # Wi0 estimates
  Wi0 = graph_draws['wi0']
  wi0_est = samples['wi0']
  wi0_est = wi0_est.reshape(int(num_samples/thinning*num_chains), L)
  wi0_est = hpdi(wi0_est, prob=0.95)
  # Wj0 estimates
  Wj0 = graph_draws['wj0']
  wj0_est = samples['wj0']
  wj0_est = wj0_est.reshape(int(num_samples/thinning*num_chains), L_prime)
  wj0_est = hpdi(wj0_est, prob=0.95)
  print(f'Wi0 Shape:{wi0_est.shape}\nWj0 Shape:{wj0_est.shape}')
  ind_min_wi0 = np.argsort(Wi0[tgraph_index, 0, :])[:m]
  w0i_ci_min = wi0_est[:, ind_min_wi0]

  ind_min_wj0 = np.argsort(Wj0[tgraph_index, 0, :])[:m]
  w0j_ci_min = wj0_est[:, ind_min_wj0]

  x = [(i, i) for i in range(m)]
  y = [(w0i_ci_min[0, i], w0i_ci_min[1, i]) for i in range(m)]
  y_prime = [(w0j_ci_min[0, i], w0j_ci_min[1, i]) for i in range(m)]
  #Plots
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
  # Upper Nodes
  for i, j in zip(x, y):
      ax1.plot((i[0], i[1]), (np.log(j[0]), np.log(j[1])), color='blue')
      ax1.scatter(i[0], np.log(Wi0[tgraph_index, 0, ind_min_wi0[i[0]]]), color='red')
  ax1.plot((i[0], i[1]), (j[0], j[1]), color='blue', label='95% CI')
  ax1.scatter(i[0], np.log(Wi0[tgraph_index, 0, ind_min_wi0[i[0]]]), color='red', label='true')
  ax1.title.set_text('Upper Nodes')
  ax1.set_xlabel('index of nodes sorted by decreasing wi0')
  ax1.set_ylabel('Log_wi0')

  # Lower Nodes
  for i, j in zip(x, y_prime):
      ax2.plot((i[0], i[1]), (np.log(j[0]), np.log(j[1])), color='blue')
      ax2.scatter(i[0], np.log(Wj0[tgraph_index, 0, ind_min_wj0[i[0]]]), color='red')
  ax2.plot((i[0], i[1]), (j[0], j[1]), color='blue', label='95% CI')
  ax2.scatter(i[0], np.log(Wj0[tgraph_index, 0, ind_min_wj0[i[0]]]), color='red', label='true')
  ax2.title.set_text('Lower Nodes')
  ax2.set_xlabel('index of nodes sorted by decreasing wj0')
  ax2.set_ylabel('Log_wj0')
  fig.tight_layout()
  plt.show()

######################################## Post Degree distribution ########################################

def network_weights_w_wprime(samples_dict, iters, iters_prime, p):
  # Calculate the estimated network weights for upper and lower nodes.
  # Upper Nodes
  W_upper_est = np.zeros((iters, L_mcmc, p))
  for i in range(iters):
    #Multiply each score in `scores` by the corresponding wi0 weight
      W_upper_est[i, :, :] = np.transpose(np.multiply(np.transpose(samples_dict['scores'][i, :, :]), samples_dict['wi0'][i, :]))
  # Lower Nodes
  W_lower_est = np.zeros((iters_prime, L_prime_mcmc, p))
  for j in range(iters_prime):
    #Multiply each score in `scores` by the corresponding wj0 weight
      W_lower_est[j, :, :] = np.transpose(np.multiply(np.transpose(samples_dict['scores_prime'][j, :, :]), samples_dict['wj0'][j, :]))
  return W_upper_est, W_lower_est

def exptiltBFRY_jax(rng_key, alpha, sigma, tau, L):  
    #Generate samples from an exponentially tilted-(BFRY) distribution
    t = ((L * sigma / alpha) ** (1.0/ sigma)).astype(float)
    g=jax.random.gamma(rng_key, 1-sigma, shape=(L,1))
    unif = jax.random.uniform(rng_key,shape=(L,1))
    s = jnp.multiply(g, jnp.power(((t + tau) ** sigma) * (1 - unif) + (tau ** sigma) * unif, -1 / sigma))
    return s

def model_etbfry_todeschini_jax(rng_key, args_dict,  alpha, alpha_prime,
                                sigma, sigma_prime, tau, tau_prime,
                                a, a_prime, b, b_prime,
                                scores=None, scores_prime=None,
                                wi0=None, wj0=None, weights=None, weights_prime=None):
  
    ##### A JAX-based implementation of the Exponentially Tilted Beta-Fractional Random Graph Model by Todeschini
    ## Extract node counts and feature dimension from args_dict
    L, L_prime = args_dict['L'], args_dict['L_prime']
    p = args_dict['p']

    # Initialize weights wi0 and wj0 for upper nodes if not provided
    if wi0 is None:
        wi0 = jnp.zeros((L, 1), dtype='float')
        wi0 = exptiltBFRY_jax(rng_key, alpha, sigma, tau, L).reshape(L)
        print(wi0.shape)
    if wj0 is None:
        wj0 = jnp.zeros((L_prime, 1), dtype='float32')
        wj0 = exptiltBFRY_jax(rng_key, alpha_prime, sigma_prime, tau_prime, L_prime).reshape(L_prime)
        print(wj0.shape)
    
    # Generate latent scores for upper and lower nodes if not provided
    if scores is None:
        scores = jnp.zeros((L, p), dtype='float32')
        scores = jax.random.gamma(rng_key, a, shape=(L, p))/b
        print(scores.shape)
    if scores_prime is None:
        scores_prime = jnp.zeros((L_prime, p), dtype='float32')
        scores_prime = jax.random.gamma(rng_key, a_prime, shape=(L_prime, p))/b_prime
        print(scores_prime.shape)
    
    # Compute weights for lower and upper nodes by scaling scores_prime with wi0 and wj0
    if weights is None:
        weights = jnp.zeros((L, p), dtype='float32')
        weights=jnp.transpose(jnp.multiply(jnp.transpose(scores), wi0))
        print(weights.shape)

    if weights_prime is None:
        weights_prime = jnp.zeros((L_prime, p), dtype='float32')
        weights_prime = jnp.transpose(jnp.multiply(jnp.transpose(scores_prime), wj0))
        print(weights_prime.shape)


    print(f'Wi0:{wi0.shape}, Wj0:{wj0.shape}, scores:{scores.shape}, scores_prime:{scores_prime.shape}, weights_prime:{weights_prime.shape}, weights:{weights.shape}')
    
    # # Calculate the weight matrix 'ww' for the network connections
    ww = jnp.matmul( weights, jnp.transpose(weights_prime))## (L,T,p) x (L_prime,T,p)= (L, L_prime)
    # Generate observed connections (Poisson-distributed) based on 'ww'
    N_obs=jax.random.poisson(rng_key, ww, shape=((L, L_prime)))
    print(f'N_obs:{N_obs.shape}')
    ## Generate binary connection indicator matrix
    Z_obs = (N_obs) >0 
    print(f'Z_obs:{Z_obs.shape}')

    return N_obs, Z_obs, weights, weights_prime, scores, scores_prime, wi0, wj0


def plot_degree_sparse(G, step=1, color='b'):
  ### Plots the degree distribution of a sparse graph represented by an adjacency matrix `G` using logarithmic binning.
    
    pd.plotting.register_matplotlib_converters()
    # Calculate Degrees
    deg_upper = np.squeeze(np.sum(G,0))
    deg_lower = np.squeeze(np.sum(G,1))
    # Remove nodes with no connections
    any_upper = np.squeeze(np.asarray(deg_upper > 0))
    any_lower = np.squeeze(np.asarray(deg_lower > 0))
    ##  Filter rows and columns
    G = G[any_lower, :]
    G = G[:, any_upper]
    ## Recalculate degrees after filtering
    deg_upper = np.squeeze(np.sum(G,0))
    deg_lower = np.squeeze(np.sum(G,1))

    # Uses logarithmic binning to get a less noisy estimate of the pdf of the degree distribution
    edgebins = 2**np.arange(0, 17, step)
    sizebins = edgebins[1:] - edgebins[:-1]
    sizebins = np.append(sizebins, 1)
    centerbins = edgebins

    ## Calculate and plot the degree distribution for upper nodes
    counts_upper = np.histogram(deg_upper, np.append(edgebins, np.inf))
    freq_upper = np.divide(counts_upper[0],sizebins)/G.shape[0]
    h2_upper = plt.loglog(centerbins, freq_upper,'o',color=color)

    ## Calculate and plot the degree distribution for lower nodes
    counts_lower = np.histogram(deg_lower, np.append(edgebins, np.inf))
    freq_lower = np.divide(counts_lower[0], sizebins)/G.shape[1]
    h2_lower = plt.loglog(centerbins, freq_lower,'o',color=color)

    plt.xlabel('Degree', fontsize=16)
    plt.ylabel('Distribution', fontsize=16)
    plt.gca().set_xlim(left=1)
    # h2, centerbins, freq
    return [h2_upper, h2_lower, centerbins, freq_upper, freq_lower ]


def plot_figure(freq, centerbins, freq_true):
    ##### Plots a degree distribution with posterior predictive intervals along with the true frequency distribution.
    
    #Calculate the 2.5% and 97.5% quantiles of the posterior predictive distribution
    quantile_freq = np.quantile(freq, [.025, .975],0)
    ## Avoid division by zero for nodes with zero degree
    ind1 = quantile_freq[0,:]==0
    quantile_freq[0, ind1] = quantile_freq[1, ind1] / 100000

    plt.figure()
    ## Plot the quantile range for the posterior predictive distribution
    plt.plot(centerbins, np.transpose(quantile_freq), color='b', alpha=0.2, label='_nolegend_')
    ind = quantile_freq[0,:]>0

    ## Fill the area between the quantile ranges with transparency
    plt.fill_between(centerbins[ind], np.transpose(quantile_freq[0,ind]), np.transpose(quantile_freq[1,ind]), alpha=0.2)
    plt.xscale('log')
    plt.yscale('log')
    # Plot the true frequency data
    ind = freq_true>0
    plt.loglog(centerbins[ind], freq_true[ind], 'o', color='b')
    
    plt.xlabel('Degree')
    plt.ylabel('Distribution')
    plt.legend(labels=('Data', '95% posterior predictive'),frameon=False, loc='lower left')
    plt.xlim([.8, 1e4])
    plt.tight_layout()

    return quantile_freq

def plot_post_degree_distribution(num_draws, samples, original_adj_matrix, args_dict, use_weights=1):
  ### Main function to plot post degree distributions 

    rng_key, rng_key_predict = random.split(random.PRNGKey(0)) # random key 
    # Inters- Calculate the number of iterations (draws) for each chain
    iters = int(samples['wi0'].shape[0] / num_chains)
    iters_prime = int(samples['wj0'].shape[0] / num_chains)
    ## samples used to parallelize the sampling process
    vmap_args = (
            random.split(rng_key_predict,  num_draws),
            samples["alpha"][iters-num_draws:iters],
            samples["sigma"][iters-num_draws:iters],
            samples["tau"][iters-num_draws:iters],
            samples["a"][iters-num_draws:iters],
            samples["b"][iters-num_draws:iters],

            samples["alpha_prime"][iters_prime-num_draws:iters_prime],
            samples["sigma_prime"][iters_prime-num_draws:iters_prime],
            samples["tau_prime"][iters_prime-num_draws:iters_prime],
            samples["a_prime"][iters_prime-num_draws:iters_prime],
            samples["b_prime"][iters_prime-num_draws:iters_prime] )

    if use_weights:
      # If weights are being used, calculate them and add to the samples
        W_upper_est, W_lower_est = network_weights_w_wprime(samples, iters, iters_prime, p =3) # helper function 
        samples['weights'] = W_upper_est
        samples['weights_prime'] = W_lower_est
        print(f'Weights Type A:{W_upper_est.shape}, Weights Type B:{W_lower_est.shape}')
        
        vmap_args=vmap_args+ (
        samples["wi0"][iters-num_draws:iters],
        samples["scores"][iters-num_draws:iters],
        samples["weights"][iters-num_draws:iters],
        samples["wj0"][iters-num_draws:iters],
        samples["scores_prime"][iters-num_draws:iters],
        samples["weights_prime"][iters-num_draws:iters])
        print('Use Weights')
        ## # Use vmap to sample from the model with the weights
        N_obs_pred, Z_obs_pred, weights_pred, weights_prime_pred, scores_pred, scores_prime_pred, wi0_pred, wj0_pred = vmap(
            lambda rng_key, alpha, sigma, tau, a, b, alpha_prime, sigma_prime, tau_prime, a_prime, b_prime, wi0, scores, weights, wj0, scores_prime, weights_prime : model_etbfry_todeschini_jax(     
                rng_key=rng_key, args_dict=args_dict, alpha=alpha, alpha_prime=alpha_prime, 
                sigma=sigma, sigma_prime=sigma_prime, tau=tau, tau_prime=tau_prime,
                 a=a, a_prime=a_prime, b=b, b_prime=b_prime, wi0=wi0, wj0=wj0, 
                scores=scores, scores_prime=scores_prime, weights=weights, weights_prime=weights_prime))(*vmap_args)
    else:
         ## # Use vmap to sample from the model with no weights
        print('No Weights ')
        N_obs_pred, Z_obs_pred, weights_pred, weights_prime_pred, scores_pred, scores_prime_pred, wi0_pred, wj0_pred  = vmap(
                lambda rng_key, alpha, sigma, tau, a, b, alpha_prime, sigma_prime, tau_prime, a_prime, b_prime : model_etbfry_todeschini_jax( 
                    rng_key=rng_key, args_dict=args_dict, alpha=alpha, alpha_prime=alpha_prime, 
                sigma=sigma, sigma_prime=sigma_prime, tau=tau, tau_prime=tau_prime,
                 a=a, a_prime=a_prime, b=b, b_prime=b_prime))(*vmap_args)

    #  Initialize arrays to store frequency distributions for degree distribution plotting
    freq_samp, freq_samp_prime  = np.zeros((num_draws, 17)),  np.zeros((num_draws, 17))
    centerbins1 = np.zeros((17))
    freq_true, freq_true_prime = np.zeros((17)), np.zeros((17))
    ## Arrays to store quantile frequency distributions
    quantile_freq_upper, quantile_freq_lower = np.zeros((2,17)),  np.zeros((2,17)) 
    
    ##each draw to compute the degree distribution for each sample
    for i in range(num_draws):
        Gsamp = Z_obs_pred[i]
        [_, _, _, freq_samp[i, :], freq_samp_prime[i, :] ] = plot_degree_sparse(Gsamp) ## Calculate the degree distribution for the current sample 
        [_, _, centerbins1[:], freq_true[:], freq_true_prime[:] ] = plot_degree_sparse(jnp.array(original_adj_matrix)) ## Calculate the degree distribution for the true sample
        plt.close()

    ## Plot the upper  and lower part of the degree distribution 
    print('plot_figure Upper')
    quantile_freq_upper[:,:] = plot_figure(freq_samp[:,:], centerbins1[:], freq_true[:])
    print('plot_figure Lower')
    quantile_freq_lower[:,:] = plot_figure(freq_samp_prime[:,:], centerbins1[:], freq_true_prime[:])

    return quantile_freq_upper, quantile_freq_lower, N_obs_pred, Z_obs_pred, weights_pred, weights_prime_pred, scores_pred, scores_prime_pred, wi0_pred, wj0_pred
