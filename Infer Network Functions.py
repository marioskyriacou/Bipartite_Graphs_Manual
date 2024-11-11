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



