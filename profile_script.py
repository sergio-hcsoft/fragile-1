from fragile.atari.swarm import MontezumaSwarm

swarm = MontezumaSwarm.create_swarm(n_walkers=1000,
                                    max_iters=2,
                                    prune_tree=True,
                                    use_tree=False,
                                    reward_scale=4,
                                    dist_scale=0.5,
                                    plot_step=100,
                                    critic_scale=2,
                                    episodic_rewad=True)
_ = swarm.run_swarm(print_every=100)