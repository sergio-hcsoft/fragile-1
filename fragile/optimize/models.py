import numpy as np
from fragile.core.models import RandomNormal
from fragile.core.states import States


class ESModel(RandomNormal):

    def __init__(
            self,
            mutation: float = 0.5,
            recombination: float = 0.7,
            random_step_prob: float = 0.1,
            *args,
            **kwargs
    ):
        super(ESModel, self).__init__(*args, **kwargs)
        self.mutation = mutation
        self.recombination = recombination
        self.random_step_prob = random_step_prob

    def sample(
            self,
            batch_size: int,
            model_states: States = None,
            env_states: States = None,
            walkers_states: "StatesWalkers" = None,
    ) -> States:
        """
        Calculate the corresponding data to interact with the Environment and \
        store it in model states.

        Args:
            batch_size: Number of new points to the sampled.
            model_states: States corresponding to the environment data.
            env_states: States corresponding to the model data.
            walkers_states: States corresponding to the walkers data.

        Returns:
            Tuple containing a tensor with the sampled actions and the new model states variable.

        """
        if np.random.random() < self.random_step_prob:
            return super(ESModel, self).sample(batch_size=batch_size, env_states=env_states,
                                               model_states=model_states)
        observs = (env_states.observs if env_states is not None
                   else np.zeros(((batch_size,) + self.shape)))
        has_best = walkers_states is not None and walkers_states.best_found is not None
        best = walkers_states.best_found if has_best else observs
        # Choose 2 random indices
        a_rand = self.random_state.permutation(np.arange(observs.shape[0]))
        b_rand = self.random_state.permutation(np.arange(observs.shape[0]))
        proposal = best + self.recombination * (observs[a_rand] - observs[b_rand])
        # Randomly mutate the each coordinate of the original vector
        assert observs.shape == proposal.shape
        rands = np.random.random(observs.shape)
        perturbations = np.where(rands < self.mutation, observs, proposal).copy()
        new_states = perturbations - observs
        actions = self.bounds.clip(new_states) if self.bounds is not None else new_states
        dt = (1 if self.dt_sampler is None else
              self.dt_sampler.calculate(batch_size=batch_size, model_states=model_states,
                                        env_states=env_states, walkers_states=walkers_states))
        model_states.update(actions=actions, dt=dt)
        return model_states
