import numpy as np
from fragile.core.models import RandomNormal
from fragile.core.states import States
from fragile.core.utils import calculate_clone, calculate_virtual_reward, relativize


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


class CompasJump(RandomNormal):

    def __init__(self, dist_coef: float = 1.0, reward_coef: float = 1.0, eps=1e-8,
                 *args,  **kwargs):
        super(CompasJump, self).__init__(*args, **kwargs)
        self.dist_coef = dist_coef
        self.reward_coef = reward_coef
        self.eps = eps

    def calculate(
            self,
            batch_size: int = None,
            model_states: States = None,
            env_states: States = None,
            walkers_states: "StatesWalkers" = None,
    ) -> np.ndarray:
        """
        Calculate the target time step values.

        Args:
            batch_size: Number of new points to the sampled.
            model_states: States corresponding to the model data.
            env_states: States corresponding to the environment data.
            walkers_states: States corresponding to the walkers data.

        Returns:
            Array containing the target time step.

        """
        virtual_rewards, compas_dist = calculate_virtual_reward(observs=env_states.observs,
                                                                rewards=env_states.rewards,
                                                                dist_coef=self.dist_coef,
                                                                reward_coef=self.reward_coef,
                                                                return_compas=True)
        compas_clone, will_clone = calculate_clone(ends=env_states.ends, eps=self.eps,
                                                   virtual_rewards=virtual_rewards)
        dif_dist = env_states.observs[compas_dist] - env_states.observs
        dif_clone = env_states.observs[compas_clone] - env_states.observs
        action_no_clone = dif_dist * self.random_state.normal(loc=self.loc, scale=self.scale)
        actions = np.where(will_clone, dif_clone, action_no_clone)
        return actions


class BestCompasJump(RandomNormal):

    def __init__(self, dist_coef: float = 1.0, reward_coef: float = 1.0, eps=1e-8,
                 mutation: float = 0.5, recombination: float = 0.7, *args,  **kwargs):
        super(BestCompasJump, self).__init__(*args, **kwargs)
        self.dist_coef = dist_coef
        self.reward_coef = reward_coef
        self.eps = eps
        self.mutation = mutation
        self.recombination = recombination

    def calculate(
            self,
            batch_size: int = None,
            model_states: States = None,
            env_states: States = None,
            walkers_states: "StatesWalkers" = None,
    ) -> np.ndarray:
        """
        Calculate the target time step values.

        Args:
            batch_size: Number of new points to the sampled.
            model_states: States corresponding to the model data.
            env_states: States corresponding to the environment data.
            walkers_states: States corresponding to the walkers data.

        Returns:
            Array containing the target time step.

        """
        distance = np.linalg.norm(env_states.observs - walkers_states.best_found, axis=1)
        distance_norm = relativize(distance.flatten())
        rewards_norm = relativize(env_states.rewards)

        virtual_rewards = distance_norm ** self.dist_coef * rewards_norm ** self.reward_coef
        compas_clone, will_clone = calculate_clone(ends=env_states.ends, eps=self.eps,
                                                   virtual_rewards=virtual_rewards)
        dif_best = env_states.observs - walkers_states.best_found
        action_no_clone = dif_best + self.random_state.normal(loc=self.loc, scale=self.scale)
        rands = np.random.random(env_states.observs.shape)
        grad = self.recombination * (env_states.observs - env_states.observs[compas_clone])
        proposal = walkers_states.best_found + grad
        best_mutations = np.where(rands < self.mutation, env_states.observs, proposal).copy()
        best_actions = best_mutations - env_states.observs
        actions = np.where(will_clone, best_actions, action_no_clone)
        actions = self.bounds.clip(actions) if self.bounds is not None else actions
        return actions