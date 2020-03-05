import numpy

from fragile.core.walkers import Walkers, StatesWalkers
from fragile.core.utils import float_type, relativize, statistics_from_array


class AmplitudeStates(StatesWalkers):
    def __init__(self, *args, **kwargs):
        super(AmplitudeStates, self).__init__(*args, *kwargs)
        self.mutual_reward = None
        self.vectors = None

    def get_params_dict(self) -> dict:
        params = super(AmplitudeStates, self).get_params_dict()
        new_params = {"mutual_reward": {"dtype": float_type}}
        params.update(new_params)
        return params

    def reset(self):
        super(AmplitudeStates, self).reset()
        self.update(mutual_reward=numpy.ones(self.n, dtype=float_type))


class AmplitudeWalkers(Walkers):
    STATE_CLASS = AmplitudeStates

    def calculate_virtual_reward(self):
        rewards = -1 * self.states.cum_rewards if self.minimize else self.states.cum_rewards
        processed_rewards = relativize(rewards)
        score_reward = processed_rewards ** self.reward_scale
        reward_prob = score_reward / score_reward.sum()
        score_dist = self.states.distances ** self.dist_scale
        dist_prob = score_dist / score_dist.sum()
        vectors = numpy.vstack([score_dist.reshape(-1, 1), score_reward.reshape(-1, 1)])
        virt_rw = 2 - dist_prob ** reward_prob
        total_entropy = numpy.prod(virt_rw)
        self._min_entropy = numpy.prod(2 - reward_prob ** reward_prob)
        self.efficiency = self._min_entropy / total_entropy
        self.update_states(virtual_rewards=virt_rw, processed_rewards=processed_rewards)
        if self.critic is not None:
            self.critic.calculate(
                walkers_states=self.states,
                model_states=self.model_states,
                env_states=self.env_states,
            )
            vectors = numpy.vstack([vectors, self.states.critic_score.reshape(-1, 1)])
        amplitude_sq = numpy.linalg.norm(vectors, axis=1) ** 2
        self.states.update(virtual_rewards=amplitude_sq, vectors=vectors)

    def update_clone_probs(self):
        """
        Calculate the new probability of cloning for each walker.

        Updates the internal state with both the probability of cloning and the index of the
        randomly chosen companions that were selected to compare the virtual rewards.
        """
        all_virtual_rewards_are_equal = (
            self.states.virtual_rewards == self.states.virtual_rewards[0]
        ).all()
        if all_virtual_rewards_are_equal:
            clone_probs = numpy.zeros(self.n, dtype=float_type)
            compas_ix = numpy.arange(self.n)
        else:
            compas_ix = self.get_alive_compas()
            compa_vectors = self.states.vectors[compas_ix]
            mutual_reward = numpy.linalg.norm(compa_vectors + self.states.vectors, axis=1) ** 2
            virtual_reward = self.states.virtual_rewards

            clone_probs = (virtual_reward - mutual_reward) / virtual_reward
            freeze_probs = (mutual_reward - virtual_reward) / mutual_reward

        self.update_states(clone_probs=clone_probs, compas_clone=compas_ix)
