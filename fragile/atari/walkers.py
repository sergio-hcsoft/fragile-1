import numpy as np

from fragile.core.walkers import Walkers
from fragile.core.utils import relativize

import line_profiler

class AtariWalkers(Walkers):
    """
    This class is in charge of performing all the mathematical operations involved in evolving a \
    cloud of walkers.

    """

    def __init__(self, max_reward: int=None, *args, **kwargs):
        super(AtariWalkers, self).__init__(*args, **kwargs)
        self.max_reward = max_reward

    def calculate_end_condition(self) -> bool:
        """
        Process data from the current state to decide if the iteration process should stop.

        Returns:
            Boolean indicating if the iteration process should be finished. True means \
            it should be stopped, and False means it should continue.

        """
        end = super(AtariWalkers, self).calculate_end_condition()
        return self.env_states.game_ends.all() and end or self.states.cum_rewards > self.max_reward


class MontezumaWalkers(Walkers):

    #@profile
    def calculate_distances(self):
        """Calculate the corresponding distance function for each state with \
        respect to another state chosen at random.

        The internal state is update with the relativized distance values.
        """
        compas_ix = np.random.permutation(np.arange(self.n))  # self.get_alive_compas()
        #obs = self.env_states.observs.reshape(self.n, -1)[:, -3:]
        rams = self.env_states.states.reshape(self.n, -1)[:, :-12].astype(np.uint8)#[:, :-15]
        vec = (rams - rams[compas_ix])
        dist_ram = np.linalg.norm(vec, axis=1).flatten()
        #dist_ram = distance(rams, compas_ix)
        #dist_pos = np.linalg.norm(obs[:, :-1] - obs[compas_ix, :-1], axis=1).flatten()
        #dist_room = np.linalg.norm(obs[:, -1] - obs[compas_ix, -1]).flatten()
        #distances = relativize(dist_pos) * relativize(dist_room) * relativize(dist_ram)
        distances = relativize(dist_ram)
        self.update_states(distances=distances, compas_dist=compas_ix)
