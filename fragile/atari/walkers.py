from fragile.core.walkers import Walkers


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
