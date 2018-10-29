import gym
from fragile.base_classes import BaseEnvironment

class CartPole(BaseEnvironment):

    def __init__(self):
        self._env = gym.make("CartPole-v0")

    @property
    def n_actions(self):
        return self._env.action_space.n

    def step(self, actions, env_states):
        for s, a in zip(env_states, actions):
            
    def reset(self, batch_size: int = 1):
        raise NotImplementedError