import torch
import numpy as np
from fragile.swarm import States
from fragile.base_classes import BaseModel


class RandomDiscrete(BaseModel):
    def __init__(self, n_actions: int, *args, **kwargs):
        super(RandomDiscrete, self).__init__(*args, **kwargs)
        self._n_actions = n_actions

    @property
    def n_actions(self):
        return self._n_actions

    def get_params_dict(self) -> dict:
        params = {
            "actions": {"sizes": tuple([1]), "dtype": torch.uint8},
            "init_actions": {"sizes": tuple([1]), "dtype": torch.uint8},
        }
        return params

    def reset(self, batch_size: int = 1, *args, **kwargs) -> tuple:
        states = States(state_dict=self.get_params_dict(), n_walkers=batch_size)
        actions = np.random.randint(0, self.n_actions, size=batch_size)
        return actions, states

    def predict(self, env_states, batch_size: int=1, model_states=None) -> np.ndarray:
        actions = np.random.randint(0, self.n_actions, size=len(env_states.rewards))
        return actions

    def actor_pred(self, *args, **kwargs):
        pass

    def critic_pred(self, *args, **kwargs):
        pass

    def world_emb_pred(self, *args, **kwargs):
        pass

    def simulation_pred(self, *args, **kwargs):
        pass

    def calculate_skipframe(self):
        pass
