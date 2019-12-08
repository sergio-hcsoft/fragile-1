import numpy as np
from fragile.core.states import States
from fragile.core.models import BinarySwap


class EarthSampler(BinarySwap):
    def __init__(
        self, force_name: list = None, force_lag: list = None, names: list = None, *args, **kwargs
    ):
        super(EarthSampler, self).__init__(*args, **kwargs)
        self.names = names if names is not None else []
        self.force_name = force_name if force_name is not None else []
        self.force_mask = self._create_force_mask(self.force_lag)
        self.force_lag = force_lag if force_lag is not None else []
        self._name_masks = self._create_names_mask()

    def sample(
        self, env_states: States = None, batch_size: int = 1, model_states: States = None, **kwargs
    ) -> States:
        states = super(EarthSampler, self).sample(
            env_states=env_states, batch_size=batch_size, model_states=model_states, **kwargs
        )
        actions = states.actions.astype(bool)
        actions = np.logical_or(actions, np.tile(self.force_mask, (actions.shape[0, 1])))
        actions = self._enforce_soft_constraints(actions)
        states.update(actions=actions)

    def _create_force_mask(self, names):
        return np.array([n in names for n in self.names]).astype(bool)

    def _create_names_mask(self):
        return {n: np.array([n in t for t in self.names]).astype(bool) for n in self.force_name}

    def _enforce_soft_constraints(self, actions):
        for name, mask in self._name_masks.items():
            actions = self._fix_points(actions, mask)
        return actions

    @staticmethod
    def _fix_points(points, mask):
        need_fix = np.logical_not(np.logical_and(points, mask).any(axis=1))

        def new_point_in_mask(mask):
            points = np.zeros_like(mask)
            ix = np.random.choice(np.arange(len(mask))[mask])
            points[ix] = 1
            return points

        fixes = np.array([new_point_in_mask(mask) for _ in range(need_fix.sum())])
        if len(fixes) > 0:
            points[need_fix] = np.logical_or(points[need_fix], fixes)
        return points
