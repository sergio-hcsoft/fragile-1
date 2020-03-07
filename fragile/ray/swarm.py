from collections import deque
import copy
from typing import Callable
import warnings

warnings.filterwarnings("ignore")

import holoviews as hv
from holoviews.streams import Pipe
import hvplot.pandas
import hvplot.streamz
import numpy as np
import pandas as pd
import ray
from streamz import Stream
from streamz.dataframe import DataFrame

from fragile.core.swarm import Swarm
from fragile.core.utils import float_type, relativize


@ray.remote
class RemoteSwarm:
    def __init__(self, swarm: Callable, n_comp_add: int = 2, minimize: bool = False):
        self.minimize = minimize
        self._swarm_callable = swarm
        self.swarm: Swarm = None
        self.n_comp_add = n_comp_add
        self.init_swarm()

    def init_swarm(self):
        self.swarm = self._swarm_callable()
        self.swarm.reset()

    def make_iteration(self, best_other):
        self.add_walker(best_other)
        self.run_step()
        return self.get_best()

    def reset(self):
        self.swarm.reset()
        return self.get_best(), self.get_best()

    def run_step(self):
        return self.swarm.run_step()

    def get_best(self):
        try:
            best_ix = (
                self.swarm.walkers.states.cum_rewards.argmin()
                if self.minimize
                else self.swarm.walkers.states.cum_rewards.argmax()
            )
        except:
            return None, None, np.inf if self.minimize else -np.inf
        state = self.swarm.walkers.env_states.states[best_ix].copy()
        obs = self.swarm.walkers.env_states.observs[best_ix].copy()
        reward = self.swarm.walkers.states.cum_rewards[best_ix].copy()
        return (state, obs, reward)

    def add_walker(self, walkers):
        if walkers[0] is None or walkers is None:
            return
        try:
            best_state, best_obs, best_rew = walkers[0]
        except Exception:
            print("WALKERS", walkers)
            raise Exception(str(walkers))

        update_reward = (
            best_rew < self.swarm.walkers.states.best_reward
            if self.minimize
            else best_rew > self.swarm.walkers.states.best_reward
        )
        if update_reward:
            self.swarm.walkers.states.update(
                best_reward=best_rew, best_state=best_state, best_obs=best_obs
            )
            self.swarm.walkers.fix_best()
        else:
            self._clone_to_walker(best_state, best_obs, best_rew)

        (state, obs, reward) = walkers[1]
        if state is not None:
            self._clone_to_walker(state, obs, reward)
            self.swarm.walkers.update_best()

    def get_end_condition(self):
        return self.swarm.calculate_end_condition()

    def _clone_to_walker(self, state, obs, reward):
        if obs is None or state is None:
            return
        # Virtual reward with respect to the new state
        indexes = np.random.choice(np.arange(self.swarm.walkers.n), size=self.n_comp_add)
        n_walkers = len(indexes)
        assert n_walkers == self.n_comp_add
        w_rewards = self.swarm.walkers.states.cum_rewards[indexes]
        walkers_obs = self.swarm.walkers.env_states.observs[indexes].reshape(n_walkers, -1)
        distances = np.linalg.norm(walkers_obs - obs.reshape(1, -1), axis=1)
        distances = relativize(distances.flatten()) ** self.swarm.walkers.dist_scale
        distances = distances / distances.sum()
        rewards = (
            relativize(np.concatenate([w_rewards, [reward]])) ** self.swarm.walkers.reward_scale
        )
        rewards = rewards / rewards.sum()
        w_virt_rew = 2 - distances ** rewards[:-1]
        other_ix = np.random.permutation(np.arange(n_walkers))
        other_virt_rew = 2 - distances[other_ix] ** rewards[-1]
        # Clone probabilities with respect to new state
        all_virtual_rewards_are_equal = (w_virt_rew == other_virt_rew).all()
        if all_virtual_rewards_are_equal:
            clone_probs = np.zeros(n_walkers, dtype=float_type)
        else:
            clone_probs = (other_virt_rew - w_virt_rew) / w_virt_rew
            clone_probs = np.sqrt(np.clip(clone_probs, 0, 1.1))
        # Clone the new state to the selected walkers
        will_clone = clone_probs > self.swarm.walkers.random_state.random_sample(n_walkers)
        if will_clone.sum() == 0:
            return
        new_rewards = np.ones(n_walkers)[will_clone].copy() * reward
        try:
            self.swarm.walkers.states.cum_rewards[indexes][will_clone] = new_rewards
            for ix, wc in zip(indexes, will_clone):
                if wc:
                    self.swarm.walkers.env_states.states[ix] = copy.deepcopy(state)
                    self.swarm.walkers.env_states.observs[ix] = copy.deepcopy(obs)
            self.swarm.walkers.update_best()
        except Exception as e:
            return
            orig_states = self.swarm.walkers.env_states.states
            msg = "indexes: %s will_clone: %s new_states: %s states shape: %s\n"
            data = (indexes, will_clone, [], orig_states.shape)
            msg_2 = "clone_probs: %s rewards: %s reward: %s state: %s\n"
            data_2 = (clone_probs, rewards, reward, state)
            x = orig_states[indexes][will_clone]
            msg_3 = "will_clone shape: %s clone_probs shape: %s SHAPE: %s DATA: %s" % (
                will_clone.shape,
                clone_probs.shape,
                type(x),
                x,
            )
            print((msg % data) + (msg_2 % data_2) + msg_3)
            raise e


@ray.remote
class ParamServer:
    def __init__(self, maxlen: int = 20, minimize: bool = False):
        self._maxlen = maxlen
        self.minimize = minimize
        self.buffer = deque([], self._maxlen)
        self.best = (None, None, np.inf if self.minimize else -np.inf)

    def get_best(self):
        return self.best

    def reset(self):
        self.buffer = deque([], self._maxlen)
        self.best = (None, None, np.inf if self.minimize else -np.inf)

    def exchange_walker(self, walker):
        if walker is not None and walker[0] is not None:
            self.append_walker(walker)
            return self.get_walker()

    def append_walker(self, walker):
        self.buffer.append(copy.deepcopy(walker))
        self._update_best()

    def get_walker(self):
        if len(self.buffer) == 0:
            return (
                (None, None, np.inf if self.minimize else -np.inf),
                (None, None, np.inf if self.minimize else -np.inf),
            )
        ix = np.random.choice(np.arange(len(self.buffer)))
        return copy.deepcopy(self.best), copy.deepcopy(self.buffer[ix])

    def _update_best(self):
        rewards = [r for _, _, r in self.buffer]
        best_ix = np.argmin(rewards) if self.minimize else np.argmax(rewards)
        state, obs, reward = self.buffer[best_ix]
        if reward <= self.best[2] if self.minimize else reward >= self.best[2]:
            self.best = copy.deepcopy((state, obs, reward))


class DistributedSwarm:
    def __init__(
        self,
        swarm: Callable,
        n_swarms: int,
        n_param_servers: int,
        max_iters_ray: int = 10,
        log_every: int = 100,
        n_comp_add: int = 5,
        minimize: bool = False,
        ps_maxlen: int = 100,
        init_reward: float = None,
        log_reward: bool = False,
    ):
        self.n_swarms = n_swarms
        self.minimize = minimize
        self.log = log_reward
        self.init_reward = (
            init_reward if init_reward is not None else (np.inf if minimize else -np.inf)
        )
        self.log_every = log_every
        self.param_servers = [
            ParamServer.remote(minimize=minimize, maxlen=ps_maxlen) for _ in range(n_param_servers)
        ]
        self.swarms = [
            RemoteSwarm.remote(copy.copy(swarm), int(n_comp_add), minimize=minimize)
            for _ in range(self.n_swarms)
        ]
        self.max_iters_ray = max_iters_ray
        self.frame_pipe: Pipe = None
        self.stream = None
        self.buffer_df = None
        self.score_dmap = None
        self.frame_dmap = None
        self.init_plot()
        self.n_iters = 0
        self.best = (None, None, None)

    def init_plot(self):
        self.frame_pipe = Pipe(data=[])
        self.frame_dmap = hv.DynamicMap(hv.RGB, streams=[self.frame_pipe])
        self.frame_dmap = self.frame_dmap.opts(
            xlim=(-0.5, 0.5), ylim=(-0.5, 0.5), xaxis=None, yaxis=None, title="Game screen"
        )
        example = pd.DataFrame({"reward": []})
        self.stream = Stream()
        self.buffer_df = DataFrame(stream=self.stream, example=example)
        self.score_dmap = self.buffer_df.hvplot(y=["reward"]).opts(
            height=200, width=500, title="Game score"
        )

    def plot(self):
        return self.frame_dmap + self.score_dmap

    def stream_progress(self, state, observation, reward):
        example = pd.DataFrame({"reward": [reward]}, index=[self.n_iters // self.n_swarms])
        self.stream.emit(example)
        obs = observation.reshape((210, 160, 3)).astype(np.uint8)
        self.frame_pipe.send(obs)

    def run_swarm(self):
        self.n_iters = 0
        best_ids = [s.reset.remote() for s in self.swarms]
        steps = {}
        param_servers = deque([])
        for worker, best in zip(self.swarms, best_ids):
            steps[worker.make_iteration.remote(best)] = worker

        bests = []
        for ps, walker in zip(self.param_servers, list(steps.keys())[: len(self.param_servers)]):
            bests.append(ps.exchange_walker.remote(walker))
            param_servers.append(ps)
        ray.get(bests)

        for i in range(self.max_iters_ray * len(self.swarms)):
            self.n_iters += 1
            ready_bests, _ = ray.wait(list(steps))
            ready_best_id = ready_bests[0]
            worker = steps.pop(ready_best_id)
            ps = param_servers.popleft()

            new_best = ps.exchange_walker.remote(ready_best_id)
            param_servers.append(ps)
            steps[worker.make_iteration.remote(new_best)] = worker

            if i % (self.log_every * len(self.swarms)) == 0:
                id_, _ = ray.wait([param_servers[-1].get_best.remote()])
                (state, best_obs, best_reward) = ray.get(id_)[0]
                if state is not None:
                    self.best = (state, best_obs, float(best_reward))
                    if (
                        (best_reward > self.init_reward)
                        if self.minimize
                        else (best_reward < self.init_reward)
                    ):
                        best_reward = self.init_reward
                    best_reward = np.log(best_reward) if self.log else best_reward
                    self.stream_progress(state, best_obs, best_reward)
                else:
                    print("skipping, not ready")
