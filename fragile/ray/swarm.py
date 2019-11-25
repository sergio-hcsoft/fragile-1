from collections import deque
import copy
from typing import Callable
import warnings
warnings.filterwarnings("ignore")

import holoviews as hv
from holoviews.streams import Pipe
import numpy as np
import pandas as pd
import ray
from streamz import Stream
from streamz.dataframe import DataFrame

from fragile.core.swarm import Swarm
from fragile.core.utils import float_type, relativize


@ray.remote
class RemoteSwarm:

    def __init__(self, swarm: Callable, n_comp_add: int = 2):
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
        return self.get_best()

    def run_step(self):
        return self.swarm.run_step()

    def get_best(self):
        try:
            best_ix = self.swarm.walkers.states.cum_rewards.argmax()
        except:
            return None, None, None
        state = self.swarm.walkers.env_states.states[best_ix].copy()
        obs = self.swarm.walkers.env_states.observs[best_ix].copy()
        reward = self.swarm.walkers.states.cum_rewards[best_ix].copy()
        return (state, obs, reward)

    def add_walker(self, best):
        (state, obs, reward) = best
        if state is not None:
            x = self.swarm.walkers.env_states.states[0]
            try:
                assert np.prod(state.shape) == np.prod(x.shape), "%s %s" % (state.shape, x.shape)
            except Exception as e:
                for i, s in enumerate(state.tolist()):
                    if s != x[i]:
                        print("FAILED i %s, %s x: %s" % (i, s, x[i]))
                raise e
            self._clone_to_walker(state, obs, reward)
            self.swarm.walkers.update_best()

    def get_end_condition(self):
        return self.swarm.calculate_end_condition()

    def _clone_to_walker(self, state, obs, reward):
        # Virtual reward with respect to the new state
        indexes = np.random.choice(np.arange(self.swarm.walkers.n), size=self.n_comp_add)
        n_walkers = len(indexes)
        assert n_walkers == self.n_comp_add
        w_rewards = self.swarm.walkers.states.cum_rewards[indexes]
        walkers_obs = self.swarm.walkers.env_states.observs[indexes].reshape(n_walkers, -1)
        distances = np.linalg.norm(walkers_obs - obs.reshape(1, -1), axis=1)
        distances = relativize(distances.flatten()) ** self.swarm.walkers.dist_scale
        distances = distances / distances.sum()
        rewards = relativize(np.concatenate([w_rewards,
                                             [reward]])) ** self.swarm.walkers.reward_scale
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
                                                                    type(x), x)
            print((msg % data) + (msg_2 % data_2) + msg_3)
            raise e


@ray.remote
class _ParamServer:

    def __init__(self, maxlen: int=100):
        self._maxlen = maxlen
        self.states = deque([], self._maxlen)
        self.observs = deque([], self._maxlen)
        self.rewards = deque([], self._maxlen)

    def get_best(self):
        best_ix = np.argmax(np.array(self.rewards))
        return self.states[best_ix], self.observs[best_ix], self.rewards[best_ix]

    def reset(self):
        self.states = deque([], self._maxlen)
        self.observs = deque([], self._maxlen)
        self.rewards = deque([], self._maxlen)

    def exchange_walker(self, walker):
        self.append_walker(walker)
        return self.get_walker()

    def append_walker(self, walker):
        state, obs, reward = walker
        self.states.append(copy.deepcopy(state))
        self.observs.append(copy.deepcopy(obs))
        self.rewards.append(reward)

    def get_walker(self):
        if len(self.states) == 0:
            return None, None, None
        ix = np.random.choice(np.arange(len(self.states)))
        state = copy.deepcopy(self.states[ix])
        obs = copy.deepcopy(self.observs[ix])
        reward = float(self.rewards[ix])
        return state, obs, reward

@ray.remote
class ParamServer:

    def __init__(self, maxlen: int = 20):
        self._maxlen = maxlen
        self.buffer = deque([], self._maxlen)

    def get_best(self):
        if len(self.buffer) == 0:
            return None, None, None
        best_ix = np.argmax([r for _, _, r in self.buffer])
        return self.buffer[best_ix]

    def reset(self):
        self.buffer = deque([], self._maxlen)

    def exchange_walker(self, walker):
        self.append_walker(walker)
        return self.get_walker()

    def append_walker(self, walker):
        self.buffer.append(copy.deepcopy(walker))

    def get_walker(self):
        if len(self.buffer) == 0:
            return None, None, None
        ix = np.random.choice(np.arange(len(self.buffer)))
        return copy.deepcopy(self.buffer[ix])


class DistributedSwarm:

    def __init__(self, swarm: Callable,
                 n_swarms: int,
                 max_iters_ray: int=10,
                 log_every: int=100, n_com_add: int=5):
        self.n_swarms = n_swarms
        self.log_every = log_every
        self.param_server = ParamServer.remote()
        ray.get(self.param_server.reset.remote())
        self.swarms = [RemoteSwarm.remote(copy.copy(swarm), int(n_com_add))
                       for _ in range(self.n_swarms)]
        self.max_iters_ray = max_iters_ray
        self.frame_pipe: Pipe = None
        self.stream = None
        self.buffer_df = None
        self.score_dmap = None
        self.frame_dmap = None
        self.init_plot()
        self.n_iters = 0

    def init_plot(self):
        self.frame_pipe = Pipe(data=[])
        self.frame_dmap = hv.DynamicMap(hv.RGB, streams=[self.frame_pipe])
        self.frame_dmap = self.frame_dmap.opts(xlim=(-0.5, 0.5), ylim=(-0.5, 0.5),
                                               xaxis=None, yaxis=None, title="Game screen")
        example = pd.DataFrame({"reward": []})
        self.stream = Stream()
        self.buffer_df = DataFrame(stream=self.stream,
                                   example=example)
        self.score_dmap = self.buffer_df.hvplot(y=["reward"]).opts(height=200, width=500,
                                                                   title="Game score")

    def plot(self):
       return self.frame_dmap + self.score_dmap

    def stream_progress(self, observation, reward):
        example = pd.DataFrame({"reward": [reward]}, index=[self.n_iters // self.n_swarms])
        self.stream.emit(example)
        obs = observation[:-3].reshape((210, 160, 3)).astype(np.uint8)
        self.frame_pipe.send(obs)

    def run_swarm(self):
        self.n_iters = 0
        best_ids = [s.reset.remote() for s in self.swarms]
        steps = {}
        for worker, best in zip(self.swarms, best_ids):
            steps[worker.make_iteration.remote(best)] = worker

        for i in range(self.max_iters_ray * len(self.swarms)):
            self.n_iters += 1
            ready_bests, _ = ray.wait(list(steps))
            ready_best_id = ready_bests[0]
            worker = steps.pop(ready_best_id)
            new_best = self.param_server.exchange_walker.remote(ready_best_id)
            steps[worker.make_iteration.remote(new_best)] = worker

            if i % (self.log_every * len(self.swarms)) == 0:
                id_, _ = (ray.wait([self.param_server.get_best.remote()]))
                (state, best_obs, best_reward) = ray.get(id_)[0]
                if state is not None:
                    self.stream_progress(best_obs, best_reward)
                else:
                    print("skipping, not ready")


