import atexit
import warnings

warnings.filterwarnings("ignore")
import multiprocessing
import sys
import traceback
from typing import Callable

import numpy as np
import ray

from fragile.core.states import States
from fragile.optimize.env import Function as SequentialFunction


def split_similar_chunks(vector: list, n_chunks: int):
    chunk_size = int(np.ceil(len(vector) / n_chunks))
    for i in range(0, len(vector), chunk_size):
        yield vector[i : i + chunk_size]


@ray.remote
class RemoteFunction:
    def __init__(self, env_callable: Callable):
        self.function = env_callable().function

    def function(self, points: np.ndarray):
        return self.function(points)


class ExternalProcess(object):
    """
    Step environment in a separate process for lock free paralellism.
    The environment will be created in the external process by calling the
    specified callable. This can be an environment class, or a function
    creating the environment and potentially wrapping it. The returned
    environment should not access global variables.

    Args:
      constructor: Callable that creates and returns an OpenAI gym environment.

    Attributes:
      observation_space: The cached observation space of the environment.
      action_space: The cached action space of the environment.

    ..notes:
        This is mostly a copy paste from
        https://github.com/tensorflow/agents/blob/master/agents/tools/wrappers.py,
        but it lets us set and read the environment state.

    """

    # Message types for communication via the pipe.
    _ACCESS = 1
    _CALL = 2
    _RESULT = 3
    _EXCEPTION = 4
    _CLOSE = 5

    def __init__(self, constructor):

        self._conn, conn = multiprocessing.Pipe()
        self._process = multiprocessing.Process(target=self._worker, args=(constructor, conn))
        atexit.register(self.close)
        self._process.start()
        self._observ_space = None
        self._action_space = None

    def __getattr__(self, name):
        """Request an attribute from the environment.
        Note that this involves communication with the external process, so it can
        be slow.

        Args:
          name: Attribute to access.

        Returns:
          Value of the attribute.
        """
        self._conn.send((self._ACCESS, name))
        return self._receive()

    def call(self, name, *args, **kwargs):
        """Asynchronously call a method of the external environment.

        Args:
          name: Name of the method to call.
          *args: Positional arguments to forward to the method.
          **kwargs: Keyword arguments to forward to the method.

        Returns:
          Promise object that blocks and provides the return value when called.
        """
        payload = name, args, kwargs
        self._conn.send((self._CALL, payload))
        return self._receive

    def close(self):
        """Send a close message to the external process and join it."""
        try:
            self._conn.send((self._CLOSE, None))
            self._conn.close()
        except IOError:
            # The connection was already closed.
            pass
        self._process.join()

    def step_batch(self, points: np.ndarray, blocking: bool = False):
        """
        Vectorized version of the `step` method. It allows to step a vector of
        states and actions. The signature and behaviour is the same as `step`, but taking
        a list of states, actions and n_repeat_actions as input.

        Args:
           points: Numpy array containing the input of the function. \
                   It expects shape = (batch_size, 1).
           blocking: If True, execute sequentially.
        Returns:
          if states is None returns (observs, rewards, ends, infos)
          else returns(new_states, observs, rewards, ends, infos)
        """
        promise = self.call("function", points)
        if blocking:
            return promise()
        else:
            return promise

    def _receive(self):
        """Wait for a message from the worker process and return its payload.

        Raises:
          Exception: An exception was raised inside the worker process.
          KeyError: The received message is of an unknown type.

        Returns:
          Payload object of the message.
        """
        message, payload = self._conn.recv()
        # Re-raise exceptions in the main process.
        if message == self._EXCEPTION:
            stacktrace = payload
            raise Exception(stacktrace)
        if message == self._RESULT:
            return payload
        raise KeyError("Received message of unexpected type {}".format(message))

    def _worker(self, constructor, conn):
        """The process waits for actions and sends back environment results.
        Args:
          constructor: Constructor for the OpenAI Gym environment.
          conn: Connection for communication to the main process.
        Raises:
          KeyError: When receiving a message of unknown type.
        """
        try:
            env = constructor()
            while True:
                try:
                    # Only block for short times to have keyboard exceptions be raised.
                    if not conn.poll(0.1):
                        continue
                    message, payload = conn.recv()
                except (EOFError, KeyboardInterrupt):
                    break
                if message == self._ACCESS:
                    name = payload
                    result = getattr(env, name)
                    conn.send((self._RESULT, result))
                    continue
                if message == self._CALL:
                    name, args, kwargs = payload
                    result = getattr(env, name)(*args, **kwargs)
                    conn.send((self._RESULT, result))
                    continue
                if message == self._CLOSE:
                    assert payload is None
                    break
                raise KeyError("Received message of unknown type {}".format(message))
        except Exception:  # pylint: disable=broad-except
            stacktrace = "".join(traceback.format_exception(*sys.exc_info()))
            conn.send((self._EXCEPTION, stacktrace))
            conn.close()


class BatchEnv(object):
    """Combine multiple environments to step them in batch.
    It is mostly a copy paste from
    https://github.com/tensorflow/agents/blob/master/agents/tools/wrappers.py
    that also allows to set and get the states.

    To step environments in parallel, environments must support a
        `blocking=False` argument to their step and reset functions that makes them
        return callables instead to receive the result at a later time.

    Args:
      envs: List of environments.
      blocking: Step environments after another rather than in parallel.

    Raises:
      ValueError: Environments have different observation or action spaces.
    """

    def __init__(self, envs, blocking):
        self._envs = envs
        self._blocking = blocking

    def __len__(self):
        """Number of combined environments."""
        return len(self._envs)

    def __getitem__(self, index):
        """Access an underlying environment by index."""
        return self._envs[index]

    def __getattr__(self, name):
        """Forward unimplemented attributes to one of the original environments.

        Args:
          name: Attribute that was accessed.

        Returns:
          Value behind the attribute name one of the wrapped environments.
        """
        return getattr(self._envs[0], name)

    def _make_transitions(self, points):
        chunks = len(self._envs)
        states_chunk = split_similar_chunks(points, n_chunks=chunks)
        results = [
            env.step_batch(states_batch) for env, states_batch in zip(self._envs, states_chunk)
        ]
        rewards = [result if self._blocking else result() for result in results]
        return rewards

    def step_batch(self, points):
        """Forward a batch of actions to the wrapped environments.
        Args:
          actions: Batched action to apply to the environment.
          states: States to be stepped. If None, act on current state.
          n_repeat_action: Number of consecutive times the action will be applied.

        Raises:
          ValueError: Invalid actions.

        Returns:
          Batch of observations, rewards, and done flags.
        """
        rewards = self._make_transitions(points)
        try:
            rewards = np.stack(rewards)
        except BaseException as e:  # Lets be overconfident for once TODO: remove this.
            for obs in rewards:
                print(obs.shape)
        return np.concatenate([r.flatten() for r in rewards])

    def close(self):
        """Send close messages to the external process and join them."""
        for env in self._envs:
            if hasattr(env, "close"):
                env.close()


class ParallelFunction:
    """
    Wrap any environment to be stepped in parallel when step_batch is called.

    """

    def __init__(self, env_callable, n_workers: int = 8, blocking: bool = False):
        self._env = env_callable()
        envs = [ExternalProcess(constructor=env_callable) for _ in range(n_workers)]
        self._batch_env = BatchEnv(envs, blocking)

    def __getattr__(self, item):
        return getattr(self._env, item)

    def step_batch(self, points: np.ndarray):
        """
        Vectorized version of the `step` method. It allows to step a vector of
        states and actions. The signature and behaviour is the same as `step`,
        but taking a list of states, actions and n_repeat_actions as input.

        Args:
            actions: Iterable containing the different actions to be applied.
            states: Iterable containing the different states to be set.
            n_repeat_action: int or array containing the frameskips that will be applied.

        Returns:
            if states is None returns (observs, rewards, ends, infos) else (new_states,
            observs, rewards, ends, infos)

        """
        return self._batch_env.step_batch(points=points.copy())


class Function(SequentialFunction):
    def __init__(self, env_callable: Callable, n_workers: int = 1, blocking: bool = False):
        self.n_workers = n_workers
        self.blocking = blocking
        self.parallel_function = ParallelFunction(
            env_callable=env_callable, n_workers=n_workers, blocking=blocking
        )
        self.local_function = env_callable()

    def __getattr__(self, item):
        return getattr(self.local_function, item)

    def step(self, model_states: States, env_states: States) -> States:
        """
        Sets the environment to the target states by applying the specified actions an arbitrary
        number of time steps.

        Args:
            model_states: States corresponding to the model data.
            env_states: States class containing the state data to be set on the Environment.

        Returns:
            States containing the information that describes the new state of the Environment.
        """
        new_points = (
            # model_states.actions * model_states.dt.reshape(env_states.n, -1) + env_states.observs
            model_states.actions
            + env_states.observs
        )
        ends = self.calculate_end(points=new_points)
        rewards = self.parallel_function.step_batch(new_points)

        last_states = self._get_new_states(new_points, rewards, ends, model_states.n)
        return last_states

    def __parallel_function(self, points):
        reward_ids = [
            env.function.remote(p)
            for env, p in zip(self.workers, split_similar_chunks(points, self.n_workers))
        ]
        rewards = ray.get(reward_ids)
        # rewards = self.pool.map(self.local_function.function,
        #                        split_similar_chunks(points, self.n_workers))
        return np.concatenate([r.flatten() for r in rewards])
