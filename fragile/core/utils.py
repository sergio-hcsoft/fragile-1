from typing import Any, Callable, Dict, Generator, Tuple, Union

import numpy
from plangym.env import Environment, ParallelEnvironment


RANDOM_SEED = 160290
random_state = numpy.random.RandomState(seed=RANDOM_SEED)

float_type = numpy.float32
Scalar = Union[int, numpy.int, float, numpy.float]
StateDict = Dict[str, Dict[str, Any]]
DistanceFunction = Callable[[numpy.ndarray, numpy.ndarray], numpy.ndarray]

NUMPY_IGNORE_WARNINGS_PARAMS = {
    "divide": "ignore",
    "over": "ignore",
    "under": "ignore",
    "invalid": "ignore",
}


def running_in_ipython() -> bool:
    """Return ``True`` if the code is this function is being called from an IPython kernel."""
    try:
        from IPython import get_ipython

        return get_ipython() is not None
    except ImportError:
        return False


def get_plangym_env(swarm: "Swarm") -> Environment:  # noqa: F821
    """Return the :class:`plangym.Environment` of the target Swarm."""
    from fragile import core
    from fragile.atari import env

    valid_env_types = (core.DiscreteEnv, env.AtariEnv)
    if isinstance(swarm.env, valid_env_types):
        if not isinstance(swarm.env._env, Environment):
            raise TypeError("swarm.env needs to represent a `plangym.Environment`.")
    elif not isinstance(swarm.env, valid_env_types):
        raise TypeError("swarm.env needs to represent a `plangym.Environment`")
    plangym_env = swarm.env._env
    if isinstance(plangym_env, ParallelEnvironment):
        return plangym_env._env
    else:
        return plangym_env


def remove_notebook_margin(output_width_pct: int = 80):
    """Make the notebook output wider."""
    from IPython.core.display import HTML

    html = (
        "<style>"
        ".container { width:" + str(output_width_pct) + "% !important; }"
        ".input{ width:70% !important; }"
        ".text_cell{ width:70% !important;"
        " font-size: 16px;}"
        ".title {align:center !important;}"
        "</style>"
    )
    return HTML(html)


def hash_numpy(x: numpy.ndarray) -> int:
    """Return a value that uniquely identifies a numpy array."""
    return hash(x.tostring())


def resize_frame(
    frame: numpy.ndarray, width: int, height: int, mode: str = "RGB"
) -> numpy.ndarray:
    """
    Use PIL to resize an RGB frame to an specified height and width.

    Args:
        frame: Target numpy array representing the image that will be resized.
        width: Width of the resized image.
        height: Height of the resized image.
        mode: Passed to Image.convert.

    Returns:
        The resized frame that matches the provided width and height.

    """
    from PIL import Image

    frame = Image.fromarray(frame)
    frame = frame.convert(mode).resize(size=(width, height))
    return numpy.array(frame)


def statistics_from_array(x: numpy.ndarray):
    """Return the (mean, std, max, min) of an array."""
    try:
        return x.mean(), x.std(), x.max(), x.min()
    except AttributeError:
        return numpy.nan, numpy.nan, numpy.nan, numpy.nan


def similiar_chunks_indexes(n_values, n_chunks) -> Generator[Tuple[int, int], None, None]:
    """
    Return the indexes for splitting an array in similar chunks.

    Args:
        n_values: Length of the array that will be split.
        n_chunks: Number of similar chunks.

    Returns:
        Generator containing the indexes of every new chunk.

    """
    chunk_size = int(numpy.ceil(n_values / n_chunks))
    for i in range(0, n_values, chunk_size):
        yield i, i + chunk_size


def split_similar_chunks(
    vector: Union[list, numpy.ndarray], n_chunks: int
) -> Generator[Union[list, numpy.ndarray], None, None]:
    """
    Split an indexable object into similar chunks.

    Args:
        vector: Target object to be split.
        n_chunks: Number of similar chunks.

    Returns:
        Generator that returns the chunks created after splitting the target object.

    """
    for start, end in similiar_chunks_indexes(len(vector), n_chunks):
        yield vector[start:end]
