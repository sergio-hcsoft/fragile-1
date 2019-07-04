from functools import partial
import multiprocessing

import numpy as np

__pool__ = multiprocessing.Pool()


def split_similar_chunks(vector: list, n_chunks: int):
    chunk_size = int(np.ceil(len(vector) / n_chunks))
    for i in range(0, len(vector), chunk_size):
        yield vector[i : i + chunk_size]


def recover_chunks(x):
    return x


def apply_multiprocessing(function, values, n_chunks: int = None, **kwargs):
    if kwargs:
        function = partial(func=function, **kwargs)
    n_chunks = n_chunks if n_chunks is not None else multiprocessing.cpu_count()
    result = __pool__.map(function, values, chunksize=n_chunks)
    return result


def apply_ray(function, values, n_chunks: int = None, **kwargs):
    if kwargs:
        function = partial(func=function, **kwargs)
    n_chunks = n_chunks if n_chunks is not None else multiprocessing.cpu_count()
    chunks = split_similar_chunks(values, n_chunks=n_chunks)
    # Execute remote function in ray
    batched_result = __pool__.map(function, chunksize=n_chunks)
    recovered = recover_chunks(batched_result)
    return recovered
