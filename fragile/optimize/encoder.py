import numpy as np
import torch

from fragile.core.utils import device, to_numpy, to_tensor


def unique_columns2(data):

    dt = np.dtype((np.void, data.dtype.itemsize * data.shape[0]))
    dataf = np.asfortranarray(data).view(dt)
    u, uind = np.unique(dataf, return_inverse=True)
    u = u.view(data.dtype).reshape(-1, data.shape[0]).T
    return (u, uind)


class Vector:
    def __init__(
        self,
        origin: torch.Tensor = None,
        end: torch.Tensor = None,
        front_data: torch.Tensor = 0,
        back_data: torch.Tensor = 0,
        timeout: int = 1e100,
        timeout_threshold: int = 100000,
    ):
        self.origin = origin
        self.end = end
        self.base = end - origin
        self.front_data = front_data
        self.back_data = back_data
        self._age = 0
        self.timeout = timeout
        self.last_regions = []
        self._pos_track = True
        self._neg_track = True
        self.timeout_threshold = timeout_threshold

    def __repr__(self):
        text = "Origin: {}, End: {} Age: {} Outdated: {}\n".format(
            self.origin, self.end, self._age, self.is_outdated()
        )
        return text

    def __hash__(self):
        return hash(str(to_numpy(torch.cat([self.origin, self.end]))))

    def scalar_product(self, other: torch.Tensor):
        return torch.dot(self.base, self.end - other)

    def assign_region(self, other: torch.Tensor) -> int:

        region = 1 if self.scalar_product(other=other) > 0 else 0
        if len(self.last_regions) < self.timeout:
            self.last_regions.append(region)
        else:
            self.last_regions[:-1] = self.last_regions[1:]
            self.last_regions[-1] = region
        return region

    def get_data(
        self, other, value: torch.Tensor = 0, return_region: bool = False
    ) -> torch.Tensor:

        region = self.assign_region(other=other)
        if region == 1:
            self.front_data = self.front_data + value
            return (self.front_data, region) if return_region else self.front_data
        else:
            self.back_data = self.back_data + value
            return (self.back_data, region) if return_region else self.back_data

    def decode_list(self, points: list):
        return [self.assign_region(p) for p in points]

    def is_outdated(self):
        if len(self.last_regions) > self.timeout_threshold:
            return all(self.last_regions) or not any(self.last_regions)
        else:
            return False


def diversity_score(x, total=None):
    n_different_rows = np.unique(to_numpy(x), axis=0).shape[0]
    return n_different_rows if total is None else float(n_different_rows / total)


class Encoder:
    def __init__(self, n_vectors: int, timeout: int = 1e100, timeout_threshold: int = 100):
        self.n_vectors = n_vectors
        self.timeout = timeout
        self.timeout_threshold = timeout_threshold
        self._vectors = []
        self._last_encoded = None

    @property
    def vectors(self):
        return self._vectors

    def __repr__(self):
        div_score = -1 if self._last_encoded is None else diversity_score(self._last_encoded, 1)
        den = float(self._last_encoded.shape[0]) if self._last_encoded is not None else 1.0
        text = (
            "Encoder with {} vectors, score {:.3f}, {} different hashes and {} "
            "available spaces\n".format(
                self.n_vectors,
                div_score / den,
                div_score,
                min(self.n_vectors - len(self), len(self.vectors)),
            )
        )
        return text + "".join(v.__repr__() for v in self.vectors)

    def __len__(self):
        return len(self._vectors)

    def __getitem__(self, item):
        return self.vectors[item]

    def reset(self):
        self._vectors = []

    def append(self, *args, **kwargs):
        kwargs["timeout"] = kwargs.get("timeout", self.timeout)
        kwargs["timeout_threshold"] = kwargs.get("timeout_threshold", self.timeout_threshold)
        vector = Vector(*args, **kwargs)
        self.append_vector(vector=vector)

    def append_vector(self, vector: Vector):
        if len(self) < self.n_vectors:
            self.vectors.append(vector)
        else:
            self.vectors[:-1] = self.vectors[1:]
            self.vectors[-1] = vector

    def pct_different_hashes(self, points: torch.Tensor) -> float:
        x = self.encode(points)
        array = x.detach().cpu().numpy()
        return float(np.unique(array, axis=0).shape[0] / int(points.shape[0]))

    def is_valid_base(self, vector: [Vector, int], points: list):
        if isinstance(vector, int):
            vector = self[vector]
        binary = vector.decode_list(points)
        return not all(binary) and any(binary)

    def encode(self, points):
        values = torch.stack([self._encode_one(point=points[i]) for i in range(points.shape[0])])
        self._last_encoded = values
        return values

    def _encode_one(self, point):
        values = torch.tensor(
            [vector.assign_region(point) for vector in self.vectors], device=device
        )
        return values

    def remove_duplicates(self):
        hashdict = {hash(v): v for v in self.vectors}
        self._vectors = [v for _, v in hashdict.items()]

    def remove_bases(self, points):

        # self._vectors = [v for v in self.vectors if not v.is_outdated()]
        self._vectors = [v for v in self.vectors if self.is_valid_base(vector=v, points=points)]
        self.remove_duplicates()

    def update_bases(self, vectors):
        n_vec = len(vectors)
        available_spaces = min(self.n_vectors - len(self), n_vec)

        if available_spaces > 0:
            chosen_vectors = np.random.choice(np.arange(n_vec), available_spaces, replace=False)
            for ix in chosen_vectors:
                origin, end = vectors[ix]
                vec = Vector(
                    origin=origin.detach().clone(), end=end.detach().clone(), timeout=self.timeout
                )
                self.append_vector(vec)
