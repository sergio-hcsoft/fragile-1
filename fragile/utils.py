import copy
import numpy as np
import torch

device = "cpu" if not torch.cuda.is_available() else "cuda"


def params_to_tensors(param_dict, n_walkers: int):
    tensor_dict = {}
    copy_dict = copy.deepcopy(param_dict)
    for key, val in copy_dict.items():
        sizes = tuple([n_walkers]) + val["sizes"]
        del val["sizes"]
        tensor_dict[key] = torch.empty(sizes, **val)
    return tensor_dict


def relativize(x, device=device):
    std = x.std()
    if float(std) == 0:
        return torch.ones(len(x), device=device)
    standard = (x - x.mean()) / std
    standard[standard > 0] = torch.log(1.0 + standard[standard > 0]) + 1.0
    standard[standard <= 0] = torch.exp(standard[standard <= 0])
    return standard


def to_numpy(x: [np.ndarray, torch.Tensor, list]) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    elif isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    else:
        return np.ndarray(x)


def to_tensor(x: [torch.Tensor, np.ndarray, list],
              device=device, *args, **kwargs) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device)
    else:
        return torch.Tensor(x, device=device, *args, **kwargs)

