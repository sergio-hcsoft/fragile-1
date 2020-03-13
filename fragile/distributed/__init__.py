"""Module that includes scalable search algorithms."""
import sys

try:
    from fragile.distributed.ray.export_swarm import DistributedExport
except (ImportError, ModuleNotFoundError) as e:
    if sys.version_info == (3, 7):
        raise e
