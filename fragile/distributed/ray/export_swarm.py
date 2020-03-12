from typing import Callable

import ray

from fragile.distributed.export_swarm import ExportedWalkers, ExportSwarm
from fragile.distributed.param_server import ParamServer


@ray.remote
class RemoteParamServer(ParamServer):
    def get_data(self, name):
        return getattr(self, name)


@ray.remote
class RemoteExportSwarm:
    def __init__(
        self,
        swarm: Callable,
        n_import: int = 2,
        n_export: int = 2,
        export_best: bool = True,
        import_best: bool = True,
        swarm_kwargs: dict = None,
    ):
        swarm_kwargs = swarm_kwargs if swarm_kwargs is not None else {}
        swarm = swarm(**swarm_kwargs)
        self.swarm = ExportSwarm(
            swarm=swarm,
            n_export=n_export,
            n_import=n_import,
            import_best=import_best,
            export_best=export_best,
        )

    def reset(self):
        self.swarm.reset()

    def get_empty_export_walkers(self):
        return ExportedWalkers(0)

    def run_exchange_step(self, walkers: ExportedWalkers) -> ExportedWalkers:
        return self.swarm.run_exchange_step(walkers)

    def get_data(self, name):
        if hasattr(self.swarm.walkers.states, name):
            return getattr(self.swarm.walkers.states, name)
        elif hasattr(self.swarm.walkers.env_states, name):
            return getattr(self.swarm.walkers.env_states, name)
        elif hasattr(self.swarm.walkers.model_states, name):
            return getattr(self.swarm.walkers.model_states, name)
        elif hasattr(self.swarm.walkers, name):
            return getattr(self.swarm.walkers, name)
        elif hasattr(self.swarm, name):
            return getattr(self.swarm, name)
        else:
            raise ValueError("%s is not an attribute of the states, swarm or walkers." % name)


class DistributedExport:
    def __init__(
        self,
        swarm: Callable,
        n_swarms: 2,
        n_import: int = 2,
        n_export: int = 2,
        export_best: bool = True,
        import_best: bool = True,
        max_len: int = 20,
        add_global_best: bool = True,
        swarm_kwargs: dict = None,
    ):
        self.swarms = [
            RemoteExportSwarm.remote(
                swarm=swarm,
                n_export=n_export,
                n_import=n_import,
                import_best=import_best,
                export_best=export_best,
                swarm_kwargs=swarm_kwargs,
            )
            for _ in range(n_swarms)
        ]
        self.n_swarms = n_swarms
        self.minimize = ray.get(self.swarms[0].get_data.remote("minimize"))
        self.max_iters = ray.get(self.swarms[0].get_data.remote("max_iters"))
        self.reward_limit = ray.get(self.swarms[0].get_data.remote("reward_limit"))
        self.param_server = RemoteParamServer.remote(
            max_len=max_len, minimize=self.minimize, add_global_best=add_global_best
        )
        self.epoch = 0

    def get_best(self):
        return ray.get(self.param_server.get_data.remote("best"))

    def reset(self):
        self.epoch = 0
        reset_param_server = self.param_server.reset.remote()
        reset_swarms = [swarm.reset.remote() for swarm in self.swarms]
        ray.get(reset_param_server)
        ray.get(reset_swarms)

    def run(self, print_every=1e10):
        self.reset()
        current_import_walkers = self.swarms[0].get_empty_export_walkers.remote()
        steps = {}
        for swarm in self.swarms:
            steps[swarm.run_exchange_step.remote(current_import_walkers)] = swarm

        for i in range(self.max_iters * self.n_swarms):
            self.epoch = i
            ready_export_walkers, _ = ray.wait(list(steps))
            ready_export_walker_id = ready_export_walkers[0]
            swarm = steps.pop(ready_export_walker_id)

            # Compute and apply gradients.
            current_import_walkers = self.param_server.exchange_walkers.remote(
                *[ready_export_walker_id]
            )
            steps[swarm.run_exchange_step.remote(current_import_walkers)] = swarm

            if self.epoch % print_every == 0 and self.epoch > 0:
                # Evaluate the current model after every 10 updates.
                best = self.get_best()
                print("iter {} best_reward_found: {:.3f}".format(i, best.rewards))
