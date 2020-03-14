try:
    import ray

    def init_ray():
        MB = 1024 * 1024
        ray.init(redis_max_memory=200 * MB, object_store_memory=200 * MB, ignore_reinit_error=True)


except ImportError:

    from fragile.distributed.ray import ray

    def init_ray():
        pass
