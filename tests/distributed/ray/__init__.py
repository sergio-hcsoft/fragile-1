try:
    import ray

    def init_ray():
        MB = 1024 * 1024
        ray.init(redis_max_memory=500 * MB, object_store_memory=500 * MB, ignore_reinit_error=True)


except ImportError:
    pass
