from qtlm import xp


def free_mempool():
    """Free the memory pool of the array module."""
    if xp.__name__ == "cupy":
        # Free all blocks in the default memory pool.
        mempool = xp.get_default_memory_pool()
        mempool.free_all_blocks()
    else:
        # Numpy does not have a memory pool, so nothing to do.
        pass
