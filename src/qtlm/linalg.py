from qtlm import NDArray, xp


def inv(a: NDArray) -> NDArray:
    return xp.linalg.inv(a)


if xp.__name__ == "cupy":
    name = xp.cuda.runtime.getDeviceProperties(0)["name"].decode("utf-8")
    if name.startswith("NVIDIA"):
        from cupy.cublas import set_batched_gesv_limit

        set_batched_gesv_limit(4096)

        def inv(a: NDArray) -> NDArray:
            return xp.linalg.solve(a, xp.broadcast_to(xp.eye(a.shape[-1]), a.shape))
