from __future__ import annotations

import warnings

import ray


def create_custom_gpu_res():
    # check if we already called this
    for key in ray.cluster_resources():
        if key.startswith("fluid_GPU"):
            return

    # add a unique resource type for each GPU on each node
    for node in ray.nodes():
        clientId = node["NodeID"]
        num_gpus = node["Resources"].get("GPU", 0)
        if num_gpus == 0:
            warnings.warn(
                "No GPU resources found, assuming local test, using CPU resources instead"
            )
            # local test
            num_gpus = node["Resources"].get("CPU", 0)
        for i in range(int(num_gpus)):
            name = f"fluid_GPU_{i}_{clientId}"
            ray.experimental.set_resource(name, 1, clientId)


def gpu_idx_from_name(res_name: str) -> int:
    if not res_name.startswith("fluid_GPU"):
        raise ValueError(res_name)

    _, _, idx, _ = res_name.split("_")
    return int(idx)
