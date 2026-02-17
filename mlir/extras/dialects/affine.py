# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from ...dialects import sdy
from ...ir import StringAttr


def tensor_sharding(mesh_name: str, axis: list[int]) -> sdy.TensorShardingAttr:
    return sdy.TensorShardingAttr.get(
        mesh_name,
        [
            sdy.DimensionShardingAttr.get(
                axes=([sdy.AxisRefAttr.get(name=f"{ax}")] if ax >= 0 else []),
                is_closed=True,
            )
            for ax in axis
        ],
    )


def manual_computation(
    results: list,
    inputs: list,
    in_shard: list[sdy.TensorShardingAttr],
    out_shard: list[sdy.TensorShardingAttr],
    axis: int,
) -> sdy.ManualComputationOp:
    return sdy.ManualComputationOp(
        results,
        inputs,
        sdy.TensorShardingPerValueAttr.get(in_shard),
        sdy.TensorShardingPerValueAttr.get(out_shard),
        sdy.ManualAxesAttr.get([StringAttr.get(f"{i}") for i in range(axis)]),
    )


def mesh(mesh_name: str, grid: list[int]) -> sdy.MeshOp:
    axes = [sdy.MeshAxisAttr.get(f"{i}", dim) for i, dim in enumerate(grid)]
    return sdy.mesh(
        sym_name=mesh_name,
        mesh=sdy.MeshAttr.get(axes),
    )
