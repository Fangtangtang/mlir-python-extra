# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from ...dialects import sdy
from ...ir import StringAttr, FlatSymbolRefAttr


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


class SPMD:
    mesh_attr = "mesh"  # global symbol name of the program mesh / grid

    @staticmethod
    def manual_computation(
        results: list,
        inputs: list,
        in_shard: list[sdy.TensorShardingAttr],
        out_shard: list[sdy.TensorShardingAttr],
        axis: int,
        mesh_name: str,
    ) -> sdy.ManualComputationOp:
        op = sdy.ManualComputationOp(
            results,
            inputs,
            sdy.TensorShardingPerValueAttr.get(in_shard),
            sdy.TensorShardingPerValueAttr.get(out_shard),
            sdy.ManualAxesAttr.get([StringAttr.get(f"{i}") for i in range(axis)]),
        )
        op.attributes[SPMD.mesh_attr] = FlatSymbolRefAttr.get(mesh_name)
        return op

    @staticmethod
    def get_mesh(op: sdy.ManualComputationOp):
        assert SPMD.mesh_attr in op.attributes
        return op.attributes[SPMD.mesh_attr]


def mesh(mesh_name: str, grid: list[int]) -> sdy.MeshOp:
    axes = [sdy.MeshAxisAttr.get(f"{i}", dim) for i, dim in enumerate(grid)]
    return sdy.mesh(
        sym_name=mesh_name,
        mesh=sdy.MeshAttr.get(axes),
    )
