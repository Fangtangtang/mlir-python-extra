# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from ...dialects import allo as allo_d
from ..types import i64
from ...ir import (
    ArrayAttr,
    DenseI64ArrayAttr,
    IntegerAttr,
    MemRefType,
    InsertionPoint,
    StringAttr,
)


class GridMap:
    interface_attr = "interface"  # i: input, o: ouput

    @staticmethod
    def build(inputs, outputs, shardings: list[list[int]], grid: list[int]):
        sharding_attr = ArrayAttr.get(
            [
                ArrayAttr.get([IntegerAttr.get(i64(), s) for s in sharding])
                for sharding in shardings
            ]
        )
        grid_attr = DenseI64ArrayAttr.get(grid)
        args = inputs + outputs

        op = allo_d.grid_map(args, sharding_attr, grid_attr)
        op.attributes[GridMap.interface_attr] = StringAttr.get(
            "i" * len(inputs) + "o" * len(outputs)
        )
        arg_types = []
        for i, arg in enumerate(args):
            memref_type = MemRefType(arg.type)
            shape = list(memref_type.shape)
            for k, s in enumerate(shardings[i]):
                if s >= 0:
                    assert s < len(grid)
                    shape[k] = shape[k] // grid[s]
                else:
                    assert s == -1
            new_type = MemRefType.get(
                shape,
                memref_type.element_type,
                memref_type.layout,
                memref_type.memory_space,
            )
            arg_types.append(new_type)

        block = op.body.blocks.append(*arg_types)
        with InsertionPoint(block):
            allo_d.yield_([])
        return block

    def get_interface(op: allo_d.GridMapOp):
        attr = op.attributes[GridMap.interface_attr].value
        return [i == "i" for i in attr]
