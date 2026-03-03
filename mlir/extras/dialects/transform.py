# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from ...dialects import transform as xform_d
from ...dialects.builtin import ModuleOp
from ...ir import InsertionPoint, UnitAttr


def module(name: str = None):
    if name is None:
        name = "__transform_main"  # default interpreter entry point
    mod = ModuleOp()
    mod.attributes["transform.with_named_sequence"] = UnitAttr.get()
    with InsertionPoint(mod.body):
        seq = xform_d.named_sequence(name, [xform_d.any_op_t()], [])
        with InsertionPoint(seq.body):
            xform_d.YieldOp()
    return mod
