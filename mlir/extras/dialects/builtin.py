# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from ...dialects import _builtin_ops_gen as builtin


def module(name=None):
    mod = builtin.module(sym_name=name)
    mod.regions[0].blocks.append()
    return mod
