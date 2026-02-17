# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from ...dialects import affine
from ...ir import AffineConstantExpr, AffineExpr


class AffExpr:
    @staticmethod
    def constant(value):
        return AffineConstantExpr.get(value)

    @staticmethod
    def symbol(idx: int):
        return AffineExpr.get_symbol(idx)

    @staticmethod
    def dim(idx: int):
        return AffineExpr.get_dim(idx)

    @staticmethod
    def add(lhs, rhs):
        return lhs + rhs

    @staticmethod
    def sub(lhs, rhs):
        return lhs - rhs

    @staticmethod
    def mul(lhs, rhs):
        return lhs * rhs

    @staticmethod
    def div(lhs, rhs):
        return AffineExpr.get_floor_div(lhs, rhs)

    @staticmethod
    def mod(lhs, rhs):
        return lhs % rhs
