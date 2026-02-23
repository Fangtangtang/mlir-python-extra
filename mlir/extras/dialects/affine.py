# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import sympy
from ...dialects import affine
from ...ir import (
    AffineExpr,
    AffineConstantExpr,
    AffineAddExpr,
    AffineMulExpr,
    AffineModExpr,
    AffineFloorDivExpr,
    AffineSymbolExpr,
    AffineDimExpr,
)


class AffExpr:
    @staticmethod
    def constant(value):
        return AffineConstantExpr.get(value)

    @staticmethod
    def symbol(idx: int):
        return AffineSymbolExpr.get(idx)

    @staticmethod
    def dim(idx: int):
        return AffineDimExpr.get(idx)

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
        return AffineFloorDivExpr.get(lhs, rhs)

    @staticmethod
    def mod(lhs, rhs):
        return lhs % rhs

    @staticmethod
    def dim_name(d: AffineDimExpr) -> str:
        return f"d{d.position}"

    @staticmethod
    def symbol_name(d: AffineDimExpr) -> str:
        return f"s{d.position}"

    @staticmethod
    def parse(expr):
        def try_cast(cls, expr_):
            try:
                return cls(expr_)
            except ValueError:
                return None

        if (c := try_cast(AffineConstantExpr, expr)) is not None:
            return sympy.Integer(c.value)
        if (d := try_cast(AffineDimExpr, expr)) is not None:
            return sympy.Symbol(AffExpr.dim_name(d))
        if (s := try_cast(AffineSymbolExpr, expr)) is not None:
            return sympy.Symbol(AffExpr.symbol_name(s))
        if (a := try_cast(AffineAddExpr, expr)) is not None:
            lhs = AffExpr.parse(a.lhs)
            rhs = AffExpr.parse(a.rhs)
            return lhs + rhs
        if (m := try_cast(AffineMulExpr, expr)) is not None:
            lhs = AffExpr.parse(m.lhs)
            rhs = AffExpr.parse(m.rhs)
            return lhs * rhs
        if (mod := try_cast(AffineModExpr, expr)) is not None:
            lhs = AffExpr.parse(mod.lhs)
            rhs = AffExpr.parse(mod.rhs)
            return lhs % rhs
        if (fd := try_cast(AffineFloorDivExpr, expr)) is not None:
            lhs = AffExpr.parse(fd.lhs)
            rhs = AffExpr.parse(fd.rhs)
            return lhs // rhs
        raise TypeError(f"Unsupported AffineExpr type: {type(expr)}")
