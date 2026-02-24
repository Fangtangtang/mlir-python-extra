# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from ..ir import OpAttributeMap


def copy_attr(src, dst, attrs: set[str], allow_missing=True):
    assert hasattr(src, "attributes") and isinstance(src.attributes, OpAttributeMap)
    assert hasattr(dst, "attributes") and isinstance(dst.attributes, OpAttributeMap)
    for attr in attrs:
        if attr in src.attributes:
            dst.attributes[attr] = src.attributes[attr]
        elif not allow_missing:
            raise ValueError(f"source op do not have {attr}")
