from ...dialects import func
from ...ir import StringAttr, FunctionType


def function(
    func_name,
    itypes,
    otypes,
    is_private=False,
    itype_hints=None,
    otype_hints=None,
    ip=None,
):
    func_type = FunctionType.get(itypes, otypes)
    func_op = func.FuncOp(name=func_name, type=func_type, ip=ip)
    if is_private:
        func_op.attributes["sym_visibility"] = StringAttr.get("private")
    else:
        func_op.add_entry_block()
    if itype_hints is not None:
        func_op.attributes["itypes"] = StringAttr.get("".join(itype_hints))
    if otype_hints is not None:
        func_op.attributes["otypes"] = StringAttr.get("".join(otype_hints))
    return func_op
