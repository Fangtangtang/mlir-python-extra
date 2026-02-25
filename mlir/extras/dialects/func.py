from ...dialects import func
from ...ir import StringAttr, FunctionType


def set_string_attr_if_present(op, name, value):
    if value is None:
        return
    if isinstance(value, list):
        value = "".join(value)
    if isinstance(value, str):
        op.attributes[name] = StringAttr.get(value)
    else:
        raise TypeError(f"{name} must be str or list[str], got {type(value)}")


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
    set_string_attr_if_present(func_op, "itypes", itype_hints)
    set_string_attr_if_present(func_op, "otypes", otype_hints)
    return func_op
