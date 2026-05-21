from ...dialects import func
from ...ir import StringAttr, FunctionType

def microkernel(
    kernel_name,
    link_file,
    itypes,
    otypes,
    ip=None,
):
    func_type = FunctionType.get(itypes, otypes)
    func_op = func.FuncOp(name=kernel_name, type=func_type, ip=ip)
    func_op.attributes["sym_visibility"] = StringAttr.get("private")
    func_op.attributes["link_with"] = StringAttr.get(str(link_file))
    return func_op
    