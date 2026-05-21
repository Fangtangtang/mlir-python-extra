try:
    from aie.ir import StringAttr
except ImportError:
    pass


class Microkernel:
    link_attr = "link_with"

    def set_link(func_op, link_file):
        with func_op.context:
            func_op.attributes[Microkernel.link_attr] = StringAttr.get(str(link_file))
