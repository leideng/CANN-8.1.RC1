from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.dynamic.cum_computer import CumsumComputer


@register_operator("Cumsum")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def cumsum(x, axis, y, exclusive=False, reverse=False, kernel_name="cumsum"):
    obj = CumsumComputer(x, axis, y, exclusive, reverse, kernel_name)
    tik_instance = obj.cum_computer()
    return tik_instance
