# Copyright 2019 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
log_softmax_v2
"""
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import build
from impl.util.platform_adapter import auto_schedule
import te.lang.cce as tbe
import impl.dynamic as dyn_impl
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util import util_select_op_base
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import OpImplMode
from impl.util.util_common import check_op_impl_mode
from impl.util.platform_adapter import register_operator_compute


# 'pylint: disable = unused-argument,unused-variable,too-many-locals
def get_op_support_info(input_x, output_y, axis=-1, kernel_name="log_softmax_v2"):
    """
    get_op_support_info
    """
    dims_x = len(input_x.get("shape"))
    if not hasattr(axis, 'index'):
        new_axis = axis
    else:
        new_axis = axis[0]
    if new_axis < 0:
        new_axis = new_axis + len(input_x.get("shape"))
    axis_split_matrix = []
    for i in range(dims_x):
        if i != new_axis:
            split_0 = [util_select_op_base.SplitInput([0, [i], [-1], [-1]]), util_select_op_base.SplitOutput([0, [i]])]
            axis_split_matrix.append(split_0)
    axis_reduce_list = None
    op_cal_info_in_json = util_select_op_base.get_op_cal_info(axis_split_matrix, axis_reduce_list, 0, 0)
    return op_cal_info_in_json


def is_white_shape(shape):
    """
    is_white_shape
    """
    white_list_shape = [[2105352, 21], [8, 81, 25276], [1003520, 11]]
    shape_t = list(shape)
    if shape_t in white_list_shape:
        return True
    return False


# 'pylint: disable = locally-disabled,unused-argument
@register_operator_compute("log_softmax_v2", op_mode="static", support_fusion=True)
def log_softmax_v2_compute(input_x, output_y, axis=-1, kernel_name="log_softmax_v2",
                           impl_mode="high_performance"):
    """
    process of calculating data's log_softmax, x - log(sum(exp(x)))
    this x is x - xmax

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input data
    output: dict
        shape and dtype of output, should be same shape and type as input
    axis: int, list or tuple
        the data's axis, range is [-d, d-1]
    kernel_name : str
        cce kernel name, default value is log_softmax_v2

    Returns
    -------
    result: TVM tensor.
    """
    inp_dtype = input_x.dtype
    shape = shape_util.shape_to_list(input_x.shape)

    if impl_mode == "high_precision":
        data_max = tbe.reduce_max(input_x, axis=axis, keepdims=True, priority_flag=True)
    else:
        data_max = tbe.reduce_max(input_x, axis=axis, keepdims=True)
    data_max_broadcast = tbe.broadcast(data_max, shape)
    data_sub = tbe.vsub(input_x, data_max_broadcast)

    # increase accuracy
    has_improve_precision = False
    if inp_dtype == "float16" and \
            tbe_platform.api_check_support("tbe.dsl.vexp",
                                                    "float32"):
        data_sub = tbe.cast_to(data_sub, "float32")
        has_improve_precision = True

    data_exp = tbe.vexp(data_sub)
    data_sum = tbe.sum(data_exp, axis=axis, keepdims=True)
    data_log = tbe.vlog(data_sum)
    data_log_broadcast = tbe.broadcast(data_log, shape)
    res = tbe.vsub(data_sub, data_log_broadcast)

    # cast output type same as input type
    if has_improve_precision:
        res = tbe.cast_to(res, "float16")

    return res


# 'pylint: disable=variable_type_changed
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            (para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_LIST_INT),
                            para_check.KERNEL_NAME, para_check.OPTION_ATTR_STR)
def log_softmax_v2(input_x, output_y, axis=-1, kernel_name="log_softmax_v2", impl_mode="high_performance"):
    """
    algorithm: log_softmax
    calculating data's log_softmax, x - log(sum(exp(x)))

    Parameters
    ----------
    input_x : dict
        shape and dtype of input, only support float16, float32
    output_y: dict
        shape and dtype of output, should be same shape and type as input
    axis: int, list or tuple
        the data's axis, range is [-d, d-1]
    kernel_name : str
        cce kernel name, default value is log_softmax_v2

    Returns
    -------
    None
    """
    check_op_impl_mode(impl_mode, [OpImplMode.HIGH_PERFORMANCE, OpImplMode.HIGH_PRECISION], kernel_name)

    check_list = ("float16", "float32")
    shape = input_x.get("shape")
    input_dtype = input_x.get("dtype").lower()
    shape_len = len(shape)

    range_x = []
    for dim in input_x.get("shape"):
        range_x.append((dim, dim))
    input_x["range"] = range_x

    tbe_product = tbe_platform.get_soc_spec("SHORT_SOC_VERSION")
    is_support = tbe_product in ("Ascend610", "BS9SX1A", "Ascend310P", "Ascend910")

    if is_support and is_white_shape(shape):
        context = tbe_context.op_context.get_context()
        if context is not None:
            context.set_op_mode("static")
            dyn_impl.log_softmax_v2(input_x, output_y, axis, kernel_name, impl_mode)
        else:
            with tbe_context.op_context.OpContext("static"):
                dyn_impl.log_softmax_v2(input_x, output_y, axis, kernel_name, impl_mode)
        return

    if not isinstance(axis, int):
        axis = list(axis)

    para_check.check_shape(shape, param_name="input_x")
    para_check.check_dtype(input_dtype, check_list, param_name="input_x")

    axis = shape_util.axis_check(shape_len, axis)

    shape, axis = shape_util.shape_refine(list(shape), axis)
    shape, axis = shape_util.simplify_axis_shape(shape, axis)

    data_input = tvm.placeholder(shape, name="data_input", dtype=input_dtype)
    result = log_softmax_v2_compute(data_input, output_y, axis=axis,
                                    kernel_name=kernel_name, impl_mode=impl_mode)

    with tvm.target.cce():
        sch = auto_schedule(result)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": [data_input, result]}

    build(sch, config)
