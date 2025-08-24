# Copyright 2020 Huawei Technologies Co., Ltd
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
dynamic logsoftmax_v2
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import operation
from impl.util.platform_adapter import OpPatternMode
from impl.util.norm_pattern_adapter import NormPattern
from impl.util.platform_adapter import OpImplMode
from impl.util.util_common import check_op_impl_mode
from impl.util import util_select_op_base


# 'pylint: disable = unused-argument,unused-variable,too-many-locals
def get_op_support_info(input_x, output_y, axes=-1, kernel_name="log_softmax_v2"):
    """
    get_op_support_info
    """
    dims_x = len(input_x.get("shape"))
    if not hasattr(axes, 'index'):
        new_axis = axes
    else:
        new_axis = axes[0]
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


# 'pylint: disable=locally-disabled,unused-argument
@register_operator_compute("LogSoftmaxV2", op_mode="dynamic", support_fusion=False, support_bfp16=True)
def log_softmax_v2_compute(input_x, output_y, axes=-1, kernel_name="log_softmax_v2",
                           impl_mode=OpImplMode.HIGH_PERFORMANCE):
    """
    process of calculating data's log_softmax, x - log(sum(exp(x)))
    this x is x - xmax

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input data
    output: dict
        shape and dtype of output, should be same shape and type as input
    axes: int, list or tuple
        the data's axes, range is [-d, d-1]
    kernel_name : str
        cce kernel name, default value is log_softmax_v2

    Returns
    -------
    result: TVM tensor.
    """
    inp_dtype = input_x.dtype
    vcmax_flag = False
    last_dim = len(input_x.shape) - 1
    for i in axes:
        if i in (-1, last_dim):
            vcmax_flag = True

    if inp_dtype == "float32" and impl_mode == OpImplMode.HIGH_PERFORMANCE and vcmax_flag:
        data_max_input = tbe.cast_to(input_x, "float16")
        data_max_output = tbe.reduce_max(data_max_input, axis=axes, keepdims=True)
        data_max = tbe.cast_to(data_max_output, "float32")
    else:
        input_x = tbe.cast_to(input_x, "float32")
        data_max = tbe.reduce_max(input_x, axis=axes, keepdims=True)

    data_max_broadcast = tbe.broadcast(data_max, input_x.shape)
    data_sub = tbe.vsub(input_x, data_max_broadcast)

    # increase accuracy
    has_improve_precision = False
    if inp_dtype == "float16" and \
            tbe_platform.api_check_support("te.lang.cce.vexp",
                                           "float32"):
        data_sub = tbe.cast_to(data_sub, "float32")
        has_improve_precision = True

    data_exp = tbe.vexp(data_sub)
    data_sum = tbe.reduce_sum(data_exp, axis=axes, keepdims=True)
    data_log = tbe.vlog(data_sum)
    data_log_broadcast = tbe.broadcast(data_log, input_x.shape)
    res = tbe.vsub(data_sub, data_log_broadcast)

    # cast output type same as input type
    if has_improve_precision:
        res = tbe.cast_to(res, "float16")

    return res


# 'pylint: disable=too-many-locals,variable_type_changed
@register_operator("LogSoftmaxV2")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            (para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_LIST_INT),
                            para_check.KERNEL_NAME, para_check.OPTION_ATTR_STR)
def log_softmax_v2(input_x, output_y, axes=-1, kernel_name="log_softmax_v2", impl_mode=OpImplMode.HIGH_PERFORMANCE):
    """
    algorithm: log_softmax
    calculating data's log_softmax, x - log(sum(exp(x)))

    Parameters
    ----------
    input_x : dict
        shape and dtype of input, only support bfloat16, float16, float32
    output_y: dict
        shape and dtype of output, should be same shape and type as input
    axes: int, list or tuple
        the data's axes, range is [-d, d-1]
    kernel_name : str
        cce kernel name, default value is log_softmax_v2

    Returns
    -------
    None
    """
    check_op_impl_mode(impl_mode, [OpImplMode.HIGH_PERFORMANCE, OpImplMode.HIGH_PRECISION], kernel_name)

    shape = input_x.get("shape")
    dtype = input_x.get("dtype").lower()
    para_check.check_shape(shape, param_name="x")
    para_check.check_dtype(dtype, ("float16", "float32", "bfloat16"), param_name="x")

    extra_params = dict()
    if axes is None:
        # when axes is None, it is binary case, go unknown axes schedule
        list_axis = NormPattern.REDUCE_UNKNOWN_MODE
        extra_params.update(NormPattern.REDUCE_SINGLE_TYPE)
        operation.add_compile_info(NormPattern.REDUCE_ATTR_IDX, 0)
        operation.add_compile_info(NormPattern.REDUCE_ATTR_NAME, "axes")
        operation.add_compile_info(NormPattern.REDUCE_ATTR_DTYPE, "ListInt")
    elif not isinstance(axes, int):
        list_axis = list(axes)
    else:
        list_axis = [axes]

    schedules = []
    tensors = []
    ins = classify([input_x, list_axis], OpPatternMode.NORM, extra_params)

    for (x, reduce_axis) in ins:
        with tbe.compute():
            shape_var_new = shape_util.variable_shape([x], op_mode="norm")[0]
            input_x = tvm.placeholder(shape_var_new, dtype=dtype, name="input_x")
            output = log_softmax_v2_compute(input_x, output_y, reduce_axis, kernel_name, impl_mode)
            tensors.append([input_x, output])

        with tvm.target.cce():
            sch = tbe.auto_schedule(output)
        schedules.append(sch)

    # build
    config = {"name": kernel_name,
              "tensor_list": tensors}
    tbe.build(schedules, config)
