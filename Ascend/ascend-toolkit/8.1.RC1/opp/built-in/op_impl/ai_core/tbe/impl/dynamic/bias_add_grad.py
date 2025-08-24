# Copyright (c) Huawei Technologies Co., Ltd. 2020-2022. All rights reserved.
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
bias_add_grad
"""
# 'pylint: disable=locally-disabled,invalid-name
# 'pylint: disable=unnecessary-comprehension,global-statement
from impl.util import util_common
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.util_compute import only_static_support
from impl.bias_add_grad import get_op_support_info as bias_add_grad_get_op_support_info


# 'pylint: disable=too-few-public-methods,too-many-instance-attributes
class SortParam:
    """
    The class for SortParam
    """
    REDUCE_LIST = None


def get_op_support_info(x, y, data_format, kernel_name="bias_add_grad"):
    """
    get_op_support_info
    """
    return bias_add_grad_get_op_support_info(x, y, data_format, kernel_name)


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument,too-many-branches
# 'pylint: disable=locally-disabled,too-many-locals
def _infer_axes(input_data_format, data_format, shape):
    """
    To infer sum operate axis by input_data format and data_format
    to keep compute Architecture, so use global parameter send variable
    Parameters:
    ----------
    input_data_format: str
        op's input data format
    data_format: str
        'NCHW' or 'NHWC'
    shape : tuple or list
        the input data shape

    Returns
    -------
    g_shape_list. list
    """
    g_shape_list = []
    if input_data_format == 'FRACTAL_NZ':
        if data_format == "NCHW":
            if len(shape) == 4:
                for i in range(-1 * len(shape), 0):
                    if i not in (-1, -4):
                        g_shape_list += [i + len(shape)]
            elif len(shape) == 5:
                for i in range(-1 * len(shape), 0):
                    if i not in (-2, -3):
                        g_shape_list += [i + len(shape)]
            else:
                g_shape_list.append(0)
                for i in range(2, len(shape)):
                    g_shape_list = g_shape_list + [i]
        else:
            if len(shape) < 4:
                error_manager_vector.raise_err_specific_reson("bias_add_grad",
                                                              "cce_bias_add_grad_nz_2_nhwc \
                                                              only support shape larger than 4D")
            for i in range(-1 * len(shape), 0):
                if i not in (-1, -4):
                    g_shape_list += [i + len(shape)]
    elif input_data_format in ("FRACTAL_Z", "FRACTAL_Z_3D", "NC1HWC0", "NDC1HWC0"):
        if input_data_format == "FRACTAL_Z":
            # mean format is FRACTAL_Z, shape is C1HWNiNoC0
            g_shape_list = [1, 2, 3, 4]
        elif input_data_format == "FRACTAL_Z_3D":
            # mean format is FRACTAL_Z_3D, shape is DC1HWNiNoC0
            g_shape_list = [0, 2, 3, 4, 5]
        elif input_data_format == "NC1HWC0":
            # mean format is NC1HWC0, shape is NC1HWC0
            g_shape_list = [0, 2, 3]
        elif input_data_format == "NDC1HWC0":
            # mean format is NDC1HWC0, shape is NDC1HWC0
            g_shape_list = [0, 1, 3, 4]
    else:
        if data_format in ("NCHW", "NCDHW"):
            g_shape_list = [0]
            for i in range(2, len(shape)):
                g_shape_list = g_shape_list + [i]
        else:
            if len(shape) < 2:
                error_manager_vector.raise_err_specific_reson("bias_add_grad", "cce_bias_add_grad \
                                                              only support shape larger than 2D")
            g_shape_list = list(range(len(shape) - 1))

    return g_shape_list


@register_operator_compute("BiasAddGrad", op_mode="dynamic", support_fusion=only_static_support, support_bfp16=True)
def bias_add_grad_compute(x, y, data_format, kernel_name="bias_add_grad"):
    """
    Reduce a tensor on last dimension in axis based on sum.

    Parameters:
    ----------
    x: TVM tensor
        the placeholder of y input data
    y: dict
        shape and dtype of output, should be same shape and type as input
    data_format: str
        'NCHW' or 'NHWC'
    kernel_name : str
        cce kernel name, default value is bias_add_grad

    Returns
    -------
    TVM tensor by bias add grad
    """
    dtype = x.dtype
    y_dtype = y.get("dtype").lower()

    if dtype == "float16" and tbe_platform.api_check_support("te.lang.cce.sum", "float32"):
        x = tbe.cast_to(x, "float32")

    result = tbe.reduce_sum(x, SortParam.REDUCE_LIST)
    result = tbe.cast_to(result, y_dtype)

    return result


@register_operator("BiasAddGrad")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_STR,
                            para_check.KERNEL_NAME)
def bias_add_grad(x, y, data_format, kernel_name="bias_add_grad"):
    """
    Reduce a tensor on last dimension in axis based on sum.

    Parameters:
    ----------
    x : dict
        shape and dtype of input, only support float16, float32
    y: dict
        shape and dtype of output, should be same shape and type as input
    data_format: str
        'NCHW' or 'NHWC'
    kernel_name : str
        cce kernel name, default value is bias_add_grad
    Returns
    -------
    None
    """
    x = util_common.update_shape_base_other_format_dynamic(x)
    y = util_common.update_shape_base_other_format_dynamic(y)
    dtype = x.get("dtype").lower()
    shape = x.get("shape")
    check_list = ("bfloat16", "float16", "float32")
    para_check.check_dtype(dtype, check_list, param_name="x")
    input_data_format = x.get("format").upper()
    x["rel_pos_to_reduce"] = "before"
    is_unknown_rank = util_common.is_unknown_rank_input(x)
    if is_unknown_rank:
        input_axis = {"shape": [-1], "rel_pos_to_reduce": "axis"}
    else:
        data_format = data_format.upper()
        data_format_tuple = ("NCHW", "NHWC", "NDHWC", "NCDHW")
        para_check.check_format(data_format, data_format_tuple, param_name="x")
        if len(shape) < 2:
            error_detail = "cce_bias_add_grad shape should be larger than 2D"
            error_manager_vector.raise_err_input_shape_invalid(kernel_name, "x", error_detail)

        g_shape_list = _infer_axes(input_data_format, data_format, shape)
        input_axis = {"shape": [len(g_shape_list), ], "value": g_shape_list, "rel_pos_to_reduce": "axis"}
    ins = classify([x, input_axis], OpPatternMode.REDUCE, {"keepdims": False})
    schedules, tensors = [], []
    for (_x, axes) in ins:
        with tbe.compute():
            shape_x = shape_util.variable_shape([_x, axes], op_mode="reduce")[0]
            input_data = tvm.placeholder(shape_x, name="input_data", dtype=dtype)
            SortParam.REDUCE_LIST = shape_util.axis_check(len(shape_x), axes.get("value"))

            res = bias_add_grad_compute(input_data, y, data_format, kernel_name)
            tensors.append([input_data, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"print_ir": False, "name": kernel_name, "tensor_list": tensors}
    tbe_context.get_context().add_compile_info("is_unknown_rank", is_unknown_rank)
    tbe.build(schedules, config)
