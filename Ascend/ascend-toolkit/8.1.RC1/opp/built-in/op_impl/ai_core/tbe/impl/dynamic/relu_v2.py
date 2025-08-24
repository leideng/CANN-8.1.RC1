# Copyright 2021 Huawei Technologies Co., Ltd
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
relu_v2

'  Op_description :
'    Algrithm: relu_v2(x) = x and 1 when x > 0 , else 0, 0
'
'    # relu_v2(
'    #   x,
'    #   y,
'    #   mask,
'    #   kernel_name='relu_v2')
'
'  Supportive_dtype_format :
'    ['float16', 'float32', 'int8', 'int32', 'uint8']
'    ['ND', 'NCHW', 'NHWC', 'NC1HWC0']
'
'  Constraint :
'    [1] All : the last dim of `x` must be mutiply of 8.
'    [2] All : shape size limit is 2147483648.
"""
# noinspection PyInterpreter
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.util_select_op_base import SplitInput
from impl.util.util_select_op_base import SplitOutput
from impl.util.util_select_op_base import get_op_cal_info
from tbe.dsl.base import operation


# 'pylint: disable=locally-disabled,too-many-argument,unused-argument,invalid-name
def get_op_support_info(x, y, mask, kernel_name="relu_v2"):
    """
    get_op_support_info
    """
    format_x = x.get("format").upper()
    if format_x == "NC1HWC0":
        axis_split_matrix = [[SplitInput([0, [0], [-1], [-1]]), SplitOutput([0, [0]], [1, [0]])]]
    else:
        axis_split_matrix = None
    axis_reduce_list = None
    op_cal_info_in_json = get_op_cal_info(axis_split_matrix, axis_reduce_list, 0, 0)
    return op_cal_info_in_json


# 'pylint: disable=locally-disabled,too-many-locals,unused-argument,invalid-name
@register_operator_compute("ReluV2", op_mode="dynamic", support_fusion=True)
def relu_v2_compute(x, y, mask, kernel_name="relu_v2"):
    """
    Algrithm : relu_v2(x) = x and 1 when x > 0 , else 0, 0

    Parameters
    ----------
    x: the placeholder of data input

    y : the dict of output

    mask : the dict of output

    kernel_name : cce kernel name

    Returns
    -------
    res : result of relu_v2_res

    mask: result of relu_v2_mask
    """

    inp_dtype = x.dtype
    shape = x.shape
    compatible_dtype = x.dtype
    vcmp_support = tbe_platform.api_check_support("tbe.dsl.vcmp", compatible_dtype)

    if inp_dtype == 'bfloat16':
        x = tbe.cast_to(x, 'float32')
        compatible_dtype = 'float32'
    elif inp_dtype in ("int8", "uint8") or not vcmp_support:
        x = tbe.cast_to(x, 'float16')
        compatible_dtype = 'float16'

    if tbe_platform.api_check_support('tbe.dsl.vrelu', compatible_dtype):
        data_res = tbe.vrelu(x)
    else:
        tensor_zero = tbe.broadcast(tvm.const(0, compatible_dtype), shape)
        data_res = tbe.vmax(x, tensor_zero)

    if inp_dtype == 'bfloat16':
        data_res = tbe.round(data_res, inp_dtype)
    elif inp_dtype in ("int8", "uint8") or not vcmp_support:
        data_res = tbe.cast_to(data_res, inp_dtype)

    if tbe_platform.get_soc_spec("SHORT_SOC_VERSION") in ("Ascend910B", "Ascend910_93",) and \
        inp_dtype == "int32":
        x = tbe.cast_to(x, 'float32')

    mask = tbe.vcmp(x, 0, "gt", "bit")

    return data_res, mask


@register_operator("ReluV2")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def relu_v2(x, y, mask, kernel_name="relu_v2"):
    """
    Algrithm: relu_v2(x) = x and 1 when x > 0 , else 0, 0

    Parameters
    ----------
    Algorithm: relu_v2

    Parameters:

    x: the dict of input data, support float16, float32, int8, int32, uint8

    y: the dict of output

    mask: the dict of mask_output

    kernel_name: cce kernel name, default value is "relu_v2".

    Returns
    -------
    None
    """

    cce_product = tbe_platform.get_soc_spec("SHORT_SOC_VERSION")
    if cce_product == "Ascend310P":
        operation.get_context().add("ElewiseMaskFlag", "Mask")
    shape = x.get("shape")
    dtype = x.get("dtype").lower()

    para_check.check_shape(shape, param_name="x")

    check_list = ("float16", "float32", "int8", "int32", "uint8", "bfloat16")
    para_check.check_dtype(dtype, check_list, param_name="x")

    ins = classify([x], OpPatternMode.ELEWISE)
    schedules, tensors = [], []
    for (_x,) in ins:
        with tbe.compute():
            shape_x = shape_util.variable_shape([_x])

            input_data = tvm.placeholder(shape_x[0], dtype, "input_data")
            res, res_mask = relu_v2_compute(input_data, y, mask, kernel_name)

            tensors.append([input_data, res, res_mask])
        with tvm.target.cce():
            sch = tbe.auto_schedule([res, res_mask])
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors, "print_ir": False}

    tbe.build(schedules, config)
