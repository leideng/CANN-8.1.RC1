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
dynamic data_format_dim_map

Op_description :
Returns the dimension index in the destination data format given the one in.

# data_format_dim_map(
#   x,
#   y,
#   src_format,
#   dst_format,
#   kernel_name='data_format_dim_map')

Supportive_dtype_format :
['int32']
['ND', 'NCHW', 'NHWC', 'NC1HWC0']

Constraint :
[1] All : `x` Must be in the range [-4, 4).
[2] All : `src_format` and `dst_format` must be length of 4.
[3] All : shape size limit is 2147483648.
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import error_manager_vector

from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.data_format_dim_map import get_op_support_info as data_format_dim_map_get_op_support_info


def get_op_support_info(x,
                        y,
                        src_format="NHWC",
                        dst_format="NCHW",
                        kernel_name="data_format_dim_map"):
    """
    get_op_support_info
    """
    return data_format_dim_map_get_op_support_info(x, y, src_format, dst_format, kernel_name)


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    # mod rhs
    MOD_RHS = 4


# 'pylint: disable=invalid-name
def _data_format_dim_map_mod(data):
    """
    mod function based on TF

    Parameters
    ----------
    data: the shape of input, only support int32.

    Returns
    -------
    data mod by 4
    """
    # quarter
    quarter = 0.25
    data = tbe.cast_to(data, 'float16')
    data = tbe.vadds(data, Constant.MOD_RHS)
    data_div_4 = tbe.vmuls(data, quarter)
    data_floor = tbe.floor(data_div_4)
    data_floor = tbe.cast_to(data_floor, 'float16')
    data_mul_4 = tbe.vmuls(data_floor, Constant.MOD_RHS)
    data_mod = tbe.vsub(data, data_mul_4)
    return data_mod


def _dimension_index(data_mod, ind):
    """
    dimension index function

    Parameters
    ----------
    data_mod: the data after modulo
    ind: mapping of index

    Returns
    -------
    dimension index
    """
    return tbe.cast_to(tbe.vcmpsel(data_mod, 0., 'eq', ind[0],
                                   tbe.vcmpsel(data_mod, 1., 'eq', ind[1],
                                   tbe.vcmpsel(data_mod, 2., 'eq', ind[2],
                                   ind[3]))), "int32")


# 'pylint: disable=locally-disabled,unused-argument,invalid-name
@register_operator_compute("DataFormatDimMap", op_mode="dynamic", support_fusion=True)
def _data_format_dim_map_compute(x,
                                 y,
                                 src_format='NHWC',
                                 dst_format='NCHW',
                                 kernel_name='data_format_dim_map'):
    """
    Parameters
    ----------
    x: the dict of input, only support int32.
    y : the dict of y, reserved parameter, not used now.
    src_format : the original type of x, default value is "NHWC" (optional).
    dst_format : the aim type of x, default value is "NCHW" (optional).
    kernel_name : cce kernel name, default value is "data_format_dim_map" (optional).

    Returns
    -------
    Tensor after dataformatdimmap compute
    """

    data_mod = _data_format_dim_map_mod(x)
    src_format = src_format.upper()
    dst_format = dst_format.upper()

    ind = [0] * len(src_format)
    for i, src in enumerate(src_format):
        for j, dst in enumerate(dst_format):
            if src == dst:
                ind[i] = j
                break

    return _dimension_index(data_mod, ind)


# 'pylint: disable=too-many-locals
@register_operator("DataFormatDimMap")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_STR, para_check.OPTION_ATTR_STR,
                            para_check.KERNEL_NAME)
def data_format_dim_map(x,
                        y,
                        src_format="NHWC",
                        dst_format="NCHW",
                        kernel_name="data_format_dim_map"):
    """
    Returns the dimension index in the destination data format given the one in.

    Parameters
    ----------
    x : A Tensor with each element as a dimension index in source data format.
        Must be the following types: `int32`. Must be in the range [-4, 4).
    y : Shape and dtype of y, reserved parameter, not used now.
    src_format : An optional `string`. Defaults to `"NHWC"`. source data format.
    dst_format : An optional `string`. Defaults to `"NCHW"`. destination data format.
    kernel_name : CCE kernel name, default value is "data_format_dim_map" (optional).

    Returns
    -------
    None
    """
    dtype_input = x.get("dtype").lower()
    check_list = ("int32",)
    para_check.check_dtype(dtype_input, check_list, param_name="x")
    # check length of format
    if len(src_format) != 4:
        error_manager_vector.raise_err_input_value_invalid(kernel_name, "src_format", \
                                                           "length 4", len(src_format))
    if len(dst_format) != 4:
        error_manager_vector.raise_err_input_value_invalid(kernel_name, "dst_format", \
                                                           "length 4", len(dst_format))
    schedules, tensors = [], []
    ins = classify([x], OpPatternMode.ELEWISE)
    for (_x,) in ins:
        with tbe.compute():
            x_shape = shape_util.variable_shape([_x])
            data_input = tvm.placeholder(x_shape[0], dtype=dtype_input,
                                         name="data_input")
            res = _data_format_dim_map_compute(data_input, y, src_format, dst_format, kernel_name)
            tensors.append([data_input, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name,
              "tensor_list": tensors,
              "bool_storage_as_1bit": False}
    tbe.build(schedules, config)
