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
masked_scale
"""

from tbe import tvm
import te.lang.cce as tbe
from te.utils import para_check
from te.utils.error_manager import error_manager_vector
from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json


# op generate_param by dtype and format
def generate_param(dtypes, formats):
    """
    Parameters
    ----------
    dtypes: list
        supported dtype of input and output
    formats: list
        supported format of input and output
    Returns
    -------
    None
    """
    dtype_x, dtype_mask, dtype_output = dtypes
    format_x, format_mask, format_output = formats

    input0 = gen_param(classify="input0", name="x",
                       datatype=",".join(dtype_x),
                       format=",".join(format_x))
    input1 = gen_param(classify="input1", name="mask",
                       datatype=",".join(dtype_mask),
                       format=",".join(format_mask))
    output0 = gen_param(classify="output0", name="y",
                        datatype=",".join(dtype_output),
                        format=",".join(format_output))
    return input0, input1, output0


# op get_format_same
def get_format_same(dtype_list, format_list, dtype_total, mask_dtypes):
    """
    Parameters
    ----------
    dtype_list: list
        supported dtype list of x
    format_list: list
        supported format list of mask and x
    mask_dtypes: list
        supported dtype list of mask
    dtype_total: list
        dtype of output and input
    Returns
    -------
    None
    """
    for dtype in dtype_list:
        dtype_total = dtype_total + [dtype] * len(format_list)

    mask_dtype_total = mask_dtypes * len(dtype_total)
    dtype_total = dtype_total * len(mask_dtypes)
    format_list = format_list * len(dtype_list) * len(mask_dtypes)

    dtypes = [dtype_total, mask_dtype_total, dtype_total]
    formats = [format_list, format_list, format_list]
    return dtypes, formats


# 'pylint: disable=unused-argument,too-many-locals
def op_select_format(x, mask, y, value=1.0, kernel_name="masked_scale"):
    """
    Parameters
    ----------
    x: dict
        shape and dtype of input, only support float16, float32
    mask: dict
        shape and dtype of input, only support int8
    value: scaler
        dtype is float, default value is 1.0
    y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name: str
        kernel name, default value is "masked_scale"

    Returns
    -------
    None
    """
    shape_x = x.get("ori_shape")
    shape_mask = mask.get("ori_shape")

    dtype_list = ["float16", "float32"]
    mask_dtypes = ["float16", "float32", "int8"]

    dtype_total = []

    # if shape is same, then all formats are supported.
    if list(shape_x) == list(shape_mask):
        format_list = ["ND", "FRACTAL_NZ", "NC1HWC0", "FRACTAL_Z", "C1HWNCoC0"]
        dtypes, formats = get_format_same(dtype_list, format_list, dtype_total, mask_dtypes)
        input0, input1, output0 = generate_param(dtypes, formats)

        param_list = [input0, input1, output0]
        param_dynamic_in_json = get_dynamic_param_in_json(param_list)
        return param_dynamic_in_json

    raise RuntimeError("The shape of x and mask must be the same.")


# 'pylint: disable=unused-argument
def masked_scale_compute(x, mask, y, value=1.0, kernel_name="masked_scale"):
    """
    function: compute of masked_scale
    Parameters
    ----------
    x: tensor
    mask: tensor
        shape must be same with x
    value: scaler
        dtype is float, default value is 1.0
    y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name: str
        kernel name, default value is "masked_scale"

    Returns
    -------
    None
    """
    # mask_dst_dtype="float16" case cast_to only support int8->float16
    mask_dtype = mask.dtype
    if mask_dtype == "int8":
        mask = tbe.cast_to(mask, dtype="float16")
        mask = tbe.cast_to(mask, dtype="float32")
    elif mask_dtype == "float16":
        mask = tbe.cast_to(mask, dtype="float32")

    if x.dtype != "float32":
        x = tbe.cast_to(x, dtype="float32")

    res_vmul = tbe.vmul(x, mask)
    data_value = tvm.const(value, dtype="float32")
    res = tbe.vmuls(res_vmul, data_value)

    if res.dtype != y.get("dtype"):
        res = tbe.cast_to(res, dtype=y.get("dtype"))

    return res


# 'pylint: disable=unused-argument
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_FLOAT,
                            para_check.KERNEL_NAME)
def masked_scale(x, mask, y, value=1.0, kernel_name="masked_scale"):
    """
    algorithm: masked_scale
    calculating data's reciprocal, y = x * mask * value

    Parameters
    ----------
    x: dict
        shape and dtype of input, only support float16, float32
    mask: dict
        shape and dtype of input, only support int8
    value: scaler
        dtype is float, default value is 1.0
    y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name: str
        kernel name, default value is "masked_scale"

    Returns
    -------
    None
    """
    x_shape = x.get("shape")
    x_dtype = x.get("dtype")

    mask_shape = mask.get("shape")
    mask_dtype = mask.get("dtype")

    # check shape
    if x_shape != mask_shape:
        error_manager_vector.raise_err_two_input_shape_invalid("masked_scale", "x",
                                                               "mask", "shou1d be same shape")

    # do compute
    data_x = tvm.placeholder(x_shape, name="data_x", dtype=x_dtype)
    data_mask = tvm.placeholder(mask_shape, name="data_mask", dtype=mask_dtype)

    res = masked_scale_compute(data_x, data_mask, y, value, kernel_name)

    with tvm.target.cce():
        sch = tbe.auto_schedule(res)

    config = {"name": kernel_name, "tensor_list": [data_x, data_mask, res]}
    tbe.cce_build_code(sch, config)
