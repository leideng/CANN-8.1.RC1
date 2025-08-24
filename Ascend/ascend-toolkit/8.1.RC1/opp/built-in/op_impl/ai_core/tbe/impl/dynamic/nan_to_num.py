# Copyright 2023 Huawei Technologies Co., Ltd
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
dynamic nan_to_num
"""
from impl.util import util_common
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.util_attr_common import OpAttr
from impl.util.util_attr_common import get_attr_by_cls


# 'pylint: disable=too-many-branches,too-many-arguments,too-many-locals
# 'pylint: disable=unused-argument,invalid-name
@register_operator_compute("NanToNum", op_mode="dynamic", support_fusion=False)
def nan_to_num_compute(x, y, nan, posinf, neginf, kernel_name="nan_to_num"):
    """
    algorithm: nan_to_num
    Replace Nan, positive infinity, and negative infinity values in input
    with the values specified by nan, posinf, and neginf, respectively.
    By default, Nans are replcced with zero, positive infinity is replaced
    with the greatest finite value representable by input's dtype, and
    negative infinity is replaced with the least finite value representable
    by input's dtype.

    Parameters
    ----------
    input_x : TVM tensor
        the placeholders of input data
    output_y : dict
        dict with keys(shape and dtype) of output
    nan : float
        nan, default value is 0.0
    posinf : float
        posinf, default value is None
    neginf : float
        neginf, default value is None
    kernel_name : str
        kernel name, default value is "nan_to_num"

    Returns
    -------
    output tensor
    """
    ori_xdtype = x.dtype.lower()
    xshape = x.shape

    xdtype = ori_xdtype
    if ori_xdtype == "bfloat16":
        x = tbe.cast_to(x, "float32")
        xdtype = "float32"

    constant_dict = {
        "float32_max_value": 3.4028235e+38,
        "float32_min_value": -3.4028235e+38,
        "float16_max_value": 65504.0,
        "float16_min_value": -65504.0,
        "bfloat16_max_value": 3.3895314e+38,
        "bfloat16_min_value": -3.3895314e+38
    }

    if not util_common.is_unknown([y]):
        nan_value = tvm.const(nan, dtype=xdtype)
        if posinf is None:
            max_value = constant_dict.get(f'{ori_xdtype}_max_value')
            posinf_value = tvm.const(max_value, xdtype)
        else:
            posinf_value = tvm.const(posinf, xdtype)
        if neginf is None:
            min_value = constant_dict.get(f'{ori_xdtype}_min_value')
            neginf_value = tvm.const(min_value, xdtype)
        else:
            neginf_value = tvm.const(neginf, xdtype)
    else:
        nan_value = get_attr_by_cls(nan, OpAttr(0, "nan", "Float"), xdtype)
        posinf_value = tbe.var(name="posinf", dtype=xdtype)
        if xdtype == "float16":
            tbe.var(name="posinf_empty", dtype=xdtype)
        neginf_value = tbe.var(name="neginf", dtype=xdtype)
        if xdtype == "float16":
            tbe.var(name="neginf_empty", dtype=xdtype)

    positive_infs = tbe.broadcast(tvm.const(float("inf"), xdtype), xshape)
    negative_infs = tbe.broadcast(tvm.const(float("-inf"), xdtype), xshape)

    # nan-->posinf-->neginf
    nan_res_temp = tbe.vcmp(x, x, "eq", "bit")
    nan_res = tbe.vsel(nan_res_temp, x, nan_value)
    pos_res_temp = tbe.vcmp(nan_res, positive_infs, "eq", "bit")
    pos_res = tbe.vsel(pos_res_temp, posinf_value, nan_res)
    res_temp = tbe.vcmp(pos_res, negative_infs, "eq", "bit")
    res = tbe.vsel(res_temp, neginf_value, pos_res)

    if ori_xdtype == "bfloat16":
        res = tbe.round(res, "bfloat16")

    return res


# 'pylint:disable=redefined-argument-from-local
@register_operator("NanToNum")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_FLOAT,
                            para_check.OPTION_ATTR_FLOAT, para_check.KERNEL_NAME)
def nan_to_num(input_x, output_y, nan, posinf, neginf, kernel_name="nan_to_num"):
    """
    algorithm: nan_to_num
    Replace Nan, positive infinity, and negative infinity values in input
    with the values specified by nan, posinf, and neginf, respectively.
    By default, Nans are replcced with zero, positive infinity is replaced
    with the greatest finite value representable by input's dtype, and
    negative infinity is replaced with the least finite value representable
    by input's dtype.

    Parameters
    ----------
    input_x : TVM tensor
        the placeholders of input data
    output_y : dict
        dict with keys(shape and dtype) of output
    nan : float
        nan, default value is 0.0
    posinf : float
        posinf, default value is None
    neginf : float
        neginf, default value is None
    kernel_name : str
        kernel name, default value is "nan_to_num"

    Returns
    -------
    output tensor
    """
    x_dtype = input_x.get("dtype").lower()
    # check the dtype of input_X, only supports fp16, bfp16 and fp32
    para_check.check_dtype(x_dtype, ("float16", "bfloat16", "float32"), param_name="input_x")

    schedules, tensors = [], []
    ins = classify([input_x], OpPatternMode.ELEWISE)

    for (_input_x, ) in ins:
        with tbe.compute():
            x_shape = shape_util.variable_shape([_input_x])
            data_x = tvm.placeholder(x_shape[0], dtype=x_dtype, name="data_x")
            res = nan_to_num_compute(data_x, output_y, nan, posinf, neginf, kernel_name)
            tensors.append([data_x, res])

        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {
              "name": kernel_name,
              "tensor_list": tensors
              }
    tbe.build(schedules, config)
