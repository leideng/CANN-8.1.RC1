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
dynamic xdivy
"""
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.util_soc_common import after_v200


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant.
    """
    SCALAR_ONE = 1
    MININUM_NUM_FLOAT = 2 ** (-126)
    MININUM_NUM_HALF = 2 ** (-24)
    MAX_ONE_CONST_FLOAT = 2 ** 62
    MAX_TWO_CONST_FLOAT = 2 ** 2
    MAX_CONST_HALF = 2 ** 12


# 'pylint: disable=locally-disabled,too-many-locals,unused-argument
@register_operator_compute("Xdivy", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def xdivy_compute(input_x, input_y, output_z, kernel_name="xdivy"):
    """
    xdivy compute
    calculating data's xdivy,return 0 if x==0 and x/y otherwise, elementwise

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input_x
    input_y: TVM tensor
        the placeholder of input_y
    output_z: dict
        dict with keys(shape and dtype) of output
    kernel_name : str
        kernel name, default value is "xdivy"

    Returns
    -------
    res: TVM tensor
        the result of xdivy compute
    """
    shape_list = shape_util.broadcast_shapes(input_x.shape, input_y.shape,
                                             param_name_input1="input_x",
                                             param_name_input2="input_y")
    dtype = input_x.dtype

    broadcast_x = tbe.broadcast(input_x, shape_list[2])
    broadcast_y = tbe.broadcast(input_y, shape_list[2])
    broadcast_one = tbe.broadcast(tvm.const(Constant.SCALAR_ONE, dtype), shape_list[2], dtype)

    float_list = ("float16", "float32")
    if after_v200() and dtype in float_list:
        res_div = tbe.vdiv(broadcast_x, broadcast_y)
        res_cmp = tbe.vcmp(broadcast_x, 0.0, 'eq', mode='bit')
        return tbe.vsel(res_cmp, 0.0, res_div)

    abs_x = tbe.vabs(broadcast_x)
    abs_y = tbe.vabs(broadcast_y)
    add_x_y = tbe.vadd(abs_x, abs_y)

    if dtype == "float32":
        data_min = tbe.broadcast(tvm.const(Constant.MININUM_NUM_FLOAT, dtype=dtype),
                                 shape_list[2], dtype)
    elif dtype == "float16":
        data_min = tbe.broadcast(tvm.const(Constant.MININUM_NUM_HALF, dtype=dtype),
                                 shape_list[2], dtype)

    zero_x_y = tbe.vmin(add_x_y, data_min)

    if dtype == "float32":
        data_mul1 = tbe.vmuls(zero_x_y, tvm.const(Constant.MAX_ONE_CONST_FLOAT,
                                                  dtype=dtype))
        data_mul2 = tbe.vmuls(data_mul1, tvm.const(Constant.MAX_ONE_CONST_FLOAT,
                                                   dtype=dtype))
        mul_data = tbe.vmuls(data_mul2, tvm.const(Constant.MAX_TWO_CONST_FLOAT,
                                                  dtype=dtype))
    elif dtype == "float16":
        data_mul1 = tbe.vmuls(zero_x_y, tvm.const(Constant.MAX_CONST_HALF,
                                                  dtype=dtype))
        mul_data = tbe.vmuls(data_mul1, tvm.const(Constant.MAX_CONST_HALF,
                                                  dtype=dtype))

    sub_x_y_zero = tbe.vsub(mul_data, broadcast_one)
    abs_x_y_zero = tbe.vabs(sub_x_y_zero)
    input_y_revised = tbe.vadd(broadcast_y, abs_x_y_zero)

    has_improve_precision = False
    if dtype == "float16" and \
            tbe_platform.api_check_support("tbe.dsl.vdiv",
                                           "float32"):
        broadcast_x = tbe.cast_to(broadcast_x, "float32")
        input_y_revised = tbe.cast_to(input_y_revised, "float32")
        has_improve_precision = True

    res = tbe.vdiv(broadcast_x, input_y_revised)

    if has_improve_precision:
        res = tbe.cast_to(res, dtype)

    return res


@register_operator("Xdivy")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def xdivy(input_x, input_y, output_z, kernel_name="xdivy"):
    """
    algorithm: xdivy
    calculating data's xdivy,return 0 if x==0 and x/y otherwise, elementwise

    Parameters
    ----------
    input_x: dict
        dict with keys(shape and dtype) of input_x
    input_y: dict
        dict with keys(shape and dtype) of input_y
    output_z: dict
        dict with keys(shape and dtype) of output
    kernel_name : str
        kernel name, default value is "xdivy"

    Returns
    -------
    None
    """
    dtype_x = input_x.get("dtype")
    dtype_y = input_y.get("dtype")
    input_dtype_x = dtype_x.lower()
    input_dtype_y = dtype_y.lower()
    check_list = ("bfloat16", "float16", "float32")
    para_check.check_dtype(input_dtype_x, check_list, param_name="input_x")
    para_check.check_dtype(input_dtype_y, check_list, param_name="input_y")
    if input_dtype_x != input_dtype_y:
        error_detal = "input_x and input_y should have same type"
        error_manager_vector.raise_err_two_input_dtype_invalid(kernel_name,
                                                               "input_x",
                                                               "input_y",
                                                               error_detal)
    ins = classify([input_x, input_y], OpPatternMode.ELEWISE_WITH_BROADCAST)
    schedules, tensors = [], []
    for (_input_x, _input_y) in ins:
        with tbe.compute():
            # shape
            shape_x1, shape_x2 = shape_util.variable_shape([_input_x, _input_y])
            # mul_compute
            data_x1 = tvm.placeholder(shape_x1, dtype=input_dtype_x, name="data_x1")
            data_x2 = tvm.placeholder(shape_x2, dtype=input_dtype_y, name="data_x2")
            res = xdivy_compute(data_x1, data_x2, output_z, kernel_name)
            tensors.append((data_x1, data_x2, res))
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    # build
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
