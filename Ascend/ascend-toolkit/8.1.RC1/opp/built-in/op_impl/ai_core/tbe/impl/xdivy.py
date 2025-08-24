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
xdivy
"""
from tbe.dsl.api import auto_schedule
from tbe.dsl.api import build
import te.lang.cce as tbe
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm


# 'pylint: disable=too-few-public-methods, too-many-instance-attributes
class Constant:
    """
    The class for constant.
    """
    SCALAR_ONE = 1
    MININUM_NUM_FLOAT = 2**(-126)
    MININUM_NUM_HALF = 2**(-24)
    MAX_ONE_CONST_FLOAT = 2**62
    MAX_TWO_CONST_FLOAT = 2**2
    MAX_CONST_HALF = 2**12


# 'pylint: disable=locally-disabled,too-many-locals,unused-argument
@register_operator_compute("xdivy", op_mode="static", support_fusion=True)
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
    input_data1 = shape_util.shape_to_list(input_x.shape)
    input_data2 = shape_util.shape_to_list(input_y.shape)
    shape_list = shape_util.broadcast_shapes(input_data1, input_data2,
                                             param_name_input1="input_x",
                                             param_name_input2="input_y")
    dtype = input_x.dtype

    broadcast_x = tbe.broadcast(input_x, shape_list[2])
    broadcast_y = tbe.broadcast(input_y, shape_list[2])
    broadcast_one = tbe.broadcast(tvm.const(Constant.SCALAR_ONE, dtype), shape_list[2], dtype)

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
            tbe_platform.api_check_support("tbe.dsl.vdiv", "float32"):
        broadcast_x = tbe.cast_to(broadcast_x, "float32")
        input_y_revised = tbe.cast_to(input_y_revised, "float32")
        has_improve_precision = True

    res = tbe.vdiv(broadcast_x, input_y_revised)

    if has_improve_precision:
        res = tbe.cast_to(res, dtype)

    return res


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
    shape_x = input_x.get("shape")
    dtype = input_x.get("dtype")
    shape_y = input_y.get("shape")
    dtype_y = input_y.get("dtype")

    shape_util.compare_tensor_dict_key(input_x, input_y, "dtype")
    para_check.check_shape(shape_x, param_name="input_x")
    para_check.check_shape(shape_y, param_name="input_y")
    shape_list = shape_util.broadcast_shapes(shape_x, shape_y,
                                             param_name_input1="input_x",
                                             param_name_input2="input_y")
    input_dtype = dtype.lower()
    input_dtype_y = dtype_y.lower()
    check_list = ("float16", "float32")
    para_check.check_dtype(input_dtype, check_list, param_name="input_x")
    para_check.check_dtype(input_dtype_y, check_list, param_name="input_y")

    reshape_x, reshape_y = shape_util.refine_shapes_for_broadcast(shape_list[0], shape_list[1])
    data_x = tvm.placeholder(reshape_x, dtype=input_dtype, name="data_x")
    data_y = tvm.placeholder(reshape_y, dtype=input_dtype, name="data_y")

    res = xdivy_compute(data_x, data_y, output_z, kernel_name)
    with tvm.target.cce():
        sch = auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_x, data_y, res]}
    build(sch, config)
