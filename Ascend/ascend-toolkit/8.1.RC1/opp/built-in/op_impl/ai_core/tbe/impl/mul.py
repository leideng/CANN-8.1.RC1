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
mul
"""
import te.platform as tbe_platform
from tbe import tvm
from te.lang import cce as tbe
from te.utils import shape_util
from te.utils import para_check
from te.utils.error_manager import error_manager_vector
from impl.util import util_common
from impl.dynamic.mul import static_reshape
from impl.dynamic.mul import op_select_format as mul_op_select_format
from impl.dynamic.mul import calc_input_tensor
from impl.dynamic.mul import mul_compute_for_batchmatmul
from impl.dynamic.mul import calc_input_shape


# 'pylint: disable=unused-argument,too-many-locals,invalid-name
def op_select_format(input1, input2, output, kernel_name="mul"):
    """
    Returns the dtype and format for Mul
    """
    return mul_op_select_format(input1, input2, output, kernel_name)


@tbe_platform.fusion_manager.fusion_manager.register("mul")
def mul_compute(input_x, input_y, output_data, is_scene_1d=False, kernel_name="mul"):
    """
    calculating element-wise mul

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of first input data
    input_y: TVM tensor
        the placeholder of second input data
    output_data: dict
        shape and dtype of output, should be broadcast shape and type as input
    kernel_name: str
        cce kernel name, default value is "mul"

    Returns
    -------
    output of the element-wise mul
    """
    shape_x = shape_util.shape_to_list(input_x.shape)
    x_dtype = input_x.dtype.lower()
    y_dtype = input_y.dtype.lower()
    is_mix_dtype = x_dtype != y_dtype

    if is_scene_1d:
        if x_dtype in ("uint8", "int8") or is_mix_dtype:
            input_x = tbe.cast_to(input_x, "float32")
            input_y = tbe.cast_to(input_y, "float32")

        input_y = tbe.broadcast(input_y, shape_x)
    else:
        batchmatmul_flag, input_x, input_y = calc_input_tensor(input_x, input_y)
        if batchmatmul_flag:
            return mul_compute_for_batchmatmul(input_x, input_y)

        shape_max, input_x, input_y = calc_input_shape(input_x, input_y)

        if x_dtype in ("uint8", "int8") or is_mix_dtype:
            input_x = tbe.cast_to(input_x, "float32")
            input_y = tbe.cast_to(input_y, "float32")

        input_x = tbe.broadcast(input_x, shape_max)
        input_y = tbe.broadcast(input_y, shape_max)

    res = tbe.vmul(input_x, input_y)

    if x_dtype in ("uint8", "int8"):
        res = util_common.uint8_int8_overflow_proc(res, x_dtype)

    output_dtype = output_data.get("dtype")
    if res.dtype != output_dtype:
        res = tbe.cast_to(res, output_dtype)

    return res


# 'pylint: disable=unused-argument, too-many-locals
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def mul(x, y, output, kernel_name="mul"):
    """
    do element-wise mul operation between two input tensors

    Parameters:
    ----------
    x : dict.
        shape, dtype of input x
    y : dict.
        shape, dtype of input y
    output : dict.
        shape, dtype of ouput
    kernel_name : str.
        cce kernel name, default value is "mul"

    Returns
    -------
    None
    """
    dtype_x = x.get("dtype").lower()
    dtype_y = y.get("dtype").lower()
    dtype_out = output.get("dtype").lower()

    mix_dtype_list = (("float16", "float32", "float32"), ("float32", "float16", "float32"))
    is_valid_mix_dtpye = (dtype_x, dtype_y, dtype_out) in mix_dtype_list

    if not is_valid_mix_dtpye and dtype_x != dtype_y:
        error_manager_vector.raise_err_inputs_dtype_not_equal(kernel_name, 'x', 'y', dtype_x, dtype_y)
    check_list = ("int32", "float16", "float32", "int16", "uint8", "int8")
    para_check.check_dtype(dtype_x, check_list, param_name="x")

    vmul_support = tbe_platform.api_check_support("te.lang.cce.vmul", "float32")
    if not vmul_support:
        new_check_list = list(check_list)
        new_check_list.remove("float32")
        para_check.check_dtype(dtype_x, new_check_list, param_name="x")

    shape_x, shape_y, is_scene_1d = static_reshape(x, y)

    input_x = tvm.placeholder(shape_x, dtype=dtype_x, name="x")
    input_y = tvm.placeholder(shape_y, dtype=dtype_y, name="y")

    res = mul_compute(input_x, input_y, output, is_scene_1d, kernel_name)

    with tvm.target.cce():
        sch = tbe.auto_schedule(res)

    config = {"name": kernel_name, "tensor_list": (input_x, input_y, res)}
    tbe.cce_build_code(sch, config)
