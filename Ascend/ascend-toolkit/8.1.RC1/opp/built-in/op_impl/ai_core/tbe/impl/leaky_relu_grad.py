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
leaky_relu_grad
"""
from tbe.dsl.api import build
from tbe.dsl.api import auto_schedule
import te.lang.cce as tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import register_operator_compute


# 'pylint: disable=unused-argument,invalid-name,too-many-locals
@register_operator_compute("leaky_relu_grad", op_mode="static", support_fusion=True)
def leaky_relu_grad_compute(g, x, y, negative_slope=0,
                            kernel_name="leaky_relu_grad"):
    """
    calculate the backpropagation of leaky_relu operation
    y = gradients(x>0) or negative_slope*gradients(x<=0).

    Parameters
    ----------
    g : TVM tensor
        the placeholder of input g
    x : TVM tensor
        the placeholder of input x
    y : dict
        dict of output y, include keys(shape and dtype)
    negative_slope : float or int
        allow non-zero slope for negative inputs to speed up optimization
    kernel_name : str
        kernel name, default value is "leaky_relu_grad"

    Returns
    -------
    res: TVM tensor
        the result of leaky_relu_grad_compute
    """

    shape_list = shape_util.produce_shapes(
        shape_util.shape_to_list(g.shape),
        shape_util.shape_to_list(x.shape))
    para_check.check_tensor_shape_size(shape_list[2])

    dtype = g.dtype
    g = tbe.broadcast(g, shape_list[2])
    x = tbe.broadcast(x, shape_list[2])

    if dtype == "float32":
        help_min = tvm.const(2 ** (-126), "float32")
        help_rec_one = tvm.const(2 ** 38, "float32")
        help_rec_sec = tvm.const(2 ** 44, "float32")
    elif dtype == "float16":
        help_min = tvm.const(2 ** (-24), "float16")
        help_rec_one = tvm.const(2 ** 12, "float16")
        help_rec_sec = help_rec_one

    tmp_min_x = tbe.vmins(x, help_min)
    tmp_max_x = tbe.vmaxs(tmp_min_x, tvm.const(0, "float32"))
    tmp_mul_x = tbe.vmuls(tmp_max_x, help_rec_one)

    if dtype == "float32":
        tmp_mul_x = tbe.vmuls(tmp_mul_x, help_rec_sec)

    result_tmp_right = tbe.vmuls(tmp_mul_x, help_rec_sec)

    result_sub = tbe.vadds(result_tmp_right, tvm.const(-1, "float32"))
    result_abs = tbe.vabs(result_sub)
    result_tmp_left = tbe.vmuls(result_abs, negative_slope)

    result_tmp = tbe.vadd(result_tmp_left, result_tmp_right)

    res = tbe.vmul(g, result_tmp)
    return res


# @register_operator("LeakyReluGrad")
@para_check.check_input_type(dict, dict, dict, (int, float), str)
def leaky_relu_grad(g, x, y, negative_slope=0, kernel_name="leaky_relu_grad"):
    """
    calculate the backpropagation of leaky_relu operation
    y = gradients(x>0) or negative_slope*gradients(x<=0).
    support dtype:float16,float32

    Parameters
    ----------
    g : dict
        the backpropagated gradients to the corresponding leaky_relu operation
    x : dict
        the x passed as output of leaky_relu operation
    y : dict
        the output of leaky_relu back propagation
    negative_slope : float or int
        allow non-zero slope for negative inputs to speed up optimization
    kernel_name : str
        kernel name, default value is "leaky_relu_grad"

    Returns
    -------
    None
    """

    shape_g = g.get("shape")
    shape_x = x.get("shape")
    dtype_g = g.get("dtype").lower()
    dtype_x = x.get("dtype").lower()

    para_check.check_kernel_name(kernel_name)
    para_check.check_shape_rule(shape_g)
    para_check.check_shape_rule(shape_x)
    para_check.check_tensor_shape_size(shape_g)
    para_check.check_tensor_shape_size(shape_x)

    shape_list = shape_util.produce_shapes(shape_g, shape_x)
    para_check.check_tensor_shape_size(shape_list[2])

    # check input tensor data_type
    check_list = ["float16", "float32"]
    para_check.check_dtype_rule(dtype_g, check_list)
    para_check.check_dtype_rule(dtype_x, check_list)
    shape_util.compare_tensor_dict_key(g, x, "dtype")

    shape_g, shape_x = shape_util.refine_shapes_for_broadcast(shape_list[0], shape_list[1])
    data_g = tvm.placeholder(shape_g, name="data_g", dtype=dtype_g)
    data_x = tvm.placeholder(shape_x, name="data_x", dtype=dtype_g)
    res = leaky_relu_grad_compute(data_g, data_x, y,
                                  negative_slope, kernel_name)

    with tvm.target.cce():
        schedule = auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_g, data_x, res]}

    build(schedule, config)
