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
in_training_update_grad
"""

from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tuple_sum
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    This class for Constant.
    """
    # minimum positive number greater than 0
    EPSLON = 1e-6


# 'pylint: disable=too-many-locals,too-many-arguments,unused-argument,invalid-name
def in_training_update_grad_compute(dy,
                                    x,
                                    variance,
                                    mean,
                                    res_gamma,
                                    res_beta,
                                    format_dy,
                                    kernel_name="in_training_update_grad"):
    """
    DSL description of the layernorm_grad operator's mathematical

    Parameters
    ----------
    dy: TVM tensor
        the placeholder of input dy
    x: TVM tensor
        the placeholder of input x
    variance: TVM tensor
        the placeholder of input variance
    mean: TVM tensor
        the placeholder of input mean
    res_gamma: dict
        shape and dtype of output res_gamma
    res_beta: dict
        shape and dtype of output res_beta
    kernel_name: str
        cce kernel name, default value is "in_training_update_grad"

    Returns
    -------
    res_list: list
        [res_gamma, res_beta]
    """
    shape_dy = shape_util.shape_to_list(dy.shape)
    shape_var = shape_util.shape_to_list(variance.shape)
    dtype_dy = dy.dtype.lower()
    reduce_axis = []
    if format_dy == "NDC1HWC0":  # only support NDC1HWC0 and NC1HWC0
        reduce_axis = [1, 3, 4]
    else:
        reduce_axis = [2, 3]

    if dtype_dy == "float16":
        dy = tbe.cast_to(dy, "float32")
        x = tbe.cast_to(x, "float32")

    mean_inverse = tbe.vmuls(mean, tvm.const(-1, dtype="float32"))
    mean_inverse_broadcast = tbe.broadcast(mean_inverse, shape_dy)
    x_sub = tbe.vadd(x, mean_inverse_broadcast)

    data_adds = tbe.vadds(variance, Constant.EPSLON)
    data_rsqrt = tbe.vsqrt(data_adds)
    data_one = tbe.broadcast(tvm.const(1, "float32"), shape_var)

    data_rsqrts = tbe.vdiv(data_one, data_rsqrt)
    rsqrts_broadcast = tbe.broadcast(data_rsqrts, shape_dy)
    x_norm = tbe.vmul(x_sub, rsqrts_broadcast)

    scale_mul = tbe.vmul(dy, x_norm)

    res_gamma, res_beta = tuple_sum([scale_mul, dy], reduce_axis, True)

    return [res_gamma, res_beta]


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def in_training_update_grad(dy, x, variance, mean, res_gamma, res_beta, kernel_name="in_training_update_grad"):
    """
    in_training_update_grad operator interface implementation

    Parameters
    ----------
    dy: dict
        shape and dtype of input dy, only support float16, float32
    x: dict
        shape and dtype of input x, only support float16, float32
    variance: dict
        shape and dtype of input variance, only support float32
    mean: dict
        shape and dtype of input mean, only support float32
    res_gamma: dict
        shape and dtype of output res_gamma, only support float32
    res_beta: dict
        shape and dtype of output res_beta, only support float32
    kernel_name: str
        cce kernel name, default value is "in_training_update_grad"

    Returns
    -------
    None
    """
    shape_dy = dy.get("shape")
    shape_x = x.get("shape")
    shape_var = variance.get("shape")
    shape_mean = mean.get("shape")
    dtype_dy = dy.get("dtype").lower()
    dtype_var = variance.get("dtype").lower()
    format_dy = dy.get("format")

    para_check.check_dtype(dtype_dy, ("float16", "float32"), param_name="dy")
    para_check.check_dtype(dtype_var, ("float32",), param_name="variance")
    para_check.check_shape(shape_dy, param_name="dy")
    para_check.check_shape(shape_x, param_name="x")
    para_check.check_shape(shape_var, param_name="variance")
    para_check.check_shape(shape_mean, param_name="mean")

    data_dy = tvm.placeholder(shape_dy, name="data_dy", dtype=dtype_dy)
    data_x = tvm.placeholder(shape_x, name="data_x", dtype=dtype_dy)
    data_var = tvm.placeholder(shape_var, name="data_var", dtype=dtype_var)
    data_mean = tvm.placeholder(shape_mean, name="data_mean", dtype=dtype_var)

    res = in_training_update_grad_compute(data_dy, data_x, data_var, data_mean, res_gamma, res_beta, format_dy,
                                          kernel_name)

    with tvm.target.cce():
        sch = tbe.auto_schedule(res)

    config = {"name": kernel_name, "tensor_list": [data_dy, data_x, data_var, data_mean] + list(res)}

    tbe.build(sch, config)
