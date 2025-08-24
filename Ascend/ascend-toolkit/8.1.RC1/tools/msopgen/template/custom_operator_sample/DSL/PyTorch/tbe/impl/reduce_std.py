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
reduce_std
"""
import te.lang.cce as tbe

from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util

SHAPE_SIZE_LIMIT = 2147483648


# pylint: disable=invalid-name,too-many-locals,unused-argument,too-many-arguments
# Analysis parameter dim
def reduce_std_check_dim(axis_dim, shape_x, dim):
    """
    reduce_std_check_dim
    """
    dims = len(shape_x)
    if isinstance(dim, int):
        axis_dim.append(dim)
    elif ((dim is None) or (len(dim) == 0)):
        for i in range(dims):
            axis_dim.append(i)
    else:
        for i in dim:
            if ((i < 0) and ((i + dims) in axis_dim)) or (i in axis_dim):
                continue
            axis_dim.append(int((i + dims) % dims))
    return axis_dim


@register_operator_compute("ReduceStd", op_mode="static", support_fusion=True)
def reduce_std_compute(x, dim, unbiased, keepdim, kernel_name="reduce_std"):
    """
    calculating data

    Parameters
    ----------
    x : TVM tensor
        the placeholder of x
    dim : int or intlist
        dimension to calculate, default value is None
    unbiased : bool
        control Bessel deviation, default value is True
    keepdim : bool
        hold dimension or not, default value is False
    kernel_name : str
        kernel name, default value is "pt_add"

    Returns
    -------
    output tensorlist
    """

    # Analysis parameter dim
    shape_x = shape_util.shape_to_list(x.shape)

    axis_dim = []
    axis_dim = reduce_std_check_dim(axis_dim, shape_x, dim)

    # got total number of tensor
    reduce_ele = 1.0
    for i in axis_dim:
        reduce_ele *= shape_x[i]
    cof = reduce_ele**(-1)

    # calculate the mu_muls
    mu_muls = tbe.vmuls(x, cof)

    # calulate mu
    mu = tbe.sum(mu_muls, axis=axis_dim, keepdims=True)

    # broadcast
    mu_broadcast = tbe.broadcast(mu, shape_x)

    # calculate x-mubroadcast
    x_mu_sub = tbe.vsub(x, mu_broadcast)

    # calculate x_mu_sub^2
    var_mul = tbe.vmul(x_mu_sub, x_mu_sub)

    # Divided by N or (N-1)
    if unbiased:
        cof_unbiased = (reduce_ele-1.0)**(-1)
        var_muls = tbe.vmuls(var_mul, cof_unbiased)
    else:
        var_muls = tbe.vmuls(var_mul, cof)

    # sum
    var = tbe.sum(var_muls, axis=axis_dim, keepdims=keepdim)

    # calculate the square root
    y = tbe.vsqrt(var)

    # calculate mu_res and return
    mu_res = tbe.sum(mu_muls, axis=axis_dim, keepdims=keepdim)

    # form a list and return
    return [y, mu_res]


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_LIST_INT,
                            para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_BOOL,
                            para_check.KERNEL_NAME)
def reduce_std(x, y1, y2, dim=None, unbiased=True, keepdim=False,
               kernel_name="reduce_std"):

    # calculating data parameters
    check_list = ("float16", "float32")

    shape_x = x.get("shape")
    dtype_x = x.get("dtype").lower()
    para_check.check_dtype(dtype_x, check_list, param_name="x")
    para_check.check_shape(shape_x, param_name="x")

    data_x = tvm.placeholder(x.get("shape"), dtype=x.get("dtype"), name="data_x")

    res = reduce_std_compute(data_x, dim, unbiased, keepdim, kernel_name)

    with tvm.target.cce():
        schedule = tbe.auto_schedule(res)

    config = {"name": kernel_name,
              "enable_group_inplace": True,
              "tensor_list": [data_x] + list(res)}
    tbe.build(schedule, config)
