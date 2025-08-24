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
fused_mul_add_n
"""

from tbe.dsl.api import build
from tbe.dsl.api import auto_schedule
import te.lang.cce as tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check


def _mul_add_n_compute(data_x, data_y, data_z):
    """
    fused mul+add_n, output = x * z + y
    res : output of the data's mul+add_n
    """
    data_z = tbe.broadcast(data_z, data_x.shape)
    res = tbe.vmul(data_x, data_z)
    res = tbe.vadd(data_y, res)
    return res


# 'pylint: disable=unused-argument,too-many-locals
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def fused_mul_add_n(input_x, input_y, input_z, output, kernel_name="fused_mul_add_n"):
    """
    algorithm: fused mul+add_n
    calculating output = input_x * input_z + input_y

    Parameters
    ----------
    input_x : dict of input_x, tensor
    input_y: dict of input_y, tensor
    input_z: dict of input_z, scalar
    output : dict of output

    kernel_name : string
        cce kernel name, default value is fused_mul_add_n

    Returns
    -------
    None
    """

    check_list = ("float16", "float32", "int32", "int16")
    shape_x = input_x.get("shape")
    dtype_x = input_x.get("dtype")
    para_check.check_shape(shape_x, param_name="input_x")
    para_check.check_dtype(dtype_x, check_list, param_name="input_x")
    shape_y = input_y.get("shape")
    dtype_y = input_y.get("dtype")
    para_check.check_shape(shape_y, param_name="input_y")
    para_check.check_dtype(dtype_y, check_list, param_name="input_y")
    dtype_z = input_z.get("dtype")
    shape_z = [1] * len(shape_x)
    para_check.check_shape(shape_z, param_name="input_z")
    para_check.check_dtype(dtype_z, check_list, param_name="input_z")

    data_x = tvm.placeholder(shape_x, name="input_x", dtype=dtype_x)
    data_y = tvm.placeholder(shape_y, name="input_y", dtype=dtype_y)
    data_z = tvm.placeholder(shape_z, name="input_z", dtype=dtype_z)

    res = _mul_add_n_compute(data_x, data_y, data_z)

    with tvm.target.cce():
        schedule = auto_schedule(res)

    tensor_list = [data_x, data_y, data_z, res]

    config = {"print_ir": False, "name": kernel_name, "tensor_list": tensor_list}

    build(schedule, config)
