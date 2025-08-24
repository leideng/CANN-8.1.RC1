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
fused_mul_add_n
"""

from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import register_operator


@register_operator_compute("FusedMulAddN", op_mode="dynamic", support_fusion=False, support_bfp16=True)
def fused_mul_add_n_compute(data_x, data_y, data_z, output, kernel_name):
    """
    fused mul+add_n, output = x * z + y
    res : output of the data's mul+add_n
    """
    res = tbe.vmuls(data_x, data_z[0])
    res = tbe.vadd(data_y, res)
    return res


# 'pylint: disable=unused-argument,too-many-locals
@register_operator("FusedMulAddN")
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

    check_list = ("bfloat16", "float16", "float32", "int32", "int16")
    shape_x = input_x.get("shape")
    dtype_x = input_x.get("dtype")
    para_check.check_shape(shape_x, param_name="input_x")
    para_check.check_dtype(dtype_x, check_list, param_name="input_x")
    shape_y = input_y.get("shape")
    dtype_y = input_y.get("dtype")
    para_check.check_shape(shape_y, param_name="input_y")
    para_check.check_dtype(dtype_y, check_list, param_name="input_y")
    dtype_z = input_z.get("dtype")
    para_check.check_dtype(dtype_z, check_list, param_name="input_z")

    z_shape = [1]
    data_z = tvm.placeholder(z_shape, dtype=dtype_z, name="data_z")

    ins = classify([input_x, input_y], OpPatternMode.ELEWISE)
    schedules, tensors = [], []
    for (_x, _y) in ins:
        with tbe.compute():
            x_shape, y_shape = shape_util.variable_shape([_x, _y])
            data_x = tvm.placeholder(x_shape, dtype=dtype_x, name="data_x")
            data_y = tvm.placeholder(y_shape, dtype=dtype_y, name="data_y")

            res = fused_mul_add_n_compute(data_x, data_y, data_z, output, kernel_name)
            tensors.append((data_x, data_y, data_z, res))
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
