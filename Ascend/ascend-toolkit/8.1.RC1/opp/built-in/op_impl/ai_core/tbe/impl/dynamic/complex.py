# Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
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
complex
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute


@register_operator_compute("Complex", op_mode="dynamic")
def complex_compute(real, imag, out, Tout=None, kernel_name="complex"):
    """
    Construct a complex tensor with its real part equal to real and its imaginary part equal to imag.
    :param real: dict
        shape, format, and datatype of real
    :param imag: dict
        shape, format, and datatype of imag
    :param out: dict
        shape, format, and datatype of out
    :param Tout: dict
        None
    :param kernel_name:
    :return:
    """
    _, _, shape_max = shape_util.broadcast_shapes(real.shape, imag.shape)
    broadcasted_real = tbe.broadcast(real, shape_max)
    broadcasted_imag = tbe.broadcast(imag, shape_max)
    return tbe.complex(broadcasted_real, broadcasted_imag)


@register_operator("Complex")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_INT, para_check.KERNEL_NAME)
def complex(real, imag, out, Tout=None, kernel_name="complex"):
    """
    Construct a complex tensor with its real part equal to real and its imaginary part equal to imag.
    :param real: dict
        shape, format, and datatype of real
    :param imag: dict
        shape, format, and datatype of imag
    :param out: dict
        shape, format, and datatype of out
    :param Tout: dict
        None
    :param kernel_name:
    :return:
    """
    input_dtype = real.get("dtype").lower()
    ins = classify([real, imag], OpPatternMode.ELEWISE_WITH_BROADCAST)
    schedules, tensors = [], []
    for (_real, _imag) in ins:
        with tbe.compute():
            real_shape, imag_shape = shape_util.variable_shape([_real, _imag])
            re = tvm.placeholder(real_shape, name="real", dtype=input_dtype)
            im = tvm.placeholder(imag_shape, name="imag", dtype=input_dtype)
            res = complex_compute(re, im, out, None, kernel_name)
            tensors.append([re, im, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
