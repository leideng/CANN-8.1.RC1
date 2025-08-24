#!/usr/bin/python
# -*- coding: utf-8 -*-
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
mish
"""
import functools

from tbe.dsl.api import build
from tbe.dsl.api import auto_schedule
import te.lang.cce as tbe
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import OpImplMode
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tbe_platform
from impl.util.util_common import check_op_impl_mode


# 'pylint: disable=unused-argument
@register_operator_compute("mish", op_mode="static", support_fusion=True, support_bfp16=True)
def mish_compute(input_x, output_y, kernel_name="mish", impl_mode=OpImplMode.HIGH_PERFORMANCE):
    """
    algorithm: mish
    calculating data's mish,y= x*(1 - 2/(1+(1+exp(x))^2))

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input data
    output_y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name: str
        cce kernel name, default value is mish

    Returns
    -------
    res : tvm.tensor
        the result of mish
    """
    dtype = input_x.dtype
    if impl_mode == OpImplMode.SUPER_PERFORMANCE:
        """
        `x*(1-2/(1+(1+(1+x/64)^64)^2))`
        """
        res_1 = tbe.vadds(tbe.vmuls(input_x, 0.015625), 1)
        res_pow_2 = tbe.vmul(res_1, res_1)
        res_pow_4 = tbe.vmul(res_pow_2, res_pow_2)
        res_pow_8 = tbe.vmul(res_pow_4, res_pow_4)
        res_pow_16 = tbe.vmul(res_pow_8, res_pow_8)
        res_pow_32 = tbe.vmul(res_pow_16, res_pow_16)
        res_pow_64 = tbe.vmul(res_pow_32, res_pow_32)
        res_2 = tbe.vadds(res_pow_64, 1)
        res_3 = tbe.vmul(res_2, res_2)
        res_4 = tbe.vadds(res_3, 1)
        res_rec = tbe.vrec(res_4, priority_flag=0)
        res_5 = tbe.vmuls(res_rec, -2)
        res_6 = tbe.vadds(res_5, 1)
        res = tbe.vmul(res_6, input_x)
        return res

    else:
        exp_val = tbe.vexp(input_x)
        add_exp_val = tbe.vadds(exp_val, tvm.const(1, dtype))
        pow_var = tbe.vmul(add_exp_val, add_exp_val)
        add_val = tbe.vadds(pow_var, tvm.const(1, dtype))
        if impl_mode == OpImplMode.HIGH_PERFORMANCE:
            rec_val = tbe.vrec(add_val, priority_flag=0)
            out_val = tbe.vmuls(rec_val, tvm.const(-2, dtype=dtype))
        else:
            is_support_inf_nan = tbe_platform.get_soc_spec("SHORT_SOC_VERSION") in ("Ascend910B", "Ascend910_93")
            if is_support_inf_nan:
                data_two = tbe.broadcast(tvm.const(-2.0, dtype), shape_util.shape_to_list(input_x.shape))
                out_val = tbe.vdiv(data_two, add_val)
            else:
                rec_val = tbe.vrec(add_val, priority_flag=1)
                out_val = tbe.vmuls(rec_val, tvm.const(-2, dtype=dtype))

        add_val2 = tbe.vadds(out_val, tvm.const(1, dtype=dtype))
        res = tbe.vmul(input_x, add_val2)
    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def mish(input_x, output_y, kernel_name="mish", impl_mode=OpImplMode.HIGH_PERFORMANCE):
    """
    algorithm: mish
    calculating data's mish,y= x*(1 - 2/(1+(1+exp(x))^2))

    Parameters
    ----------
    input_x : dict
        shape and dtype of input, only support float16, float32, bfloat16
    output_y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        cce kernel name, default value is mish

    Returns
    -------
    None
    """
    check_op_impl_mode(impl_mode,
                       [OpImplMode.HIGH_PERFORMANCE, OpImplMode.HIGH_PRECISION, OpImplMode.SUPER_PERFORMANCE],
                       kernel_name)

    input_shape = input_x.get("shape")
    input_format = input_x.get("format")
    input_dtype = input_x.get("dtype").lower()
    para_check.check_shape(input_shape, param_name="input_x")
    check_list = ("float16", "float32", "bfloat16")
    para_check.check_dtype(input_dtype, check_list, param_name="input_x")
    para_check.check_format(input_format)

    # fuse single axis
    fuseshape = [1]
    fuseshape[0] = functools.reduce(lambda x, y: x * y, input_shape)

    data_x = tvm.placeholder(fuseshape, dtype=input_dtype, name="data_x")
    res = mish_compute(data_x, output_y, kernel_name, impl_mode)
    with tvm.target.cce():
        schedule = auto_schedule(res)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": [data_x, res]}
    build(schedule, config)
