#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Copyright 2021 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

AddMatMatElements
"""
from impl.util.util_common import is_unknown
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import OpPatternMode


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument,invalid-name
@register_operator_compute("AddMatMatElements", op_mode="dynamic", support_fusion=True)
def add_mat_mat_elements_compute(c, a, b, beta, alpha, c_out, kernel_name="add_mat_mat_elements"):
    """
    add_mat_mat_elements compute function

    c_out = c * beta + alpha * a * b

    Parameters
    ----------
    c : mutable tensor c.

    a : mutable tensor a.

    b : mutable tensor b.

    beta : mutable scalar beta.

    alpha : mutable scalar alpha.

    c_out : the dict of output.

    kernel_name : str
        cce kernel name, default value is "add_mat_mat_elements"

    Returns
    -------
    res : tvm.tensor
        tensor of result
    """
    a = tbe.broadcast(a, c.shape)
    b = tbe.broadcast(b, c.shape)

    ab = tbe.vmul(a, b)

    ab_val = tbe.vmuls(ab, alpha[0])

    c_out = tbe.vmuls(c, beta[0])

    res = tbe.vadd(c_out, ab_val)
    return res


# 'pylint: disable=too-many-locals
@register_operator("AddMatMatElements")
@para_check.check_op_params(para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def add_mat_mat_elements(c, a, b, beta, alpha, c_out, kernel_name="add_mat_mat_elements"):
    """
    Update var by subtracting value from it.
    c_out = c * beta + alpha * a * b

    Parameters:
    ----------
    c : the dict of mutable tensor c, only support float16, float32

    a : the dict of mutable tensor c, Must have the same data type as `c`.

    b : the dict of mutable tensor c, Must have the same data type as `c`.

    beta : the dict of scalar tensor c, Must have the same data type as `c`.

    alpha : the dict of scalar tensor c, Must have the same data type as `c`.

    out : the dict of output.

    kernel_name : str
        cce kernel name, default value is "add_mat_mat_elements"

    Returns
    -------
    None
    """
    check_list = ('float16', 'float32')
    dtype_c = c.get("dtype").lower()
    para_check.check_dtype(dtype_c, check_list, param_name="c")

    shape_util.compare_tensor_dict_key(c, a, "dtype")
    shape_util.compare_tensor_dict_key(c, b, "dtype")
    shape_util.compare_tensor_dict_key(c, beta, "dtype")
    shape_util.compare_tensor_dict_key(c, alpha, "dtype")

    beta_shape = beta.get("shape")
    alpha_shape = alpha.get("shape")

    if tuple(beta_shape) != (1,) and not is_unknown([beta]):
        raise RuntimeError("beta has to be scalar")
    if tuple(alpha_shape) != (1,) and not is_unknown([alpha]):
        raise RuntimeError("alpha has to be scalar")

    ins = classify([c, a, b], OpPatternMode.ELEWISE)
    schedules, tensors = [], []
    for (_c, _a, _b) in ins:
        with tbe.compute():

            c_shape, a_shape, b_shape = shape_util.variable_shape([_c, _a, _b])

            tensor_c = tvm.placeholder(c_shape, dtype_c, "tensor_c")
            tensor_a = tvm.placeholder(a_shape, dtype_c, "tensor_a")
            tensor_b = tvm.placeholder(b_shape, dtype_c, "tensor_b")
            tensor_beta = tvm.placeholder([1], dtype_c, "tensor_beta")
            tensor_alpha = tvm.placeholder([1], dtype_c, "tensor_alpha")

            res = add_mat_mat_elements_compute(tensor_c, tensor_a, tensor_b, tensor_beta,
                                               tensor_alpha, c_out, kernel_name)

        tensors.append([tensor_c, tensor_a, tensor_b, tensor_beta, tensor_alpha, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
