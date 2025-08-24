#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright 2022 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""
from __future__ import absolute_import

from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.util_attr_common import get_attr_by_cls
from impl.util.util_attr_common import OpAttr


def scatter_update(dst, src, index, node_name):
    k = tvm.sparse_axis((0, index.shape[0]), name='k')
    sparse_compute = tvm.compute(dst.shape,
                                 lambda *i: tvm.sparse(tvm.select(index[k] == i[0], src(k, *i[1:])), axis=[k]),
                                 name=node_name)
    return sparse_compute


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument
# 'pylint: disable=too-many-locals,invalid-name
def sparse_apply_adagrad_v2_d_compute(var, accum, grad, indices, var_out, accum_out, lr, epsilon, use_locking=False,
                                      update_slots=True, kernel_name="sparse_apply_adagrad_v2_d"):
    """
    Adds sparse updates to the variable referenced by resource.

    Parameters
    ----------
    var: TVM tensor
        data of input.
        source data type, support  "float32"
    accum: TVM tensor
        data of input.
        source data type, support "float32"
    grad: TVM tensor
        data of input
        source data type should ne same as var
    indices: TVM tensor
         A tensor of indices into var, support "int32"
    var_out: dict
        A Tensor. Has the same type and format as input "var".
    accum_out: dict
        A Tensor. Has the same type and format as input "var".
    lr: float
        scalar
    epsilon: float
        scalar
    use_locking: bool
        not used in this compute
    kernel_name: str
        kernel name, default value is "sparse_apply_adagrad_v2_d"

    Returns:
    -----------
    res: TVM tensor list
        the result of sparse_apply_adagrad_v2_d
    """
    lr = get_attr_by_cls(lr, OpAttr(0, "lr", "Float", 0.01), "float32")
    epsilon = get_attr_by_cls(epsilon,
                              OpAttr(1, "epsilon", "Float", 0.0000001),
                              "float32")
    var_gathered = tbe.gather(var, indices, axis=0, support_out_of_bound_index=False)
    accum_gathered = tbe.gather(accum, indices, axis=0, support_out_of_bound_index=False)
    # `accum += grad * grad if update_slots is true`
    if update_slots:
        grad_square = tbe.vmul(grad, grad)
        accum_out = tbe.vadd(accum_gathered, grad_square)
    else:
        accum_out = accum_gathered
    accum_out_gm = scatter_update(accum, accum_out, indices, "accum_out_gm")
    # `var -= lr * grad / (sqrt(accum) + epsilon)`
    sqrt = tbe.vsqrt(accum_out)
    adds = tbe.vadds(sqrt, epsilon)
    div = tbe.vdiv(grad, adds)
    muls = tbe.vmuls(div, lr)
    sub = tbe.vsub(var_gathered, muls)
    var_out = scatter_update(var, sub, indices, "var_out")
    return [var_out, accum_out_gm]


# 'pylint: disable=too-many-arguments,unused-argument,invalid-name
@register_operator("SparseApplyAdagradV2D", pattern="SparseApply")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_BOOL,
                            para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def sparse_apply_adagrad_v2_d(var,
                              accum,
                              grad,
                              indices,
                              var_out,
                              accum_out,
                              lr,
                              epsilon,
                              use_locking=False,
                              update_slots=True,
                              kernel_name="sparse_apply_adagrad_v2_d"):
    """
    Adds sparse updates to the variable referenced by resource.

    Parameters
    ----------
    var: dict
        data of input.
        source data type, support  "float32"
    accum: dict
        data of input.
        source data type, support "float32"
    grad: dict
        data of input
        source data type should ne same as var
    indices: dict
         A tensor of indices into var, support "int32"
    var_out: dict
        A Tensor. Has the same type and format as input "var".
    accum_out: dict
        A Tensor. Has the same type and format as input "var".
    lr: float
        scalar
    epsilon: float
        scalar
    use_locking: bool
        not used in this compute
    kernel_name: str
        kernel name, default value is "sparse_apply_adagrad_v2_d"

    Returns:
    None
    """
    extra_params = {"input_shape_type": [0, 0, 1, 2]}  # the same number means the same shape
    ins = classify([var, accum, grad, indices], OpPatternMode.SPARSE_APPLY, extra_params)
    schedules, tensors = [], []
    for (ins_var, ins_accum, ins_grad, ins_indices) in ins:
        with tbe.compute():
            var_shape, accum_shape, grad_shape, indices_shape = shape_util.variable_shape([ins_var, ins_accum,
                                                                                           ins_grad, ins_indices],
                                                                                          op_mode="sparse_apply")
            data_var = tvm.placeholder(var_shape, var.get("dtype"), "data_var")
            data_accum = tvm.placeholder(accum_shape, accum.get("dtype"), "data_accum")
            data_grad = tvm.placeholder(grad_shape, grad.get("dtype"), "data_grad")
            data_indices = tvm.placeholder(indices_shape, indices.get("dtype"), "data_indices")
            res = sparse_apply_adagrad_v2_d_compute(data_var, data_accum, data_grad, data_indices, var_out, accum_out,
                                                    lr, epsilon, use_locking, update_slots, kernel_name)
            tensors.append([data_var, data_accum, data_grad, data_indices, res[0], res[1]])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    config = {"name": kernel_name,
              "tensor_list": tensors}
    tbe.build(schedules, config)
