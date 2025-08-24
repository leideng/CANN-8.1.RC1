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
from impl.util.util_attr_common import get_attr_by_cls
from impl.util.util_attr_common import OpAttr


def scatter_update(dst, src, index, node_name):
    k = tvm.sparse_axis((0, index.shape[0]), name='k')
    sparse_compute = tvm.compute(
        dst.shape,
        lambda *i: tvm.sparse(tvm.select(index[k] == i[0], src(k, *i[1:])), axis=[k]),
        name=node_name
    )

    return sparse_compute


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument
# 'pylint: disable=too-many-locals,invalid-name
def sparse_apply_adadelta_d_compute(var, accum, accum_update, lr, rho, grad, indices, var_out, accum_out,
                                    accum_update_out, epsilon, use_locking=False,
                                    kernel_name="sparse_apply_adadelta_d"):
    epsilon = get_attr_by_cls(epsilon, OpAttr(0, "epsilon", "Float", 0.0000001), "float32")
    var_gathered = tbe.gather(var, indices, axis=0, support_out_of_bound_index=False)
    accum_gathered = tbe.gather(accum, indices, axis=0, support_out_of_bound_index=False)
    accum_update_gathered = tbe.gather(accum_update, indices, axis=0, support_out_of_bound_index=False)

    # `rho_gs = 1 - rho`
    scalar_one = tbe.broadcast(tvm.const(1.0, "float32"), (1,))
    rho_gs = tbe.vsub(scalar_one, rho)

    # `accum = rho * accum + rho_gs * square(grad)`
    lhs = tbe.vmuls(accum_gathered, rho[0])
    grad_square = tbe.vmul(grad, grad)
    rhs = tbe.vmuls(grad_square, rho_gs[0])
    accum_out = tbe.vadd(lhs, rhs)

    # `update = sqrt(accum_update + epsilon) * sqrt(accum + epsilon) * grad`
    lhs = tbe.vadds(accum_update_gathered, epsilon)
    lhs = tbe.vsqrt(lhs)
    rhs = tbe.vadds(accum_out, epsilon)
    rhs = tbe.vsqrt(rhs)
    rhs = tbe.vdiv(grad, rhs)
    update = tbe.vmul(lhs, rhs)

    # `var = var - update * lr`
    rhs = tbe.vmuls(update, lr[0])
    var_out = tbe.vsub(var_gathered, rhs)

    # `accum_update = rho * accum_update + rho_gs * square(update)`
    rhs = tbe.vmuls(accum_update_gathered, rho[0])
    lhs = tbe.vmul(update, update)
    lhs = tbe.vmuls(lhs, rho_gs[0])
    accum_update_out = tbe.vadd(lhs, rhs)

    var_res = scatter_update(var, var_out, indices, "var_out")
    accum_res = scatter_update(accum, accum_out, indices, "accum_out")
    accum_update_res = scatter_update(accum_update, accum_update_out, indices, "accum_update_out")

    return [var_res, accum_res, accum_update_res]


@register_operator("SparseApplyAdadeltaD", pattern="SparseApply")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_BOOL,
                            para_check.KERNEL_NAME)
def sparse_apply_adadelta_d(var,
                            accum,
                            accum_update,
                            lr,
                            rho,
                            grad,
                            indices,
                            var_out,
                            accum_out,
                            accum_update_out,
                            epsilon,
                            use_locking=False,
                            kernel_name="sparse_apply_adadelta_d"):
    """
    Updates "var" in specified index according to the Adadelta algorithm.

    accum = rho * accum + (1 - rho) * square(grad)
    update = sqrt(accum_update + epsilon) * sqrt(accum + epsilon)*grad
    accum_update = rho * accum_update + (1 - rho) * square(update);
    var = var - update * lr;

    Parameters
    ----------
    var: dict
        dict of tensor var, include shape and dtype,
        dtype only support float32.
    accum: dict
        dict of tensor accum, include shape and dtype.
        Must have the same dtype and shape as var.
    accum_update: dict
        dict of tensor accum_update, include shape and dtype.
        Must have the same dtype and shape as var.
    lr: dict
        dict of scalar lr,
        Must have the same dtype as var.
    grad: dict
        dict of tensor grad,
        Must have the same dtype  as var.
    indices: dict
       dict of tensor indices, include shape and dtype, only support int32.
    out_var: dict
        dict of out_var, include shape and dtype.
    out_accum: dict
        dict of out_accum, include shape and dtype.
    out_accum_update: dict
        dict of out_accum_update, include shape and dtype.
    rho: float
        scalar
    accum_updateentum: float
        scalar
    epsilon: float
        scalar
    use_locking: bool
        not used in this compute
    kernel_name: str
        kernel name, default value is "sparse_apply_adadelta_d"

    Returns:
    None
    """
    # The same input_shape_type values means inputs have identical shapes.
    extra_params = {"input_shape_type": [0, 0, 0, 1, 2]}
    ins = classify([var, accum, accum_update, grad, indices], OpPatternMode.SPARSE_APPLY, extra_params)
    schedules, tensors = [], []
    for (ins_var, ins_accum, ins_accum_update, ins_grad, ins_indices) in ins:
        with tbe.compute():
            var_shape, accum_shape, accum_update_shape, grad_shape, indices_shape = \
                shape_util.variable_shape([ins_var, ins_accum, ins_accum_update, ins_grad, ins_indices],
                                          op_mode="sparse_apply")
            data_var = tvm.placeholder(var_shape, var.get("dtype"), "data_var")
            data_accum = tvm.placeholder(accum_shape, accum.get("dtype"), "data_accum")
            data_accum_update = tvm.placeholder(accum_update_shape, accum.get("dtype"), "data_accum_update")
            data_grad = tvm.placeholder(grad_shape, grad.get("dtype"), "data_grad")
            data_indices = tvm.placeholder(indices_shape, indices.get("dtype"), "data_indices")
            lr = tvm.placeholder((1,), lr.get("dtype"), "lr")
            rho = tvm.placeholder((1,), rho.get("dtype"), "rho")
            res = sparse_apply_adadelta_d_compute(data_var, data_accum, data_accum_update, lr, rho, data_grad,
                                                  data_indices, var_out, accum_out, accum_update_out, epsilon,
                                                  use_locking, kernel_name)
            tensors.append([data_var, data_accum, data_accum_update, lr, rho, data_grad, data_indices, res[0], res[1],
                            res[2]])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": tensors}
    tbe.build(schedules, config)
