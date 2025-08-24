# Copyright 2022 Huawei Technologies Co., Ltd
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
sparse_apply_proximal_adagrad
"""
from __future__ import absolute_import

from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util import util_compute


def scatter_update(dst, src, index, node_name):
    k = tvm.sparse_axis((0, index.shape[0]), name='k')
    sparse_compute = tvm.compute(
        dst.shape,
        lambda *i: tvm.sparse(tvm.select(index[k] == i[0], src(k, *i[1:])), axis=[k]),
        name=node_name
    )

    return sparse_compute


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument
# 'pylint: disable=too-many-locals,invalid-name,huawei-too-many-arguments
def sparse_apply_proximal_adagrad_compute(var, accum, lr, l1, l2, grad, indices, var_out, accum_out,
                                          use_locking=False, kernel_name="sparse_apply_proximal_adagrad"):
    var_gather = tbe.gather(var, indices, axis=0, support_out_of_bound_index=False)
    accum_gather = tbe.gather(accum, indices, axis=0, support_out_of_bound_index=False)

    # `accum += grad * grad`
    grad_square = tbe.vmul(grad, grad)
    accum_out = tbe.vadd(accum_gather, grad_square)

    # `lr1 = tbe.vrsqrt(accum_out)`
    lr1 = tbe.vlog(accum_out, impl_mode="high_presicion")
    lr1 = tbe.vmuls(lr1, -0.5)
    lr1 = tbe.vexp(lr1)
    lr1 = tbe.vmuls(lr1, lr[0])

    # `lr2 = (1.0 + l2 * lr1)`
    lr2 = tbe.vmuls(lr1, l2[0])
    lr2 = tbe.vadds(lr2, 1.0)

    # `prox_var = var - grad * lr1`
    prox_var = tbe.vmul(grad, lr1)
    prox_var = tbe.vsub(var_gather, prox_var)

    # `var = sign(prox_var) * max(abs(prox_var) - lr1 * max(l1, 0), 0.0) * lr2`
    # `grad_as_temp = abs(prox_var)`
    grad_as_temp = tbe.vabs(prox_var)

    # `var = lr1 * max(l1, 0)`
    var_out = tbe.vmaxs(l1, 0.0)
    var_out = tbe.vmuls(lr1, var_out[0])
    # `var = grad_as_temp - var = abs(prox_var) - lr1 * max(l1, 0)`
    var_out = tbe.vsub(grad_as_temp, var_out)
    # `var = max(var. 0.0) = max(abs(prox_var) - lr1 * max(l1, 0.0), 0.0)`
    var_out = tbe.vmaxs(var_out, 0.0)

    # `tmp = sign(prox_var)`
    tmp = util_compute.sign(prox_var)
    # `var = tmp * var`
    var_out = tbe.vmul(tmp, var_out)
    # `var = var * lr2`
    var_out = tbe.vdiv(var_out, lr2)

    var_res = scatter_update(var, var_out, indices, "var_out")
    accum_res = scatter_update(accum, accum_out, indices, "accum_out")
    return [var_res, accum_res]


# 'pylint: disable=too-many-locals,invalid-name,too-many-arguments,huawei-too-many-arguments
@register_operator("SparseApplyProximalAdagrad", pattern="SparseApply")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def sparse_apply_proximal_adagrad(var,
                                  accum,
                                  lr,
                                  l1,
                                  l2,
                                  grad,
                                  indices,
                                  var_out,
                                  accum_out,
                                  use_locking=False,
                                  kernel_name="sparse_apply_proximal_adagrad"):
    """
    Adds sparse updates to the variable referenced by resource.

    accum += grad * grad
    learning_rate = lr_broad * rsqrt(accum)
    prox_v = var - grad * learning_rate
    sign(prox_var) * max(|prox_var| - lr * max(l1, 0), 0) / (1 + l2 * learning_rate)

    Parameters
    ----------
    var: dict
        data of input.
        source data type, support "float32"
    accum: dict
        data of input.
        source data type, support "float32"
    lr: dict
        A Tensor of the same type as "var".
        Scaling factor. Must be a scalar. Should be greater than zero.
    l1: dict
        A Tensor of the same type as "var".
        L1 regularization. Must be a scalar. Should be greater than or equal to zero.
    l2: dict
        A Tensor of the same type as "var".
        L2 regularization. Must be a scalar. Should be greater than or equal to zero.
    grad: dict
        data of input
        source data type should be same as var
    indices: dict
        A tensor of indices into var, support "int32"
    var_out: dict
        A Tensor. Has the same type and format as input "var".
    accum_out: dict
        A Tensor. Has the same type and format as input "var".
    use_locking: bool
        not used in this compute
    kernel_name: str
        kernel name, default value is "sparse_apply_proximal_adagrad"

    Returns:
    None
    """
    # The same input_shape_type values means inputs have identical shapes.
    extra_params = {"input_shape_type": [0, 0, 1, 2]}
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
            data_lr = tvm.placeholder((1,), lr.get("dtype"), "lr")
            data_l1 = tvm.placeholder((1,), lr.get("dtype"), "l1")
            data_l2 = tvm.placeholder((1,), lr.get("dtype"), "l2")
            res = sparse_apply_proximal_adagrad_compute(data_var, data_accum, data_lr, data_l1, data_l2, data_grad,
                                                        data_indices, var_out, accum_out, use_locking, kernel_name)
            tensors.append([data_var, data_accum, data_lr, data_l1, data_l2, data_grad, data_indices, res[0], res[1]])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": tensors}
    tbe.build(schedules, config)
