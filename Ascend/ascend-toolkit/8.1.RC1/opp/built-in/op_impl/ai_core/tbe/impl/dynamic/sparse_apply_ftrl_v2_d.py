#!/usr/bin/env python
# -*- coding: UTF-8 -*-
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
def sparse_apply_ftrl_v2_d_compute(var, accum, linear, grad, indices, var_out, accum_out, linear_out,
                                   lr, l1, l2, l2_shrinkage, lr_power,
                                   use_locking=False, kernel_name="sparse_apply_ftrl_v2_d"):
    """
    algorithm: sparse_apply_ftrl_v2_d

    Parameters
    ----------
    var: TVM tensor
        data of input var
        datatype suports float32,float16
    accum: TVM tensor
        data of input accum
        datatype suports float32,float16
    linear: TVM tensor
        data of input linear
        datatype suports float32,float16
    grad: TVM tensor
        data of grad
        datatype supports float32,float16
    indices: TVM tensor
        data of indices
        datatype supports int32
    lr: const
        data of lr
        datatype supports float32,float16,int32
    l1: const
        data of l1
        datatype supports float32,float16,int32
    l2: const
        data of l2
        datatype supports float32,float16,int32
    lr_power: const
        data of lr_power
        datatype supports float32,float16,int32
    use_locking: bool
        not used in this compute
    kernel_name: str
        kernel name, default value is "sparse_apply_ftrl_v2_d"

    Returns:
    -------
    res: TVM tensor list
        the result of sparse_apply_ftrl_v2_d compute
    """
    lr = get_attr_by_cls(lr, OpAttr(0, "lr", "Float", 0.1), "float32")
    l1 = get_attr_by_cls(l1, OpAttr(1, "l1", "Float", 0.1), "float32")
    l2 = get_attr_by_cls(l2, OpAttr(2, "l2", "Float", 0.1), "float32")
    l2_shrinkage = get_attr_by_cls(l2_shrinkage, OpAttr(3, "l2_shrinkage", "Float", 0.1), "float32")
    lr_power = get_attr_by_cls(lr_power, OpAttr(4, "lr_power", "Float", -0.5), "float32")

    var_bak = var
    accum_bak = accum
    linear_bak = linear
    var_gathered = tbe.gather(var, indices, axis=0, support_out_of_bound_index=False)
    accum_gathered = tbe.gather(accum, indices, axis=0, support_out_of_bound_index=False)
    linear_gathered = tbe.gather(linear, indices, axis=0, support_out_of_bound_index=False)

    # `accum_new = accum + grad * grad`
    grad_sq = tbe.vmul(grad, grad)
    # `grad_with_shrinkage = grad + 2 * l2_shrinkage * var`
    grad_w_s_2 = tbe.vmuls(var_gathered, 2 * l2_shrinkage)
    grad_w_s = tbe.vadd(grad, grad_w_s_2)
    # `linear += grad_with_shrinkage + (accum^(-lr_power) - accum_new^(-lr_power))/lr * var`
    linear = tbe.vadd(grad_w_s, linear_gathered)
    linear_2 = tbe.vlog(accum_gathered)
    linear_2 = tbe.vmuls(linear_2, -lr_power)
    linear_2 = tbe.vexp(linear_2)
    accum_new = tbe.vadd(accum_gathered, grad_sq)
    linear_3 = tbe.vlog(accum_new)
    linear_3 = tbe.vmuls(linear_3, -lr_power)
    linear_3 = tbe.vexp(linear_3)
    # `quadtratic = 1.0/(accum_new^lr_power * lr) + 2 * l2`
    quadtratic_1 = tbe.vmuls(linear_3, 1.0 / lr)
    linear_23 = tbe.vsub(linear_2, linear_3)
    linear_23 = tbe.vmuls(linear_23, 1.0 / lr)
    linear_23 = tbe.vmul(linear_23, var_gathered)
    linear = tbe.vadd(linear_23, linear)
    # `var = (sign(linear) * l1 - linear) / quadratic if |linear| > l1 else 0.0`
    var_1 = tbe.vmins(linear, l1)
    var_1 = tbe.vmaxs(var_1, -l1)
    var_1 = tbe.vsub(var_1, linear)
    quadtratic = tbe.vadds(quadtratic_1, 2 * l2)
    var = tbe.vdiv(var_1, quadtratic)

    var_out_gm = scatter_update(var_bak, var, indices, "var_out")
    accum_out_gm = scatter_update(accum_bak, accum_new, indices, "accum_out")
    linear_out_gm = scatter_update(linear_bak, linear, indices, "linear_out")

    return [var_out_gm, accum_out_gm, linear_out_gm]


# 'pylint: disable=too-many-arguments,unused-argument,invalid-name,too-many-locals
@register_operator("SparseApplyFtrlV2D", pattern="SparseApply")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_FLOAT,
                            para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_FLOAT,
                            para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_BOOL,
                            para_check.KERNEL_NAME)
def sparse_apply_ftrl_v2_d(var,
                           accum,
                           linear,
                           grad,
                           indices,
                           var_out,
                           accum_out,
                           linear_out,
                           lr,
                           l1,
                           l2,
                           l2_shrinkage,
                           lr_power,
                           use_locking=False,
                           kernel_name="sparse_apply_ftrl_v2_d"):
    """
    Update the variable referenced by resource.

    Parameters
    ----------
    var: dict
        data of input var
        datatype suports float32,float16
    accum: dict
        data of input accum
        datatype suports float32,float16
    linear: dict
        data of input linear
        datatype suports float32,float16
    grad: dict
        data of grad
        datatype supports float32,float16
    indices: dict
        data of indices
        datatype supports int32
    lr: const
        data of lr
        datatype supports float32,float16,int32
    l1: const
        data of l1
        datatype supports float32,float16,int32
    l2: const
        data of l2
        datatype supports float32,float16,int32
    lr_power: const
        data of lr_power
        datatype supports float32,float16,int32
    use_locking: bool
        not used in this compute
    kernel_name: str
        kernel name, default value is "sparse_apply_ftrl_v2_d"

    Returns:
    None
    """
    extra_params = {"input_shape_type": [0, 0, 0, 1, 2]}  # # the same number means the same shape
    ins = classify([var, accum, linear, grad, indices], OpPatternMode.SPARSE_APPLY, extra_params)
    schedules, tensors = [], []
    for (ins_var, ins_accum, ins_linear, ins_grad, ins_indices) in ins:
        with tbe.compute():
            var_shape, accum_shape, linear_shape, grad_shape, indices_shape = \
                shape_util.variable_shape([ins_var, ins_accum, ins_linear, ins_grad, ins_indices],
                                          op_mode="sparse_apply")
            data_var = tvm.placeholder(var_shape, var.get("dtype"), "data_var")
            data_accum = tvm.placeholder(accum_shape, accum.get("dtype"), "data_accum")
            data_linear = tvm.placeholder(linear_shape, linear.get("dtype"), "data_linear")
            data_grad = tvm.placeholder(grad_shape, grad.get("dtype"), "data_grad")
            data_indices = tvm.placeholder(indices_shape, indices.get("dtype"), "data_indices")

            res = sparse_apply_ftrl_v2_d_compute(data_var, data_accum, data_linear, data_grad, data_indices, var_out,
                accum_out, linear_out, lr, l1, l2, l2_shrinkage, lr_power, use_locking, kernel_name)
            tensors.append([data_var, data_accum, data_linear, data_grad, data_indices, res[0], res[1], res[2]])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    config = {"name": kernel_name,
              "tensor_list": tensors}
    tbe.build(schedules, config)
