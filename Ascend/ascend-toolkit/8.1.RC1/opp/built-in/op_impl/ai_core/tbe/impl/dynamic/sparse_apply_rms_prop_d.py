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
sparse_apply_rms_prop_d
"""
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import error_manager_vector
from tbe.dsl.base import operation
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.util_attr_common import get_attr_by_cls
from ..util.util_attr_common import SparseApplyRMSPropDAttrInfo


def scatter_update(dst, src, index, node_name):
    k = tvm.sparse_axis((0, index.shape[0]), name='k')
    sparse = tvm.compute(dst.shape, lambda *i: tvm.sparse(tvm.select(index[k] == i[0], src(k, *i[1:])), axis=[k]),
                         name=node_name)
    return sparse


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument
# 'pylint: disable=too-many-locals,invalid-name
def sparse_apply_rms_prop_d_compute(var,
                                    ms,
                                    mom,
                                    lr,
                                    grad,
                                    indices,
                                    out_var,
                                    out_ms,
                                    out_mom,
                                    rho,
                                    momentum,
                                    epsilon,
                                    use_locking=False,
                                    kernel_name="sparse_apply_rms_prop_d"):
    """
    Adds sparse updates to the variable referenced by resource.

    Parameters
    ----------
    var: tensor
       tensor var, include shape and dtype,
        dtype only support float32.
    ms: tensor
        tensor ms, include shape and dtype.
        Must have the same dtype and shape as var.
    mom: tensor
        tensor mom, include shape and dtype.
        Must have the same dtype and shape as var.
    lr: tensor
        tensor lr,
        Must have the same dtype as var.
    grad: tensor
        tensor grad,
        Must have the same dtype  as var.
    indices: tensor
       tensor indices, include shape and dtype, only support int32.
    out_var: tensor
        tensor out_var, include shape and dtype.
    out_ms: tensor
        tensor out_ms, include shape and dtype.
    out_mom: tensor
        tensor out_mom, include shape and dtype.
    rho: float
        scalar
    momentum: float
        scalar
    epsilon: float
        scalar
    use_locking: bool
        not used in this compute
    kernel_name: str
        kernel name, default value is "sparse_apply_rms_prop_d"

    Returns:
    None
    """
    # `gm sparse -> ub dense`
    var_gather = tbe.gather(var, indices, axis=0, support_out_of_bound_index=False)
    ms_gather = tbe.gather(ms, indices, axis=0, support_out_of_bound_index=False)
    mom_gather = tbe.gather(mom, indices, axis=0, support_out_of_bound_index=False)

    # `ms_out = ms * rho + grad * grad * (1 - rho)`
    ms_tmp = tbe.vmuls(ms_gather, rho)
    grad_tmp = tbe.vmuls(grad, 1.0 - rho)
    grad_tmp = tbe.vmul(grad, grad_tmp)
    ms_out = tbe.vadd(ms_tmp, grad_tmp)

    # `mom_out = mom * momentum + (ms_out + epsilon).rsqrt() * lr * grad`
    mom_tmp = tbe.vmuls(mom_gather, momentum)
    grad_tmp = tbe.vmuls(grad, lr[0])
    ms_out_tmp = tbe.vadds(ms_out, epsilon)
    sqrt_tmp = tbe.vsqrt(ms_out_tmp)
    ms_out_tmp = tbe.vdiv(grad_tmp, sqrt_tmp)
    mom_out = tbe.vadd(mom_tmp, ms_out_tmp)

    # `var_out = var - mom_out`
    var_out = tbe.vsub(var_gather, mom_out)

    # `ub dense -> gm sparse`
    var_out_gm = scatter_update(var, var_out, indices, "sparse1")
    ms_out_gm = scatter_update(ms, ms_out, indices, "sparse2")
    mom_out_gm = scatter_update(mom, mom_out, indices, "sparse3")
    return [var_out_gm, ms_out_gm, mom_out_gm]


# 'pylint: disable=too-many-arguments,unused-argument,invalid-name,too-many-locals
@register_operator("SparseApplyRMSPropD", pattern="SparseApply")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_FLOAT,
                            para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def sparse_apply_rms_prop_d(var,
                            ms,
                            mom,
                            lr,
                            grad,
                            indices,
                            out_var,
                            out_ms,
                            out_mom,
                            rho,
                            momentum,
                            epsilon,
                            use_locking=False,
                            kernel_name="sparse_apply_rms_prop_d"):
    """
    Adds sparse updates to the variable referenced by resource.

    Parameters
    ----------
    var: dict
        dict of tensor var, include shape and dtype,
        dtype only support float32.
    ms: dict
        dict of tensor ms, include shape and dtype.
        Must have the same dtype and shape as var.
    mom: dict
        dict of tensor mom, include shape and dtype.
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
    out_ms: dict
        dict of out_ms, include shape and dtype.
    out_mom: dict
        dict of out_mom, include shape and dtype.
    rho: float
        scalar
    momentum: float
        scalar
    epsilon: float
        scalar
    use_locking: bool
        not used in this compute
    kernel_name: str
        kernel name, default value is "sparse_apply_rms_prop_d"

    Returns:
    None
    """
    dtype = var.get("dtype")
    dtype_idx = indices.get("dtype")
    # input_shape_type correspond to three type var(0), grad(1) and indices(2)
    extra_params = {"input_shape_type": [0, 0, 0, 1, 2]}
    ins = classify([var, ms, mom, grad, indices], OpPatternMode.SPARSE_APPLY, extra_params)
    schedules, tensors = [], []
    for (ins_var, ins_ms, ins_mom, ins_grad, ins_indices) in ins:
        with tbe.compute():
            var_shape, ms_shape, mom_shape, grad_shape, indices_shape = shape_util.variable_shape(
                [ins_var, ins_ms, ins_mom, ins_grad, ins_indices], op_mode="sparse_apply")
            data_var = tvm.placeholder(var_shape, dtype, "data_var")
            data_ms = tvm.placeholder(ms_shape, dtype, "data_ms")
            data_mom = tvm.placeholder(mom_shape, dtype, "data_mom")
            data_lr = tvm.placeholder((1,), dtype, "lr")
            data_grad = tvm.placeholder(grad_shape, dtype, "data_grad")
            data_indices = tvm.placeholder(indices_shape, dtype_idx, "data_indices")
            rho = get_attr_by_cls(rho, SparseApplyRMSPropDAttrInfo.ATTR_RHO, "float32")
            momentum = get_attr_by_cls(momentum, SparseApplyRMSPropDAttrInfo.ATTR_MOM, "float32")
            epsilon = get_attr_by_cls(epsilon, SparseApplyRMSPropDAttrInfo.ATTR_EPS, "float32")
            res = sparse_apply_rms_prop_d_compute(data_var, data_ms, data_mom, data_lr,
                                                  data_grad, data_indices, out_var, out_ms,
                                                  out_mom, rho, momentum, epsilon, use_locking, kernel_name)
            tensors.append([data_var, data_ms, data_mom, data_lr, data_grad, data_indices, res[0], res[1], res[2]])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": tensors}
    tbe.build(schedules, config)
