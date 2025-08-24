# Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
dynamic in_training_reduce_grad
"""
from impl.util import util_common
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.util_common import is_unknown_rank_input
from impl.util.util_compute import only_static_support


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    This class for Constant.
    """
    # minimum positive number greater than 0
    EPSLON = 1e-6


@register_operator_compute("INTrainingReduceGrad", op_mode="dynamic", support_fusion=only_static_support)
def in_training_reduce_grad_compute(dy,
                                    x,
                                    variance,
                                    mean,
                                    res_gamma,
                                    res_beta,
                                    gamma,
                                    pd_x,
                                    kernel_name="in_training_reduce_grad"):
    """
    DSL description of the layernorm_grad operator's mathematical

    Parameters
    ----------
    dy: TVM tensor
        the placeholder of input dy
    x: TVM tensor
        the placeholder of input x
    variance: TVM tensor
        the placeholder of input variance
    mean: TVM tensor
        the placeholder of input mean
    res_gamma: TVM tensor
        the placeholder of input res_gamma
    res_beta: TVM tensor
        the placeholder of input res_beta
    gamma: TVM tensor
        the placeholder of input gamma
    pd_x: dict
        shape and dtype of output pd_x
    kernel_name: str
        cce kernel name, default value is "in_training_reduce_grad"

    Returns
    -------
    res_list: list
        [res]
    """
    is_cast = False
    if dy.dtype == "float16":
        is_cast = True
        dy = tbe.cast_to(dy, "float32")

    if x.dtype == "float16":
        is_cast = True
        x = tbe.cast_to(x, "float32")

    if not util_common.is_unknown([pd_x]):
        data_format = pd_x.get("format").upper()
        shape_pd_x = pd_x.get("shape")
        if data_format in ("NC1HWC0",):
            reduce_dims = [shape_pd_x[2], shape_pd_x[3]]
        elif data_format in ("NDC1HWC0",):
            reduce_dims = [shape_pd_x[1], shape_pd_x[3], shape_pd_x[4]]
        else:
            reduce_dims = None

        num = 1
        if reduce_dims:
            for dim in reduce_dims:
                num *= dim

        num_bw = 1.0 / num
        num_rec = tvm.const(num_bw, dtype="float32")
        neg_num_rec = tvm.const(-num_bw, dtype="float32")
    else:
        num_rec = tbe.var("num_rec", dtype="float32")
        neg_num_rec = tbe.var("neg_num_rec", dtype="float32")

    data_sqrt = tbe.vsqrt(tbe.vadds(variance, Constant.EPSLON))
    scale_inv = tbe.vmuls(res_gamma, num_rec)
    scale_inv_reverse = tbe.vmuls(res_gamma, neg_num_rec)
    offset_inv_reverse = tbe.vmuls(res_beta, neg_num_rec)

    multiplier = tbe.vdiv(scale_inv_reverse, data_sqrt)
    addend_div = tbe.vdiv(mean, data_sqrt)
    addend_mul = tbe.vmul(addend_div, scale_inv)
    addend = tbe.vadd(addend_mul, offset_inv_reverse)

    multiplier_broadcast = tbe.broadcast(multiplier, dy.shape)
    addend_broadcast = tbe.broadcast(addend, dy.shape)

    coef_mul = tbe.vmul(multiplier_broadcast, x)
    coef_add = tbe.vadd(dy, coef_mul)
    coef = tbe.vadd(coef_add, addend_broadcast)

    gamma_broadcast = tbe.broadcast(gamma, variance.shape)
    mul_scale = tbe.vdiv(gamma_broadcast, data_sqrt)
    mul_scale_broadcast = tbe.broadcast(mul_scale, dy.shape)

    res = tbe.vmul(coef, mul_scale_broadcast)

    if is_cast:
        res = tbe.cast_to(res, "float16")
    return res


@register_operator("INTrainingReduceGrad")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def in_training_reduce_grad(dy,
                            x,
                            variance,
                            mean,
                            res_gamma,
                            res_beta,
                            gamma,
                            pd_x,
                            kernel_name="in_training_reduce_grad"):
    """
    in_training_reduce_grad operator interface implementation

    Parameters
    ----------
    dy: dict
        shape and dtype of input dy, only support float16, float32
    x: dict
        shape and dtype of input x, only support float16, float32
    variance: dict
        shape and dtype of input variance, only support float32
    mean: dict
        shape and dtype of input mean, only support float32
    res_gamma: dict
        shape and dtype of input res_gamma, only support float32
    res_beta: dict
        shape and dtype of input res_beta, only support float32
    gamma: dict
        shape and dtype of input gamma, only support float32
    pd_x: dict
        shape and dtype of output pd_x, only support float16, float32
    kernel_name: str
        cce kernel name, default value is "in_training_reduce_grad"

    Returns
    -------
    None
    """
    input_list = [dy, x, variance, mean, res_gamma, res_beta, gamma]
    dy_dtype = dy.get("dtype").lower()
    x_dtype = x.get("dtype").lower()
    variance_dtype = variance.get("dtype").lower()
    mean_dtype = mean.get("dtype").lower()
    res_gamma_dtype = res_gamma.get("dtype").lower()
    res_beta_dtype = res_beta.get("dtype").lower()
    gamma_dtype = gamma.get("dtype").lower()

    para_check.check_dtype(dy_dtype, ("float32", "float16"), param_name="dy")
    para_check.check_dtype(x_dtype, ("float32", "float16"), param_name="x")
    para_check.check_dtype(variance_dtype, ("float32",), param_name="variance")
    para_check.check_dtype(mean_dtype, ("float32",), param_name="mean")
    para_check.check_dtype(res_gamma_dtype, ("float32",), param_name="res_gamma")
    para_check.check_dtype(res_beta_dtype, ("float32",), param_name="res_beta")
    para_check.check_dtype(gamma_dtype, ("float32",), param_name="gamma")

    if is_unknown_rank_input(input_list):
        unknown_rank_shape = [-1, -1, -1, -1, -1, -1]
        unknown_rank_range = [(1, None), (1, None), (1, None), (1, None), (1, None), (1, None)]
        for input_dict in input_list:
            input_dict["shape"] = unknown_rank_shape
            input_dict["range"] = unknown_rank_range

    ins = classify([dy, x, variance, mean, res_gamma, res_beta, gamma], OpPatternMode.ELEWISE_WITH_BROADCAST)

    schedules, tensors = [], []

    for (_dy, _x, _variance, _mean, _res_gamma, _res_beta, _gamma) in ins:
        with tbe.compute():
            _shape_dy, _, _, _, _shape_res_gamma, _, _ = shape_util.variable_shape(
                [_dy, _x, _variance, _mean, _res_gamma, _res_beta, _gamma])

            dy_input = tvm.placeholder(_shape_dy, name="dy_input", dtype=dy_dtype)
            x_input = tvm.placeholder(_shape_dy, name="x_input", dtype=x_dtype)
            variance_input = tvm.placeholder(_shape_res_gamma, name="variance_input", dtype=variance_dtype)
            mean_input = tvm.placeholder(_shape_res_gamma, name="mean_input", dtype=mean_dtype)
            res_gamma_input = tvm.placeholder(_shape_res_gamma, name="res_gamma_input", dtype=res_gamma_dtype)
            res_beta_input = tvm.placeholder(_shape_res_gamma, name="res_beta_input", dtype=res_beta_dtype)
            gamma_input = tvm.placeholder(_shape_res_gamma, name="gamma_input", dtype=gamma_dtype)

            res = in_training_reduce_grad_compute(dy_input,
                                                  x_input,
                                                  variance_input,
                                                  mean_input,
                                                  res_gamma_input,
                                                  res_beta_input,
                                                  gamma_input,
                                                  pd_x,
                                                  kernel_name=kernel_name)

            tensor_list = [
                dy_input, x_input, variance_input, mean_input, res_gamma_input, res_beta_input, gamma_input, res
            ]
            tensors.append(tensor_list)

        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    # build
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
