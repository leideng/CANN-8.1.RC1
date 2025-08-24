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
dynamic in_training_update_grad_gamma_beta
"""
from impl.util import util_common
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import tuple_sum


@register_operator_compute("INTrainingUpdateGradGammaBeta", op_mode="dynamic", support_fusion=True)
def in_training_update_grad_gamma_beta_compute(res_gamma,
                                               res_beta,
                                               pd_gamma,
                                               pd_beta,
                                               kernel_name="in_training_update_grad_gamma_beta",
                                               reduce_axis=None):
    """
    DSL description of the layernorm_grad operator's mathematical

    Parameters
    ----------
    res_gamma: TVM tensor
        the placeholder of input res_gamma
    res_beta: TVM tensor
        the placeholder of input res_beta
    pd_gamma: dict
        shape and dtype of output pd_gamma
    pd_beta: dict
        shape and dtype of input pd_beta
    kernel_name: str
        cce kernel name, default value is "in_training_update_grad_gamma_beta"
    reduce_axis: list
        reduce axis of input shape

    Returns
    -------
    res_list: list
        [pd_gamma, pd_beta]
    """
    if not reduce_axis:
        axis = [0]
    else:
        axis = reduce_axis

    pd_gamma, pd_beta = tuple_sum([res_gamma, res_beta], axis, True)

    return [pd_gamma, pd_beta]


@register_operator("INTrainingUpdateGradGammaBeta")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def in_training_update_grad_gamma_beta(res_gamma,
                                       res_beta,
                                       pd_gamma,
                                       pd_beta,
                                       kernel_name="in_training_update_grad_gamma_beta"):
    """
    in_training_update_grad_gamma_beta operator interface implementation

    Parameters
    ----------
    res_gamma: dict
        shape and dtype of input res_gamma, only support float32
    res_beta: dict
        shape and dtype of input res_beta, only support float32
    pd_gamma: dict
        shape and dtype of output pd_gamma, only support float32
    pd_beta: dict
        shape and dtype of input pd_beta, only support float32
    kernel_name: str
        cce kernel name, default value is "in_training_update_grad_gamma_beta"

    Returns
    -------
    None
    """
    dtype_res_gamma = res_gamma.get("dtype").lower()
    dtype_res_beta = res_beta.get("dtype").lower()
    para_check.check_dtype(dtype_res_gamma, ("float32",), param_name="res_gamma")
    para_check.check_dtype(dtype_res_beta, ("float32",), param_name="res_beta")

    if util_common.is_unknown_rank_input([res_gamma, res_beta]):
        unknown_rank_shape = [-1, -1, -1, -1, -1, -1]
        unknown_rank_range = [(1, None), (1, None), (1, None), (1, None), (1, None), (1, None)]
        res_gamma["shape"] = unknown_rank_shape
        res_gamma["range"] = unknown_rank_range
        res_beta["shape"] = unknown_rank_shape
        res_beta["range"] = unknown_rank_range

    list_axis = [0]
    ins = classify([res_gamma, res_beta, list_axis], OpPatternMode.TUPLE_REDUCE)
    schedules, tensors = [], []
    for (_res_gamma, _res_beta, _reduce_axis) in ins:
        with tbe.compute():
            shape_res = shape_util.variable_shape([_res_gamma, _res_beta])
            input_res_gamma = tvm.placeholder(shape_res[0], name="input_gamma", dtype=dtype_res_gamma)
            input_res_beta = tvm.placeholder(shape_res[1], name="input_beta", dtype=dtype_res_beta)
            res = in_training_update_grad_gamma_beta_compute(input_res_gamma, input_res_beta, pd_gamma, pd_beta,
                                                             kernel_name, _reduce_axis)

            tensor_list = [input_res_gamma, input_res_beta] + list(res)
            tensors.append(tensor_list)

        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    # build
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
