# Copyright 2021 Huawei Technologies Co., Ltd
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
in_training_update_grad
"""

from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tuple_sum
from impl.util.platform_adapter import para_check


# 'pylint: disable=too-many-locals,too-many-arguments,unused-argument,invalid-name
def in_training_update_grad_gamma_beta_compute(res_gamma,
                                               res_beta,
                                               pd_gamma,
                                               pd_beta,
                                               kernel_name="in_training_update_grad_gamma_beta"):
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

    Returns
    -------
    res_list: list
        [pd_gamma, pd_beta]
    """
    pd_gamma, pd_beta = tuple_sum([res_gamma, res_beta], [0], True)

    return [pd_gamma, pd_beta]


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
    shape_gamma = res_gamma.get("shape")
    shape_beta = res_beta.get("shape")
    dtype_gamma = res_gamma.get("dtype").lower()

    para_check.check_dtype(dtype_gamma, ("float32",), param_name="variance")
    para_check.check_shape(shape_gamma, param_name="gamma")
    para_check.check_shape(shape_beta, param_name="beta")

    data_gamma = tvm.placeholder(shape_gamma, name="data_gamma", dtype=dtype_gamma)
    data_beta = tvm.placeholder(shape_beta, name="data_beta", dtype=dtype_gamma)

    res = in_training_update_grad_gamma_beta_compute(data_gamma, data_beta, pd_gamma, pd_beta, kernel_name)

    with tvm.target.cce():
        sch = tbe.auto_schedule(res)

    config = {"name": kernel_name, "tensor_list": [data_gamma, data_beta] + list(res)}

    tbe.build(sch, config)
