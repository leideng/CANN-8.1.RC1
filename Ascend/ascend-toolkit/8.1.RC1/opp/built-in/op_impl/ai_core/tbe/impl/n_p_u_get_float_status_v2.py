#!/usr/bin/python
# -*- coding: utf-8 -*-
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
n_p_u_get_float_status_v2
"""
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_build
from impl.util.platform_adapter import build_config


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    ELE_NUM_PER_BLOCK_INT32 = 8


# 'pylint:disable=invalid-name,too-many-locals,unused-argument,unused-variable
@para_check.check_op_params(para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def n_p_u_get_float_status_v2(data, kernel_name="n_p_u_get_float_status_v2"):
    """
    the main function of n_p_u_get_float_status_v2

    Parameters
    ----------
    addr: dict,shape and datatype,datatype supports float32
    data: dict,shape and datatype,datatype supports float32
    kernel_name: cce kernel name, default value is "n_p_u_get_float_status_v2"

    Returns
    -------
    tik_instance: tik_instance
    """
    tik_instance = tik.Tik()

    spec_workspace = tik_instance.Tensor("int32", (Constant.ELE_NUM_PER_BLOCK_INT32,),
                                         name="spec_workspace", scope=tik.scope_gm, is_global_tensor=True)
    ub_tensor = tik_instance.Tensor("int32", (Constant.ELE_NUM_PER_BLOCK_INT32,), name="ub_tensor",
                                    scope=tik.scope_ubuf)
    scalar_zero = tik_instance.Scalar(dtype="int32", init_value=0)
    output_data = tik_instance.Tensor("int32", (Constant.ELE_NUM_PER_BLOCK_INT32,), name="output_data",
                                      scope=tik.scope_gm)

    tik_instance.data_move(ub_tensor, spec_workspace, 0, 1, 1, 0, 0)
    for i in range(1, Constant.ELE_NUM_PER_BLOCK_INT32):
        ub_tensor[i].set_as(scalar_zero)
    tik_instance.data_move(output_data, ub_tensor, 0, 1, 1, 0, 0)
    with build_config(status_check=False):
        tik_instance.BuildCCE(kernel_name, inputs=[], outputs=[output_data])
    return tik_instance
