#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright 2019-2020 Huawei Technologies Co., Ltd
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
gemm_compute
"""
import warnings
from tbe.dsl.compute.gemm_compute import gemm as gemm_tbe


def gemm(tensor_a, tensor_b, para_dict):
    """
    algorithm: gemm and matmul
    for gemm: calculating matrix multiplication, C = alpha_num*A*B+  beta_num*C
    for matmul: caculating matrix multiplication with bias, C = A*B + bias

    Parameters:
    tensor_a: the first tensor a

    tensor_b: second tensor b with the same type and shape with a

    If tensor_a/tensor_b is int8/uint8,then L0A must be 16*32,L0B
    must be 32*16.
    If A is transpose , then AShape classification matrix must be
    32*16 in gm/L1,then it is 16*32 in L0A.
    If B is transpose , then BShape classification matrix must be
    16*32 in gm/L1,then it is 32*16 in L0B.

    para_dict:

    Returns result
    """
    warnings.warn("te.lang.cce.te_compute.gemm_compute is expired, "
        "please replace it with the func tbe.dsl.compute.gemm_compute",
        DeprecationWarning)

    return gemm_tbe(tensor_a, tensor_b, para_dict)
