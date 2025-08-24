#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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
test_sigmoid_cross_entropy_with_logits_grad_v2_impl
"""

import os
from op_test_frame.ut import OpUT

ut_case = OpUT("SigmoidCrossEntropyWithLogitsGradV2", None, None)

case1 = {"params": [{"shape": (15, 32), "dtype": "float32", "format": "ND", "ori_shape": (15, 32),
                     "ori_format": "ND"},
                    {"shape": (15, 32), "dtype": "float32", "format": "ND", "ori_shape": (15, 32),
                     "ori_format": "ND"},
                    {"shape": (15, 32), "dtype": "float32", "format": "ND", "ori_shape": (15, 32),
                     "ori_format": "ND"},
                    {"shape": (15, 32), "dtype": "float32", "format": "ND", "ori_shape": (15, 32),
                     "ori_format": "ND"},
                    {"shape": (15, 32), "dtype": "float32", "format": "ND", "ori_shape": (15, 32),
                     "ori_format": "ND"},
                    {"shape": (15, 32), "dtype": "float32", "format": "ND", "ori_shape": (15, 32),
                     "ori_format": "ND"},
                    "mean"
                    ],
         "case_name": "case1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (15, 32, 64), "dtype": "float32", "format": "ND", "ori_shape": (15, 32, 64),
                     "ori_format": "ND"},
                    {"shape": (15, 32, 64), "dtype": "float32", "format": "ND", "ori_shape": (15, 32, 64),
                     "ori_format": "ND"},
                    {"shape": (15, 32, 64), "dtype": "float32", "format": "ND", "ori_shape": (15, 32, 64),
                     "ori_format": "ND"},
                    {"shape": (15, 32, 64), "dtype": "float32", "format": "ND", "ori_shape": (15, 32, 64),
                     "ori_format": "ND"},
                    None,
                    {"shape": (15, 32, 64), "dtype": "float32", "format": "ND", "ori_shape": (15, 32, 64),
                     "ori_format": "ND"},
                    "none"
                    ],
         "case_name": "case2",
         "expect": "success",
         "support_expect": True}

case3 = {"params": [{"shape": (15, 32, 64, 128), "dtype": "float32", "format": "ND", "ori_shape": (15, 32, 64, 128),
                     "ori_format": "ND"},
                    {"shape": (15, 32, 64, 128), "dtype": "float32", "format": "ND", "ori_shape": (15, 32, 64, 128),
                     "ori_format": "ND"},
                    {"shape": (15, 32, 64, 128), "dtype": "float32", "format": "ND", "ori_shape": (15, 32, 64, 128),
                     "ori_format": "ND"},
                    None,
                    {"shape": (15, 32, 64, 128), "dtype": "float32", "format": "ND", "ori_shape": (15, 32, 64, 128),
                     "ori_format": "ND"},
                    {"shape": (15, 32, 64, 128), "dtype": "float32", "format": "ND", "ori_shape": (15, 32, 64, 128),
                     "ori_format": "ND"},
                    "sum"
                    ],
         "case_name": "case3",
         "expect": "success",
         "support_expect": True}

ut_case.add_case(["Ascend910A"], case1)
ut_case.add_case(["Ascend910A"], case2)
ut_case.add_case(["Ascend910A"], case3)

if __name__ == '__main__':
    ut_case.run(["Ascend910A"])
