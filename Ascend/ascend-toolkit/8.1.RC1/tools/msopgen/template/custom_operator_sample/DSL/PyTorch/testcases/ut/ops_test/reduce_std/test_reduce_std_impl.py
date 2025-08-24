#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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
test_reduce_std_impl ut
"""
import sys
import numpy as np
from op_test_frame.ut import BroadcastOpUT

ut_case = BroadcastOpUT("reduce_std")

ut_case.add_case("Ascend910A", {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (3, 4), "shape": (3, 4),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (3, 1), "shape": (3, 1),
                "param_type": "output"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (3, 1), "shape": (3, 1),
                "param_type": "output"},
               [1,],
               True,
               True],
    "case_name": "test_reduce_std_case_1"
})

ut_case.add_case("Ascend910A", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (3, 4, 5), "shape": (3, 4, 5),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (3, 4), "shape": (3, 4),
                "param_type": "output"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (3, 4), "shape": (3, 4),
                "param_type": "output"},
               [2,],
               True,
               False],
    "case_name": "test_reduce_std_case_2",
})

ut_case.add_case("Ascend910A", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (3, 4, 5), "shape": (3, 4, 5),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (3, 4), "shape": (3, 4),
                "param_type": "output"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (3, 4), "shape": (3, 4),
                "param_type": "output"},
               [2,],
               False,
               False],
    "case_name": "test_reduce_std_case_3",
})
