#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
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
test_masked_fill_impl
"""

import numpy as np
from op_test_frame.common import precision_info
from op_test_frame.ut import OpUT
ut_case = OpUT("MaskedFill", None, None)


case1 = {"params": [{"shape": (8192, 1), "dtype": "float32", "format": "ND", "ori_shape": (8192, 1),"ori_format": "ND"},
                    {"shape": (8192, 1), "dtype": "int8", "format": "ND", "ori_shape": (8192, 1),"ori_format": "ND"},
                    {"shape": (1, ), "dtype": "float32", "format": "ND", "ori_shape": (1, ),"ori_format": "ND"},
                    {"shape": (8192, 1), "dtype": "float32", "format": "ND", "ori_shape": (8192, 1),"ori_format": "ND"}
                    ],
         "case_name": "masked_fill_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case2 = {"params": [{"shape": (2, 1, 16), "dtype": "int32", "format": "ND", "ori_shape": (2, 1, 16),"ori_format": "ND"},
                    {"shape": (2, 1, 16), "dtype": "bool", "format": "ND", "ori_shape": (2, 1, 16),"ori_format": "ND"},
                    {"shape": (1, ), "dtype": "int32", "format": "ND", "ori_shape": (1, ),"ori_format": "ND"},
                    {"shape": (2, 1, 16), "dtype": "int32", "format": "ND", "ori_shape": (2, 1, 16),"ori_format": "ND"}
                    ],
         "case_name": "masked_fill_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case3 = {"params": [{"shape": (2, 1, 16), "dtype": "float32", "format": "ND", "ori_shape": (2, 1, 16),"ori_format": "ND"},
                    {"shape": (1, ), "dtype": "int8", "format": "ND", "ori_shape": (1, ),"ori_format": "ND"},
                    {"shape": (1, ), "dtype": "float32", "format": "ND", "ori_shape": (1, ),"ori_format": "ND"},
                    {"shape": (2, 1, 16), "dtype": "float32", "format": "ND", "ori_shape": (2, 1, 16),"ori_format": "ND"}
                    ],
         "case_name": "masked_fill_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case4 = {"params": [{"shape": (2, 1, 16), "dtype": "float16", "format": "ND", "ori_shape": (2, 1, 16),"ori_format": "ND"},
                    {"shape": (1, ), "dtype": "int8", "format": "ND", "ori_shape": (1, ),"ori_format": "ND"},
                    {"shape": (1, ), "dtype": "float16", "format": "ND", "ori_shape": (1, ),"ori_format": "ND"},
                    {"shape": (2, 1, 16), "dtype": "float16", "format": "ND", "ori_shape": (2, 1, 16),"ori_format": "ND"}
                    ],
         "case_name": "masked_fill_4",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend910A"], case1)
ut_case.add_case(["Ascend310", "Ascend910A"], case2)
ut_case.add_case(["Ascend910A"], case3)
ut_case.add_case(["Ascend310", "Ascend910A"], case4)

if __name__ == '__main__':
    ut_case.run()
