#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright 2020 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
============================================================================
"""
import numpy as np
from op_test_frame.ut import BroadcastOpUT

ut_case = BroadcastOpUT("square_impl", op_func_name="square_impl")


def calc_expect_func(input_x, output_y):
    """
    :except result calculate function
    :param input_x:
    :param output_y:
    :return: except result
    """
    res = np.square(input_x["value"])
    return [res, ]


ut_case.add_precision_case("all", {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND",
                "ori_shape": (32,), "shape": (32,),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND",
                "ori_shape": (32,), "shape": (32,),
                "param_type": "output"}],
    "calc_expect_func": calc_expect_func
})
