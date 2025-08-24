#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

test_add_impl
"""
from op_test_frame.ut import BroadcastOpUT

ut_case = BroadcastOpUT("Add")


def calc_expect_func(input_x, input_y, output_z):
    """
    get expect output
    :param input_x:tensor x
    :param input_y: tensor y
    :param output_z: output placeholder
    :return:output tensor
    """
    res = input_x["value"] + input_y["value"]
    return [res, ]


ut_case.add_precision_case("all", {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND",
                "ori_shape": (32,), "shape": (32,),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND",
                "ori_shape": (32,), "shape": (32,),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND",
                "ori_shape": (32,), "shape": (32,),
                "param_type": "output"}],
    "calc_expect_func": calc_expect_func
})
