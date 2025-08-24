#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import sys
from op_test_frame.ut import BroadcastOpUT
import numpy as np

ut_case = BroadcastOpUT("fill_v2d", "impl.fill_v2d", "fill_v2d")

# [TODO] coding expect function here
def calc_expect_func(output_z, value, shape):
    res = np.ones(shape) * value
    return [res, ]


# [TODO] coding cases here
ut_case.add_precision_case("all", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (32,), "shape": (32,),
                "param_type": "output"}, 16.0, (32,)],
    "calc_expect_func": calc_expect_func
})

ut_case.add_precision_case("all", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (32,), "shape": (32,),
                "param_type": "output"}, 5.0, (32,)],
    "calc_expect_func": calc_expect_func
})

ut_case.add_precision_case("all", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (32,25,64), "shape": (32,25,64),
                "param_type": "output"}, 5.0, (32,25,64)],
    "calc_expect_func": calc_expect_func
})

shape_ = (10000,25,64)
ut_case.add_precision_case("all", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": shape_, "shape": shape_,
                "param_type": "output"}, 5.0, shape_],
    "calc_expect_func": calc_expect_func
})

shape_ = (10,25,64,4,5,6,7,8)
ut_case.add_precision_case("all", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": shape_, "shape": shape_,
                "param_type": "output"}, 5.0, shape_],
    "calc_expect_func": calc_expect_func
})