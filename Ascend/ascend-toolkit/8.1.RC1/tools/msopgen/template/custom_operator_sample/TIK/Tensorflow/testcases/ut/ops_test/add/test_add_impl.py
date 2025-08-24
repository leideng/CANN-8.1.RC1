# # -*- coding:utf-8 -*-
import sys
from op_test_frame.ut import BroadcastOpUT

ut_case = BroadcastOpUT("add")

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
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND",
                "ori_shape": (32,), "shape": (32,),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND",
                "ori_shape": (32,), "shape": (32,),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND",
                "ori_shape": (32,), "shape": (32,),
                "param_type": "output"}],
    "calc_expect_func": calc_expect_func
})