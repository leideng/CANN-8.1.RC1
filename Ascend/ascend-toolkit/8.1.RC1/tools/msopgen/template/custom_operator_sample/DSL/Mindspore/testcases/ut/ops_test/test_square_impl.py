# -- coding:utf-8 --
import numpy as np
from op_test_frame.ut import BroadcastOpUT

ut_case = BroadcastOpUT("square_impl", op_func_name="square_impl")


# pylint: disable=unused-argument
def calc_expect_func(input_x, output_y):
    res = np.square(input_x.get("value"))
    return [res, ]


ut_case.add_precision_case("all", {
    "params": [
        {"dtype": "float32", "format": "NCHW", "ori_format": "NCHW", "ori_shape": (1, 2), "shape": (1, 2),
         "param_type": "input"},
        {"dtype": "float32", "format": "NCHW", "ori_format": "NCHW", "ori_shape": (1, 2), "shape": (1, 2),
         "param_type": "output"}],
    "calc_expect_func": calc_expect_func
})
