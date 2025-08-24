# -- coding:utf-8 --
import numpy as np
from op_test_frame.ut import BroadcastOpUT

ut_case = BroadcastOpUT("add3_impl", op_func_name="add3_impl")


# pylint: disable=unused-argument
def calc_expect_func(input1, input2, sum1, const_bias):
    const_bias_array = np.ones((1,)).astype(input1.get('dtype')) * const_bias
    res = input1.get("value") + input2.get("value") + const_bias_array
    return [res, ]


ut_case.add_precision_case("all", {
    "params": [
        {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1,), "shape": (1,),
         "param_type": "input"},
        {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1,), "shape": (1,),
         "param_type": "input"},
        {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1,), "shape": (1,),
         "param_type": "output"},
        1.0],
    "calc_expect_func": calc_expect_func
})
