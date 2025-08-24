# -- coding:utf-8 --
import numpy as np
from op_test_frame.ut import BroadcastOpUT

ut_case = BroadcastOpUT("cus_correction_mul_impl", op_func_name="cus_correction_mul")


# pylint: disable=unused-argument
def calc_expect_func(x, batch_std, running_std, y, channel_axis):
    shape_x = x.get("shape")
    factor = batch_std.get("value") / running_std.get("value")
    factor_b = np.broadcast_to(factor, shape_x)
    res = x.get("value") * factor_b
    return [res, ]


ut_case.add_precision_case("all", {
    "params": [
        {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (3, 4), "shape": (3, 4),
         "param_type": "input"},
        {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (4,), "shape": (4,),
         "param_type": "input"},
        {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (4,), "shape": (4,),
         "param_type": "input"},
        {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (3, 4), "shape": (3, 4),
         "param_type": "output"},
        1],
    "calc_expect_func": calc_expect_func
})
