# -- coding:utf-8 --
import numpy as np
from op_test_frame.ut import BroadcastOpUT

ut_case = BroadcastOpUT("batch_matmul_impl", op_func_name="batch_matmul")


ut_case.add_case("Ascend910A", {
    "params": [
        {"dtype": "float32", "format": "NCHW", "ori_format": "NCHW", "ori_shape": (1, 64, 64), "shape": (1, 64, 64),
         "param_type": "input"},
        {"dtype": "float32", "format": "NCHW", "ori_format": "NCHW", "ori_shape": (1, 64, 64), "shape": (1, 64, 64),
         "param_type": "input"},
        {"dtype": "float32", "format": "NCHW", "ori_format": "NCHW", "ori_shape": (1, 64, 64), "shape": (1, 64, 64),
         "param_type": "output"}],
    "calc_expect_func": ""
})
