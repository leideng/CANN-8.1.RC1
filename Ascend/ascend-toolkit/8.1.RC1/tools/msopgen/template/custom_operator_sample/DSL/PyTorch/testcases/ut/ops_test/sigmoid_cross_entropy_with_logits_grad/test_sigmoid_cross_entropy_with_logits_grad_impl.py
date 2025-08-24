#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
sigmoid_cross_entropy_with_logits_grad_test
'''
import os
from op_test_frame.ut import OpUT

ut_case = OpUT("SigmoidCrossEntropyWithLogitsGrad", None, None)

case1 = {"params": [{"shape": (1, 2, 4), "dtype": "float16", "format": "ND",
                     "ori_shape": (1, 2, 4), "ori_format": "ND"},
                    {"shape": (1, 2, 4), "dtype": "float16", "format": "ND",
                     "ori_shape": (1, 2, 4), "ori_format": "ND"},
                    {"shape": (1, 2, 4), "dtype": "float16", "format": "ND",
                     "ori_shape": (1, 2, 4), "ori_format": "ND"},
                    {"shape": (1, 2, 4), "dtype": "float16", "format": "ND",
                     "ori_shape": (1, 2, 4), "ori_format": "ND"}],
         "case_name": "sigmoid_cross_entropy_with_logits_grad_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (16, 16), "dtype": "float16", "format": "ND",
                     "ori_shape": (16, 16), "ori_format": "ND"},
                    {"shape": (16, 16), "dtype": "float16", "format": "ND",
                     "ori_shape": (16, 16), "ori_format": "ND"},
                    {"shape": (16, 16), "dtype": "float16", "format": "ND",
                     "ori_shape": (16, 16), "ori_format": "ND"},
                    {"shape": (16, 16), "dtype": "float16", "format": "ND",
                     "ori_shape": (16, 16), "ori_format": "ND"}],
         "case_name": "sigmoid_cross_entropy_with_logits_grad_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (32, 2, 4, 16), "dtype": "float16", "format": "ND",
                     "ori_shape": (32, 2, 4, 16), "ori_format": "ND"},
                    {"shape": (32, 2, 4, 16), "dtype": "float16", "format": "ND",
                     "ori_shape": (32, 2, 4, 16), "ori_format": "ND"},
                    {"shape": (32, 2, 4, 16), "dtype": "float16", "format": "ND",
                     "ori_shape": (32, 2, 4, 16), "ori_format": "ND"},
                    {"shape": (32, 2, 4, 16), "dtype": "float16", "format": "ND",
                     "ori_shape": (32, 2, 4, 16), "ori_format": "ND"}],
         "case_name": "sigmoid_cross_entropy_with_logits_grad_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case4 = {"params": [{"shape": (32, 2, 4, 16), "dtype": "float32", "format": "ND",
                     "ori_shape": (32, 2, 4, 16), "ori_format": "ND"},
                    {"shape": (32, 2, 4, 16), "dtype": "float32", "format": "ND",
                     "ori_shape": (32, 2, 4, 16), "ori_format": "ND"},
                    {"shape": (32, 2, 4, 16), "dtype": "float32", "format": "ND",
                     "ori_shape": (32, 2, 4, 16), "ori_format": "ND"},
                    {"shape": (32, 2, 4, 16), "dtype": "float32", "format": "ND",
                     "ori_shape": (32, 2, 4, 16), "ori_format": "ND"}],
         "case_name": "sigmoid_cross_entropy_with_logits_grad_4",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case5 = {"params": [{"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 2), "ori_format": "ND"},
                    {"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 2), "ori_format": "ND"},
                    {"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 2), "ori_format": "ND"},
                    {"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 2), "ori_format": "ND"}],
         "case_name": "sigmoid_cross_entropy_with_logits_grad_5",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
ut_case.add_case(["Ascend310P3", "Ascend910A"], case1)
ut_case.add_case(["Ascend310P3", "Ascend910A"], case2)
ut_case.add_case(["Ascend310P3", "Ascend910A"], case3)
ut_case.add_case(["Ascend310P3", "Ascend910A"], case4)
ut_case.add_case(["Ascend310P3", "Ascend910A"], case5)

# ============ auto gen ["Ascend910"] test cases end =================

if __name__ == '__main__':
    user_home_path = os.path.expanduser("~")
    simulator_lib_path = os.path.join(user_home_path, ".mindstudio/huawei/adk/1.75.T15.0.B150/toolkit/tools/simulator")
    ut_case.run(["Ascend910A"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)
