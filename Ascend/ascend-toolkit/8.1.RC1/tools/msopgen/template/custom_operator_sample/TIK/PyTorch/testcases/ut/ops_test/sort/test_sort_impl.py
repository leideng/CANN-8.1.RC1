# # -*- coding:utf-8 -*-
import sys
from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info

ut_case = OpUT("Sort")

ut_case.add_case("all", {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (2, 10), "shape": (2, 10),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (2, 10), "shape": (2, 10),
                "param_type": "output"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (2, 10), "shape": (2, 10),
                "param_type": "output"},
               -1, False],
    "case_name": "test0",
    "expect": "success",
    "format_expect": ["ND"],
    "support_expect": True})

ut_case.add_case("all", {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (2, 5000), "shape": (2, 5000),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (2, 5000), "shape": (2, 5000),
                "param_type": "output"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (2, 5000), "shape": (2, 5000),
                "param_type": "output"},
               -1, True],
    "case_name": "test1",
    "expect": "success",
    "format_expect": ["ND"],
    "support_expect": True})

ut_case.add_case("all", {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (2, 50000), "shape": (2, 50000),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (2, 50000), "shape": (2, 50000),
                "param_type": "output"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (2, 50000), "shape": (2, 50000),
                "param_type": "output"},
               -1, True],
    "case_name": "test1",
    "expect": "success",
    "format_expect": ["ND"],
    "support_expect": True})

ut_case.add_case("all", {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (2, 3, 4, 10, 32, 8, 3),
               "shape": (2, 3, 4, 10, 32, 8, 3), "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (2, 3, 4, 10, 32, 8, 3),
               "shape": (2, 3, 4, 10, 32, 8, 3), "param_type": "input"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (2, 3, 4, 10, 32, 8, 3),
               "shape": (2, 3, 4, 10, 32, 8, 3), "param_type": "output"},
               -1, True],
    "case_name": "test1",
    "expect": "success",
    "format_expect": ["ND"],
    "support_expect": True})

if __name__ == '__main__':
    ut_case.run("Ascend910A")
    exit(0)