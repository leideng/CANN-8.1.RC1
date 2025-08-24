#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
ut_case = OpUT("Iou", "impl.pt_iou", "iou")

case1 = {"params": [{"shape": (1,4), "dtype": "float32", "format": "ND", "ori_shape": (1,4),"ori_format": "ND"},
                    {"shape": (1,4), "dtype": "float32", "format": "ND", "ori_shape": (1,4),"ori_format": "ND"},
                    {"shape": (1,1), "dtype": "float32", "format": "ND", "ori_shape": (1,1),"ori_format": "ND"}],
         "case_name": "iou_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (16,4), "dtype": "float32", "format": "ND", "ori_shape": (16,4),"ori_format": "ND"},
                    {"shape": (16,4), "dtype": "float32", "format": "ND", "ori_shape": (16,4),"ori_format": "ND"},
                    {"shape": (16,16), "dtype": "float32", "format": "ND", "ori_shape": (16,16),"ori_format": "ND"}],
         "case_name": "iou_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (32, 4), "dtype": "float32", "format": "ND", "ori_shape": (32, 4),"ori_format": "ND"},
                    {"shape": (32, 4), "dtype": "float32", "format": "ND", "ori_shape": (32, 4),"ori_format": "ND"},
                    {"shape": (32, 32), "dtype": "float32", "format": "ND", "ori_shape": (32, 32),"ori_format": "ND"}],
         "case_name": "iou_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case4 = {"params": [{"shape": (32, 4), "dtype": "float32", "format": "ND", "ori_shape": (32, 4),"ori_format": "ND"},
                    {"shape": (32, 4), "dtype": "float32", "format": "ND", "ori_shape": (32, 4),"ori_format": "ND"},
                    {"shape": (32, 32), "dtype": "float32", "format": "ND", "ori_shape": (32, 32),"ori_format": "ND"}],
         "case_name": "iou_4",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case5 = {"params": [{"shape": (1, 4), "dtype": "float32", "format": "ND", "ori_shape": (1, 4),"ori_format": "ND"},
                    {"shape": (16, 4), "dtype": "float32", "format": "ND", "ori_shape": (16, 4),"ori_format": "ND"},
                    {"shape": (16, 1), "dtype": "float32", "format": "ND", "ori_shape": (16, 1),"ori_format": "ND"}],
         "case_name": "iou_5",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910"], case4)
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910"], case5)

if __name__ == '__main__':
    ut_case.run("Ascend910")
    exit(0)
