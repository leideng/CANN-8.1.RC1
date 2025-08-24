#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
ut_case = OpUT("InplaceUpdate", None, None)

case1 = {"params": [{"shape": (32, 5), "dtype": "int32", "format": "ND", "ori_shape": (32, 5),"ori_format": "ND"},
                    {"shape": (32, ), "dtype": "int32", "format": "ND", "ori_shape": (32, ),"ori_format": "ND"},
                    {"shape": (32, 5), "dtype": "int32", "format": "ND", "ori_shape": (32, 5),"ori_format": "ND"},
                    {"shape": (32, 5), "dtype": "int32", "format": "ND", "ori_shape": (32, 5),"ori_format": "ND"}],
         "case_name": "inplace_update_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (32, 5), "dtype": "float16", "format": "ND", "ori_shape": (32, 5),"ori_format": "ND"},
                    {"shape": (32, ), "dtype": "int32", "format": "ND", "ori_shape": (32, ),"ori_format": "ND"},
                    {"shape": (32, 5), "dtype": "float16", "format": "ND", "ori_shape": (32, 5),"ori_format": "ND"},
                    {"shape": (32, 5), "dtype": "float16", "format": "ND", "ori_shape": (32, 5),"ori_format": "ND"}],
         "case_name": "inplace_update_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (16,4), "dtype": "int32", "format": "ND", "ori_shape": (16,4),"ori_format": "ND"},
                    {"shape": (16, ), "dtype": "int32", "format": "ND", "ori_shape": (16, ),"ori_format": "ND"},
                    {"shape": (16,4), "dtype": "int32", "format": "ND", "ori_shape": (16,4),"ori_format": "ND"},
                    {"shape": (16,4), "dtype": "int32", "format": "ND", "ori_shape": (16,4),"ori_format": "ND"}],
         "case_name": "inplace_update_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case4 = {"params": [{"shape": (4,16,16), "dtype": "int32", "format": "ND", "ori_shape": (4,16,16),"ori_format": "ND"},
                    {"shape": (4, ), "dtype": "int32", "format": "ND", "ori_shape": (4, ),"ori_format": "ND"},
                    {"shape": (4,16,16), "dtype": "int32", "format": "ND", "ori_shape": (4,16,16),"ori_format": "ND"},
                    {"shape": (4,16,16), "dtype": "int32", "format": "ND", "ori_shape": (4,16,16),"ori_format": "ND"}],
         "case_name": "inplace_update_4",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case5 = {"params": [{"shape": (32,16), "dtype": "int32", "format": "ND", "ori_shape": (32,16),"ori_format": "ND"},
                    {"shape": (32, ), "dtype": "int32", "format": "ND", "ori_shape": (32, ),"ori_format": "ND"},
                    {"shape": (32,16), "dtype": "int32", "format": "ND", "ori_shape": (32,16),"ori_format": "ND"},
                    {"shape": (32,16), "dtype": "int32", "format": "ND", "ori_shape": (32,16),"ori_format": "ND"}],
         "case_name": "inplace_update_5",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case6 = {"params": [{"shape": (32,16), "dtype": "float32", "format": "ND", "ori_shape": (32,16),"ori_format": "ND"},
                    {"shape": (32, ), "dtype": "int32", "format": "ND", "ori_shape": (32, ),"ori_format": "ND"},
                    {"shape": (32,16), "dtype": "float32", "format": "ND", "ori_shape": (32,16),"ori_format": "ND"},
                    {"shape": (32,16), "dtype": "float32", "format": "ND", "ori_shape": (32,16),"ori_format": "ND"}],
         "case_name": "inplace_update_6",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910"], case4)
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910"], case5)
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910"], case6)

if __name__ == '__main__':
    ut_case.run("Ascend910")
