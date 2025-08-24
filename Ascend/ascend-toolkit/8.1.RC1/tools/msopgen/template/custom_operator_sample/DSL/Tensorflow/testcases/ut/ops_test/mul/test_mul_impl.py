"""
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

Mul ut case
"""
# pylint: disable=unused-import, pointless-string-statement
import numpy as np
from op_test_frame.common import precision_info
from op_test_frame.ut import OpUT
ut_case = OpUT("Mul", None, None)

case1 = {"params": [{"shape": (8192, 1), "dtype": "float32", "format": "NHWC",
                     "ori_shape": (8192, 1), "ori_format": "NHWC"},
                    {"shape": (8192, 100), "dtype": "float32", "format": "NHWC",
                     "ori_shape": (8192, 100), "ori_format": "NHWC"},
                    {"shape": (8192, 1), "dtype": "float32", "format": "NHWC",
                     "ori_shape": (8192, 1), "ori_format": "NHWC"}],
         "case_name": "mul_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case2 = {"params": [{"shape": (10241,), "dtype": "float16", "format": "NHWC",
                     "ori_shape": (10241,), "ori_format": "NHWC"},
                    {"shape": (10, 10241), "dtype": "float16", "format": "NHWC",
                     "ori_shape": (10, 10241), "ori_format": "NHWC"},
                    {"shape": (10241,), "dtype": "float16", "format": "NHWC",
                     "ori_shape": (10241,), "ori_format": "NHWC"}
                    ],
         "case_name": "mul_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case3 = {"params": [{"shape": (10241,), "dtype": "float32", "format": "NHWC",
                     "ori_shape": (10241,), "ori_format": "NHWC"},
                    {"shape": (10, 10241), "dtype": "float32", "format": "NHWC",
                     "ori_shape": (10, 10241), "ori_format": "NHWC"},
                    {"shape": (10241,), "dtype": "float32", "format": "NHWC",
                     "ori_shape": (10241,), "ori_format": "NHWC"}
                    ],
         "case_name": "mul_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case4 = {"params": [{"shape": (10241,), "dtype": "int8", "format": "NHWC",
                     "ori_shape": (10241,), "ori_format": "NHWC"},
                    {"shape": (10, 10241), "dtype": "int8", "format": "NHWC",
                     "ori_shape": (10, 10241), "ori_format": "NHWC"},
                    {"shape": (10241,), "dtype": "int8", "format": "NHWC",
                     "ori_shape": (10241,), "ori_format": "NHWC"}
                    ],
         "case_name": "mul_4",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}

case5 = {"params": [{"shape": (10241,), "dtype": "uint8", "format": "NHWC",
                     "ori_shape": (10241,), "ori_format": "NHWC"},
                    {"shape": (10, 10241), "dtype": "uint8", "format": "NHWC",
                     "ori_shape": (10, 10241), "ori_format": "NHWC"},
                    {"shape": (10241,), "dtype": "uint8", "format": "NHWC",
                     "ori_shape": (10241,), "ori_format": "NHWC"}
                    ],
         "case_name": "mul_5",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}

case6 = {"params": [{"shape": (1,), "dtype": "float32", "format": "ND",
                     "ori_shape": (1,), "ori_format": "ND"},
                    {"shape": (3, 16, 16), "dtype": "float32", "format": "FRACTAL_NZ",
                     "ori_shape": (3, 16, 16), "ori_format": "FRACTAL_NZ"},
                    {"shape": (3, 16, 16), "dtype": "float32", "format": "FRACTAL_NZ",
                     "ori_shape": (3, 16, 16), "ori_format": "FRACTAL_NZ"}],
         "case_name": "mul_6",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case7 = {"params": [{"shape": (3, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ",
                     "ori_shape": (3, 16, 16), "ori_format": "FRACTAL_NZ"},
                    {"shape": (1,), "dtype": "float16", "format": "ND",
                     "ori_shape": (1,), "ori_format": "ND"},
                    {"shape": (3, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ",
                     "ori_shape": (3, 16, 16), "ori_format": "FRACTAL_NZ"}],
         "case_name": "mul_7",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}


# pylint: disable=unused-argument
def test_op_select_format(test_arg):
    """
    test_op_select_format
    """
    from impl.mul import op_select_format
    op_select_format({"shape": (20, 28, 16, 16), "dtype": "float16", "format": "NCHW",
                      "ori_shape": (20, 28, 16, 16), "ori_format": "NCHW"},
                     {"shape": (1, 1), "dtype": "float16", "format": "ND",
                      "ori_shape": (1, 1), "ori_format": "ND"},
                     {"shape": (20, 28, 16, 16), "dtype": "float16", "format": "NCHW",
                      "ori_shape": (20, 28, 16, 16), "ori_format": "NCHW"})
    op_select_format({"shape": (1, 1), "dtype": "float16", "format": "ND",
                      "ori_shape": (1, 1), "ori_format": "ND"},
                     {"shape": (20, 28, 16, 16), "dtype": "float16", "format": "NCHW",
                      "ori_shape": (20, 28, 16, 16), "ori_format": "NCHW"},
                     {"shape": (20, 28, 16, 16), "dtype": "float16", "format": "NCHW",
                      "ori_shape": (20, 28, 16, 16), "ori_format": "NCHW"})
    op_select_format({"shape": (20, 28, 16, 16), "dtype": "float16", "format": "NCHW",
                      "ori_shape": (20, 28, 16, 16), "ori_format": "NCHW"},
                     {"shape": (20, 28, 16, 16), "dtype": "float16", "format": "NCHW",
                      "ori_shape": (20, 28, 16, 16), "ori_format": "NCHW"},
                     {"shape": (20, 28, 16, 16), "dtype": "float16", "format": "NCHW",
                      "ori_shape": (20, 28, 16, 16), "ori_format": "NCHW"})
    op_select_format({"shape": (20, 28, 3, 3, 16), "dtype": "float32", "format": "NDHWC",
                      "ori_shape": (20, 28, 16, 16), "ori_format": "NDHWC"},
                     {"shape": (20, 28, 3, 3, 16), "dtype": "float32", "format": "NDHWC",
                      "ori_shape": (20, 28, 16, 16), "ori_format": "NDHWC"},
                     {"shape": (20, 28, 3, 3, 16), "dtype": "float32", "format": "NDHWC",
                      "ori_shape": (20, 28, 16, 16), "ori_format": "NDHWC"})
    op_select_format({"shape": (20, 28, 16, 16), "dtype": "float16", "format": "NCHW",
                      "ori_shape": (20, 28, 16, 16), "ori_format": "NCHW"},
                     {"shape": (1,), "dtype": "float16", "format": "ND",
                      "ori_shape": (1,), "ori_format": "ND"},
                     {"shape": (20, 28, 16, 16), "dtype": "float16", "format": "NCHW",
                      "ori_shape": (20, 28, 16, 16), "ori_format": "NCHW"})
    op_select_format({"shape": (20, 28, 16, 16), "dtype": "float16", "format": "NCHW",
                      "ori_shape": (20, 28, 16, 16), "ori_format": "NCHW"},
                     {"shape": (1,), "dtype": "float16", "format": "ND",
                      "ori_shape": (1,), "ori_format": "ND"},
                     {"shape": (20, 28, 16, 16), "dtype": "float16", "format": "NCHW",
                      "ori_shape": (20, 28, 16, 16), "ori_format": "NCHW"})
    op_select_format({"shape": (20, 28, 16), "dtype": "float16", "format": "ND",
                      "ori_shape": (20, 28, 16), "ori_format": "ND"},
                     {"shape": (1,), "dtype": "float16", "format": "ND",
                      "ori_shape": (1,), "ori_format": "ND"},
                     {"shape": (20, 28, 16), "dtype": "float16", "format": "ND",
                      "ori_shape": (20, 28, 16), "ori_format": "ND"})
    op_select_format({"shape": (1, 1, 1), "dtype": "float16", "format": "NHWC",
                      "ori_shape": (1, 1, 1), "ori_format": "NHWC"},
                     {"shape": (96, 1, 56, 56, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (96, 56, 56, 8), "ori_format": "NHWC"},
                     {"shape": (96, 1, 56, 56, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (96, 56, 56, 8), "ori_format": "NHWC"})
    op_select_format({"shape": (), "dtype": "float32", "format": "NHWC",
                      "ori_shape": (), "ori_format": "NHWC"},
                     {"shape": (96, 256), "dtype": "float32", "format": "FRACTAL_NZ",
                      "ori_shape": (96, 256), "ori_format": "NHWC"},
                     {"shape": (96, 256), "dtype": "float32", "format": "FRACTAL_NZ",
                      "ori_shape": (96, 256), "ori_format": "NHWC"})
    op_select_format({"shape": (25, 1, 16, 16), "dtype": "float32", "format": "FRACTAL_Z",
                      "ori_shape": (6, 1, 5, 5), "ori_format": "NCHW"},
                     {"shape": (), "dtype": "float32", "format": "NCHW",
                      "ori_shape": (), "ori_format": "NCHW"},
                     {"shape": (25, 1, 16, 16), "dtype": "float32", "format": "FRACTAL_Z",
                      "ori_shape": (6, 1, 5, 5), "ori_format": "NCHW"})
    op_select_format({"shape": (512,), "dtype": "float32", "format": "NCHW",
                      "ori_shape": (512,), "ori_format": "NCHW"},
                     {"shape": (16, 16, 512, 512), "dtype": "float32", "format": "NCHW",
                      "ori_shape": (16, 16, 512, 512), "ori_format": "NCHW"},
                     {"shape": (16, 16, 512, 512), "dtype": "float32", "format": "NCHW",
                      "ori_shape": (16, 16, 512, 512), "ori_format": "NCHW"})
    op_select_format({"shape": (33, 17, 3, 5, 3), "dtype": "float16", "format": "ND",
                      "ori_shape": (33, 17, 3, 5, 3), "ori_format": "ND"},
                     {"shape": (1,), "dtype": "float16", "format": "ND",
                      "ori_shape": (1,), "ori_format": "ND"},
                     {"shape": (33, 17, 3, 5, 3), "dtype": "float16", "format": "ND",
                      "ori_shape": (33, 17, 3, 5, 3), "ori_format": "ND"})
    op_select_format({"shape": (16, 32, 16), "dtype": "float32", "format": "ND",
                      "ori_shape": (16, 32, 16), "ori_format": "ND"},
                     {"shape": (1,), "dtype": "float32", "format": "ND",
                      "ori_shape": (1,), "ori_format": "ND"},
                     {"shape": (16, 32, 16), "dtype": "float32", "format": "ND",
                      "ori_shape": (16, 32, 16), "ori_format": "ND"})
    op_select_format({"shape": (1,), "dtype": "float32", "format": "ND",
                      "ori_shape": (1,), "ori_format": "ND"},
                     {"shape": (33, 17, 3, 5, 3), "dtype": "float32", "format": "ND",
                      "ori_shape": (33, 17, 3, 5, 3), "ori_format": "ND"},
                     {"shape": (33, 17, 3, 5, 3), "dtype": "float32", "format": "ND",
                      "ori_shape": (33, 17, 3, 5, 3), "ori_format": "ND"})
    op_select_format({"shape": (1,), "dtype": "float16", "format": "ND",
                      "ori_shape": (1,), "ori_format": "ND"},
                     {"shape": (16, 32, 16), "dtype": "float16", "format": "ND",
                      "ori_shape": (16, 32, 16), "ori_format": "ND"},
                     {"shape": (16, 32, 16), "dtype": "float16", "format": "ND",
                      "ori_shape": (16, 32, 16), "ori_format": "ND"})
    op_select_format({"shape": (1,), "dtype": "int32", "format": "ND",
                      "ori_shape": (1,), "ori_format": "ND"},
                     {"shape": (33, 17, 3, 5, 3), "dtype": "int32", "format": "ND",
                      "ori_shape": (33, 17, 3, 5, 3), "ori_format": "ND"},
                     {"shape": (33, 17, 3, 5, 3), "dtype": "int32", "format": "ND",
                      "ori_shape": (33, 17, 3, 5, 3), "ori_format": "ND"})
    op_select_format({"shape": (16, 32, 16), "dtype": "int32", "format": "ND",
                      "ori_shape": (16, 32, 16), "ori_format": "ND"},
                     {"shape": (1,), "dtype": "int32", "format": "ND",
                      "ori_shape": (1,), "ori_format": "ND"},
                     {"shape": (16, 32, 16), "dtype": "int32", "format": "ND",
                      "ori_shape": (16, 32, 16), "ori_format": "ND"})
    op_select_format({"shape": (1,), "dtype": "uint8", "format": "ND",
                      "ori_shape": (1,), "ori_format": "ND"},
                     {"shape": (33, 17, 3, 5, 3), "dtype": "uint8", "format": "ND",
                      "ori_shape": (33, 17, 3, 5, 3), "ori_format": "ND"},
                     {"shape": (33, 17, 3, 5, 3), "dtype": "uint8", "format": "ND",
                      "ori_shape": (33, 17, 3, 5, 3), "ori_format": "ND"})
    op_select_format({"shape": (16, 32, 16), "dtype": "int8", "format": "ND",
                      "ori_shape": (16, 32, 16), "ori_format": "ND"},
                     {"shape": (1,), "dtype": "int8", "format": "ND",
                      "ori_shape": (1,), "ori_format": "ND"},
                     {"shape": (16, 32, 16), "dtype": "int8", "format": "ND",
                      "ori_shape": (16, 32, 16), "ori_format": "ND"})
    op_select_format({"shape": (1,), "dtype": "float16", "format": "ND",
                      "ori_shape": (1,), "ori_format": "ND"},
                     {"shape": (3, 32, 32), "dtype": "float16", "format": "ND",
                      "ori_shape": (3, 32, 32), "ori_format": "ND"},
                     {"shape": (3, 32, 32), "dtype": "float16", "format": "ND",
                      "ori_shape": (3, 32, 32), "ori_format": "ND"})
    op_select_format({"shape": (16, 16, 512, 512), "dtype": "float32", "format": "NCHW",
                      "ori_shape": (16, 16, 512, 512), "ori_format": "NCHW"},
                     {"shape": (1,), "dtype": "float32", "format": "NCHW",
                      "ori_shape": (1,), "ori_format": "NCHW"},
                     {"shape": (16, 16, 512, 512), "dtype": "float32", "format": "NCHW",
                      "ori_shape": (16, 16, 512, 512), "ori_format": "NCHW"})

ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910"], case4)
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910"], case5)
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910"], case6)
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910"], case7)
ut_case.add_cust_test_func(test_func=test_op_select_format)

"""
The ca_model of CI is faulty.Related cases are commented out temporaily.
def calc_expect_func(input_a, input_b, output_y):
    return np.multiply(input_a["value"], input_b["value"]).astype(input_a["dtype"])

ut_case.add_precision_case("all", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND",
                     "ori_shape": (92, 1), "shape": (92, 1), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND",
                     "ori_shape": (92, 100), "shape": (92, 100), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND",
                     "ori_shape": (92, 100), "shape": (92, 100), "param_type": "output"}],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

ut_case.add_precision_case("all", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND",
                "ori_shape": (1024, 3), "shape": (1024, 3), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND",
                "ori_shape": (1024, 3), "shape": (1024, 3), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND",
                "ori_shape": (1024, 3), "shape": (1024, 3), "param_type": "output"}],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

ut_case.add_precision_case("all", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND",
                "ori_shape": (10, 11, 1), "shape": (10, 11, 1), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND",
                "ori_shape": (10, 11, 1), "shape": (10, 11, 1), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND",
                "ori_shape": (10, 11, 1), "shape": (10, 11, 1), "param_type": "output"}],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

ut_case.add_precision_case("all", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND",
                "ori_shape": (3, 3, 144, 1), "shape": (3, 3, 144, 1), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND",
                "ori_shape": (3, 3, 144, 1), "shape": (3, 3, 144, 1), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND",
                "ori_shape": (3, 3, 144, 1), "shape": (3, 3, 144, 1), "param_type": "output"}],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})
"""


if __name__ == '__main__':
    ut_case.run()
