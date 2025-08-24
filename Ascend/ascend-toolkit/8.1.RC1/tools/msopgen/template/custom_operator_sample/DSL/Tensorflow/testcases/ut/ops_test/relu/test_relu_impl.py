#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("Relu", "impl.dynamic.relu", "relu")


def gen_dynamic_relu_case(shape_x, range_x, dtype_val, kernel_name_val,
                          expect):
    return {"params": [
        {"ori_shape": shape_x, "shape": shape_x, "ori_format": "ND",
         "format": "ND", "dtype": dtype_val, "range": range_x},
        {"ori_shape": shape_x, "shape": shape_x, "ori_format": "ND",
         "format": "ND", "dtype": dtype_val, "range": range_x}],
        "case_name": kernel_name_val, "expect": expect, "format_expect": [],
        "support_expect": True}


ut_case.add_case("all", gen_dynamic_relu_case((-1,), ((1, None),), "float16",
                                              "dynamic_relu_fp16_ND",
                                              "success"))
ut_case.add_case("all", gen_dynamic_relu_case((-1,), ((1, None),), "float32",
                                              "dynamic_relu_fp32_ND",
                                              "success"))
ut_case.add_case("all", gen_dynamic_relu_case((-1,), ((1, None),), "int32",
                                              "dynamic_relu_int32_ND",
                                              "success"))
ut_case.add_case("all", gen_dynamic_relu_case((-1,), ((1, None),), "int8",
                                              "dynamic_relu_int8_ND",
                                              "success"))

if __name__ == '__main__':
    ut_case.run("Ascend910A")
