#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("ReduceAll", "impl.dynamic.reduce_all", "reduce_all")

def gen_dynamic_reduce_sum_case(shape_x, range_x, shape_axes, range_axes, dtype_val, format,
                                  ori_shape_x, ori_shape_axes, keepdims, kernel_name_val, expect):

    return {"params": [{"shape": shape_x, "dtype": dtype_val,
                        "range": range_x, "format": format,
                        "ori_shape": ori_shape_x, "ori_format": format},
                       {"shape": shape_axes, "dtype": "int32",
                        "range": range_axes, "format": format,
                        "ori_shape": ori_shape_axes, "ori_format": format},
                       {"shape": shape_x, "dtype": dtype_val,
                        "range": range_x, "format": format,
                        "ori_shape": ori_shape_x, "ori_format": format}, keepdims],
            "case_name": kernel_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": True}

ut_case.add_case(
    "all", gen_dynamic_reduce_sum_case((-1,3,-1,2), [(1,None), (3,3), (1,None), (2,2)],
                                       (1,), [(1,1),], "int8", "ND", (-1,3,-1,2), (1, ), 
                                       True, "dynamic_reduce_all_fp16_ND", "success"))

if __name__ == '__main__':
    ut_case.run("Ascend910A")
