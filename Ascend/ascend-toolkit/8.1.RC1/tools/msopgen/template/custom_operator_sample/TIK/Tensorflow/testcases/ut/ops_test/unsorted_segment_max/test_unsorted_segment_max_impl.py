#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("UnsortedSegmentMax", "impl.dynamic.unsorted_segment_max", "unsorted_segment_max")


def gen_dynamic_unsorted_case(shape_input, range_input, num_segments, range_num_segments, dtype_val, expect):
    shape_output = [num_segments] + shape_input[1:]
    range_output = range_num_segments + range_input[1:]
    return {"params":
                [{"shape": shape_input, "dtype": dtype_val, "ori_shape": shape_input, "ori_format": "ND",
                  "format": "ND", "range": range_input},
                 {"shape": shape_input[:1], "dtype": "int32", "ori_shape": shape_input[:1], "ori_format": "ND",
                  "format": "ND", "range": range_input[:1]},
                 {"shape": [1], "dtype": "int32", "ori_shape": [1], "ori_format": "ND", "format": "ND",
                  "range": [[1, 1]]},
                 {"shape": shape_output, "dtype": dtype_val, "ori_shape": shape_output, "ori_format": "ND",
                  "format": "ND", "range": range_output}],
            "expect": expect,
            "format_expect": [],
            "support_expect": True}


ut_case.add_case("Ascend910A",
                 gen_dynamic_unsorted_case([-1, 16, 10419, 3], [[3, 16], [16, 16], [10419, 10419], [3, 3]],
                                           -1, [[8, 8]], "float32", "success"))

ut_case.add_case("Ascend910A", gen_dynamic_unsorted_case([-1, 16, 419], [[1508, 1518], [16, 16], [419, 419]],
                                                         -1, [[8, 16]], "float16", "success"))

ut_case.add_case("Ascend910A", gen_dynamic_unsorted_case([4, 31, 4], [[4, 4], [31, 31], [4, 4]],
                                                         -1, [[16, 512]], "float32", "success"))

ut_case.add_case("Ascend910A", gen_dynamic_unsorted_case([2, -1], [[2, 2], [13, 256]],
                                                         -1, [[256, 512]], "float16", "success"))

ut_case.add_case("Ascend910A", gen_dynamic_unsorted_case([1, -1], [[1, 1], [5, 15]],
                                                         -1, [[72, 72]], "float16", "success"))
if __name__ == '__main__':
    ut_case.run("Ascend910A")
