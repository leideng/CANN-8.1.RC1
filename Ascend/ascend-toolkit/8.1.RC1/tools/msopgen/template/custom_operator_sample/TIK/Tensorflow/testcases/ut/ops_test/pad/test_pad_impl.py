"""
Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

Dynamic Pad ut case
"""
from op_test_frame.ut import OpUT


ut_case = OpUT("Pad", "impl.dynamic.pad", "pad")


# pylint: disable=unused-argument
def get_special_shape(ori_shape, ori_format, dst_format, align_num=16):
    """
    get_special_shape
    """
    def _ceil_div(dim):
        return (dim + align_num - 1) // align_num

    dst_shape = []
    if dst_format in ("FRACTAL_NZ",):
        dst_shape = ori_shape[:-2] + [_ceil_div(ori_shape[-1]), _ceil_div(ori_shape[-2]), align_num, align_num]
    dst_shape_len = len(dst_shape)
    return dst_shape if dst_shape_len != 0 else ori_shape


def tensor_dict(tensor_ori_shape, tensor_ori_format, tensor_type, tensor_format=None):
    """
    return a dict
    """
    if tensor_format is None:
        tensor_format = tensor_ori_format
    tensor_shape = get_special_shape(tensor_ori_shape, tensor_ori_format, tensor_format)

    gen_dict = dict()
    gen_dict["ori_shape"] = tensor_ori_shape
    gen_dict["ori_format"] = tensor_ori_format
    gen_dict["dtype"] = tensor_type
    gen_dict["shape"] = tensor_shape
    gen_dict["format"] = tensor_format
    gen_dict["range"] = [(1, 100000)] * len(tensor_shape)

    return gen_dict


ut_case.add_case(["Ascend910A"],
                 {"params": [tensor_dict([-1, -1, -1], "ND", "float16"),
                             tensor_dict([-1, -1, -1], "ND", "int32"),
                             tensor_dict([-1, -1, -1], "ND", "float16")
                            ],
                  "case_name": "dynamic_pad_case_1",
                  "expect": "success",
                  "support_expect": True})
ut_case.add_case(["Ascend910A"],
                 {"params": [tensor_dict([-2], "ND", "float32"),
                             tensor_dict([-2], "ND", "int64"),
                             tensor_dict([-2], "ND", "float16")
                            ],
                  "case_name": "dynamic_pad_case_2",
                  "expect": "success",
                  "support_expect": True})

if __name__ == '__main__':
    ut_case.run("Ascend910A")
