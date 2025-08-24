#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
import numpy as np
import tensorflow as tf
ut_case = OpUT("MaxPool", None, None)

case1 = {"params": [{"shape": (1,3,35,49,16), "dtype": "float16", "format": "NHWC", "ori_shape": (1, 3, 35, 49, 16),"ori_format": "NHWC"},
                    {"shape": (1,3,17,24,16), "dtype": "float16", "format": "NHWC", "ori_shape": (1, 3, 17, 24, 16),"ori_format": "NHWC"},
                    [1, 1, 3, 3],
                    [1, 1, 2, 2],
                    "VALID"],
         "case_name": "max_pool_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (1,4,23,111,16), "dtype": "float16", "format": "NHWC", "ori_shape": (1, 4, 23, 111, 16),"ori_format": "NHWC"},
                    {"shape": (1,4,11,55,16), "dtype": "float16", "format": "NHWC", "ori_shape": (1, 4, 11, 55, 16),"ori_format": "NHWC"},
                    [1, 1, 3, 3],
                    [1, 1, 2, 2],
                    "VALID"],
         "case_name": "max_pool_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910"], case2)

from impl.max_pool import check_supported

# pylint: disable=unused-argument,unused-variable
def test_check_support(test_arg):
    # input_x, output_y, output_argmax, ksize, strides, padding
    res = check_supported(
                    {"shape": (1,4,23,111,16), "dtype": "float16", "format": "NHWC", "ori_shape": (1, 4, 23, 111, 16),"ori_format": "NHWC"},
                    {"shape": (1,4,11,55,16), "dtype": "float16", "format": "NHWC", "ori_shape": (1, 4, 11, 55, 16),"ori_format": "NHWC"},
                    [1, 1, 3, 3],
                    [1, 1, 2, 2],
                    "VALID",
                    "NCHW"
                    "max_pool_check_support_case_001")
    res = check_supported(
                    {"shape": (1,3,35,49,16), "dtype": "float16", "format": "NHWC", "ori_shape": (1, 3, 35, 49, 16),"ori_format": "NHWC"},
                    {"shape": (1,3,17,24,16), "dtype": "float16", "format": "NHWC", "ori_shape": (1, 3, 17, 24, 16),"ori_format": "NHWC"},
                    [1, 16, 16, 1],
                    [1, 16, 16, 1],
                    "VALID",
                    "NHWC"
                    "max_pool_check_support_case_002")
    res = check_supported(
                    {"shape": (1,3,35,49,16), "dtype": "float16", "format": "NHWC", "ori_shape": (1, 3, 35, 49, 16),"ori_format": "NHWC"},
                    {"shape": (1,3,17,24,16), "dtype": "float16", "format": "NHWC", "ori_shape": (1, 3, 17, 24, 16),"ori_format": "NHWC"},
                    [1, 21, 21, 1],
                    [1, 21, 21, 1],
                    "SAME",
                    "NHWC"
                    "max_pool_check_support_case_003")

ut_case.add_cust_test_func(test_func=test_check_support)

#precision cases
def NC1HWC02NCHW(fmi, fmi_shape, precise):
    if precise=='int8':
        fmo = np.zeros((fmi_shape[0], fmi_shape[1]*fmi_shape[4], fmi_shape[2], fmi_shape[3]), dtype=np.int8)
    else:
        fmo = np.zeros((fmi_shape[0], fmi_shape[1]*fmi_shape[4], fmi_shape[2], fmi_shape[3]), dtype=np.float16)
    for n in range(fmi_shape[0]):
        for c1 in range(fmi_shape[1]):
            for h in range(fmi_shape[2]):
                for w in range(fmi_shape[3]):
                    for c0 in range(fmi_shape[4]):
                        fmo[n][c1*fmi_shape[4]+c0][h][w] = fmi[n][c1][h][w][c0]
    return fmo

#NCHW2NC1HWC0
def NCHW2NC1HWC0(fmi, fmo_shape, precise):
    if precise=='int8':
        fmo = np.zeros((fmo_shape[0], fmo_shape[1], fmo_shape[2], fmo_shape[3], fmo_shape[4]), dtype=np.int8)
    else:
        fmo = np.zeros((fmo_shape[0], fmo_shape[1], fmo_shape[2], fmo_shape[3], fmo_shape[4]), dtype=np.float16)
    for n in range(fmo_shape[0]):
        for c1 in range(fmo_shape[1]):
            for h in range(fmo_shape[2]):
                for w in range(fmo_shape[3]):
                    for c0 in range(fmo_shape[4]):
                        fmo[n][c1][h][w][c0] = fmi[n][c1*fmo_shape[4]+c0][h][w]
    return fmo

def calc_expect_func(x, y, ksize, strides, padding):
    inputArr = x['value']
    shape = x['shape']
    inputArr_NCHW = NC1HWC02NCHW(inputArr, shape, "float16")
    inputArr_NHWC = np.transpose(inputArr_NCHW, (0, 2, 3, 1))
    mat_2 = tf.nn.max_pool(inputArr_NHWC, ksize=ksize, strides=strides, padding=padding, data_format="NHWC")
    with tf.compat.v1.Session() as sess:
        outputArr_NHWC = sess.run(mat_2)
    outputArr_NCHW = np.transpose(outputArr_NHWC, (0, 3, 1, 2))
    #output shape
    batch, channel, height, width = outputArr_NCHW.shape
    C0 = shape[4]
    C1 = (channel + C0 - 1) // shape[4]
    shape_output = [batch, C1, height, width, C0]
    outputArr = NCHW2NC1HWC0(outputArr_NCHW, shape_output, "float16")
    return outputArr

# ut_case.add_precision_case("Ascend910", {"params": [{"shape": (1,3,35,49,16), "dtype": "float16", "format": "NHWC", "ori_shape": (1, 3, 35, 49, 16),"ori_format": "NHWC", "param_type": "input", "value_range":[-65504, 65504]},
#                                                     {"shape": (1,3,17,24,16), "dtype": "float16", "format": "NHWC", "ori_shape": (1, 3, 17, 24, 16),"ori_format": "NHWC", "param_type": "output"},
#                                                     [1, 1, 1, 3],
#                                                     [1, 1, 1, 2],
#                                                     "VALID"],
#                                          "expect": "success",
#                                          "calc_expect_func": calc_expect_func,
#                                          "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)})
# ut_case.add_precision_case("Ascend910", {"params": [{"shape": (2,3,11,33,16), "dtype": "float16", "format": "NHWC", "ori_shape": (2,3,11,33,16),"ori_format": "NHWC", "param_type": "input", "value_range":[-65504, 65504]},
#                                                     {"shape": (2,3,11,33,16), "dtype": "float16", "format": "NHWC", "ori_shape": (1, 3, 17, 24, 16),"ori_format": "NHWC", "param_type": "output"},
#                                                     [1, 2, 2, 1],
#                                                     [1, 2, 2, 1],
#                                                     "VALID"],
#                                          "expect": "success",
#                                          "calc_expect_func": calc_expect_func,
#                                          "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)})
ut_case.add_precision_case("Ascend910", {"params": [{"shape": (1, 1, 5, 3, 16), "dtype": "float16", "format": "NHWC", "ori_shape": (1, 3, 35, 49, 16),"ori_format": "NHWC", "param_type": "input", "value_range":[-65504, 65504]},
                                                    {"shape": (1, 1, 5, 3, 16), "dtype": "float16", "format": "NHWC", "ori_shape": (1, 3, 17, 24, 16),"ori_format": "NHWC", "param_type": "output"},
                                                    [1, 1, 1, 1],
                                                    [1, 1, 1, 1],
                                                    "SAME"],
                                         "expect": "success",
                                         "calc_expect_func": calc_expect_func,
                                         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)})
