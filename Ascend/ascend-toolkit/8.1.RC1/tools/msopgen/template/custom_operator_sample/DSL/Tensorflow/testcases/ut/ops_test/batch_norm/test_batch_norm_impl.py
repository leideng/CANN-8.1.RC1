#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
import numpy as np

ut_case = OpUT("BatchNorm", "impl.batch_norm", "batch_norm")

def gen_x_shapes(x_dim):
    shape = np.random.randint(3, 8, (x_dim, )).tolist()
    return shape

def calc_expect_func_infer(x, scale, offset, mean, variance, y, batch_mean, batch_variance, reserve_space_1, reserve_space_2,
                           epsilon, data_format, is_training):
    x_value = x.get("value")
    data_format = x.get("format")
    scale_value = scale.get("value")
    offset_value = offset.get("value")
    mean_value = mean.get("value")
    variance_value = variance.get("value")
    x_shape = x_value.shape
    x_dim = len(x_shape)
    if data_format in ["NCHW", "ND"]:
        ori_dim_order = list(range(x_dim))
        change_dim_order = [ori_dim_order[0]] + ori_dim_order[2:] + [ori_dim_order[1]]
        updated_x_value = np.transpose(x_value, change_dim_order)
    elif data_format in ["NC1HWC0"]:
        updated_x_value = np.transpose(x_value, [0, 2, 3, 1, 4]).reshape((x_shape[0], x_shape[2], x_shape[3], -1))
    elif data_format in ["NDC1HWC0"]:
        updated_x_value = np.transpose(x_value, [0, 1, 3, 4, 2, 5]).reshape((x_shape[0], x_shape[1], x_shape[3], x_shape[4], -1))
    elif data_format in ["NHWC"]:
        updated_x_value = x_value
    res_value = scale_value * (updated_x_value - mean_value)/np.sqrt(variance_value+epsilon) + offset_value
    if data_format in ["NCHW", "ND"]:
        ori_dim_order = list(range(x_dim))
        change_dim_order = [ori_dim_order[0], ori_dim_order[-1]] + ori_dim_order[1:-1]
        updated_res_value = np.transpose(res_value, change_dim_order)
    elif data_format in ["NC1HWC0"]:
        updated_res_value = np.transpose(res_value.reshape(x_shape[0], x_shape[2], x_shape[3], x_shape[1], -1), [0, 3, 1, 2, 4])
    elif data_format in ["NDC1HWC0"]:
        updated_res_value = np.transpose(res_value.reshape(x_shape[0], x_shape[1], x_shape[3], x_shape[4], x_shape[2], -1), [0, 1, 4, 2, 3, 5])
    elif data_format in ["NHWC"]:
        updated_res_value = res_value
    res_batch_mean = mean_value
    res_batch_var = variance_value
    res_reserve_space_1= mean_value
    res_reserve_space_2 = variance_value
    return [updated_res_value, res_batch_mean, res_batch_var, res_reserve_space_1, res_reserve_space_2]

def gen_batch_norm_case(x_shape, shape_scale, shape_reserve, dtype_x,
                        dtype_other, format_x, case_name_val, expect, epsilon=0.0001, data_format="NHWC", is_training=True):
    if is_training:
        mean = None
        variance = None
    else:
        mean = {"shape": shape_scale, "dtype": dtype_other, "ori_shape": shape_scale, "ori_format": format_x, "format": format_x}
        variance = {"shape": shape_scale, "dtype": dtype_other, "ori_shape": shape_scale, "ori_format": format_x, "format": format_x}
    return {"params": [{"shape": x_shape, "dtype": dtype_x, "ori_shape": x_shape, "ori_format": format_x, "format": format_x},
                       {"shape": shape_scale, "dtype": dtype_other, "ori_shape": shape_scale, "ori_format": format_x, "format": format_x},
                       {"shape": shape_scale, "dtype": dtype_other, "ori_shape": shape_scale, "ori_format": format_x, "format": format_x},
                       mean,
                       variance,
                       {"shape": x_shape, "dtype": dtype_x, "ori_shape": x_shape, "ori_format": format_x, "format": format_x},
                       {"shape": shape_scale, "dtype": dtype_other, "ori_shape": shape_scale, "ori_format": format_x, "format": format_x},
                       {"shape": shape_scale, "dtype": dtype_other, "ori_shape": shape_scale, "ori_format": format_x, "format": format_x},
                       {"shape": shape_reserve, "dtype": dtype_other, "ori_shape": shape_reserve, "ori_format": format_x, "format": format_x},
                       {"shape": shape_reserve, "dtype": dtype_other, "ori_shape": shape_reserve, "ori_format": format_x, "format": format_x},
                       None,
                       epsilon,
                       data_format,
                       is_training],
            "case_name": case_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": True}

def gen_batch_norm_precious_case(x_shape, shape_scale, shape_reserve, dtype_x,
                        dtype_other, format_x, case_name_val, epsilon=0.0001, data_format="NHWC", is_training=True):
    if is_training:
        mean = None
        variance = None
    else:
        mean = {"shape": shape_scale, "dtype": dtype_other, "ori_shape": shape_scale, "ori_format": format_x, "format": format_x, "param_type": "input", "value_range": [1.0, 2.0]}
        variance = {"shape": shape_scale, "dtype": dtype_other, "ori_shape": shape_scale, "ori_format": format_x, "format": format_x, "param_type": "input", "value_range": [1.0, 2.0]}
    return {"params": [{"shape": x_shape, "dtype": dtype_x, "ori_shape": x_shape, "ori_format": format_x, "format": format_x, "param_type": "input", "value_range": [1.0, 2.0]},
                       {"shape": shape_scale, "dtype": dtype_other, "ori_shape": shape_scale, "ori_format": format_x, "format": format_x, "param_type": "input", "value_range": [1.0, 2.0]},
                       {"shape": shape_scale, "dtype": dtype_other, "ori_shape": shape_scale, "ori_format": format_x, "format": format_x, "param_type": "input", "value_range": [1.0, 2.0]},
                       mean,
                       variance,
                       {"shape": x_shape, "dtype": dtype_x, "ori_shape": x_shape, "ori_format": format_x, "format": format_x, "param_type": "output"},
                       {"shape": shape_scale, "dtype": dtype_other, "ori_shape": shape_scale, "ori_format": format_x, "format": format_x, "param_type": "output"},
                       {"shape": shape_scale, "dtype": dtype_other, "ori_shape": shape_scale, "ori_format": format_x, "format": format_x, "param_type": "output"},
                       {"shape": shape_reserve, "dtype": dtype_other, "ori_shape": shape_reserve, "ori_format": format_x, "format": format_x, "param_type": "output"},
                       {"shape": shape_reserve, "dtype": dtype_other, "ori_shape": shape_reserve, "ori_format": format_x, "format": format_x, "param_type": "output"},
                       None,
                       epsilon,
                       data_format,
                       is_training
                       ],
            "case_name": case_name_val,
            "calc_expect_func": calc_expect_func_infer,
            "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910A"],
                 gen_batch_norm_case((1,2,3,4,16), (1,2,1,1,16), (), "float16",
                                     "float32", "NC1HWC0", "batch_norm_1", "success"))
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910A"],
                 gen_batch_norm_case((1,1,2,3,4,16), (1,1,2,1,1,16), (), "float16",
                                     "float32", "NDC1HWC0", "batch_norm_2", "success"))
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910A"],
                 gen_batch_norm_case((2,16,384,576,16), (1,16,1,1,16), (), "float16",
                                     "float32", "NC1HWC0", "batch_norm_3", "success"))

pre_fix = "_function_"
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910A"],
                 gen_batch_norm_case((2, 64, 7), (64, ), (64, ), "float16",
                                    "float32", "NCHW", "batch_norm" + pre_fix + "0_dim_3_traing_True", "success", is_training=False))
idx = 1
for dim_num in range(2, 9):
    for training in [True, False]:
        shape_x = gen_x_shapes(dim_num)
        ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910A"],
                 gen_batch_norm_case(shape_x, (shape_x[1],), (), "float16",
                                     "float32", "ND", "batch_norm" + pre_fix + str(idx) + "_dim_" + str(dim_num) + "_traing_" + str(training), "success", is_training=training))
        idx += 1


if __name__ == '__main__':
    ut_case.run()
    exit(0)
