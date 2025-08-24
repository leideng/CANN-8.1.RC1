# # -*- coding:utf-8 -*-
from op_test_frame.ut import OpUT
import tensorflow as tf
import numpy as np
import time, itertools
from op_test_frame.common import precision_info

platforms = ["Ascend910A", "Ascend310", "Ascend610", "BS9SX1AA", "Ascend310P3", "Ascend910"]
platforms_only_fp16 = ["Hi3796CV300CS", "Hi3796CV300ES", "SD3403"]

ut_case = OpUT("lp_norm", None, None)

case1 = {"params": [{"shape": (400, 416, 5, 69), "dtype": "float16", "format": "ND", "ori_shape": (400, 416, 5, 69), "ori_format": "ND"},
                    {"shape": (400, 416, 5, 1), "dtype": "float16", "format": "ND", "ori_shape": (400, 416, 5, 1), "ori_format": "ND"},
                    2147483647, [-1], True],
         "case_name": "lp_norm_1",
         "expect": "success",
         "format_expect": ["ND"],
         "support_expect": True}
case2 = {"params": [{"shape": (400, 416, 5, 69), "dtype": "float16", "format": "ND", "ori_shape": (400, 416, 5, 69), "ori_format": "ND"},
                    {"shape": (400, 416, 5), "dtype": "float16", "format": "ND", "ori_shape": (400, 416, 5), "ori_format": "ND"},
                    -2147483648, [-1]],
         "case_name": "lp_norm_2",
         "expect": "success",
         "format_expect": ["ND"],
         "support_expect": True}
case3 = {"params": [{"shape": (32, 2, 4, 16), "dtype": "float16", "format": "ND", "ori_shape": (32, 2, 4, 16), "ori_format": "ND"},
                    {"shape": (4, 16), "dtype": "float16", "format": "ND", "ori_shape": (4, 16),"ori_format": "ND"},
                    2, [0, 1]],
         "case_name": "lp_norm_3",
         "expect": "success",
         "format_expect": ["ND"],
         "support_expect": True}
case4 = {"params": [{"shape": (32, 2, 4, 16), "dtype": "float16", "format": "ND", "ori_shape": (32, 2, 4, 16), "ori_format": "ND"},
                    {"shape": (4, 16), "dtype": "float16", "format": "ND", "ori_shape": (4, 16), "ori_format": "ND"},
                    "-inf", [0, 1]],
         "case_name": "lp_norm_4",
         "expect": "success",
         "format_expect": ["ND"],
         "support_expect": True}
case5 = {"params": [{"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 2),"ori_format": "ND"},
                    {"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 2),"ori_format": "ND"},
                    "inf", [0, 1]],
         "case_name": "lp_norm_5",
         "expect": "success",
         "format_expect": ["ND"],
         "support_expect": True}
case6 = {"params": [{"shape": (32, 2, 4, 16), "dtype": "float32", "format": "ND", "ori_shape": (32, 2, 4, 16), "ori_format": "ND"},
                    {"shape": (4, 16), "dtype": "float32", "format": "ND", "ori_shape": (4, 16),"ori_format": "ND"},
                    0, [0, 1]],
         "case_name": "lp_norm_6",
         "expect": "success",
         "format_expect": ["ND"],
         "support_expect": True}
case7 = {"params": [{"shape": (32, 2, 4, 16), "dtype": "float32", "format": "ND", "ori_shape": (32, 2, 4, 16), "ori_format": "ND"},
                    {"shape": (4, 16), "dtype": "float32", "format": "ND", "ori_shape": (4, 16),"ori_format": "ND"},
                    1, [0, 1]],
         "case_name": "lp_norm_7",
         "expect": "success",
         "format_expect": ["ND"],
         "support_expect": True}
case8 = {"params": [{"shape": (32, 2, 4, 16), "dtype": "float32", "format": "ND", "ori_shape": (32, 2, 4, 16), "ori_format": "ND"},
                    {"shape": (4, 16), "dtype": "float32", "format": "ND", "ori_shape": (4, 16),"ori_format": "ND"},
                    3, [0, 1]],
         "case_name": "lp_norm_8",
         "expect": "success",
         "format_expect": ["ND"],
         "support_expect": True}
ut_case.add_case("all", case1)
ut_case.add_case("all", case2)
ut_case.add_case("all", case3)
ut_case.add_case("all", case4)
ut_case.add_case(platforms, case5)
ut_case.add_case(platforms, case6)
ut_case.add_case(platforms, case7)
ut_case.add_case(platforms, case8)


def gen_params(x_dim_min=1, x_dim_max=5, dim_val_min=15, dim_val_max=30):
    p_range = [ "inf", "-inf"] + [idx for idx in range(5)]
    axes_base_range = [None, "RAND    OM"]
    keepdim_range = [False]
    dtype_range = ["float16", "float32"]
    np.random.seed(10)
    for p, axes, keepdim, dtype in itertools.product(p_range, axes_base_range, keepdim_range, dtype_range):
        x_dim = np.random.randint(x_dim_min, x_dim_max)
        x_shape = np.random.randint(dim_val_min, dim_val_max, (x_dim, )).tolist()
        if axes_base_range == "RANDOM":
            axes = [idx for idx in range(x_dim) if np.random.rand() > 0.5]
            axes = 0 if len(axes) == 0 else axes
        else: # None, []
            axes = [idx for idx in range(x_dim)]
        axes_list = list(np.ravel([axes]))
        y_shape = [1  if idx in axes_list else x_shape[idx] for idx in range(x_dim)]
        if not keepdim:
            y_shape = [idx for idx in range(x_dim) if idx not in axes_list]
        y_shape = [1] if not y_shape else y_shape
        epsilon = np.random.uniform(0, 1)
        yield x_shape, y_shape, dtype, p, axes, keepdim, epsilon

def calc_expect_func_infer(x, y, p=2, axes=None, keepdim=False, epsilon=1e-12):
    x_value = x.get("value")
    if type(axes) is int:
        axes = [axes]
    elif axes is None or len(axes) == 0:
        axes =[idx for idx in range(len(x_value.shape))]
    # p == 1
    tf.compat.v1.disable_eager_execution()
    inputs = tf.compat.v1.placeholder(x_value.dtype, x_value.shape)
    if p == "inf":
        out_val = tf.reduce_max(inputs, axis=axes)
    elif p == "-inf":
        out_val = tf.reduce_min(inputs, axis=axes)
    elif p == 0:
        out_val = tf.reduce_sum(1 - tf.cast(tf.math.equal(inputs, 0), x_value.dtype), axis=axes)
    elif p == 1:
        out_val = tf.reduce_sum(inputs, axis=axes)
    elif p == 2:
        out_val = tf.sqrt(tf.reduce_sum(inputs**2, axis=axes))
    else: # p >= 3
        out_val = tf.exp(tf.math.log(tf.reduce_sum(inputs**p, axis=axes))/p)
    out_val = tf.maximum(out_val, epsilon)
    with tf.compat.v1.Session() as session:
       result = session.run(out_val, feed_dict={inputs: np.abs(x_value)})
    result = np.array([result]).reshape(y.get("shape"))
    return (result, )

idx = 1
param_gen = gen_params()
for x_shape, y_shape, dtype, p, axes, keepdim, epsilon in param_gen:
    if dtype == "float16":
        platform_select = platforms + platforms_only_fp16
    else:
        platform_select = platforms
    ut_case.add_precision_case(platform_select, {"params": [
                                    {"shape": x_shape, "dtype": dtype, "format": "ND", "ori_shape": x_shape, "ori_format": "ND", "param_type": "input", "value_range": [-2, 2]},
                                    {"shape": y_shape, "dtype": dtype, "format": "ND", "ori_shape": y_shape,"ori_format": "ND", "param_type": "output"},
                                    p,
                                    axes,
                                    keepdim,
                                    epsilon],
         "case_name": "lp_norm_precision" + str(idx),
         "calc_expect_func": calc_expect_func_infer,
         "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)
         })
    idx += 1

if __name__ == '__main__':
    ut_case.run("Ascend910A")
    exit(0)
