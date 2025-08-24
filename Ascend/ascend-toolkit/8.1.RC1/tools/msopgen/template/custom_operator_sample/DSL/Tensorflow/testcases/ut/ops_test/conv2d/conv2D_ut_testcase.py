# -*- coding: UTF-8 -*-

conv2D_ut_testcase = [
 # platform, inputs, weights, bias, offset_w, outputs, strides, pads, dilations, expect
# ============ success =====================
# ============ base case 910 ===============
["all", {'ori_shape': (4, 64, 64, 16), 'shape': (4, 1, 64, 64, 16), 'ori_format': 'NHWC', 'dtype': 'float16'}, {'ori_shape': (1, 1, 16, 1), 'ori_format': 'HWCN', 'dtype': 'float16'}, None, None, {'dtype': 'float16'}, (1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), "success"],
# ============ base case 310 int 8===============
["Ascend310", {'ori_shape': (4, 64, 64, 16), 'shape': (4, 1, 64, 64, 32), 'ori_format': 'NHWC', 'dtype': 'int8'}, {'ori_shape': (1, 1, 16, 1), 'ori_format': 'HWCN', 'dtype': 'int8'}, None, None, {'dtype': 'int32'}, (1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), "success"],
# ============ base case C0=4 ===============
["all", {'ori_shape': (4, 64, 64, 4), 'shape': (4, 1, 64, 64, 16), 'ori_format': 'NHWC', 'dtype': 'float16'}, {'ori_shape': (3, 3, 4, 1), 'ori_format': 'HWCN', 'dtype': 'float16', "format":'FRACTAL_Z_C04'}, None, None, {'dtype': 'float16'}, (1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), "success"],
# ============ base case bias ===============
["all", {'ori_shape': (4, 64, 64, 16), 'shape': (4, 1, 64, 64, 16), 'ori_format': 'NHWC', 'dtype': 'float16'}, {'ori_shape': (1, 1, 1, 16), 'ori_format': 'NHWC', 'dtype': 'float16'}, {'ori_shape': (16), "dtype": "float16"}, None, {'dtype': 'float16'}, (1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), "success"],
# ============ base case wo=1 hw!=1 ===============
["all", {'ori_shape': (1, 32, 7, 1), 'shape': (1, 1, 7, 1, 32), 'ori_format': 'NCHW', 'dtype': 'int8'}, {'ori_shape': (32, 32, 1, 1), 'ori_format': 'NCHW', 'dtype': 'int8'}, None, None, {'dtype': 'int32'}, (1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), "success"],
# ============ RuntimeError ===============
# ============ base case C0=4   H and W are both equal to 1 ===============
["all", {'ori_shape': (4, 64, 64, 16), 'shape': (4, 1, 64, 64, 16), 'ori_format': 'NHWC', 'dtype': 'float16'}, {'ori_shape': (1, 1, 4, 1), 'ori_format': 'HWCN', 'dtype': 'float16', "format":'FRACTAL_Z_C04'}, None, None, {'dtype': 'float16'}, (1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), RuntimeError],
# ============ strides no 4d ===============
["all", {'ori_shape': (4, 64, 64, 16), 'shape': (4, 1, 64, 64, 16), 'ori_format': 'NHWC', 'dtype': 'float16'}, {'ori_shape': (1, 1, 16, 1), 'ori_format': 'HWCN', 'dtype': 'float16'}, None, None, {'dtype': 'float16'}, (1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), RuntimeError],
# ============ pads no 4d ===============
["all", {'ori_shape': (4, 64, 64, 16), 'shape': (4, 1, 64, 64, 16), 'ori_format': 'NHWC', 'dtype': 'float16'}, {'ori_shape': (1, 1, 16, 1), 'ori_format': 'HWCN', 'dtype': 'float16'}, None, None, {'dtype': 'float16'}, (1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), RuntimeError],
# ============ dilations no 4d ===============
["all", {'ori_shape': (4, 64, 64, 16), 'shape': (4, 1, 64, 64, 16), 'ori_format': 'NHWC', 'dtype': 'float16'}, {'ori_shape': (1, 1, 16, 1), 'ori_format': 'HWCN', 'dtype': 'float16'}, None, None, {'dtype': 'float16'}, (1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1), RuntimeError],
# ============ weights_format not in ["NCHW", "NHWC", "HWCN"] ===============
["all", {'ori_shape': (4, 64, 64, 16), 'shape': (4, 1, 64, 64, 16), 'ori_format': 'NHWC', 'dtype': 'float16'}, {'ori_shape': (1, 1, 16, 1), 'ori_format': 'WCNH', 'dtype': 'float16'}, None, None, {'dtype': 'float16'}, (1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), RuntimeError],
# ============ inputs_format not in ["NCHW", "NHWC"] ===============
["all", {'ori_shape': (4, 64, 64, 16), 'shape': (64, 4, 16, 4, 16), 'ori_format': 'WCNH', 'dtype': 'float16'}, {'ori_shape': (1, 1, 16, 1), 'ori_format': 'HWCN', 'dtype': 'float16'}, None, None, {'dtype': 'float16'}, (1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), RuntimeError],
# ============ weights no 4d ===============
["all", {'ori_shape': (4, 64, 64, 16), 'shape': (4, 1, 64, 64, 16), 'ori_format': 'NHWC', 'dtype': 'float16'}, {'ori_shape': (1, 1, 16), 'ori_format': 'HWCN', 'dtype': 'float16'}, None, None, {'dtype': 'float16'}, (1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), RuntimeError],
# ============ inputs no 4d ===============
["all", {'ori_shape': (4, 64, 64), 'shape': (4, 1, 64, 64, 16), 'ori_format': 'NHWC', 'dtype': 'float16'}, {'ori_shape': (1, 1, 16, 1), 'ori_format': 'HWCN', 'dtype': 'float16'}, None, None, {'dtype': 'float16'}, (1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), RuntimeError],
# ============ Quant conv does not support dilate > 1 ===============
["Ascend310", {'ori_shape': (4, 64, 64, 16), 'shape': (4, 1, 64, 64, 16), 'ori_format': 'NHWC', 'dtype': 'int8'}, {'ori_shape': (1, 1, 16, 1), 'ori_format': 'HWCN', 'dtype': 'int8'}, None, None, {'dtype': 'int32'}, (1, 1, 1, 1), (0, 0, 0, 0), (2, 2, 2, 2), RuntimeError],
# ============ dma load3d time out aicore error case ===============
["Ascend310", {'ori_shape': (1, 100001, 19, 1), 'shape': (1, 1, 100001, 19, 16), 'ori_format': 'NHWC', 'dtype': 'float16'}, {'ori_shape': (256, 5, 1, 1), 'ori_format': 'HWCN', 'dtype': 'float16'}, None, None, {'dtype': 'float16'}, (1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), "success"],
# ============ dma load3d exceed L1 runtime error case ===============
["Ascend310", {'ori_shape': (1, 100001, 9, 3), 'shape': (1, 1, 100001, 9, 16), 'ori_format': 'NHWC', 'dtype': 'float16'}, {'ori_shape': (255, 9, 3, 9), 'ori_format': 'HWCN', 'dtype': 'float16'}, None, None, {'dtype': 'float16'}, (1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), RuntimeError],
["Ascend310", {'ori_shape': (1, 0, 9, 3), 'shape': (1, 1, 0, 9, 16), 'ori_format': 'NHWC', 'dtype': 'float16'}, {'ori_shape': (255, 9, 3, 9), 'ori_format': 'HWCN', 'dtype': 'float16'}, None, None, {'dtype': 'float16'}, (1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), RuntimeError],
# ============ when data h == 1, use conv1d(load3d) ===============
["Ascend310", {'ori_shape': (1, 3, 1, 50000), 'shape': (1, 1, 1, 50000, 16), 'ori_format': 'NCHW', 'dtype': 'float16'}, {'ori_shape': (1, 1, 3, 1), 'ori_format': 'HWCN', 'dtype': 'float16'}, None, None, {'dtype': 'float16'}, (1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), "success"],
# ============ test float16 case when cout % 16 == 0 ===============
["Ascend310", {'ori_shape': (1, 16, 16, 16), 'shape': (1, 1, 16, 16, 16), 'ori_format': 'NHWC', 'dtype': 'float16'}, {'ori_shape': (3, 3, 16, 16), 'ori_format': 'HWCN', 'dtype': 'float16'}, {'ori_shape': (16), "dtype": "float16"}, None, {'dtype': 'float16'}, (1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), "success"],
["Ascend310", {'ori_shape': (111, 111, 111, 111), 'shape': (1, 1, 16, 16, 16), 'ori_format': 'NHWC', 'dtype': 'float16'}, {'ori_shape': (5, 5, 111, 128), 'ori_format': 'HWCN', 'dtype': 'float16'}, {'ori_shape': (128), "dtype": "float16"}, None, {'dtype': 'float16'}, (1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), "success"],
# ============ test float16 case when cout % 16 != 0 ===============
["Ascend310", {'ori_shape': (111, 111, 111, 111), 'shape': (1, 7, 16, 16, 16), 'ori_format': 'NHWC', 'dtype': 'float16'}, {'ori_shape': (5, 5, 111, 127), 'ori_format': 'HWCN', 'dtype': 'float16'}, {'ori_shape': (127), "dtype": "float16"}, None, {'dtype': 'float16'}, (1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), "success"],
["Ascend310", {'ori_shape': (111, 111, 111, 111), 'shape': (1, 7, 16, 16, 16), 'ori_format': 'NHWC', 'dtype': 'float16'}, {'ori_shape': (5, 5, 111, 126), 'ori_format': 'HWCN', 'dtype': 'float16'}, {'ori_shape': (126), "dtype": "float16"}, None, {'dtype': 'float16'}, (1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), "success"],
["Ascend310", {'ori_shape': (111, 111, 111, 111), 'shape': (1, 7, 16, 16, 16), 'ori_format': 'NHWC', 'dtype': 'float16'}, {'ori_shape': (5, 5, 111, 123), 'ori_format': 'HWCN', 'dtype': 'float16'}, {'ori_shape': (123), "dtype": "float16"}, None, {'dtype': 'float16'}, (1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), "success"],
["Ascend310", {'ori_shape': (111, 111, 111, 111), 'shape': (1, 7, 16, 16, 16), 'ori_format': 'NHWC', 'dtype': 'float16'}, {'ori_shape': (5, 5, 111, 120), 'ori_format': 'HWCN', 'dtype': 'float16'}, {'ori_shape': (120), "dtype": "float16"}, None, {'dtype': 'float16'}, (1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), "success"],
["Ascend310", {'ori_shape': (111, 111, 111, 111), 'shape': (1, 7, 16, 16, 16), 'ori_format': 'NHWC', 'dtype': 'float16'}, {'ori_shape': (5, 5, 111, 113), 'ori_format': 'HWCN', 'dtype': 'float16'}, {'ori_shape': (113), "dtype": "float16"}, None, {'dtype': 'float16'}, (1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), "success"],
["Ascend310", {'ori_shape': (111, 111, 111, 111), 'shape': (1, 7, 16, 16, 16), 'ori_format': 'NHWC', 'dtype': 'float16'}, {'ori_shape': (5, 5, 111, 112), 'ori_format': 'HWCN', 'dtype': 'float16'}, {'ori_shape': (112), "dtype": "float16"}, None, {'dtype': 'float16'}, (1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), "success"],
]

conv2D_op_select_ut_testcase = [
# ============ success =====================
# ============ base case 910 ===============
["all", {'ori_shape': (4, 64, 64, 16), 'ori_format': 'NHWC', 'dtype': 'float16'}, {'ori_shape': (1, 1, 16, 1), 'ori_format': 'HWCN', 'dtype': 'float16'}, None, None, {'dtype': 'float16'}, (1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), "success"],
["all", {'ori_shape': (4, 64, 64, 4), 'ori_format': 'NHWC', 'dtype': 'float16'}, {'ori_shape': (1, 3, 3, 4), 'ori_format': 'NHWC', 'dtype': 'float16'}, None, None, {'dtype': 'float16'}, (1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), "success"],
["all", {'ori_shape': (4, 16, 64, 64), 'ori_format': 'NCHW', 'dtype': 'float16'}, {'ori_shape': (1, 1, 16, 1), 'ori_format': 'NCHW', 'dtype': 'float16'}, None, None, {'dtype': 'float16'}, (1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), "success"],
# ============ inputs_format not in ["NCHW", "NHWC"] ===============
["all", {'ori_shape': (71,1,1,1), 'ori_format': 'NC', 'dtype': 'float16'}, {'ori_shape': (3,3,1,1), 'ori_format': 'NCHW', 'dtype': 'float16', "format":"FRACTAL_Z_C04"}, None, (0, 0, 0, 0), None, (1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), RuntimeError],
# ============ weights no 4d ===============
["all", {'ori_shape': (4, 16, 64, 64), 'ori_format': 'NCHW', 'dtype': 'float16'}, {'ori_shape': (1, 1, 16), 'ori_format': 'HWCN', 'dtype': 'float16'}, None, None, {'dtype': 'float16'}, (1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), RuntimeError],
# ============ base case 910 ===============
["all", {'ori_shape': (4, 64, 64, 16), 'ori_format': 'NHWC', 'dtype': 'float16'}, {'ori_shape': (1, 1, 16, 1), 'ori_format': 'HWCN', 'dtype': 'float16'}, None, None, {'dtype': 'float16'}, (1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), "success"],
# ============ weights_format not in ["NCHW", "NHWC", "HWCN"] ===============
["all", {'ori_shape': (4, 64, 64, 16), 'ori_format': 'NHWC', 'dtype': 'float16'}, {'ori_shape': (1, 1, 16, 1), 'ori_format': 'WCNH', 'dtype': 'float16'}, None, None, {'dtype': 'float16'}, (1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), RuntimeError],
 ]

conv2D_dynamic_ut_testcase = [
    # ============ success =====================
    ["all", {'ori_shape': (1, 32, -1, -1), 'ori_format': 'NCHW', 'dtype': 'float16', "range": [(1, 1), (32, 32), (10, 25), (10, 25)]}, {"ori_shape": [64, 32, 1, 1], "dtype": "float16", "ori_format": "NCHW", "range": [(64, 64), (32, 32), (1, 1), (1, 1)]}, None, None, {'ori_shape': (1, 32, -1, -1), 'ori_format': 'NCHW', 'dtype': 'float16'}, (-1, -1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), 1, "NCHW", 0, "success"],
    ["all", {'ori_shape': (1, 32, -1, -1), 'ori_format': 'NCHW', 'dtype': 'float16', "range": [(1, 1), (32, 32), (10, 25), (10, 25)]}, {"ori_shape": [64, 32, 3, 3], "dtype": "float16", "ori_format": "NCHW", "range": [(64, 64), (32, 32), (3, 3), (3, 3)]}, None, None, {'ori_shape': (1, 32, -1, -1), 'ori_format': 'NCHW', 'dtype': 'float16'}, (-1, -1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), 1, "NCHW", 0, "success"],
    ["all", {'ori_shape': (-1, 32, 16, 16), 'ori_format': 'NCHW', 'dtype': 'float16', "range": [(1, 5), (32, 32), (16, 16), (16, 16)]}, {"ori_shape": [64, 32, 3, 3], "dtype": "float16", "ori_format": "NCHW", "range": [(64, 64), (32, 32), (3, 3), (3, 3)]}, None, None, {'ori_shape': (-1, 32, 16, 16), 'ori_format': 'NCHW', 'dtype': 'float16'}, (-1, -1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), 1, "NCHW", 0, "success"],
    ["all", {'ori_shape': (-1, 32, 16, 16), 'ori_format': 'NCHW', 'dtype': 'float16', "range": [(1, 10), (32, 32), (16, 16), (16, 16)]}, {"ori_shape": [64, 32, 3, 3], "dtype": "float16", "ori_format": "NCHW", "range": [(64, 64), (32, 32), (3, 3), (3, 3)]}, None, None, {'ori_shape': (-1, 32, 16, 16), 'ori_format': 'NCHW', 'dtype': 'float16'}, (-1, -1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), 1, "NCHW", 0, "success"],
    ["all", {'ori_shape': (-1, 32, -1, -1), 'ori_format': 'NCHW', 'dtype': 'float16', "range": [(1, 10), (32, 32), (10, 25), (10, 25)]}, {"ori_shape": [64, 32, 3, 3], "dtype": "float16", "ori_format": "NCHW", "range": [(64, 64), (32, 32), (3, 3), (3, 3)]}, None, None, {'ori_shape': (-1, 32, -1, -1), 'ori_format': 'NCHW', 'dtype': 'float16'}, (-1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), 1, "NCHW", 0, "success"],
    ["all", {'ori_shape': (-1, -1, -1, -1), 'ori_format': 'NCHW', 'dtype': 'float16', "range": [(1, 10), (32, 32), (10, 25), (10, 25)]}, {"ori_shape": [64, 32, 3, 3], "dtype": "float16", "ori_format": "NCHW", "range": [(64, 64), (32, 32), (3, 3), (3, 3)]}, None, None, {'ori_shape': (-1, 32, -1, -1), 'ori_format': 'NCHW', 'dtype': 'float16'}, (-1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), 1, "NCHW", 0, "success"],
    # ============ with bias ===================
    ["all", {'ori_shape': (-1, 32, -1, -1), 'ori_format': 'NCHW', 'dtype': 'float16', "range": [(1, 10), (32, 32), (10, 25), (10, 25)]}, {"ori_shape": [64, 32, 3, 3], "dtype": "float16", "ori_format": "NCHW", "range": [(64, 64), (32, 32), (3, 3), (3, 3)]},  {'ori_shape': (32, ), 'ori_format': 'NCHW', 'dtype': 'float16'}, None, {'ori_shape': (-1, 32, -1, -1), 'ori_format': 'NCHW', 'dtype': 'float16'}, (-1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), 1, "NCHW", 0, "success"],
    # ============ dilation > 1 ===================
    ["all", {'ori_shape': (-1, 32, -1, -1), 'ori_format': 'NCHW', 'dtype': 'float16', "range": [(1, 10), (32, 32), (10, 25), (10, 25)]}, {"ori_shape": [64, 32, 3, 3], "dtype": "float16", "ori_format": "NCHW", "range": [(64, 64), (32, 32), (3, 3), (3, 3)]},  {'ori_shape': (32, ), 'ori_format': 'NCHW', 'dtype': 'float16'}, None, {'ori_shape': (-1, 32, -1, -1), 'ori_format': 'NCHW', 'dtype': 'float16'}, (-1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 2, 2), 1, "NCHW", 0, "success"],
    # ============ conv1d split w ===========
    ["all", {'ori_shape': (-1, 32, 1, -1), 'ori_format': 'NCHW', 'dtype': 'float16', "range": [(1, 10), (32, 32), (1, 1), (10, 25)]}, {"ori_shape": [64, 32, 1, 3], "dtype": "float16", "ori_format": "NCHW", "range": [(64, 64), (32, 32), (1, 1), (3, 3)]}, None, None, {'ori_shape': (-1, 32, 1, -1), 'ori_format': 'NCHW', 'dtype': 'float16'}, (-1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), 1, "NCHW", 0, "success"],
    # ============ kernel > 255 ===========
    ["all", {'ori_shape': (-1, 32, 1, -1), 'ori_format': 'NCHW', 'dtype': 'float16', "range": [(1, 10), (32, 32), (1, 1), (10, 25)]}, {"ori_shape": [64, 32, 1, 300], "dtype": "float16", "ori_format": "NCHW", "range": [(64, 64), (32, 32), (1, 1), (3, 3)]}, None, None, {'ori_shape': (-1, 32, 1, -1), 'ori_format': 'NCHW', 'dtype': 'float16'}, (-1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), 1, "NCHW", 0, RuntimeError],
    ["all", {'ori_shape': (1, -1, -1, -1), 'ori_format': 'NCHW', 'dtype': 'float16', "range": [(1, 1), (32, 32), (10, 25), (10, 25)]}, {"ori_shape": [64, 32, 3, 3], "dtype": "float16", "ori_format": "NCHW", "range": [(64, 64), (32, 32), (3, 3), (3, 3)]}, None, None, {'ori_shape': (1, 32, -1, -1), 'ori_format': 'NCHW', 'dtype': 'float16'}, (-1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), 1, "NCHW", 0, "success"],
    # ============ RuntimeError ===============
    # ============ conv2d_dynamic ut test invalid range 0 ===============
    #["all", {'ori_shape': (1, 16, -1, -1), 'ori_format': 'NCHW', 'dtype': 'float16', "range": [(1, 1), (16, 16), (1, 40), (1, 40)]}, {"ori_shape": (32, 16, 3, 3), "dtype": "float16", "ori_format": "NCHW", "range": [(32, 32), (16, 16), (3, 3), (3, 3)]}, None, None, {'ori_shape': (1, 16, -1, -1), 'ori_format': 'NCHW', 'dtype': 'float16'}, (1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), 1, "NCHW", 0, RuntimeError],
    #["all", {'ori_shape': (1, 16, -1, -1), 'ori_format': 'NCHW', 'dtype': 'float16', "range": [(1, 1), (16, 16), (1, 32545), (1, 32545)]}, {"ori_shape": (32, 16, 3, 3), "dtype": "float16", "ori_format": "NCHW", "range": [(32, 32), (16, 16), (3, 3), (3, 3)]}, None, None, {'ori_shape': (1, 16, -1, -1), 'ori_format': 'NCHW', 'dtype': 'float16'}, (1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), 1, "NCHW", 0, RuntimeError],
    #["all", {'ori_shape': (1, 16, -1, -1), 'ori_format': 'NCHW', 'dtype': 'float16', "range": [(1, 1), (16, 16), (1, 50), (1, 50)]}, {"ori_shape": (32, 16, 2048, 3), "dtype": "float16", "ori_format": "NCHW", "range": [(32, 32), (16, 16), (2048, 2048), (3, 3)]}, None, None, {'ori_shape': (1, 16, -1, -1), 'ori_format': 'NCHW', 'dtype': 'float16'}, (1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), 1, "NCHW", 0, RuntimeError],
    #["all", {"ori_shape": (1, 16, -1, -1), "dtype": "float16", "ori_format":"NCHW", "range": [(1, 1), (16, 16), (3, 40), (3, 40)]}, {"ori_shape": (32, 16, 3, 3), "dtype": "float16", "ori_format": "NCHW", "range": [(32, 32), (16, 16), (3, 3), (3, 3)]}, None, None, {'ori_shape': (1, 16, -1, -1), 'ori_format': 'NCHW', 'dtype': 'float16'}, (1, 1, 2, 2), (0, 0, 0, 0), (1, 1, 1, 1), 1, "NCHW", 0, RuntimeError],
    # ============ test_conv2d_invalid_input_shape ===============
    #["all", {"ori_shape": (1, 16, -1), "dtype": "float16", "ori_format":"NCHW", "range": [(1, 1), (16, 16), (3, 40), (3, 40)]}, {"ori_shape": (32, 16, 3, 3), "dtype": "float16", "ori_format": "NCHW", "range": [(32, 32), (16, 16), (3, 3), (3, 3)]}, None, None, {'ori_shape': (1, 16, -1, -1), 'ori_format': 'NCHW', 'dtype': 'float16'}, (1, 1, 2, 2), (0, 0, 0, 0), (1, 1, 1, 1), 1, "NCHW", 0, RuntimeError],
    # ============ test_conv2d_invalid_weight_shape ===============
    ["all", {"ori_shape": (1, 16, -1, -1), "dtype": "float16", "ori_format":"NCHW", "range": [(1, 1), (16, 16), (3, 40), (3, 40)]}, {"ori_shape": (32, 16, 3), "dtype": "float16", "ori_format": "NCHW", "range": [(32, 32), (16, 16), (3, 3), (3, 3)]}, None, None, {'ori_shape': (1, 16, -1, -1), 'ori_format': 'NCHW', 'dtype': 'float16'}, (1, 1, 2, 2), (0, 0, 0, 0), (1, 1, 1, 1), 1, "NCHW", 0, RuntimeError],
    # ============ test_conv2d_invalid_stride_shape ===============
    ["all", {"ori_shape": (1, 16, -1, -1), "dtype": "float16", "ori_format":"NCHW", "range": [(1, 1), (16, 16), (3, 40), (3, 40)]}, {"ori_shape": (32, 16, 3, 3), "dtype": "float16", "ori_format": "NCHW", "range": [(32, 32), (16, 16), (3, 3), (3, 3)]}, None, None, {'ori_shape': (1, 16, -1, -1), 'ori_format': 'NCHW', 'dtype': 'float16'}, (1, 1, 2), (0, 0, 0, 0), (1, 1, 1, 1), 1, "NCHW", 0, RuntimeError],
    # ============ test_conv2d_invalid_dilation_range ===============
    ["all", {"ori_shape": (1, 16, -1, -1), "dtype": "float16", "ori_format":"NCHW", "range": [(1, 1), (16, 16), (3, 40), (3, 40)]}, {"ori_shape": (32, 16, 3, 3), "dtype": "float16", "ori_format": "NCHW", "range": [(32, 32), (16, 16), (3, 3), (3, 3)]}, None, None, {'ori_shape': (1, 16, -1, -1), 'ori_format': 'NCHW', 'dtype': 'float16'}, (1, 1, 2, 2), (0, 0, 0, 0), (1, 1, 1), 1, "NCHW", 0, RuntimeError],
    # ============ test_conv2d_invalid_format ===============
    ["all", {"ori_shape": (1, 16, -1, -1), "dtype": "float16", "ori_format":"NCHW11", "range": [(1, 1), (16, 16), (3, 40), (3, 40)]}, {"ori_shape": (32, 16, 3, 3), "dtype": "float16", "ori_format": "NCHW", "range": [(32, 32), (16, 16), (3, 3), (3, 3)]}, None, None, {'ori_shape': (1, 16, -1, -1), 'ori_format': 'NCHW', 'dtype': 'float16'}, (1, 1, 2, 2), (0, 0, 0, 0), (1, 1, 1, 1), 1, "NCHW", 0, RuntimeError],
    # ============ test_conv2d_invalid_weight_format_shape ===============
    ["all", {"ori_shape": (1, 16, -1, -1), "dtype": "float16", "ori_format":"NCHW", "range": [(1, 1), (16, 16), (3, 40), (3, 40)]}, {"ori_shape": (32, 16, 3, 3), "dtype": "float16", "ori_format": "NCHW1111", "range": [(32, 32), (16, 16), (3, 3), (3, 3)]}, None, None, {'ori_shape': (1, 16, -1, -1), 'ori_format': 'NCHW', 'dtype': 'float16'}, (1, 1, 2, 2), (0, 0, 0, 0), (1, 1, 1, 1), 1, "NCHW", 0, RuntimeError],
    # [[1, 32, 16, 16], [64, 32, 1, 1], [0, 0, 0, 0], [1, 1], "dynamic_hw", [10, 25], [10, 25], "conv_relu"],
]

conv2D_ut_precision_testcase = [
# platform, inputs, weights, bias, offset_w, outputs, strides, pads, dilations, group, format, offset_x, expect
# ============ success =====================
# ============ base case 310 ===============
["all", {'shape': (1, 1, 16, 16, 16), "format": "NC1HWC0", 'ori_shape': (1, 16, 16, 16), 'ori_format': 'NCHW', 'dtype': 'float16', "param_type": "input", "value_range": [1.0, 2.0]},
        {'shape': (1, 1, 16, 16), "format":"FRACTAL_Z", 'ori_shape': (16, 16, 1, 1), 'ori_format': 'NCHW', 'dtype': 'float16', "param_type": "input", "value_range": [1.0, 2.0]},
        None, None,
        {'shape':(1, 1, 16, 16, 16), 'ori_shape':(1, 16, 16, 16), "format": "NC1HWC0",  'ori_format': 'NCHW', "param_type": "output", 'dtype': 'float16'},
        (1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), 1, 'NCHW', 0, "success"],
["all", {'shape': (1, 1, 16, 16, 16), "format": "NC1HWC0", 'ori_shape': (1, 16, 16, 16), 'ori_format': 'NCHW', 'dtype': 'float16', "param_type": "input", "value_range": [1.0, 2.0]},
        {'shape': (1, 1, 16, 16), "format":"FRACTAL_Z", 'ori_shape': (16, 16, 1, 1), 'ori_format': 'NCHW', 'dtype': 'float16', "param_type": "input", "value_range": [1.0, 2.0]},
        None, None,
        {'shape':(1, 1, 8, 8, 16), 'ori_shape':(1, 16, 8, 8), "format": "NC1HWC0",  'ori_format': 'NCHW', "param_type": "output", 'dtype': 'float16'},
        (1, 1, 2, 2), (0, 0, 0, 0), (1, 1, 1, 1), 1, 'NCHW', 0, "success"],
["all", {'shape': (1, 1, 16, 16, 16), "format": "NC1HWC0", 'ori_shape': (1, 16, 16, 16), 'ori_format': 'NCHW', 'dtype': 'float16', "param_type": "input", "value_range": [0.1, 0.2]},
        {'shape': (225, 1, 16, 16), "format":"FRACTAL_Z", 'ori_shape': (16, 16, 15, 15), 'ori_format': 'NCHW', 'dtype': 'float16', "param_type": "input", "value_range": [1, 2]},
        None, None,
        {'shape':(1, 1, 2, 2, 16), 'ori_shape':(1, 16, 2, 2), "format": "NC1HWC0",  'ori_format': 'NCHW', "param_type": "output", 'dtype': 'float16'},
        (1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), 1, 'NCHW', 0, "success"],
["all", {'shape': (1, 1, 16, 16, 16), "format": "NC1HWC0", 'ori_shape': (1, 16, 16, 16), 'ori_format': 'NCHW', 'dtype': 'float16', "param_type": "input", "value_range": [1, 2]},
        {'shape': (9, 1, 16, 16), "format":"FRACTAL_Z", 'ori_shape': (16, 16, 3, 3), 'ori_format': 'NCHW', 'dtype': 'float16', "param_type": "input", "value_range": [1, 2]},
        None, None,
        {'shape':(1, 1, 8, 8, 16), 'ori_shape':(1, 16, 8, 8), "format": "NC1HWC0",  'ori_format': 'NCHW', "param_type": "output", 'dtype': 'float16'},
        (1, 1, 2, 2), (1, 1, 1, 1), (1, 1, 1, 1), 1, 'NCHW', 0, "success"],
["all", {'shape': (1, 1, 16, 16, 16), "format": "NC1HWC0", 'ori_shape': (1, 16, 16, 16), 'ori_format': 'NCHW', 'dtype': 'float16', "param_type": "input", "value_range": [1, 2]},
        {'shape': (256, 1, 16, 16), "format":"FRACTAL_Z", 'ori_shape': (16, 16, 16, 16), 'ori_format': 'NCHW', 'dtype': 'float16', "param_type": "input", "value_range": [1, 2]},
        None, None,
        {'shape':(1, 1, 17, 17, 16), 'ori_shape':(1, 16, 17, 17), "format": "NC1HWC0",  'ori_format': 'NCHW', "param_type": "output", 'dtype': 'float16'},
        (1, 1, 1, 1), (8, 8, 8, 8), (1, 1, 1, 1), 1, 'NCHW', 0, "success"],
["Ascend310", {'shape': (1, 1, 16, 16, 32), "format": "NC1HWC0", 'ori_shape': (1, 32, 16, 16), 'ori_format': 'NCHW', 'dtype': 'int8', "param_type": "input", "value_range": [1, 2]},
        {'shape': (256, 2, 16, 32), "format":"FRACTAL_Z", 'ori_shape': (16, 32, 16, 16), 'ori_format': 'NCHW', 'dtype': 'int8', "param_type": "input", "value_range": [1, 2]},
        None, None,
        {'shape':(1, 1, 1, 1, 16), 'ori_shape':(1, 16, 1, 1), "format": "NC1HWC0",  'ori_format': 'NCHW', "param_type": "output", 'dtype': 'int32'},
        (1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), 1, 'NCHW', 0, "success"],
]
op_support_info_testcase=[
    ["all", {'ori_shape': (4, 64, 64, 16), 'ori_format': 'NHWC', 'dtype': 'float16'}, {'ori_shape': (1, 1, 16, 1), 'ori_format': 'HWCN', 'dtype': 'float16'}, None, None, {'dtype': 'float16'}, (1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), "success"],
    ["all", {'ori_shape': (4, 64, 64, 16), 'ori_format': 'NHWC', 'dtype': 'float16'}, {'ori_shape': (1, 1, 1, 16), 'ori_format': 'NHWC', 'dtype': 'float16'}, {'ori_shape': (16), "dtype": "float16"}, None, {'dtype': 'float16'}, (1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), "success"],
]
