#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Description : UT test for Conv3D
from op_test_frame.ut import OpUT


ut_case = OpUT("Conv3D", "impl.conv3d", "conv3d")
case_list = []


# Define Utility function
def _gen_data_case(case, expect, case_name_val, support_expect=True):
    return {"params": case,
            "case_name": case_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": support_expect}


def _run_api_end_with_d(
    fmap={'ori_shape': (1, 8, 60, 88, 32), 'shape': (1, 8, 60, 88, 32),
          'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16'},
    weight={'ori_shape': (2, 2, 2, 32, 64), 'shape': (2, 2, 2, 32, 64),
            'ori_format': 'DHWCN', 'format': 'DHWCN', 'dtype': 'float16'},
    bias=None, offset_w=None,
    output={'ori_shape': (1, 4, 30, 44, 64), 'shape': (1, 4, 30, 44, 64),
            'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16'},
    strides=(1, 2, 2, 2, 1),
    pads=[0, 0, 0, 0, 0, 0],
    dilations=(1, 1, 1, 1, 1),
    groups=1, data_format="NDHWC", offset_x=0):
    return [fmap, weight, bias, offset_w, output, strides,
            pads, dilations, groups, data_format, offset_x]


def test_op_check_supported(test_arg):
    from impl.conv3d import check_supported
    fmap = {'ori_shape': (2, 32, 15, 4098, 18), 'shape': (2, 32, 15, 4098, 18),
        'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'float16'}
    (fmap, weight, bias, offset_w, output, strides,
            pads, dilations, groups, data_format, _) = _run_api_end_with_d(fmap = fmap)
    check_supported(fmap, weight, bias, offset_w, output, strides, pads, dilations, groups, data_format)


ut_case.add_cust_test_func(test_func=test_op_check_supported)

def _test_op_get_op_support_info(test_arg):
    from impl.conv3d import get_op_support_info
    (fmap, weight, bias, offset_w, output, strides,
        pads, dilations, groups, data_format, offset_x) = _run_api_end_with_d()
    get_op_support_info(
       fmap, weight, bias, offset_w, output, strides,
            pads, dilations, groups, data_format, offset_x)
    # Test Bias Cut in NDC1HWCO
    fmap_ndc1hwc0 = {'ori_shape': (1, 8, 60, 88, 32), 'shape': (1, 8, 2, 60, 88, 16), 'ori_format': 'NDHWC', 'format': 'NDC1HWC0', 'dtype': 'float16'}
    bias = {'ori_shape': (64,), 'shape': (64,),
            'ori_format': 'ND', 'format': 'ND', 'dtype': 'float16'}
    get_op_support_info(
       fmap_ndc1hwc0, weight, bias, offset_w, output, strides,
            pads, dilations, groups, data_format, offset_x)
    # Test None filter shape runtime Errors
    wrong_filter = weight.copy()
    wrong_filter['ori_format'] = 'NDDDD'
    wrong_filter['format'] = 'NDDDD'
    try:
        get_op_support_info(
        fmap, wrong_filter, None, offset_w, output, strides,
                pads, dilations, groups, data_format, offset_x)
    except Exception as e:
        print(e)
    # Test strides_formated runtime Errors
    wrong_fmap = fmap.copy()
    wrong_fmap['ori_format'] = 'NDDDD'
    wrong_fmap['format'] = 'NDDDD'
    try:
        get_op_support_info(
        wrong_fmap, weight, None, offset_w, output, strides,
                pads, dilations, groups, data_format, offset_x)
    except Exception as e:
        print(e)
    
ut_case.add_cust_test_func(test_func=_test_op_get_op_support_info)

# test_conv3dbp_succ_d
case1 = _run_api_end_with_d()

# test_conv3dbp_stride_one
fmap = {'ori_shape': (1, 32, 8, 60, 88), 'shape': (1, 32, 8, 60, 88),
        'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'float16'},
weight = {'ori_shape': (64, 32, 2, 2, 2), 'shape': (64, 32, 2, 2, 2),
          'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'float16'},
output = {'ori_shape': (1, 7, 59, 87, 64), 'shape': (1, 7, 59, 87, 64),
          'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16'}
strides = (1, 1, 1, 1, 1)
case2 = _run_api_end_with_d(output=output, strides=strides)

# test_bias_length_fail
bias = {'ori_shape': (64, 64,), 'shape': (64, 64,),
        'ori_format': 'ND', 'format': 'ND', 'dtype': 'float16'}
case3 = _run_api_end_with_d(bias=bias)

# test_conv3d_invalid_fmap_shape
fmap = {'ori_shape': (2, 32, 15, 4098, 18), 'shape': (2, 32, 15, 4098, 18),
        'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'float16'}
case4 = _run_api_end_with_d(fmap=fmap)

# test_conv3d_invalid_output
output = {'dtype': 'float32'}
case5 = _run_api_end_with_d(output=output)

# test_conv3d_invalid_dilations
dilations = (1, 2, 1, 1, 1)
case6 = _run_api_end_with_d(dilations=dilations)

# test_conv3d_invalid_fmap_shape
fmap = {'ori_shape': (1, 8, 60, 88), 'shape': (1, 8, 60, 88),
      'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16'}
case7 = _run_api_end_with_d(fmap=fmap)

# test_conv3d_invalid_pad_length
pads = (0, -1, -1, -1, 0)
case8 = _run_api_end_with_d(pads=pads)

# test_conv3d_invalid_weight
weight = {'ori_shape': (2, 2, 2, 32), 'shape': (2, 2, 2, 32),
        'ori_format': 'DHWCN', 'format': 'DHWCN', 'dtype': 'float16'}
case9 = _run_api_end_with_d(weight=weight)

# test_conv3d_invalid_weight_D
weight = {'ori_shape': (2, 2, 354, 32, 64), 'shape': (2, 2, 354, 32, 64),
        'ori_format': 'DHWCN', 'format': 'DHWCN', 'dtype': 'float16'}
case10 = _run_api_end_with_d(weight=weight)

# test_conv3d_invalid_big_fmap
fmap = {'ori_shape': (200, 3000, 4000, 4000, 3000),
      'shape': (200, 3000, 4000, 4000, 3000),
      'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16'}
case11 = _run_api_end_with_d(fmap=fmap)

# test_conv3d_invalid_bias_dtype
bias = {'ori_shape': (1,), "dtype": "float32"}
case12 = _run_api_end_with_d(bias=bias)

# test_conv3d_invalid_pads
pads = (1, 1, 1, 1, 3, 1)
case13 = _run_api_end_with_d(pads=pads)

# test_conv3d_invalid_stride_shape
dilations = (1, 1, 0, 1, 1)
case14 = _run_api_end_with_d(dilations=dilations)

# test_conv3d_invalid_fmap_format
fmap = {'ori_shape': (1, 32, 8, 60, 88), 'shape': (1, 32, 8, 60, 88),
      'ori_format': 'NDCHW', 'format': 'NDCHW', 'dtype': 'float16'}
case15 = _run_api_end_with_d(fmap=fmap)

# test_conv3d_invalid_weight
weight = {'ori_shape': (2, 2, 2, 32, 64), 'shape': (2, 2, 2, 32, 64),
        'ori_format': 'NDCHW', 'format': 'NDCHW', 'dtype': 'float16'}
case16 = _run_api_end_with_d(weight=weight)

# test_conv3d_invalid_stride_shape
strides = (1, 0, 1, 1, 1)
case17 = _run_api_end_with_d(strides=strides)

# test_conv3d_invalid_stride_shape
strides = (1, 1, 0, 1, 1)
case18 = _run_api_end_with_d(strides=strides)

# test_conv3d_invalid_stride_shape
strides = (1, 1, 1, 0, 1)
case19 = _run_api_end_with_d(strides=strides)

# test_conv3d_invalid_weight
weight = {'ori_shape': (257, 2, 2, 32, 64), 'shape': (257, 2, 2, 32, 64),
        'ori_format': 'DHWCN', 'format': 'DHWCN', 'dtype': 'float16'}
case20 = _run_api_end_with_d(weight=weight)

# test_conv3d_invalid_weight
weight = {'ori_shape': (2, 257, 2, 32, 64), 'shape': (2, 257, 2, 32, 64),
        'ori_format': 'DHWCN', 'format': 'DHWCN', 'dtype': 'float16'}
case21 = _run_api_end_with_d(weight=weight)

# test_conv3d_dilation_d_zero
dilations = (1, 0, 1, 1, 1)
case22 = _run_api_end_with_d(dilations=dilations)

# test_conv3d_invalid_pad
pads = (256, 256, 256, 256, 256, 256)
case23 = _run_api_end_with_d(pads=pads)

# test_conv3d_invalid_fmap_shape
fmap = {'ori_shape': (1, 8, 60, 4098, 32), 'shape': (1, 8, 60, 4098, 32),
        'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16'}
case24 = _run_api_end_with_d(fmap=fmap)

fmap = {'ori_shape': (1, 8, 4098, 88, 32), 'shape': (1, 8, 4098, 88, 32),
        'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16'}
case25 = _run_api_end_with_d(fmap=fmap)

# test_conv3d_invalid_weight
weight = {'ori_shape': (2, 2, 257, 32, 64), 'shape': (2, 2, 257, 32, 64),
        'ori_format': 'DHWCN', 'format': 'DHWCN', 'dtype': 'float16'}
case26 = _run_api_end_with_d(weight=weight)

# test_conv3d_invalid_pad
pads = (3, 3, 256, 256, 256, 256)
case27 = _run_api_end_with_d(pads=pads)

# Test Conv3D Bias Case
bias = {'ori_shape': (64,), 'shape': (64,),
            'ori_format': 'ND', 'format': 'ND', 'dtype': 'float16'}
case28 = _run_api_end_with_d(bias=bias)

# test_conv3d_fmap_wrong_format
# WARNING: Did not trigger anything. This is consider to be redundant
wrong_fmap={'ori_shape': (1, 8, 60, 88, 32), 'shape': (1, 8, 60, 88, 32),
            'ori_format': 'NHWC', 'format': 'NHWC', 'dtype': 'float16'}
case29 = _run_api_end_with_d(fmap=wrong_fmap)

# test_conv3d_wight_wrong_format
wrong_weight={'ori_shape': (2, 2, 2, 32, 64), 'shape': (2, 2, 2, 32, 64),
              'ori_format': 'NHWC', 'format': 'NHWC', 'dtype': 'float16'}
case30 = _run_api_end_with_d(weight=wrong_weight)

# test_stride_length_constraint
wrong_strides = [1,1,1]
case31 = _run_api_end_with_d(strides=wrong_strides)

# test_dilation_length_constraint
wrong_dilations = [1,1,1]
case32 = _run_api_end_with_d(dilations=wrong_dilations)

# test_groups_constraint
case33 = _run_api_end_with_d(groups=2)

# test_conv3d_dilation_w_zero
dilations = (1, 1, 1, 0, 1)
case34 = _run_api_end_with_d(dilations=dilations)

# test_conv3d_fmap_wrong_format
wrong_format = "NHWC"
case35 = _run_api_end_with_d(data_format=wrong_format)

# test_dilation_none
# This is Redundant test. Para_check block none
case36 = _run_api_end_with_d(dilations=None)

# test_fmap_dim + pad < filter_dim
fmap={'ori_shape': (1, 8, 8, 8, 32), 'shape': (1, 8, 8, 8, 32),
          'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16'}
weight={'ori_shape': (10, 10, 10, 32, 64), 'shape': (10, 10, 10, 32, 64),
        'ori_format': 'DHWCN', 'format': 'DHWCN', 'dtype': 'float16'}
strides=(1, 1, 1, 1, 1)
case37 = _run_api_end_with_d(fmap=fmap, weight=weight, strides=strides)

fmap={'ori_shape': (1, 8, 8, 8, 32), 'shape': (1, 8, 8, 8, 32),
          'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16'}
weight={'ori_shape': (2, 10, 10, 32, 64), 'shape': (2, 10, 10, 32, 64),
        'ori_format': 'DHWCN', 'format': 'DHWCN', 'dtype': 'float16'}
strides=(1, 1, 1, 1, 1)
case38 = _run_api_end_with_d(fmap=fmap, weight=weight, strides=strides)

fmap={'ori_shape': (1, 8, 8, 8, 32), 'shape': (1, 8, 8, 8, 32),
          'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16'}
weight={'ori_shape': (2, 2, 10, 32, 64), 'shape': (2, 2, 10, 32, 64),
        'ori_format': 'DHWCN', 'format': 'DHWCN', 'dtype': 'float16'}
strides=(1, 1, 1, 1, 1)
case39 = _run_api_end_with_d(fmap=fmap, weight=weight, strides=strides)

# Test Padding Error
pads = (0, 0, 256, 256, 256, 256)
case40 = _run_api_end_with_d(pads=pads)

pads = (0, 0, 3, 3, 0, 0)
case41 = _run_api_end_with_d(pads=pads)

pads = (0, 0, 0, 0, 256, 256)
case42 = _run_api_end_with_d(pads=pads)


# Add test Cases
# Params is the input params of the operator.
ut_case.add_case(["Ascend910", "Ascend310"],
                 _gen_data_case(case1, "success", "case1", True))

ut_case.add_case(["Ascend910", "Ascend310"],
                 _gen_data_case(case2, "success", "case2", True))

ut_case.add_case(["Ascend910", "Ascend310"],
                 _gen_data_case(case3, RuntimeError, "case3", True))

ut_case.add_case(["Ascend910", "Ascend310"],
                 _gen_data_case(case4, RuntimeError, "case4", True))

ut_case.add_case(["Ascend910", "Ascend310"],
                 _gen_data_case(case5, RuntimeError, "case5", True))

ut_case.add_case(["Ascend910", "Ascend310"],
                 _gen_data_case(case6, RuntimeError, "case6", True))

ut_case.add_case(["Ascend910", "Ascend310"],
                 _gen_data_case(case7, RuntimeError, "case7", True))

ut_case.add_case(["Ascend910", "Ascend310"],
                 _gen_data_case(case8, RuntimeError, "case8", True))

ut_case.add_case(["Ascend910", "Ascend310"],
                 _gen_data_case(case9, RuntimeError, "case9", True))

ut_case.add_case(["Ascend910", "Ascend310"],
                 _gen_data_case(case10, RuntimeError, "case10", True))

ut_case.add_case(["Ascend910", "Ascend310"],
                 _gen_data_case(case11, RuntimeError, "case11", True))

ut_case.add_case(["Ascend910", "Ascend310"],
                 _gen_data_case(case12, RuntimeError, "case12", True))

ut_case.add_case(["Ascend910", "Ascend310"],
                 _gen_data_case(case13, RuntimeError, "case13", True))

ut_case.add_case(["Ascend910", "Ascend310"],
                 _gen_data_case(case14, RuntimeError, "case14", True))

ut_case.add_case(["Ascend910", "Ascend310"],
                 _gen_data_case(case15, RuntimeError, "case15", True))

ut_case.add_case(["Ascend910", "Ascend310"],
                 _gen_data_case(case16, RuntimeError, "case16", True))

ut_case.add_case(["Ascend910", "Ascend310"],
                 _gen_data_case(case17, RuntimeError, "case17", True))

ut_case.add_case(["Ascend910", "Ascend310"],
                 _gen_data_case(case18, RuntimeError, "case18", True))

ut_case.add_case(["Ascend910", "Ascend310"],
                 _gen_data_case(case19, RuntimeError, "case19", True))

ut_case.add_case(["Ascend910", "Ascend310"],
                 _gen_data_case(case20, RuntimeError, "case20", True))

ut_case.add_case(["Ascend910", "Ascend310"],
                 _gen_data_case(case21, RuntimeError, "case21", True))

ut_case.add_case(["Ascend910", "Ascend310"],
                 _gen_data_case(case22, RuntimeError, "case22", True))

ut_case.add_case(["Ascend910", "Ascend310"],
                 _gen_data_case(case23, RuntimeError, "case23", True))

ut_case.add_case(["Ascend910", "Ascend310"],
                 _gen_data_case(case24, RuntimeError, "case24", True))

ut_case.add_case(["Ascend910", "Ascend310"],
                 _gen_data_case(case25, RuntimeError, "case25", True))

ut_case.add_case(["Ascend910", "Ascend310"],
                 _gen_data_case(case26, RuntimeError, "case26", True))

ut_case.add_case(["Ascend910", "Ascend310"],
                 _gen_data_case(case27, RuntimeError, "case27", True))

ut_case.add_case(["Ascend910", "Ascend310"],
                 _gen_data_case(case28, RuntimeError, "Conv3D_default_bias", True))

ut_case.add_case(["Ascend910", "Ascend310"],
                 _gen_data_case(case29, RuntimeError, "fmap_format_wrong", True))

ut_case.add_case(["Ascend910", "Ascend310"],
                 _gen_data_case(case30, RuntimeError, "weight_format_wrong", True))

ut_case.add_case(["Ascend910", "Ascend310"],
                 _gen_data_case(case31, RuntimeError, "wrong_strides", True))

ut_case.add_case(["Ascend910", "Ascend310"],
                 _gen_data_case(case32, RuntimeError, "wrong_dilation", True))

ut_case.add_case(["Ascend910", "Ascend310"],
                 _gen_data_case(case33, RuntimeError, "wrong_groups", True))
        
ut_case.add_case(["Ascend910", "Ascend310"],
                 _gen_data_case(case34, RuntimeError, "case34", True))

ut_case.add_case(["Ascend910", "Ascend310"],
                 _gen_data_case(case35, RuntimeError, "case35", True))
            
ut_case.add_case(["Ascend910", "Ascend310"],
                 _gen_data_case(case36, RuntimeError, "case36", True))

ut_case.add_case(["Ascend910", "Ascend310"],
                 _gen_data_case(case37, RuntimeError, "case37", True))

ut_case.add_case(["Ascend910", "Ascend310"],
                 _gen_data_case(case38, RuntimeError, "case38", True))

ut_case.add_case(["Ascend910", "Ascend310"],
                 _gen_data_case(case39, RuntimeError, "case39", True))

ut_case.add_case(["Ascend910", "Ascend310"],
                 _gen_data_case(case40, RuntimeError, "case40", True))

ut_case.add_case(["Ascend910", "Ascend310"],
                 _gen_data_case(case41, RuntimeError, "case41", True))

ut_case.add_case(["Ascend910", "Ascend310"],
                 _gen_data_case(case42, RuntimeError, "case42", True))
if __name__ == '__main__':
    ut_case.run("Ascend910")
    exit(0)
