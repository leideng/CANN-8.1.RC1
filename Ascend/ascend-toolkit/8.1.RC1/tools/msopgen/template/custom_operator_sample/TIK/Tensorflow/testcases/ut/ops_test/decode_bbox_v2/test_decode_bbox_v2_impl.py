# # -*- coding:utf-8 -*-
from op_test_frame.ut import OpUT
import numpy as np
from op_test_frame.common import precision_info

ut_case = OpUT("DecodeBboxV2", "impl.decode_bbox_v2", "decode_bbox_v2")

def param_dict(shape, dtype):

    return {"shape": shape, "dtype": dtype, "format": "ND", "ori_shape": shape, "ori_format": "ND"}

def impl_lsit(shape_a, shape_b, dtype_a, dtype_b, scales, decode_clip, reversed_box):
    input_list = [param_dict(shape_a, dtype_a), param_dict(shape_b, dtype_b)]
    output_list = [param_dict(shape_a, dtype_a)]
    param_list = [scales, float(decode_clip), reversed_box]

    return input_list + output_list + param_list


case1 = {"params": impl_lsit([29782, 4], [29782, 4], "float16", "float16", [1.0,1.0,1.0,1.0], 0, False),
         "case_name": "faster_rcnn_case_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": impl_lsit([4, 29782], [4, 29782], "float16", "float16", [0.5,0.5,0.5,0.5], 2, True),
         "case_name": "faster_rcnn_case_2",
         "expect": "success",
         "support_expect": True}

case3 = {"params": impl_lsit([1, 4], [1, 4], "float16", "float16", [0.5,0.5,0.5,0.5], 2, False),
         "case_name": "faster_rcnn_case_3",
         "expect": "success",
         "support_expect": True}

err1 = {"params": impl_lsit([1, 4], [2, 4], "float16", "float16", [0.5,0.5,0.5,0.5], 2, False),
         "case_name": "erro_case_1",
         "expect": RuntimeError,
         "support_expect": False}

err2 = {"params": impl_lsit([1, 5], [1, 5], "float16", "float16", [0.5,0.5,0.5,0.5], 2, False),
        "case_name": "erro_case_2",
        "expect": RuntimeError,
        "support_expect": False}

err3 = {"params": impl_lsit([5, 5], [5, 5], "float16", "float16", [0.5,0.5,0.5,0.5], 2, True),
        "case_name": "erro_case_3",
        "expect": RuntimeError,
        "support_expect": False}

err4 = {"params": impl_lsit([4, 5], [4, 5], "float16", "float32", [0.5,0.5,0.5,0.5], 2, True),
        "case_name": "erro_case_5",
        "expect": RuntimeError,
        "support_expect": False}

err5 = {"params": impl_lsit([4, 5], [4, 5], "float16", "float16", [0.5,0.5,0.5,0.5], 20, True),
        "case_name": "erro_case6",
        "expect": RuntimeError,
        "support_expect": False}

err6 = {"params": impl_lsit([4, 5], [4, 5], "float16", "float16", [0.5,0.5,0.5,0.5,0.5], 2, True),
        "case_name": "erro_case7",
        "expect": RuntimeError,
        "support_expect": False}

ut_case.add_case(["Ascend310"], case1)
ut_case.add_case(["Ascend310","Ascend910"], case2)
ut_case.add_case(["Ascend310"], case3)
ut_case.add_case(["Ascend310"], err1)
ut_case.add_case(["Ascend310"], err2)
ut_case.add_case(["Ascend310"], err3)
ut_case.add_case(["Ascend310"], err4)
ut_case.add_case(["Ascend310"], err5)
ut_case.add_case(["Ascend310"], err6)
# add precision case
def gen_precision_case(shape, dtype, scales, decode_clip, reversed_box):
    return {"params": [{"shape": shape, "dtype": dtype, "ori_shape": shape,
                        "ori_format": "ND", "format": "ND",
                        "param_type": "input", "value_range": [-2, 2]},
                        {"shape": shape, "dtype": dtype, "ori_shape": shape,
                        "ori_format": "ND", "format": "ND",
                        "param_type": "input", "value_range": [-2, 2]},
                       {"shape": shape, "dtype": dtype, "ori_shape": shape,
                        "ori_format": "ND", "format": "ND", "param_type": "output"},
                       scales, decode_clip, reversed_box],
            "calc_expect_func": np_decode_bbox_v2,
            "precision_standard": precision_info.PrecisionStandard(0.005, 0.001)}

def np_decode_bbox_v2(boxes, anchors, y, scales, decode_clip, reversed_box):
    boxes_arr = boxes.get("value")
    anchors_arr = anchors.get("value")
    shape = boxes.get("shape")
    if reversed_box is False:
        boxes_arr = boxes_arr.T
        anchors_arr = anchors_arr.T
    ty, tx, th, tw = boxes_arr[0], boxes_arr[1], boxes_arr[2], boxes_arr[2]
    anchor_ymin, anchor_xmin, anchor_ymax, anchor_xmax = anchors_arr[0], anchors_arr[1], anchors_arr[2], anchors_arr[3]
    y_scale, x_scale, h_scale, w_scale = scales
    anchor_h = anchor_ymax-anchor_ymin
    anchor_w = anchor_xmax-anchor_xmin
    scaled_ty = ty / y_scale
    scaled_tx = tx / x_scale
    scaled_th = th / h_scale
    scaled_tw = tw / w_scale

    def isclose(valuex, valuey, rel_tol=1e-08, abs_tol=0.0):
        """
        determines whether the values of two floating-point numbers
        are close or equal
        """
        return math.isclose(valuex, valuey, rel_tol=rel_tol, abs_tol=abs_tol)

    if not isclose(clip, 0):
        w = np.exp(np.minimum(scaled_tw, clip)) * anchor_w
        h = np.exp(np.minimum(scaled_th, clip)) * anchor_h
    else:
        w = np.exp(scaled_tw) * anchor_w
        h = np.exp(scaled_th) * anchor_h
    ycenter = scaled_ty * anchor_h + anchor_ymin + anchor_h / 2
    xcenter = scaled_tx * anchor_w + anchor_xmin + anchor_w / 2

    ymin = ycenter - h / 2
    ymax = ycenter + h / 2
    xmin = xcenter - w / 2
    xmax = xcenter + w / 2

    if reversed_box:
        data_np = np.concatenate((ymin, xmin, ymax, xmax), axis=0).reshape(shape)
    else:
        data_np = np.concatenate((ymin, xmin, ymax, xmax), axis=0).T.reshape(shape)

    return data_np


#ut_case.add_precision_case(["Ascend310"], gen_precision_case([345,4], "float16", [10.0,10.0,5.0,5.0], 0.0, False))
#ut_case.add_precision_case(["Ascend310"], gen_precision_case([4,2385], "float16", [10.0,10.0,5.0,5.0], 0.0, True))
#ut_case.add_precision_case(["Ascend310"], gen_precision_case([4,5], "float16", [10.0,10.0,5.0,5.0], 0.5, True))


if __name__ == '__main__':
    ut_case.run("Ascend310")
    exit(0)
