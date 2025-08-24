#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
yolo_v5_detection_output_d
"""

# 'pylint: disable=too-many-lines,unused-argument,too-many-locals,too-many-arguments
# 'pylint: disable=ungrouped-imports,import-error,too-many-branches
from impl import constant_util as constant
from impl.util import util_select_op_base
from impl.util.platform_adapter import tbe_platform
from impl.yolo_v3_cls_prob_v2 import check_param
from impl.yolo_v3_detection_output_v2d import DetectionOutput as YoloV3DectionOutput


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    This class for Constant.
    """
    PRE_NMS_TOPN = 1024
    UB_NUM = 10240
    # reserve size for ub
    RESERVE_SIZE = 16 * 1024
    # repeat one
    REPEAT_ONE = 1
    # one nburst
    NBURST_ONE = 1
    # value one
    VALUE_ONE = 1
    # stride eight for dma
    STRIDE_EIGHT = 8
    # stride zero for dma
    GAP_ZERO = 0
    # sid for dma
    SID = 0
    # value zero
    VALUE_ZERO = 0
    # value two
    VALUE_TWO = 2
    # value three
    VALUE_THREE = 3
    # neg one
    NEG_ONE = -1
    # neg two
    NEG_TWO = -2
    # value half
    VALUE_HALF = 0.5


# 'pylint: disable=unused-argument, too-many-locals, too-many-arguments
def get_op_support_info(input_x, box_out, box_out_num, biases,
                        boxes=3, coords=4, classes=80,
                        relative=True, obj_threshold=0.5,
                        post_nms_topn=1024, score_threshold=0.5,
                        iou_threshold=0.45, pre_nms_topn=512,
                        input_num=10, resize_origin_img_to_net=False,
                        out_box_dim=3, alpha=2.0,
                        kernel_name="yolo_v5_detection_output_d"):
    """
    get split info
    only support split N
    """
    return util_select_op_base.get_split_n_info(list(range(input_num)), [0, 1])


# 'pylint: disable=invalid-name, too-many-locals, too-many-arguments
# 'pylint: disable=unused-argument
def yolo_v5_detection_output_d(x, box_out, box_out_num, biases,
                               boxes=3, coords=4, classes=80,
                               relative=True, obj_threshold=0.5,
                               post_nms_topn=1024, score_threshold=0.5,
                               iou_threshold=0.45, pre_nms_topn=512,
                               N=10, resize_origin_img_to_net=False,
                               out_box_dim=3, alpha=2.0,
                               kernel_name="yolo_v5_detection_output_d"):

    """
    yolo_v5_detection_output_d

    Parameters
    ----------
    x: A list of `dict`. include following data
        coord_data: (N-1)/3
        obj_prob: (N-1)/3
        classes_prob: (N-1)/3
        img_info: 1
        dict include keys shape and dtype dict, shape.
        dtype:fp16,fp32 format:only support NCHW

    windex: A list of dict, shape, dtype:fp16,fp32 format:only support NCHW
        (N-1)/3
    hindex: A list of dict, shape, dtype:fp16,fp32 format:only support NCHW
        (N-1)/3

    box_out: dict, shape, dtype:fp16,fp32 format:only support NCHW
    box_out_num: dict, shape, dtype:fp16,fp32 format:only support NCHW
    biases: A list of box's biases list
    boxes: number of boxes
    coords: number of coordinates
    classes: number of classes
    relative:
    obj_threshold: threshold of probability of objects
    score_threshold: threshold for each category
    post_nms_topn: after nms, return posttopk boxes
    iou_threshold: nms threshold
    pre_nms_topn: for each category,take the number of pre nms topn
        before processing, and the maximum is 1024
    N: input(x) number
    resize_origin_img_to_net: bool
        False: images use darknet letter box resize
        True: images use direct scaling resize
    out_box_dim: output of out_box dim count
    kernel_name: kernel_name

    Returns
    -------
    tik_instance: tik_instance
    """
    box_info = []
    yolo_num = int((N - 1) / 3)
    batch = x[0]['shape'][0]
    dtype = x[0]['dtype']
    biases_list = []
    for i in range(yolo_num):
        index = x[N + i]
        h = index['shape'][0]
        w = index['shape'][1]
        box_info.append({"shape": (batch, boxes * (coords + 1 + classes), h, w),
                         "dtype": dtype, "format": "NCHW"})
        biases_list.append(biases[i * 2 * boxes: i * 2 * boxes + 2 * boxes])

    input_dict = {
        "box_info": box_info,
        "biases": biases_list,
        "coords": coords,
        "boxes": boxes,
        "classes": classes,
        "relative": relative,
        "obj_threshold": obj_threshold,
        "classes_threshold": score_threshold,
        "post_top_k": post_nms_topn,
        "nms_threshold": iou_threshold,
        "pre_nms_topn": pre_nms_topn,
        "max_box_number_per_batch": post_nms_topn,
        "resize_origin_img_to_net": resize_origin_img_to_net,
        "kernel_name": kernel_name,
        "alpha": alpha,
    }

    check_param(input_dict)
    detection_output = DetectionOutput(input_dict)
    tik_instance = detection_output.compute_detection_output(kernel_name)
    return tik_instance


# 'pylint: disable=too-many-ancestors,too-many-public-methods
class DetectionOutput(YoloV3DectionOutput):
    """
    Function: use to process DetectionOutput
    Modify : 2019-11-06
    """

    def __init__(self, input_dict):
        """
        init the detection output parameters

        Parameters
        ----------
        input_dict: input_dict is a dict, the keys as follow:
                    box1_info,box2_info,box3_info,biases1,biases2,biases3,
                    coords,boxes,classes,relative,obj_threshold,post_top_k,
                    post_top_k,nms_threshold,pre_nms_topn,
                    max_box_number_per_batch,kernel_name, for more details,
                    please check the yolov5_detection_output function

        Returns
        -------
          None
        """
        self.alpha = input_dict.get("alpha")
        super(DetectionOutput, self).__init__(input_dict)

    def compute_detection_output(self, kernel_name):
        """
        compute detection output is main function of the detection output

        Parameters
        ----------
        kernel_name:

        Returns
        -------
        None
        """
        with self.instance.for_range(0, self.block_num,
                                     block_num=self.block_num) as block_i:
            image_ub = self.instance.Tensor(self.dtype,
                                            [constant.BLOCK_SIZE // self.dsize],
                                            scope=tbe_platform.scope_ubuf,
                                            name="image_ub")
            batch = self.instance.Scalar("int32")
            with self.instance.for_range(0, self.outer_loop) as outer_i:
                batch.set_as(block_i * self.outer_loop + outer_i)
                param = {}
                self.init_param(batch, param)
                self.instance.data_move(image_ub, self.img_info[batch * 4],
                                        constant.SID, constant.DEFAULT_NBURST,
                                        constant.DEFAULT_BURST_LEN, 0, 0)
                self.correct_box(batch, image_ub)
                self.cls_prob(batch, param)
                self.multi_class(batch, image_ub, param)
            if self.outer_tail > 0:
                with self.instance.if_scope(block_i < self.outer_tail):
                    batch.set_as(self.block_num * self.outer_loop + block_i)
                    self.instance.data_move(image_ub, self.img_info[batch * 4],
                                            constant.SID,
                                            constant.DEFAULT_NBURST,
                                            constant.DEFAULT_BURST_LEN, 0, 0)
                    param = {}
                    self.init_param(batch, param)
                    self.correct_box(batch, image_ub)
                    self.cls_prob(batch, param)
                    self.multi_class(batch, image_ub, param)

        input_tuple = tuple(self.coord_data) + tuple(self.obj_datas) + \
                      tuple(self.classes_data) + tuple([self.img_info]) + \
                      tuple(self.windex) + tuple(self.hindex)
        self.instance.BuildCCE(kernel_name=kernel_name,
                               inputs=input_tuple,
                               outputs=(self.bbox, self.bbox_num),
                               enable_l2=False)

        return self.instance

    def compute_big_hw(self, batch, loop, param, repeat):
        """
        compute big shape height and weight

        Parameters
        ----------
        batch: the number of picture
        loop: loop times
        param: a dict,the keys as fllow:
                mov_len: the number of elements of each data move
                mov_loop: data move loop times
                last_len: the number of elements of last_len data move
                ub_bias: a tensor,store bias
                x_vmuls_val: a scalar
                x_vadds_val: a scalar
                y_vmuls_val: a scalar
                ub_a: a tensor,store middle compute data, store coords data
                ub_b: a tensor,store middle compute data for calculate (2*[pw,ph])^alpha*bias/[netw,neth]
                ub_c: a tensor,store middle compute data for calculate (2*[pw,ph])^alpha*bias/[netw,neth]
                last_32b: a tensor,store last_32b data
                co_id: a scalar,store co_id
                box_id:  a scalar,store box_id
                in_data: a tensor
        repeat: vector repeat times

        Returns
        -------
        None
        """
        tmp_scalar = self.instance.Scalar(self.dtype)
        bias_value = self.instance.Scalar(self.dtype)
        # comput height
        with self.instance.if_scope(param['co_id'] == Constant.VALUE_TWO):
            bias_value.set_as(
                param['ub_bias'][Constant.VALUE_TWO * param['box_id'] + Constant.VALUE_ONE])
            tmp_scalar.set_as(param['img_ub'][0])

            self.instance.vec_muls(self.mask, param['ub_c'], param['ub_a'],
                                   2, repeat, Constant.STRIDE_EIGHT, Constant.STRIDE_EIGHT) # 2 * sigmoid

            self.instance.vec_ln(self.mask, param['ub_b'], param['ub_c'],
                                  repeat, Constant.STRIDE_EIGHT, Constant.STRIDE_EIGHT) # ln(h)
            self.instance.vec_muls(self.mask, param['ub_b'], param['ub_b'], self.alpha,
                                  repeat, Constant.STRIDE_EIGHT, Constant.STRIDE_EIGHT) # alpha*ln(h)
            self.instance.vec_exp(self.mask, param['ub_b'], param['ub_b'],
                                  repeat, Constant.STRIDE_EIGHT, Constant.STRIDE_EIGHT) # e^(alpha(ln(h))

            self.instance.vec_muls(self.mask, param['ub_c'], param['ub_b'],
                                   bias_value, repeat, Constant.STRIDE_EIGHT, Constant.STRIDE_EIGHT)

            self.instance.vec_dup(self.mask, param['ub_b'], tmp_scalar, repeat,
                                  Constant.STRIDE_EIGHT)

            self.newton_div(param['ub_b'], param['ub_c'], param['ub_b'], repeat)

            self.instance.vec_muls(self.mask, param['ub_b'], param['ub_b'],
                                   param['y_vmuls_val'], repeat,
                                   Constant.STRIDE_EIGHT, Constant.STRIDE_EIGHT)

            if not self.relative:
                tmp_scalar.set_as(param['img_ub'][2])
                self.instance.vec_muls(self.mask, param['ub_b'], param['ub_b'],
                                       tmp_scalar, repeat,
                                       Constant.STRIDE_EIGHT, Constant.STRIDE_EIGHT)
            self.data_mov_out(batch, loop, param)
        # comput weight
        with self.instance.if_scope(param['co_id'] == Constant.VALUE_THREE):
            bias_value.set_as(param['ub_bias'][Constant.VALUE_TWO * param['box_id']])

            # img ub: neth,netw,scaleh,scalew
            tmp_scalar.set_as(param['img_ub'][1])

            self.instance.vec_muls(self.mask, param['ub_c'], param['ub_a'],
                                   2, repeat, Constant.STRIDE_EIGHT, Constant.STRIDE_EIGHT) # 2 * sigmoid

            self.instance.vec_ln(self.mask, param['ub_b'], param['ub_c'],
                                 repeat, Constant.STRIDE_EIGHT, Constant.STRIDE_EIGHT) # ln(w)
            self.instance.vec_muls(self.mask, param['ub_b'], param['ub_b'], self.alpha,
                                   repeat, Constant.STRIDE_EIGHT, Constant.STRIDE_EIGHT) # alpha*ln(w)
            self.instance.vec_exp(self.mask, param['ub_b'], param['ub_b'],
                                  repeat, Constant.STRIDE_EIGHT, Constant.STRIDE_EIGHT) # e^(alpha(ln(w))

            self.instance.vec_muls(self.mask, param['ub_c'], param['ub_b'],
                                   bias_value, repeat, Constant.STRIDE_EIGHT, Constant.STRIDE_EIGHT)

            self.instance.vec_dup(self.mask, param['ub_b'], tmp_scalar, repeat,
                                  Constant.STRIDE_EIGHT)

            self.newton_div(param['ub_b'], param['ub_c'], param['ub_b'], repeat)
            self.instance.vec_muls(self.mask, param['ub_b'], param['ub_b'],
                                   param['x_vmuls_val'], repeat,
                                   Constant.STRIDE_EIGHT, Constant.STRIDE_EIGHT)

            if not self.relative:
                tmp_scalar.set_as(param['img_ub'][3])
                self.instance.vec_muls(self.mask, param['ub_b'], param['ub_b'],
                                       tmp_scalar, repeat,
                                       Constant.STRIDE_EIGHT, Constant.STRIDE_EIGHT)

            self.data_mov_out(batch, loop, param)

    def compute_big_xy(self, batch, cycle, param, repeat):
        """
        compute big shape of x,y

        Parameters
        ----------
        batch: the number of picture
        loop: loop times
        param: a dict,the keys as fllow:
                mov_len: the number of elements of each data move
                mov_loop: data move loop times
                last_len: the number of elements of last_len data move
                ub_bias: a tensor,store bias
                x_vmuls_val: a scalar
                x_vadds_val: a scalar
                y_vmuls_val: a scalar
                ub_a: a tensor,store middle compute data, store coords data
                ub_b: a tensor,store middle compute data for calculate
                      ([px,py]*2-0.5+[windex,hindex])/[feature_w,feature_h] and relative coords data
                ub_c: a tensor,store middle compute data for calculate
                      ([px,py]*2-0.5+[windex,hindex])/[feature_w,feature_h] and relative coords data
                last_32b: a tensor,store last_32b data
                co_id: a scalar,store co_id
                box_id:  a scalar,store box_id
                in_data: a tensor
        repeat: vector repeat times

        Returns
        -------
        None
        """
        tmp_scalar = self.instance.Scalar(self.dtype)
        # compute x
        with self.instance.if_scope(param['co_id'] == Constant.VALUE_ZERO):
            # move windex to ub b
            self.instance.data_move(param['ub_b'], param['windex'][cycle * param['mov_len']],
                                    Constant.SID, Constant.NBURST_ONE, param['burlen'], 0, 0)

            self.instance.vec_muls(self.mask, param['ub_c'], param['ub_a'], 2.0, repeat,
                                   Constant.STRIDE_EIGHT, Constant.STRIDE_EIGHT)

            self.instance.vec_adds(self.mask, param['ub_c'], param['ub_c'], -0.5, repeat,
                                   Constant.STRIDE_EIGHT, Constant.STRIDE_EIGHT)

            self.instance.vec_add(self.mask, param['ub_b'], param['ub_c'], param['ub_b'], repeat,
                                  Constant.STRIDE_EIGHT, Constant.STRIDE_EIGHT, Constant.STRIDE_EIGHT)

            self.instance.vec_muls(self.mask, param['ub_b'], param['ub_b'], (1.0 / param['w']), repeat,
                                   Constant.STRIDE_EIGHT, Constant.STRIDE_EIGHT)

            self.instance.vec_muls(self.mask, param['ub_b'], param['ub_b'], param['x_vmuls_val'], repeat,
                                   Constant.STRIDE_EIGHT, Constant.STRIDE_EIGHT)

            self.instance.vec_adds(self.mask, param['ub_b'], param['ub_b'], param['x_vadds_val'], repeat,
                                   Constant.STRIDE_EIGHT, Constant.STRIDE_EIGHT)

            if not self.relative:
                tmp_scalar.set_as(param['img_ub'][3])
                self.instance.vec_muls(self.mask, param['ub_b'], param['ub_b'],
                                       tmp_scalar, repeat,
                                       Constant.STRIDE_EIGHT, Constant.STRIDE_EIGHT)

            self.data_mov_out(batch, cycle, param)
        #compute y
        with self.instance.if_scope(param['co_id'] == 1):
            # move hindex to ub
            self.instance.data_move(param['ub_b'],
                                    param['hindex'][cycle * param['mov_len']],
                                    Constant.SID,
                                    Constant.NBURST_ONE,
                                    param['burlen'],
                                    Constant.GAP_ZERO, Constant.GAP_ZERO)

            self.instance.vec_muls(self.mask, param['ub_c'], param['ub_a'], 2.0, repeat,
                                   Constant.STRIDE_EIGHT, Constant.STRIDE_EIGHT)

            self.instance.vec_adds(self.mask, param['ub_c'], param['ub_c'], -0.5, repeat,
                                   Constant.STRIDE_EIGHT, Constant.STRIDE_EIGHT)

            self.instance.vec_add(self.mask, param['ub_b'], param['ub_c'], param['ub_b'], repeat,
                                  Constant.STRIDE_EIGHT, Constant.STRIDE_EIGHT, Constant.STRIDE_EIGHT)

            self.instance.vec_muls(self.mask, param['ub_b'], param['ub_b'], (1.0 / param['h']), repeat,
                                   Constant.STRIDE_EIGHT, Constant.STRIDE_EIGHT)

            self.instance.vec_muls(self.mask, param['ub_b'], param['ub_b'], param['y_vmuls_val'], repeat,
                                   Constant.STRIDE_EIGHT, Constant.STRIDE_EIGHT)

            self.instance.vec_adds(self.mask, param['ub_b'], param['ub_b'], param['y_vadds_val'], repeat,
                                   Constant.STRIDE_EIGHT, Constant.STRIDE_EIGHT)

            if not self.relative:
                tmp_scalar.set_as(param['img_ub'][2])
                self.instance.vec_muls(self.mask, param['ub_b'], param['ub_b'],
                                       tmp_scalar, repeat,
                                       Constant.STRIDE_EIGHT, Constant.STRIDE_EIGHT)

            self.data_mov_out(batch, cycle, param)

    def compute_small_hw(self, batch, param, repeat, start_idx):
        """
        compute small shape of height and weight

        Parameters
        ----------
        batch: the number of picture
        param: a dict,the keys as fllow:
                ub_a: a tensor,store middle compute data, store coords data
                ub_b: a tensor,store middle compute data for calculate (2*[pw,ph])^alpha*bias/[netw,neth]
                ub_c: a tensor,store middle compute data for calculate (2*[pw,ph])^alpha*bias/[netw,neth]
                last_32b: a tensor,store last_32b data
                co_id: a scalar,store co_id
                box_id: a scalar,store box_id
                img_ub: a tensor,store img data
                x_vmuls_val: a scalar,store x_vmuls_val
                y_vmuls_val: a scalar,store y_vmuls_val
                ub_bias:  a tensor,store bias data

        repeat: vector repeat times
        start_idx: a scalar,store start_idx

        Returns
        -------
        None
        """
        tmp_scalar = self.instance.Scalar(self.dtype)
        bias_value = self.instance.Scalar(self.dtype)
        # comput height
        with self.instance.if_scope(param['co_id'] == Constant.VALUE_TWO):
            bias_value.set_as(
                param['ub_bias'][Constant.VALUE_TWO * param['box_id'] + Constant.VALUE_ONE])
            tmp_scalar.set_as(param['img_ub'][0])

            self.instance.vec_muls(self.mask, param['ub_c'], param['ub_a'][start_idx],
                                   2, repeat, Constant.STRIDE_EIGHT, Constant.STRIDE_EIGHT) # 2 * sigmoid

            self.instance.vec_ln(self.mask, param['ub_b'], param['ub_c'],
                                 repeat, Constant.STRIDE_EIGHT, Constant.STRIDE_EIGHT) # ln(h)
            self.instance.vec_muls(self.mask, param['ub_b'], param['ub_b'], self.alpha,
                                   repeat, Constant.STRIDE_EIGHT, Constant.STRIDE_EIGHT) # alpha*ln(h)
            self.instance.vec_exp(self.mask, param['ub_b'], param['ub_b'],
                                  repeat, Constant.STRIDE_EIGHT, Constant.STRIDE_EIGHT) # e^(alpha(ln(h))

            self.instance.vec_muls(self.mask, param['ub_c'], param['ub_b'],
                                   bias_value, repeat, Constant.STRIDE_EIGHT, Constant.STRIDE_EIGHT)

            self.instance.vec_dup(self.mask, param['ub_b'], tmp_scalar, repeat,
                                  Constant.STRIDE_EIGHT)

            self.newton_div(param['ub_b'], param['ub_c'], param['ub_b'],
                            repeat)
            self.instance.vec_muls(self.mask, param['ub_b'], param['ub_b'],
                                   param['y_vmuls_val'], repeat,
                                   Constant.STRIDE_EIGHT, Constant.STRIDE_EIGHT)

            if not self.relative:
                tmp_scalar.set_as(param['img_ub'][2])
                self.instance.vec_muls(self.mask, param['ub_b'], param['ub_b'],
                                       tmp_scalar, repeat,
                                       Constant.STRIDE_EIGHT, Constant.STRIDE_EIGHT)

            self.t_small_mov_to_gm(batch, param)
        # comput weight
        with self.instance.if_scope(param['co_id'] == Constant.VALUE_THREE):
            bias_value.set_as(param['ub_bias'][Constant.VALUE_TWO * param['box_id']])
            tmp_scalar.set_as(param['img_ub'][1])

            self.instance.vec_muls(self.mask, param['ub_c'], param['ub_a'][start_idx],
                                   2, repeat, Constant.STRIDE_EIGHT, Constant.STRIDE_EIGHT) # 2 * sigmoid

            self.instance.vec_ln(self.mask, param['ub_b'], param['ub_c'],
                                 repeat, Constant.STRIDE_EIGHT, Constant.STRIDE_EIGHT) # ln(w)
            self.instance.vec_muls(self.mask, param['ub_b'], param['ub_b'], self.alpha,
                                   repeat, Constant.STRIDE_EIGHT, Constant.STRIDE_EIGHT) # alpha*ln(w)
            self.instance.vec_exp(self.mask, param['ub_b'], param['ub_b'],
                                  repeat, Constant.STRIDE_EIGHT, Constant.STRIDE_EIGHT) # e^(alpha(ln(w))

            self.instance.vec_muls(self.mask, param['ub_c'], param['ub_b'],
                                   bias_value, repeat, Constant.STRIDE_EIGHT, Constant.STRIDE_EIGHT)

            self.instance.vec_dup(self.mask, param['ub_b'], tmp_scalar, repeat,
                                  Constant.STRIDE_EIGHT)

            self.newton_div(param['ub_b'], param['ub_c'], param['ub_b'],
                            repeat)
            self.instance.vec_muls(self.mask, param['ub_b'], param['ub_b'],
                                   param['x_vmuls_val'], repeat,
                                   Constant.STRIDE_EIGHT, Constant.STRIDE_EIGHT)

            if not self.relative:
                tmp_scalar.set_as(param['img_ub'][3])
                self.instance.vec_muls(self.mask, param['ub_b'], param['ub_b'],
                                       tmp_scalar, repeat,
                                       Constant.STRIDE_EIGHT, Constant.STRIDE_EIGHT)

            self.t_small_mov_to_gm(batch, param)

    def compute_small_xy(self, batch, param, repeat, start_idx):
        """
        compute small shape of x,y

        Parameters
        ----------
        batch: the number of picture
        param: a dict,the keys as fllow:
                ub_a: a tensor,store middle compute data, store coords data
                ub_b: a tensor,store middle compute data for calculate
                      ([px,py]*2-0.5+[windex,hindex])/[feature_w,feature_h] and relative coords data
                ub_c: a tensor,store middle compute data for calculate
                      ([px,py]*2-0.5+[windex,hindex])/[feature_w,feature_h] and relative coords data
                last_32b: a tensor,store last_32b data
                co_id: a scalar,store co_id
                box_id: a scalar,store box_id
                img_ub: a tensor,store img data
                x_vmuls_val: a scalar,store x_vmuls_val
                y_vmuls_val: a scalar,store y_vmuls_val
                ub_bias:  a tensor,store bias data

        repeat: vector repeat times
        start_idx: a scalar,store start_idx

        Returns
        -------
        None
        """
        tmp_scalar = self.instance.Scalar(self.dtype)
        # comput x
        with self.instance.if_scope(param['co_id'] == Constant.VALUE_ZERO):
            self.instance.data_move(param['ub_b'], param['windex'], Constant.SID,
                                    Constant.NBURST_ONE, param['burlen'],
                                    Constant.GAP_ZERO, Constant.GAP_ZERO)

            self.instance.vec_muls(self.mask, param['ub_c'], param['ub_a'][start_idx],
                                   2.0, repeat,
                                   Constant.STRIDE_EIGHT, Constant.STRIDE_EIGHT)

            self.instance.vec_adds(self.mask, param['ub_c'], param['ub_c'],
                                   -0.5, repeat,
                                   Constant.STRIDE_EIGHT, Constant.STRIDE_EIGHT)

            self.instance.vec_add(self.mask, param['ub_b'], param['ub_c'],
                                  param['ub_b'], repeat,
                                  Constant.STRIDE_EIGHT, Constant.STRIDE_EIGHT, Constant.STRIDE_EIGHT)

            self.instance.vec_muls(self.mask, param['ub_b'], param['ub_b'],
                                   (1.0 / param['w']), repeat,
                                   Constant.STRIDE_EIGHT, Constant.STRIDE_EIGHT)

            self.instance.vec_muls(self.mask, param['ub_b'], param['ub_b'],
                                   param['x_vmuls_val'], repeat,
                                   Constant.STRIDE_EIGHT, Constant.STRIDE_EIGHT)

            self.instance.vec_adds(self.mask, param['ub_b'], param['ub_b'],
                                   param['x_vadds_val'], repeat,
                                   Constant.STRIDE_EIGHT, Constant.STRIDE_EIGHT)

            if not self.relative:
                tmp_scalar.set_as(param['img_ub'][3])
                self.instance.vec_muls(self.mask, param['ub_b'], param['ub_b'],
                                       tmp_scalar, repeat,
                                       Constant.STRIDE_EIGHT, Constant.STRIDE_EIGHT)

            self.t_small_mov_to_gm(batch, param)
        # comput y
        with self.instance.if_scope(param['co_id'] == Constant.VALUE_ONE):
            self.instance.data_move(param['ub_b'], param['hindex'], Constant.SID,
                                    Constant.NBURST_ONE,
                                    param['burlen'], Constant.GAP_ZERO, Constant.GAP_ZERO)

            self.instance.vec_muls(self.mask, param['ub_c'], param['ub_a'][start_idx],
                                   2.0, repeat,
                                   Constant.STRIDE_EIGHT, Constant.STRIDE_EIGHT)

            self.instance.vec_adds(self.mask, param['ub_c'], param['ub_c'],
                                   -0.5, repeat,
                                   Constant.STRIDE_EIGHT, Constant.STRIDE_EIGHT)

            self.instance.vec_add(self.mask, param['ub_b'], param['ub_c'],
                                  param['ub_b'], repeat,
                                  Constant.STRIDE_EIGHT, Constant.STRIDE_EIGHT, Constant.STRIDE_EIGHT)

            self.instance.vec_muls(self.mask, param['ub_b'], param['ub_b'],
                                   (1.0 / param['h']), repeat,
                                   Constant.STRIDE_EIGHT, Constant.STRIDE_EIGHT)

            self.instance.vec_muls(self.mask, param['ub_b'], param['ub_b'],
                                   param['y_vmuls_val'], repeat,
                                   Constant.STRIDE_EIGHT, Constant.STRIDE_EIGHT)

            self.instance.vec_adds(self.mask, param['ub_b'], param['ub_b'],
                                   param['y_vadds_val'], repeat,
                                   Constant.STRIDE_EIGHT, Constant.STRIDE_EIGHT)
            if not self.relative:
                tmp_scalar.set_as(param['img_ub'][2])
                self.instance.vec_muls(self.mask, param['ub_b'], param['ub_b'],
                                       tmp_scalar, repeat,
                                       Constant.STRIDE_EIGHT, Constant.STRIDE_EIGHT)

            self.t_small_mov_to_gm(batch, param)