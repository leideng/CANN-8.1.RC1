#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright 2019 Huawei Technologies Co., Ltd
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
clip_boxes_d
"""
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import error_manager_vector
from tbe.tvm.topi.cce import util
from impl.util import util_select_op_base


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    the class for constant
    """
    SHAPE_SIZE_LIMIT = 65500
    CONFIG_ONE = 1
    NEG_ONE = -1
    CONFIG_TWO = 2
    CONFIG_FOUR = 4
    CONFIG_EIGHT = 8
    CONFIG_SIXTEEN = 16
    CONFIG_DATA_ALIGN = 32
    CONFIG_DATA_TRANS = 64
    CONFIG_MASK = 128
    CONFIG_UB_LIMITED = 4096
    IF_USE_V200 = ("Ascend610", "BS9SX1A", "Ascend310P")


# 'pylint: disable=unused-argument,super-with-arguments
def get_op_support_info(boxes_input,
                        boxes_output,
                        img_size,
                        kernel_name="clip_boxes"):
    """
    :param boxes_input: (N,4)
    :param boxes_output:(N,4)
    :param img_size: H,W
    :param kernel_name: clip_boxes
    :return:
    """

    format_boxes_input = boxes_input.get("format")
    dims_boxes_input = boxes_input.get("shape")
    if format_boxes_input == "ND" \
            and len(dims_boxes_input) == Constant.CONFIG_TWO \
            and dims_boxes_input[Constant.CONFIG_ONE] == Constant.CONFIG_FOUR:
        axis_split_matrix = []
        for i in dims_boxes_input:
            split_0 = [util_select_op_base.SplitInput(
                [0, [i], [Constant.NEG_ONE], [Constant.NEG_ONE]]),
                util_select_op_base.SplitOutput([0, [i]])]
            axis_split_matrix.append(split_0)
        axis_reduce_list = None
    else:
        axis_split_matrix = None
        axis_reduce_list = None
    op_cal_info_in_json = util_select_op_base.get_op_cal_info(
        axis_split_matrix, axis_reduce_list, 0, 0)
    return op_cal_info_in_json


class InitConst:
    """
    define some const numbers
    these const numbers are for vector operators
    """

    def __init__(self):
        # const number for vector operators
        self.dstorsrc_blk_stride1 = Constant.CONFIG_ONE
        self.dstorsrc_rep_stride1 = Constant.CONFIG_EIGHT
        self.dstorsrc_blk_stride2 = Constant.CONFIG_TWO
        self.dstorsrc_rep_stride2 = Constant.CONFIG_SIXTEEN
        self.mask = Constant.CONFIG_MASK

    def set_dstorsrc_blk_stride1(self, dstorsrc_blk_stride):
        """
        set dstorsrc_blk_stride1
        return: None
        """
        self.dstorsrc_blk_stride1 = dstorsrc_blk_stride

    def set_dstorsrc_rep_stride1(self, dstorsrc_rep_stride):
        """
        set detorsrc_rep_stride1
        return: None
        """
        self.dstorsrc_rep_stride1 = dstorsrc_rep_stride


class ConstList(InitConst):
    """
    define the const numbers
    these const numbers are related to the size of memory
    """

    def __init__(self):
        super(ConstList, self).__init__()
        # const number for float16, each block contains 32B
        self.num_one_blk = Constant.CONFIG_SIXTEEN
        # const number for vector op, 8 blk each
        self.num_one_vecop = Constant.CONFIG_EIGHT
        # const number for trans op
        self.num_one_trans = Constant.CONFIG_DATA_TRANS
        # const number for ND, D=4
        self.num_d = Constant.CONFIG_FOUR

    def set_num_one_blk(self, num_one_blk):
        """
        set  num_one_blk
        return: None
        """
        self.num_one_blk = num_one_blk

    def set_num_one_trans(self, num_one_trans):
        """
        set num_one_trans
        return: None
        """
        self.num_one_trans = num_one_trans


def ceil_div(num_a, num_bulk):
    """
    calculate number of bulk needed
    num_a: the num of  input boxes
    num_bulk : the num of elements each bulk
    return  the num of bulk at least needed
    """

    return (num_a + num_bulk - Constant.CONFIG_ONE) // num_bulk


class TilingFunc:
    """
    planning the method for data tiling
      tot_of_blk: total num of block
      num_of_blk: num of block for each loop
      num_of_trans: num of transpose is needed, 16*16 for each
      loop_time: loop time
      thread_num: whether pingpang buffer is needed
    """

    def __init__(self, shape):
        #  num_of_boxes <= 4096  not  double buffer, no multi_core
        num_of_boxes = shape[0]
        if num_of_boxes <= Constant.CONFIG_UB_LIMITED:
            self.thread_num = Constant.CONFIG_ONE
            self.loop_time = Constant.CONFIG_ONE
            # first should be times of 4, thus the data is times of 16, and can be moved to UB
            #  num of  block    for data move
            self.tot_of_blk = ceil_div(num_of_boxes, Constant.CONFIG_FOUR)
            self.num_of_blk = self.tot_of_blk
            # num of 16*block   for data transpose
            self.num_of_trans = ceil_div(self.num_of_blk, Constant.CONFIG_SIXTEEN)
        else:
            # the suggested num_of_block moved to UB once
            num_half_buf = Constant.CONFIG_UB_LIMITED // Constant.CONFIG_TWO
            #  Use pingpang Buffer
            self.thread_num = Constant.CONFIG_TWO
            #  tot num of blocks
            self.tot_of_blk = ceil_div(num_of_boxes, Constant.CONFIG_FOUR)
            #  the num of kernels  needed
            loop_time = ceil_div(self.tot_of_blk * Constant.CONFIG_FOUR,
                                 num_half_buf)
            #  num of boxes  each time
            num_half_buf = ceil_div(self.tot_of_blk * Constant.CONFIG_FOUR,
                                    loop_time)
            #  each time the memory size should be times of 256=32*4*2
            self.num_of_blk = ceil_div(num_half_buf, Constant.CONFIG_DATA_ALIGN) * Constant.CONFIG_EIGHT
            self.loop_time = ceil_div(self.tot_of_blk,
                                      self.num_of_blk)

            # num of 16*block   for data transpose    (each loop)
            self.num_of_trans = ceil_div(self.num_of_blk,
                                         Constant.CONFIG_SIXTEEN)

    def set_thread_num(self, thread_num):
        """
        set thread_num
        return: None
        """
        self.thread_num = thread_num

    def set_num_of_blk(self, num_of_blk):
        """
        set num_of_blk
        return: None
        """
        self.num_of_blk = num_of_blk


class InitMiddleTensor:
    """
    init the middle tensors
    these tensors are located in UB
    """

    def __init__(self, tik_instance, const_num, num_of_trans):
        # the size of image, construct a vector for computing
        if tbe_platform.get_soc_spec("SHORT_SOC_VERSION") not in Constant.IF_USE_V200:
            self.width_ub = tik_instance.Tensor("float16",
                                                (const_num.num_one_vecop*const_num.num_one_blk,
                                                 Constant.CONFIG_ONE),
                                                name="width_ub",
                                                scope=tik.scope_ubuf)
            self.height_ub = tik_instance.Tensor("float16",
                                                 (const_num.num_one_vecop*const_num.num_one_blk,
                                                  Constant.CONFIG_ONE),
                                                 name="height_ub",
                                                 scope=tik.scope_ubuf)
        self.anchors_ub = tik_instance.Tensor("float16",
                                              (num_of_trans*const_num.num_one_trans,
                                               const_num.num_d),
                                              name="anchors_ub",
                                              scope=tik.scope_ubuf)
        self.boxes_ub = tik_instance.Tensor("float16",
                                            (const_num.num_d,
                                             num_of_trans*const_num.num_one_trans),
                                            name="boxes_ub",
                                            scope=tik.scope_ubuf)
        self.res_temp1_ub = tik_instance.Tensor("float16",
                                                (const_num.num_d,
                                                 num_of_trans*const_num.num_one_trans),
                                                name="res_temp1_ub",
                                                scope=tik.scope_ubuf)
        self.res_temp2_ub = tik_instance.Tensor("float16",
                                                (const_num.num_d,
                                                 num_of_trans*const_num.num_one_trans),
                                                name="res_temp2_ub",
                                                scope=tik.scope_ubuf)
        self.res_ub = tik_instance.Tensor("float16",
                                          (num_of_trans*const_num.num_one_trans,
                                           const_num.num_d),
                                          name="res_ub",
                                          scope=tik.scope_ubuf)

    def set_imgh_vec(self, tik_instance, img_h, const_num):
        """
        set the image height vector
        return: None
        """
        if tbe_platform.get_soc_spec("SHORT_SOC_VERSION") not in Constant.IF_USE_V200:
            tik_instance.vector_dup(const_num.mask,
                                    self.height_ub[0],
                                    float(img_h),
                                    const_num.dstorsrc_blk_stride1,
                                    const_num.dstorsrc_blk_stride1,
                                    const_num.dstorsrc_rep_stride1)

    def set_imgw_vec(self, tik_instance, img_w, const_num):
        """
        set the image width vector
        return: None
        """
        if tbe_platform.get_soc_spec("SHORT_SOC_VERSION") not in Constant.IF_USE_V200:
            tik_instance.vector_dup(const_num.mask,
                                    self.width_ub[0],
                                    float(img_w),
                                    const_num.dstorsrc_blk_stride1,
                                    const_num.dstorsrc_blk_stride1,
                                    const_num.dstorsrc_rep_stride1)


# 'pylint: disable=not-use-list-comprehension,too-many-locals
def processing_one_loop(tik_instance, data_gm, tiling_para, img_size, offset):
    """
    Using Pingpang, this func is one loop processing
    param tik_instance: tik container
    param data_gm: in and out data tensors in DDR
    param tiling_para: tiling
    param img_size: (img_h. img_w)
    param offset: loop id
    return: None
    """

    const_num = ConstList()
    anchors = data_gm[0]
    res_anchors = data_gm[1]
    img_h, img_w = img_size

    data_tensor = InitMiddleTensor(tik_instance, const_num,
                                   tiling_para.num_of_trans)

    # move data from DDR to UB
    tik_instance.data_move(data_tensor.anchors_ub[0],
                           anchors[tiling_para.num_of_blk*const_num.num_one_blk*offset],
                           const_num.dstorsrc_blk_stride1,
                           const_num.dstorsrc_blk_stride1,
                           tiling_para.num_of_blk,
                           0, 0)

    #  do the transpose for the input 16*16
    dst_list = [data_tensor.boxes_ub[const_num.num_one_blk * i]
                for i in range(const_num.num_one_blk)]
    src_list = [data_tensor.anchors_ub[const_num.num_one_blk * i]
                for i in range(const_num.num_one_blk)]

    if tiling_para.num_of_trans == Constant.CONFIG_ONE:
        tik_instance.vnchwconv(True, False, dst_list, src_list,
                               tiling_para.num_of_trans, 0, 0)
    else:
        tik_instance.vnchwconv(True, False, dst_list, src_list,
                               tiling_para.num_of_trans,
                               const_num.dstorsrc_rep_stride2,
                               const_num.dstorsrc_rep_stride2)

    # do relu, comparing with 0
    tik_instance.vrelu(const_num.mask,
                       data_tensor.res_temp1_ub[0],
                       data_tensor.boxes_ub[0],
                       tiling_para.num_of_trans * Constant.CONFIG_TWO,
                       const_num.dstorsrc_blk_stride1,
                       const_num.dstorsrc_blk_stride1,
                       const_num.dstorsrc_rep_stride1,
                       const_num.dstorsrc_rep_stride1)

    # do the comparing
    if tbe_platform.get_soc_spec("SHORT_SOC_VERSION") in Constant.IF_USE_V200:
        tik_instance.vmins(const_num.mask,
                           data_tensor.res_temp2_ub[0],
                           data_tensor.res_temp1_ub[0],
                           img_w,
                           tiling_para.num_of_trans,
                           const_num.dstorsrc_blk_stride2,
                           const_num.dstorsrc_blk_stride2,
                           const_num.dstorsrc_rep_stride2,
                           const_num.dstorsrc_rep_stride2)
        tik_instance.vmins(const_num.mask,
                           data_tensor.res_temp2_ub[Constant.CONFIG_SIXTEEN],
                           data_tensor.res_temp1_ub[Constant.CONFIG_SIXTEEN],
                           img_h,
                           tiling_para.num_of_trans,
                           const_num.dstorsrc_blk_stride2,
                           const_num.dstorsrc_blk_stride2,
                           const_num.dstorsrc_rep_stride2,
                           const_num.dstorsrc_rep_stride2)
    else:
        # init the vector img_h, img_w
        data_tensor.set_imgh_vec(tik_instance, img_h, const_num)
        data_tensor.set_imgw_vec(tik_instance, img_w, const_num)
        tik_instance.vmin(const_num.mask,
                          data_tensor.res_temp2_ub[0],
                          data_tensor.res_temp1_ub[0],
                          data_tensor.width_ub[0],
                          tiling_para.num_of_trans,
                          const_num.dstorsrc_blk_stride2,
                          const_num.dstorsrc_blk_stride2,
                          const_num.dstorsrc_blk_stride1,
                          const_num.dstorsrc_rep_stride2,
                          const_num.dstorsrc_rep_stride2,
                          0)
        tik_instance.vmin(const_num.mask,
                          data_tensor.res_temp2_ub[Constant.CONFIG_SIXTEEN],
                          data_tensor.res_temp1_ub[Constant.CONFIG_SIXTEEN],
                          data_tensor.height_ub[0],
                          tiling_para.num_of_trans,
                          const_num.dstorsrc_blk_stride2,
                          const_num.dstorsrc_blk_stride2,
                          const_num.dstorsrc_blk_stride1,
                          const_num.dstorsrc_rep_stride2,
                          const_num.dstorsrc_rep_stride2,
                          0)

    # Data  transpose
    dst_list = [data_tensor.res_ub[const_num.num_one_blk * i]
                for i in range(const_num.num_one_blk)]
    src_list = [data_tensor.res_temp2_ub[const_num.num_one_blk * i]
                for i in range(const_num.num_one_blk)]
    if tiling_para.num_of_trans == Constant.CONFIG_ONE:
        tik_instance.vnchwconv(True, False, dst_list, src_list,
                               tiling_para.num_of_trans, 0, 0)
    else:
        tik_instance.vnchwconv(True, False, dst_list, src_list,
                               tiling_para.num_of_trans,
                               const_num.dstorsrc_rep_stride2,
                               const_num.dstorsrc_rep_stride2)

    #  move data from UB  to DDR
    tik_instance.data_move(res_anchors[tiling_para.num_of_blk*const_num.num_one_blk*offset],
                           data_tensor.res_ub[0],
                           const_num.dstorsrc_blk_stride1,
                           const_num.dstorsrc_blk_stride1,
                           tiling_para.num_of_blk,
                           0, 0)


# 'pylint: disable=not-use-list-comprehension,too-many-locals
def processing_tail(tik_instance, data_gm, tiling_para, img_size):
    """

    :param tik_instance:
    :param data_gm:
    :param tiling_para:
    :param img_size:
    :return:
    """

    const_num = ConstList()
    anchors = data_gm[0]
    res_anchors = data_gm[1]
    img_h, img_w = img_size

    data_tensor = InitMiddleTensor(tik_instance, const_num,
                                   tiling_para.num_of_trans)

    # move data from DDR to UB
    tik_instance.data_move(data_tensor.anchors_ub[0],
                           anchors[tiling_para.num_of_blk * const_num.num_one_blk *
                                   (tiling_para.loop_time - Constant.CONFIG_ONE)],
                           const_num.dstorsrc_blk_stride1,
                           const_num.dstorsrc_blk_stride1,
                           tiling_para.tot_of_blk -
                           tiling_para.num_of_blk * (tiling_para.loop_time - Constant.CONFIG_ONE),
                           0, 0)

    #  do the transpose for the input 16*16
    dst_list = [data_tensor.boxes_ub[const_num.num_one_blk * i]
                for i in range(const_num.num_one_blk)]
    src_list = [data_tensor.anchors_ub[const_num.num_one_blk * i]
                for i in range(const_num.num_one_blk)]

    if tiling_para.num_of_trans == Constant.CONFIG_ONE:
        tik_instance.vnchwconv(True, False, dst_list, src_list,
                               tiling_para.num_of_trans, 0, 0)
    else:
        tik_instance.vnchwconv(True, False, dst_list, src_list,
                               tiling_para.num_of_trans,
                               const_num.dstorsrc_rep_stride2,
                               const_num.dstorsrc_rep_stride2)

    # do relu, comparing with 0
    tik_instance.vrelu(const_num.mask,
                       data_tensor.res_temp1_ub[0],
                       data_tensor.boxes_ub[0],
                       tiling_para.num_of_trans * Constant.CONFIG_TWO,
                       const_num.dstorsrc_blk_stride1,
                       const_num.dstorsrc_blk_stride1,
                       const_num.dstorsrc_rep_stride1,
                       const_num.dstorsrc_rep_stride1)

    # do the comparing
    if tbe_platform.get_soc_spec("SHORT_SOC_VERSION") in Constant.IF_USE_V200:
        tik_instance.vmins(const_num.mask,
                           data_tensor.res_temp2_ub[0],
                           data_tensor.res_temp1_ub[0],
                           img_w,
                           tiling_para.num_of_trans,
                           const_num.dstorsrc_blk_stride2,
                           const_num.dstorsrc_blk_stride2,
                           const_num.dstorsrc_rep_stride2,
                           const_num.dstorsrc_rep_stride2)
        tik_instance.vmins(const_num.mask,
                           data_tensor.res_temp2_ub[Constant.CONFIG_SIXTEEN],
                           data_tensor.res_temp1_ub[Constant.CONFIG_SIXTEEN],
                           img_h,
                           tiling_para.num_of_trans,
                           const_num.dstorsrc_blk_stride2,
                           const_num.dstorsrc_blk_stride2,
                           const_num.dstorsrc_rep_stride2,
                           const_num.dstorsrc_rep_stride2)
    else:
        # init the vector height and width
        data_tensor.set_imgh_vec(tik_instance, img_h, const_num)
        data_tensor.set_imgw_vec(tik_instance, img_w, const_num)

        tik_instance.vmin(const_num.mask,
                          data_tensor.res_temp2_ub[0],
                          data_tensor.res_temp1_ub[0],
                          data_tensor.width_ub[0],
                          tiling_para.num_of_trans,
                          const_num.dstorsrc_blk_stride2,
                          const_num.dstorsrc_blk_stride2,
                          const_num.dstorsrc_blk_stride1,
                          const_num.dstorsrc_rep_stride2,
                          const_num.dstorsrc_rep_stride2,
                          0)
        tik_instance.vmin(const_num.mask,
                          data_tensor.res_temp2_ub[Constant.CONFIG_SIXTEEN],
                          data_tensor.res_temp1_ub[Constant.CONFIG_SIXTEEN],
                          data_tensor.height_ub[0],
                          tiling_para.num_of_trans,
                          const_num.dstorsrc_blk_stride2,
                          const_num.dstorsrc_blk_stride2,
                          const_num.dstorsrc_blk_stride1,
                          const_num.dstorsrc_rep_stride2,
                          const_num.dstorsrc_rep_stride2,
                          0)

    # Data  transpose
    dst_list = [data_tensor.res_ub[const_num.num_one_blk * i]
                for i in range(const_num.num_one_blk)]
    src_list = [data_tensor.res_temp2_ub[const_num.num_one_blk * i]
                for i in range(const_num.num_one_blk)]
    if tiling_para.num_of_trans == Constant.CONFIG_ONE:
        tik_instance.vnchwconv(True, False, dst_list, src_list,
                               tiling_para.num_of_trans, 0, 0)
    else:
        tik_instance.vnchwconv(True, False, dst_list, src_list,
                               tiling_para.num_of_trans,
                               const_num.dstorsrc_rep_stride2,
                               const_num.dstorsrc_rep_stride2)

    #  move data from UB  to DDR
    tik_instance.data_move(res_anchors[tiling_para.num_of_blk * const_num.num_one_blk *
                                       (tiling_para.loop_time - Constant.CONFIG_ONE)],
                           data_tensor.res_ub[0],
                           const_num.dstorsrc_blk_stride1,
                           const_num.dstorsrc_blk_stride1,
                           tiling_para.tot_of_blk -
                           tiling_para.num_of_blk * (tiling_para.loop_time-Constant.CONFIG_ONE),
                           0, 0)


def clip_boxes_d_compute(boxes_input, img_w, img_h, kernel_name="clip_boxes"):
    """
    the compute process of clip_boxes
    input:
     boxes_input:a dict, include shape, and dtype
     img_w: width of the image
     img_h: height of the image
     kernel_name: the kernel name
    return:
     the tik container
    """

    const_num = ConstList()
    tiling_para = TilingFunc(boxes_input.get("shape"))

    #  start the TIK container
    tik_instance = tik.Tik(tik.Dprofile(), True)

    anchors = tik_instance.Tensor("float16",
                                  (tiling_para.tot_of_blk*const_num.num_d,
                                   const_num.num_d),
                                  name="anchors",
                                  scope=tik.scope_gm)
    res_anchors = tik_instance.Tensor("float16",
                                      (tiling_para.tot_of_blk*const_num.num_d,
                                       const_num.num_d),
                                      name="res_anchors",
                                      scope=tik.scope_gm)

    with tik_instance.for_range(0, tiling_para.loop_time - Constant.CONFIG_ONE,
                                thread_num=tiling_para.thread_num) as loop_i:
        processing_one_loop(tik_instance,
                            (anchors, res_anchors),
                            tiling_para,
                            (img_h, img_w),
                            loop_i)

    # the tail processing
    processing_tail(tik_instance,
                    (anchors, res_anchors),
                    tiling_para,
                    (img_h, img_w))

    tik_instance.BuildCCE(kernel_name=kernel_name, inputs=[anchors], outputs=[res_anchors])
    return tik_instance


def check_clip_boxes_input_dict(boxes_input, boxes_output):
    """
    check the input parameters -- tensor
    input:
      boxes_input: an dict, include shape, and dtype of input
      boxes_output: an dict, include shape, and dtype of output
    return: None
    """

    input_shape = boxes_input.get("shape")
    input_dtype = boxes_input.get("dtype")
    output_shape = boxes_output.get("shape")
    output_dtype = boxes_output.get("dtype")

    para_check.check_shape(input_shape)

    # the shape and type of the output  should be the same as the input
    if input_shape != output_shape:
        error_detail = "the shape of output should be the same as the input"
        error_manager_vector.raise_err_two_input_shape_invalid("clip_boxes", "input", "output",
                                                            error_detail)
    # Check the size of the input shape
    if len(input_shape) != Constant.CONFIG_TWO:
        error_manager_vector.raise_err_input_value_invalid("clip_boxes", "dimension of input",
                                                        2, len(input_shape))
    n_x, n_y = input_shape
    if n_x <= 0 or n_x > Constant.SHAPE_SIZE_LIMIT:
        error_manager_vector.raise_err_input_param_not_in_range("clip_boxes", "N dimension of input",
                                                            1, Constant.SHAPE_SIZE_LIMIT, n_x)
    if n_y != Constant.CONFIG_FOUR:
        error_manager_vector.raise_err_input_value_invalid("clip_boxes", "last dimension of input",
                                                        4, n_y)
    para_check.check_dtype(input_dtype, ["float16"], param_name="x")

    if input_dtype != output_dtype:
        error_manager_vector.raise_err_inputs_dtype_not_equal("clip_boxes", "input", "output",
                                                            input_dtype, output_dtype)


def check_clip_boxes_input_attr(img_w, img_h):
    """
    check the input parameters  -- attr
    input:
      img_w: width of the image
      img_h: height of the image
    return: None
    """

    # the size of the image  should  be lager than zero
    if img_h <= 0 or img_w <= 0:
        real_value = "img_h is {}, img_w is {}".format(img_h, img_w)
        error_manager_vector.raise_err_input_value_invalid("clip_boxes", "img_h/img_w",
                                                        "larger than zero", real_value)


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_LIST_INT,
                            para_check.KERNEL_NAME)
def clip_boxes_d(boxes_input, boxes_output, img_size, kernel_name="clip_boxes"):
    """
    the External interface function
    input:
      boxes_input: an dict, include shape, and dtype of input
      boxes_output: an dict, include shape, and dtype of output
      img_w: width of the image
      img_h: height of the image
      kernel_name: the kernel name
    return:
      the tik container
    """

    if len(img_size) != Constant.CONFIG_TWO:
        error_manager_vector.raise_err_input_value_invalid("clip_boxes", "img_size",
                                                        "[img_h, img_w]", len(img_size))

    img_h, img_w = img_size
    check_clip_boxes_input_dict(boxes_input, boxes_output)
    check_clip_boxes_input_attr(img_w, img_h)
    util.check_kernel_name(kernel_name)

    tik_instance = clip_boxes_d_compute(boxes_input, img_w, img_h, kernel_name=kernel_name)
    return tik_instance
