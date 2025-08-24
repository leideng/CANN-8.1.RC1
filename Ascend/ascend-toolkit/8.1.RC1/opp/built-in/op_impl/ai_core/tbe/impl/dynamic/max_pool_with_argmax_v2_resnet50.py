# Copyright 2022 Huawei Technologies Co., Ltd
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
max_pool_with_argmax_v2_resnet50
"""
import math
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl import common_util_v1


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """

    def __init__(self):
        pass

    # min value of fp16
    MIN_VALUE_FP16 = -65504.0

    # parameters for vector instruct
    MASK = 128
    REPEAT_2 = 2
    DSTSTRIDEM0 = 1
    SRC0STRIDEM0 = 1
    SRC1STRIDEM0 = 1
    DSTSTRIDEM1 = 8
    SRC0STRIDEM1 = 8
    SRC1STRIDEM1 = 8
    DT_INT32 = 3
    DT_INT64 = 9
    DIM_C0 = 16

    # Parameters for ResNet50
    KSIZE = 3
    STRIDE = 2
    PAD = 1
    DILATION = 1
    IN_SIZE_H_W = 112
    OUT_SIZE_H_W = 56

# 'pylint: disable=locally-disabled, too-many-instance-attributes
# 'pylint: disable=too-few-public-methods
class MaxPoolWithArgmaxV2Resnet50:
    """
    MaxPoolWithArgmaxV2Resnet50
    """

    # 'pylint: disable=too-many-arguments
    def __init__(self, input_shape, input_dtype, input_gm, max_output_gm, mask_output_gm, tik_inst):
        """
        init MaxPoolWithArgmaxV2Resnet50 parameters

        Parameters
        ----------
        input_shape: input data shape
        input_dtype: input data dtype
        input_gm: input gm tensor for input data
        max_output_gm: output gm tensor for max
        mask_output_gm: output gm tensor for mask
        tik_inst: tik inst for total project
        Returns
        -------
        None
        """
        self.input_shape = [input_shape[0], input_shape[1], Constant.IN_SIZE_H_W, Constant.IN_SIZE_H_W, input_shape[4]]
        self.input_dtype = input_dtype
        self.input_type_size = common_util_v1.get_data_size(self.input_dtype)
        self.ceil_mode = False
        self.tik_inst = tik_inst
        self.ksize_h = Constant.KSIZE
        self.ksize_w = Constant.KSIZE
        self.stride_h = Constant.STRIDE
        self.stride_w = Constant.STRIDE
        self.pad_t = Constant.PAD
        self.pad_l = Constant.PAD
        self.dilation_h = Constant.DILATION
        self.dilation_w = Constant.DILATION
        self.output_h = Constant.OUT_SIZE_H_W
        self.output_w = Constant.OUT_SIZE_H_W
        self.fmap_img2col_w = self.ksize_h * self.ksize_w

        if self.input_dtype == "float16":
            self.pad_value = Constant.MIN_VALUE_FP16

        self.input_fmap_gm = input_gm
        self.output_max_gm = max_output_gm
        self.output_mask_gm = mask_output_gm

        self.check_load3d_supported = tbe_platform.api_check_support("tik.load3dv1")

        self.c1_dim = None
        self.input_h = None
        self.input_w = None
        self.filter_size = None
        self.input_idx = None
        self.output_idx = None
        self.l1_idx = None
        self.mask_idx = None
        self.fm_size = None
        self.output_block_h = None
        self.ub_load0 = None
        self.ub_load1 = None
        self.ub_max_buff = None
        self.ub_mask_buff = None
        self.ub_mask_temp = None
        self.ub_mask_or_buff = None
        self.ub_mask_not_buff = None
        self.buf_0 = None
        self.buf_1 = None
        self.loop_h = None
        self.mask_one_window = None
        self.mask_gap_element = None
        self.mask_gap = None
        self.repeat_stride = None
        self.repeat_0 = None
        self.repeat_1 = None
        self.l1_buff0 = None

    def _load3d_fm_to_ub(self, ub_buff, l1_buff, i_dim_w, i_dim_h):
        """
        :param ub_buff: ub buffer
        :param l1_buff: l1 buffer
        :param i_dim_w: i_dim_w
        :param i_dim_h: i_dim_h
        :return:
        """
        instance = self.tik_inst
        filter_size = self.ksize_h * self.ksize_w
        with instance.for_range(0, filter_size, thread_num=1) as loopk:
            k_dim_w = loopk % 3
            k_dim_h = loopk // 3
            instance.load3dv1(ub_buff[loopk * 4 * 56 * 16], l1_buff,
                              [1, self.pad_l, self.pad_t, 1],
                              self.input_h, self.input_w,
                              0, k_dim_w, k_dim_h, i_dim_w, i_dim_h,
                              self.stride_w, self.stride_h,
                              self.ksize_w, self.ksize_h,
                              self.dilation_w, self.dilation_h, 1, 1, 4 * 56 // 16, 0, self.pad_value)

    def _clear_ub_to_pad_value_fp16(self, ub_buff, length, pad_value):
        """
        :param ub_buff: ub buffer
        :param length: data length
        :param pad_value: pad value
        :return:
        """
        max_repeat_time = 255
        vec_fp16_mask = 128
        single_vec_fp16_mask = 16
        max_repeat_length = max_repeat_time * vec_fp16_mask
        vec_repeat_time = length // vec_fp16_mask
        res_vec_length = length - vec_repeat_time * vec_fp16_mask
        max_vec_repeat_block = vec_repeat_time // max_repeat_time
        res_max_vec_repeat_block = vec_repeat_time - max_vec_repeat_block * max_repeat_time

        if max_vec_repeat_block > 0:
            with self.tik_inst.for_range(0, max_vec_repeat_block) as loop:
                self.tik_inst.vector_dup(vec_fp16_mask, ub_buff[loop * max_repeat_length], pad_value,
                                         max_repeat_time, 1, 8)

        if res_max_vec_repeat_block > 0:
            self.tik_inst.vector_dup(vec_fp16_mask, ub_buff[max_vec_repeat_block * max_repeat_length],
                                     pad_value, res_max_vec_repeat_block, 1, 8)

        if res_vec_length > 0:
            self.tik_inst.vector_dup(single_vec_fp16_mask, ub_buff[vec_repeat_time * vec_fp16_mask],
                                     pad_value, res_vec_length // single_vec_fp16_mask, 1, 1)

    # 'pylint: disable=too-many-arguments,too-many-locals
    def _load_gm_to_ub_ping(self, ub_buff, output_block_h, input_fmap_gm, input_gm_idx, looph):
        """
        load data from gm to ub

        Parameters
        ----------
        ub_buff: address of ub_buff
        output_block_h: size of cut
        input_fmap_gm: address of gm
        input_gm_idx: offset of gm
        looph: index of looph

        Returns
        -------
        None
        """
        instance = self.tik_inst
        gm_len = instance.Scalar("uint64", name="gm_len")
        cur_h = instance.Scalar("int64", name="cur_h")
        start_ub_pos = instance.Scalar("int64", name="start_ub_pos")
        cur_w = instance.Scalar("int64", name="cur_w")
        filter_size = self.ksize_h * self.ksize_w
        c0_dim = 16
        self._clear_ub_to_pad_value_fp16(ub_buff, output_block_h * self.output_w * c0_dim * filter_size,
                                         self.pad_value)

        with instance.for_range(0, filter_size) as filter_index:
            w_index = filter_index % self.ksize_w
            h_index = filter_index // self.ksize_w
            with instance.if_scope(w_index == 0):
                start_ub_pos.set_as(1)
                cur_w.set_as(self.ksize_w // 2)
                gm_len.set_as(self.output_w - 1)
            with instance.else_scope():
                start_ub_pos.set_as(0)
                cur_w.set_as(w_index - self.ksize_w // 2)
                gm_len.set_as(self.output_w)

            with instance.for_range(0, output_block_h) as output_block_h_index:
                cur_h.set_as((looph * 2 * output_block_h + output_block_h_index) * \
                             self.stride_h - self.pad_t + h_index)
                with self.tik_inst.if_scope(cur_h >= 0):
                    instance.data_move(ub_buff[(filter_index * output_block_h + output_block_h_index) * \
                                       self.output_w * c0_dim + start_ub_pos * c0_dim],
                                       input_fmap_gm[input_gm_idx + cur_h * self.input_w * c0_dim + cur_w * c0_dim],
                                       0, gm_len, 1, 1, 0)

    # 'pylint: disable=too-many-locals
    def _load_gm_to_ub_pong(self, ub_buff, output_block_h, input_fmap_gm, input_gm_idx, looph):
        """
        load data from gm to ub

        Parameters
        ----------
        ub_buff: address of ub_buff
        output_block_h: size of cut
        input_fmap_gm: address of gm
        input_gm_idx: offset of gm
        looph: index of looph

        Returns
        -------
        None
        """
        instance = self.tik_inst
        gm_len = instance.Scalar("uint64", name="gm_len")
        cur_h = instance.Scalar("int64", name="cur_h")
        start_ub_pos = instance.Scalar("int64", name="start_ub_pos")
        cur_w = instance.Scalar("int64", name="cur_w")
        filter_size = self.ksize_h * self.ksize_w
        c0_dim = 16
        self._clear_ub_to_pad_value_fp16(ub_buff, output_block_h * self.output_w * c0_dim * filter_size,
                                         self.pad_value)
        with instance.for_range(0, filter_size) as filter_index:
            w_index = filter_index % self.ksize_w
            h_index = filter_index // self.ksize_w
            with instance.if_scope(w_index == 0):
                start_ub_pos.set_as(1)
                cur_w.set_as(self.ksize_w // 2)
                gm_len.set_as(self.output_w - 1)
            with instance.else_scope():
                start_ub_pos.set_as(0)
                cur_w.set_as(w_index - self.ksize_w // 2)
                gm_len.set_as(self.output_w)

            with instance.for_range(0, output_block_h) as output_block_h_index:
                cur_h.set_as(((looph * 2 + 1) * output_block_h + output_block_h_index) * \
                             self.stride_h - self.pad_t + h_index)
                instance.data_move(ub_buff[(filter_index * output_block_h + output_block_h_index) * \
                                   self.output_w * c0_dim + start_ub_pos * c0_dim],
                                   input_fmap_gm[input_gm_idx + cur_h * self.input_w * c0_dim + cur_w * c0_dim],
                                   0, gm_len, 1, 1, 0)

    def _variable_init(self):
        """
        variable init
        :return:
        """
        dtype = self.input_dtype

        self.c1_dim = self.tik_inst.Scalar("uint64", name="c1_dim")
        self.c1_dim.set_as(self.input_shape[1])
        self.input_h = self.tik_inst.Scalar("uint64", name="input_h")
        self.input_h.set_as(self.input_shape[2])
        self.input_w = self.tik_inst.Scalar("uint64", name="input_w")
        self.input_w.set_as(self.input_shape[3])
        self.filter_size = self.tik_inst.Scalar("uint64", name="filter_size")
        self.filter_size.set_as(self.ksize_h * self.ksize_w)

        self.input_idx = self.tik_inst.Scalar("uint64", name="input_idx")
        self.output_idx = self.tik_inst.Scalar("uint64", name="output_idx")
        self.l1_idx = self.tik_inst.Scalar("uint64", name="l1_idx")
        self.mask_idx = self.tik_inst.Scalar("uint64", name="mask_idx")
        self.fm_size = self.tik_inst.Scalar("uint64", init_value=0)

        self.output_block_h = 2
        if self.check_load3d_supported:
            self.output_block_h = 4
            l1_buff0_size = self.input_h * self.input_w * Constant.DIM_C0 + 32 * 1024
            self.l1_buff0 = self.tik_inst.Tensor(dtype, (l1_buff0_size,), name="l1_buff0", scope=tik.scope_cbuf)
        else:
            ub_load_size = ((self.output_block_h - 1) * self.stride_h + self.ksize_h) * \
                           self.input_w * Constant.DIM_C0 + Constant.DIM_C0
            self.ub_load0 = self.tik_inst.Tensor(dtype, (ub_load_size,),
                                                 name="ub_load0", scope=tik.scope_ubuf)
            self.ub_load1 = self.tik_inst.Tensor(dtype, (ub_load_size,),
                                                 name="ub_load1", scope=tik.scope_ubuf)

        ub_max_buff_size = self.stride_h * self.output_block_h * self.output_w * Constant.DIM_C0
        self.ub_max_buff = self.tik_inst.Tensor(dtype, (ub_max_buff_size,), name="ub_max_buff", scope=tik.scope_ubuf)

        ub_mask_buff_size = 8 * 1024
        self.ub_mask_buff = self.tik_inst.Tensor(
            "uint16", (ub_mask_buff_size,), name="ub_mask_buff", scope=tik.scope_ubuf)
        self.ub_mask_temp = self.tik_inst.Tensor(
            "uint16", (ub_mask_buff_size,), name="ub_mask_temp", scope=tik.scope_ubuf)
        self.ub_mask_or_buff = self.tik_inst.Tensor(
            "uint16", (ub_mask_buff_size,), name="ub_mask_or_buff", scope=tik.scope_ubuf)
        self.ub_mask_not_buff = self.tik_inst.Tensor(
            "uint16", (ub_mask_buff_size,), name="ub_mask_not_buff", scope=tik.scope_ubuf)

        filter_size = self.ksize_h * self.ksize_w
        ub_buff_size = self.output_block_h * self.output_w * Constant.DIM_C0 * filter_size
        self.buf_0 = self.tik_inst.Tensor(dtype, (ub_buff_size,), name="ub_buf_0", scope=tik.scope_ubuf)
        self.buf_1 = self.tik_inst.Tensor(dtype, (ub_buff_size,), name="ub_buf_1", scope=tik.scope_ubuf)

        self.loop_h = self.tik_inst.Scalar("uint64", name="loop_h")
        self.loop_h.set_as(self.output_h // self.output_block_h)
        self.mask_one_window = ((self.output_h * self.output_w + 15) // 16 + 1) * 16
        self.mask_gap_element = (self.mask_one_window - self.output_block_h * self.output_w)
        self.mask_gap = self.mask_gap_element * 2 // 32  # 3
        self.repeat_stride = Constant.MASK * 2 // 32
        self.repeat_0 = (self.output_block_h * self.output_w * Constant.DIM_C0 // Constant.MASK)
        self.repeat_1 = math.ceil(self.output_block_h * self.output_w / Constant.MASK)

    def _calc_ping_fm_size(self, looph, input_w):
        """
        :param looph: loop index for h
        :param input_w: input w
        :return:
        """
        self.fm_size.set_as(0)
        with self.tik_inst.if_scope(looph == 0):
            self.fm_size.set_as((self.output_block_h * self.stride_h + 1) * input_w)
        with self.tik_inst.else_scope():
            self.fm_size.set_as(self.output_block_h * self.stride_h * input_w)

    def _calc_pong_fm_size(self, looph, loop_h, input_w):
        """
        :param looph: loop index for h
        :param loop_h: loop for h
        :param input_w: input w
        :return:
        """
        self.fm_size.set_as(0)
        with self.tik_inst.if_scope(looph == loop_h // 2 - 1):
            self.fm_size.set_as((self.output_block_h * self.stride_h - 1) * input_w)
        with self.tik_inst.else_scope():
            self.fm_size.set_as(self.output_block_h * self.stride_h * input_w)

    def _calc_output_mask(self, filter_size, repeat_1, repeat_stride, output_w):
        """
        :param filter_size: filter size
        :param repeat_1: repeat_1
        :param repeat_stride: repeat_stride
        :param output_w: output w
        :return:
        """
        with self.tik_inst.for_range(2, filter_size) as idx:
            self.tik_inst.vnot(Constant.MASK, self.ub_mask_not_buff, self.ub_mask_or_buff, repeat_1, 1,
                               1, repeat_stride, repeat_stride)
            self.tik_inst.vor(Constant.MASK, self.ub_mask_or_buff, self.ub_mask_or_buff,
                              self.ub_mask_buff[idx * self.output_block_h * output_w], repeat_1, 1, 1, 1,
                              repeat_stride, repeat_stride, repeat_stride)
            self.tik_inst.vand(Constant.MASK, self.ub_mask_temp[(idx - 1) * self.output_block_h * output_w],
                               self.ub_mask_not_buff, self.ub_mask_buff[idx * self.output_block_h * output_w],
                               repeat_1, 1, 1, 1, repeat_stride, repeat_stride, repeat_stride)

    def _tik_instance_function_ping(self, looph, input_w, output_w, mask_gap,
                                    filter_size, repeat_0, repeat_1, repeat_stride, mask_one_window):
        """
        :param looph: loop index for h
        :param input_w: input w
        :param output_w: output w
        :param mask_gap: mask gap
        :param filter_size: filter size
        :param repeat_0: repeat_0
        :param repeat_1: repeat_1
        :param repeat_stride: repeat_stride
        :param mask_one_window: mask one window
        :return:
        """
        if self.check_load3d_supported:
            self._calc_ping_fm_size(looph, input_w)
            self.tik_inst.data_move(self.l1_buff0[self.l1_idx], self.input_fmap_gm[self.input_idx], 0, 1,
                                    self.fm_size, 0, 0)
            self._load3d_fm_to_ub(self.buf_0, self.l1_buff0, 0 - self.pad_l,
                                  looph * 2 * self.output_block_h * self.stride_h - self.pad_t)
        else:
            self._load_gm_to_ub_ping(self.buf_0, self.output_block_h, self.input_fmap_gm, self.input_idx, looph)

        self.tik_inst.vmax(Constant.MASK, self.ub_max_buff, self.buf_0,
                           self.buf_0[self.output_block_h * output_w * Constant.DIM_C0],
                           repeat_0, 1, 1, 1, repeat_stride, repeat_stride, repeat_stride)
        with self.tik_inst.for_range(2, filter_size) as idx:
            self.tik_inst.vmax(Constant.MASK, self.ub_max_buff, self.ub_max_buff,
                               self.buf_0[self.output_block_h * output_w * Constant.DIM_C0 * idx],
                               repeat_0, 1, 1, 1, repeat_stride, repeat_stride, repeat_stride)
        output_idx_tmp = (self.output_idx + looph * 2 * self.output_block_h * output_w * Constant.DIM_C0)
        self.tik_inst.data_move(self.output_max_gm[output_idx_tmp], self.ub_max_buff, 0, 1,
                                self.output_block_h * output_w * Constant.DIM_C0 * 2 // 32, 0, 0)

        with self.tik_inst.for_range(0, filter_size) as idx:
            self.tik_inst.vcmpv_eq(self.ub_mask_buff[idx * self.output_block_h * output_w * Constant.DIM_C0 // 16],
                                   self.buf_0[idx * self.output_block_h * output_w * Constant.DIM_C0],
                                   self.ub_max_buff, repeat_0, 1, 1, repeat_stride, repeat_stride)
        self.tik_inst.vnot(Constant.MASK, self.ub_mask_not_buff, self.ub_mask_buff,
                           repeat_1, 1, 1, repeat_stride, repeat_stride)
        self.tik_inst.vor(Constant.MASK, self.ub_mask_or_buff, self.ub_mask_buff,
                          self.ub_mask_buff[self.output_block_h * output_w],
                          repeat_1, 1, 1, 1, repeat_stride, repeat_stride, repeat_stride)
        self.tik_inst.vand(Constant.MASK, self.ub_mask_temp, self.ub_mask_not_buff,
                           self.ub_mask_buff[self.output_block_h * output_w],
                           repeat_1, 1, 1, 1, repeat_stride, repeat_stride, repeat_stride)
        self._calc_output_mask(filter_size, repeat_1, repeat_stride, output_w)
        self.tik_inst.data_move(self.output_mask_gm[self.mask_idx], self.ub_mask_buff, 0, 1,
                                self.output_block_h * output_w // Constant.DIM_C0, 0, 0)
        self.tik_inst.data_move(self.output_mask_gm[self.mask_idx + mask_one_window],
                                self.ub_mask_temp, 0, filter_size - 1,
                                self.output_block_h * output_w // Constant.DIM_C0, 0, mask_gap)

        self.mask_idx.set_as(self.mask_idx + self.output_block_h * output_w * Constant.DIM_C0 // 16)
        if self.check_load3d_supported:
            self.input_idx.set_as(self.input_idx + self.fm_size * Constant.DIM_C0)
            self.l1_idx.set_as(self.l1_idx + self.fm_size * 16)

    def _tik_instance_function_pong(self, looph, loop_h, input_w, output_w, mask_gap, filter_size,
                                    repeat_0, repeat_1, repeat_stride, mask_one_window):
        """
        :param looph: loop index for h
        :param input_w: input w
        :param output_w: output w
        :param mask_gap: mask gap
        :param filter_size: filter size
        :param repeat_0: repeat_0
        :param repeat_1: repeat_1
        :param repeat_stride: repeat_stride
        :param mask_one_window: mask one window
        :return:
        """
        if self.check_load3d_supported:
            self._calc_pong_fm_size(looph, loop_h, input_w)
            self.tik_inst.data_move(self.l1_buff0[self.l1_idx], self.input_fmap_gm[self.input_idx],
                                    0, 1, self.fm_size, 0, 0)

            self._load3d_fm_to_ub(self.buf_1, self.l1_buff0, -self.pad_l,
                                  (looph * 2 + 1) * self.output_block_h * self.stride_h - self.pad_t)
        else:
            self._load_gm_to_ub_pong(self.buf_1, self.output_block_h, self.input_fmap_gm,
                                     self.input_idx, looph)

        self.tik_inst.vmax(Constant.MASK, self.ub_max_buff, self.buf_1,
                           self.buf_1[self.output_block_h * output_w * Constant.DIM_C0],
                           repeat_0, 1, 1, 1, repeat_stride, repeat_stride, repeat_stride)

        with self.tik_inst.for_range(2, filter_size) as idx:
            self.tik_inst.vmax(Constant.MASK, self.ub_max_buff, self.ub_max_buff,
                               self.buf_1[self.output_block_h * output_w * Constant.DIM_C0 * idx],
                               repeat_0, 1, 1, 1, repeat_stride, repeat_stride, repeat_stride)

        output_idx_tmp = (self.output_idx + (looph * 2 + 1) * self.output_block_h * output_w * Constant.DIM_C0)
        self.tik_inst.data_move(self.output_max_gm[output_idx_tmp], self.ub_max_buff, 0, 1, self.output_block_h *
                                output_w * Constant.DIM_C0 * 2 // 32, 0, 0)

        with self.tik_inst.for_range(0, filter_size) as idx:
            self.tik_inst.vcmpv_eq(self.ub_mask_buff[idx * self.output_block_h * output_w * Constant.DIM_C0 // 16],
                                   self.buf_1[idx * self.output_block_h * output_w * Constant.DIM_C0],
                                   self.ub_max_buff, repeat_0, 1, 1, repeat_stride, repeat_stride)

        self.tik_inst.vnot(Constant.MASK, self.ub_mask_not_buff, self.ub_mask_buff, repeat_1, 1, 1,
                           repeat_stride, repeat_stride)
        self.tik_inst.vor(Constant.MASK, self.ub_mask_or_buff, self.ub_mask_buff,
                          self.ub_mask_buff[self.output_block_h * output_w],
                          repeat_1, 1, 1, 1, repeat_stride, repeat_stride, repeat_stride)
        self.tik_inst.vand(Constant.MASK, self.ub_mask_temp, self.ub_mask_not_buff,
                           self.ub_mask_buff[self.output_block_h * output_w], repeat_1, 1, 1, 1,
                           repeat_stride, repeat_stride, repeat_stride)
        self._calc_output_mask(filter_size, repeat_1, repeat_stride, output_w)

        self.tik_inst.data_move(self.output_mask_gm[self.mask_idx], self.ub_mask_buff,
                                0, 1, self.output_block_h * output_w // Constant.DIM_C0, 0, 0)
        self.tik_inst.data_move(self.output_mask_gm[self.mask_idx + mask_one_window], self.ub_mask_temp, 0,
                                filter_size - 1, self.output_block_h * output_w // Constant.DIM_C0, 0, mask_gap)
        self.mask_idx.set_as(self.mask_idx + self.output_block_h * output_w * Constant.DIM_C0 // 16)
        if self.check_load3d_supported:
            self.input_idx.set_as(self.input_idx + self.fm_size * Constant.DIM_C0)
            self.l1_idx.set_as(self.l1_idx + self.fm_size * 16)

    # 'pylint: disable=too-many-locals
    def maxpool_resnet50(self, core_idx, loop_num, one_core_loop):
        """
        implementation of max_pool_with_argmax of resnet50
        :return:
        """
        self._variable_init()
        with self.tik_inst.for_range(0, loop_num) as loop_idx:
            batch_idx = core_idx * one_core_loop + loop_idx
            batch = batch_idx / self.c1_dim
            loop_c = batch_idx % self.c1_dim
            self.input_idx.set_as(batch * self.c1_dim * self.input_h * self.input_w * Constant.DIM_C0 +
                                  loop_c * self.input_h * self.input_w * Constant.DIM_C0)
            self.output_idx.set_as(batch * self.c1_dim * self.output_h * self.output_w * Constant.DIM_C0 +
                                   loop_c * self.output_h * self.output_w * Constant.DIM_C0)
            if self.check_load3d_supported:
                self.l1_idx.set_as(0)

            self.mask_idx.set_as(batch * self.c1_dim * self.mask_one_window * self.filter_size +
                                 loop_c * self.mask_one_window * self.filter_size)

            with self.tik_inst.for_range(0, self.loop_h // 2) as looph:
                # ping
                self._tik_instance_function_ping(looph, self.input_w, self.output_w, self.mask_gap,
                                                 self.filter_size, self.repeat_0, self.repeat_1,
                                                 self.repeat_stride, self.mask_one_window)

                # pong
                self._tik_instance_function_pong(looph, self.loop_h, self.input_w, self.output_w, self.mask_gap,
                                                 self.filter_size, self.repeat_0, self.repeat_1,
                                                 self.repeat_stride, self.mask_one_window)
