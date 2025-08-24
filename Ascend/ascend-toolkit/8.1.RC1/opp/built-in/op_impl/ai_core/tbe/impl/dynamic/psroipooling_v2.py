#!/usr/bin/env python
# coding: utf-8
# Copyright 2023 Huawei Technologies Co., Ltd
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
PSROIPoolingV2
"""
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tik


# pylint:disable=too-few-public-methods
class Constant(object):
    DIGIT_2 = 2
    DIGIT_4 = 4
    DIGIT_5 = 5
    DIGIT_8 = 8
    DIGIT_64 = 64
    DIGIT_128 = 128
    DIGIT_255 = 255
    DIGIT_256 = 256
    C0 = 16
    TILING_NUM = 64
    MAX_INT32 = (1 << 31) - 1
    DEQSCALE = 1.0
    ONE_POINT = 1.0
    POINT_1 = 0.1
    NEG_ONE = -1.0
    NEG_TWO = -2.0
    UINT32 = "uint32"
    INT32 = "int32"
    FP16 = "float16"
    FP32 = "float32"
    INT16 = "int16"


# 'pylint:disable=too-few-public-methods,too-many-instance-attributes
class StreamParam(object):
    def __init__(self):
        self.row = None
        self.col = None
        self.h_start = None
        self.h_end = None
        self.w_start = None
        self.w_end = None
        self.h_width = None
        self.w_width = None
        self.bin_area = None
        self.ub_output_dim = None
        self.bin_i_offset = None
        self.scalar_roi_batch_id = None
        self.cur_roi_output_offset = None
        self.bin_start_h_floor = None
        self.bin_end_h_ceil = None
        self.bin_start_w_floor = None
        self.bin_end_w_ceil = None


# one block size takes up 32b
BLOCK_SIZE = 32
# instruction's default sid is 0
SID = 0
# instruction mask 64
MASK64 = 64
# one burst
BURST_1 = 1
# default repeat time
REPEAT_1 = 1
# repeat 2 time
REPEAT_2 = 2
# repeat 4 time
REPEAT_4 = 4
# stride zero
STRIDE_ZERO = 0
# stride one
STRIDE_ONE = 1
# default repeat stride length
REP_STRIDE_EIGHT = 8
REP_STRIDE_FOUR = 4


def _ceil_value(value, factor):
    """
    math: ceil(value/factor)
    """
    return (value + factor - 1) // factor


class PSROIPoolOpBase(object):
    TYPE_LEN_DICT = {Constant.FP16: 2, Constant.FP32: 4}
    REP_TIMES = {Constant.FP16: 2, Constant.FP32: 1}
    BLOCK_ELEM_NUM = {Constant.FP16: 16, Constant.FP32: 8}
    # number of element of fp16 and fp32 data type in one vector
    VEC_ELEM_NUM = {Constant.FP16: 128, Constant.FP32: 64}
    # repeat stride of fp16 and fp32 data type in vconv instruction
    REP_STRIDE = {Constant.FP16: 4, Constant.FP32: 8}

    def __init__(self, x_dict):
        profile = tik.Dprofile()
        self.tik_instance = tik.Tik(profile, disable_debug=True)
        self.aicore_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.ub_size = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
        self.product_name = tbe_platform.get_soc_spec(tbe_platform.SHORT_SOC_VERSION)
        self.dtype = x_dict.get("dtype").lower()
        self.dsize = self._get_type_len(self.dtype)
        self.vec_elem_num = self._get_vec_elem_num(self.dtype)
        self.mask = self.vec_elem_num
        self.const_0_127_ub = None
        self.const_1_128_ub = None

        self.is_hisi_cs = False
        if self.dtype == Constant.FP16 and self.product_name in ("Hi3796CV300CS", "SD3403"):
            self.is_hisi_cs = True
        self.deqscale = None
        self._scalar_conv = None
        if self.dtype == Constant.FP16:
            self.deqscale = Constant.DEQSCALE
            self._scalar_conv = self._scalar_conv_int32_to_float16
        else:
            self._scalar_conv = self._scalar_conv_int32_to_float32

    def _scalar_conv_int32_to_float32(self, dst_scalar, src_scalar):
        self.tik_instance.scalar_conv('', dst_scalar, src_scalar)

    def _scalar_conv_int32_to_float16(self, dst_scalar, src_scalar):
        with self.tik_instance.new_stmt_scope():
            fp32_scalar = self.tik_instance.Scalar(Constant.FP32)
            self.tik_instance.scalar_conv('', fp32_scalar, src_scalar)
            self.tik_instance.scalar_conv('', dst_scalar, fp32_scalar)

    def _get_type_len(self, data_type):
        return self.TYPE_LEN_DICT.get(data_type)

    def _get_rep_times(self, data_type):
        return self.REP_TIMES.get(data_type)

    def _get_block_elem_num(self, data_type):
        return self.BLOCK_ELEM_NUM.get(data_type)

    def _get_vec_elem_num(self, data_type):
        return self.VEC_ELEM_NUM.get(data_type)

    def _get_rep_stride(self, data_type):
        return self.REP_STRIDE.get(data_type)

    def _init_const_tensor(self):
        self.const_0_127_ub = self.tik_instance.Tensor(self.dtype,
                                                       shape=(Constant.DIGIT_128,),
                                                       name="const_0_127_ub",
                                                       scope=tbe_platform.scope_ubuf)
        self.const_1_128_ub = self.tik_instance.Tensor(self.dtype,
                                                       shape=(Constant.DIGIT_128, ),
                                                       name="const_1_128_ub",
                                                       scope=tbe_platform.scope_ubuf)
        if self.is_hisi_cs:
            const_0_127_int16 = self.tik_instance.Tensor(Constant.INT16, (Constant.DIGIT_128,),
                                                         name="const_0_127_int16", scope=tbe_platform.scope_ubuf)
            with self.tik_instance.for_range(0, Constant.DIGIT_128) as i:
                const_0_127_int16[i].set_as(i)
            self.tik_instance.vconv(Constant.DIGIT_128, '', self.const_0_127_ub,
                                    const_0_127_int16, REPEAT_1, STRIDE_ONE,
                                    STRIDE_ONE, REP_STRIDE_EIGHT, REP_STRIDE_EIGHT)
        else:
            const_0_127_int32 = self.tik_instance.Tensor(Constant.INT32, (Constant.DIGIT_128,),
                                                         name="const_0_127_int32", scope=tbe_platform.scope_ubuf)
            with self.tik_instance.for_range(0, Constant.DIGIT_128) as i:
                const_0_127_int32[i].set_as(i)
            self.tik_instance.vconv(MASK64, '', self.const_0_127_ub,
                                    const_0_127_int32, REPEAT_2, STRIDE_ONE,
                                    STRIDE_ONE, self._get_rep_stride(self.dtype),
                                    REP_STRIDE_EIGHT, deqscale=self.deqscale)

        self.tik_instance.vadds(MASK64, self.const_1_128_ub, self.const_0_127_ub, Constant.ONE_POINT,
                                Constant.DIGIT_128 // MASK64, STRIDE_ONE, STRIDE_ONE,
                                STRIDE_ZERO, STRIDE_ZERO)


# 'pylint: disable=invalid-name, too-many-locals, too-many-arguments
# 'pylint: disable=too-many-instance-attributes, too-many-lines
class PSROIPoolingV2Class(PSROIPoolOpBase):
    # 'pylint: disable = unused-argument
    def __init__(self, x_dict, rois_dict, y_dict, kernel_name):
        """
        constructor of PSROIPoolingV2Class

        Parameters
        ----------
        x_dict: dict describes input fm, NC1HWC0
        rois_dict: dict describes input rois
        op_attr: a struct,contain spatial_scale, output_dim, group_size
        kernel_name: name of kernel

        Returns
        -------
        None
        """
        PSROIPoolOpBase.__init__(self, x_dict)

        self.kernel_name = kernel_name

        # divide the available UB space into four parts
        self.ub_one_buf = self.ub_size // 4
        self.ub_one_buf_elem = self.ub_one_buf // self.dsize
        self._process_fun = None
        self._vdiv_fun = None
        if tbe_platform.api_check_support("tik.vdiv", "float32") and tbe_platform.api_check_support("tik.vdiv",
                                                                                                     "float16"):
            self._vdiv_fun = self._vdiv
        else:
            if self.dtype == Constant.FP16:
                self._vdiv_fun = self._newton_div_fp16
            else:
                self._vdiv_fun = self._newton_div_fp32
        self._init_aicore_params()

    def compute(self):
        """
        Main process of PSROIPoolingV2.
        """
        self._get_tiling_params()
        self._process()
        tbe_context.get_context().add_compile_info("vars", {"core_num": self.aicore_num})
        config = {"out_of_bound_sync_check": True, "enable_const_fold": True}
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[self.x_gm, self.rois_gm],
                                   outputs=[self.y_gm],
                                   config=config,
                                   flowtable=[self.tiling_gm])
        return self.tik_instance

    def _get_tiling_params(self):
        """
        get runtime params from tiling
        :return: None
        """
        with self.tik_instance.new_stmt_scope():
            tiling_ub = self.tik_instance.Tensor(Constant.UINT32,
                                                 shape=(Constant.TILING_NUM, ),
                                                 scope=tik.scope_ubuf,
                                                 name="tiling_ub")
            self.tik_instance.data_move(tiling_ub, self.tiling_gm,
                                        sid=0,
                                        nburst=1,
                                        burst=Constant.TILING_NUM // Constant.DIGIT_8,
                                        src_stride=0,
                                        dst_stride=0)
            self.roi_num_b.set_as(tiling_ub[0])
            self.fm_batch.set_as(tiling_ub[1])
            self.fm_h.set_as(tiling_ub[2])
            self.fm_w.set_as(tiling_ub[3])
            self.fm_c1.set_as(tiling_ub[4])
            self.fm_hw.set_as(tiling_ub[5])
            self.y_c1.set_as(tiling_ub[6])
            self.y_h.set_as(tiling_ub[7])
            self.y_w.set_as(tiling_ub[8])
            self.core_num.set_as(tiling_ub[9])
            self.outer_loop.set_as(tiling_ub[10])
            self.outer_tail.set_as(tiling_ub[11])
            self.num1.set_as(tiling_ub[12])
            self.num2.set_as(tiling_ub[13])
            self.block_num.set_as(tiling_ub[14])
            self.roi_loop1.set_as(tiling_ub[15])
            self.roi_step1_l.set_as(tiling_ub[16])
            self.roi_loop2.set_as(tiling_ub[17])
            self.roi_step2_l.set_as(tiling_ub[18])
            self.output_dim_align.set_as(tiling_ub[19])
            self.output_dim_align_c0.set_as(tiling_ub[20])
            self.output_dim.set_as(tiling_ub[21])
            self.group_size.set_as(tiling_ub[22])
            self.k2.set_as(tiling_ub[23])
            if self.dtype == Constant.FP32:
                self.spatial_scale.set_as(tiling_ub[24])
            else:
                spatial_scale_fp32 = self.tik_instance.Scalar(Constant.FP32, name="spatial_scale_fp32")
                spatial_scale_fp32.set_as(tiling_ub[24])
                self.tik_instance.scalar_conv('', dst=self.spatial_scale, src=spatial_scale_fp32)

        with self.tik_instance.if_scope(self.fm_batch == 1):
            self._process_fun = self._process_onebatch
        with self.tik_instance.else_scope():
            self._process_fun = self._process_mutilbatch

    def _process_onebatch(self, block_id):
        rois_num_offset = self.tik_instance.Scalar(Constant.UINT32, name="rois_num_offset")
        # process roi nums: num1
        with self.tik_instance.if_scope(block_id < self.outer_tail):
            # rois_num_offset is the offset in block_id aicore
            rois_num_offset.set_as(block_id * self.num1)
            self._process_rois(rois_num_offset, self.roi_loop1, self.roi_step1_l)
        # process roi nums: num2
        with self.tik_instance.else_scope():
            with self.tik_instance.if_scope(self.outer_loop > 0):
                rois_num_offset.set_as(self.outer_tail * self.num1 +
                                       (block_id - self.outer_tail) * self.num2)
                self._process_rois(rois_num_offset, self.roi_loop2, self.roi_step2_l)

    def _process_mutilbatch(self, block_id):
        rois_num_offset = self.tik_instance.Scalar(Constant.UINT32, name="rois_num_offset")
        # process roi nums: num1*fm_batch
        with self.tik_instance.if_scope(block_id < self.outer_tail):
            # rois_num_offset is the offset in block_id aicore
            with self.tik_instance.for_range(0, self.fm_batch) as batch_id:
                rois_num_offset.set_as(block_id * self.num1)
                self._process_rois_multi_batch(rois_num_offset,
                                               self.roi_loop1,
                                               self.roi_step1_l,
                                               batch_id)
        # process roi nums: num2*fm_batch
        with self.tik_instance.else_scope():
            with self.tik_instance.if_scope(self.outer_loop > 0):
                with self.tik_instance.for_range(0, self.fm_batch) as batch_id:
                    rois_num_offset.set_as(self.outer_tail * self.num1 +
                                           (block_id - self.outer_tail) * self.num2)
                    self._process_rois_multi_batch(rois_num_offset,
                                                   self.roi_loop2, self.roi_step2_l, batch_id)

    def _process(self):
        """
        process of PSROIPoolingV2
        """
        with self.tik_instance.for_range(0, self.block_num, block_num=self.block_num) as block_id:
            # process of one aicore
            self._init_const_tensor()
            self._process_fun(block_id)

    def _init_aicore_params(self):
        self.roi_num_b = self.tik_instance.Scalar(Constant.UINT32, name="roi_num_b")
        self.fm_batch = self.tik_instance.Scalar(Constant.UINT32, name="fm_batch")
        self.fm_c1 = self.tik_instance.Scalar(Constant.UINT32, name="fm_c1")
        self.fm_h = self.tik_instance.Scalar(Constant.INT32, name="fm_h")
        self.fm_w = self.tik_instance.Scalar(Constant.INT32, name="fm_w")
        self.fm_hw = self.tik_instance.Scalar(Constant.UINT32, name="fm_hw")
        self.y_c1 = self.tik_instance.Scalar(Constant.UINT32, name="y_c1")
        self.y_h = self.tik_instance.Scalar(Constant.UINT32, name="y_h")
        self.y_w = self.tik_instance.Scalar(Constant.UINT32, name="y_w")
        self.roi_num_b = self.tik_instance.Scalar(Constant.UINT32, name="roi_num_b")
        self.core_num = self.tik_instance.Scalar(Constant.UINT32, name="core_num")
        self.outer_loop = self.tik_instance.Scalar(Constant.UINT32, name="outer_loop")
        self.outer_tail = self.tik_instance.Scalar(Constant.UINT32, name="outer_tail")
        self.num1 = self.tik_instance.Scalar(Constant.UINT32, name="num1")
        self.num2 = self.tik_instance.Scalar(Constant.UINT32, name="num2")
        self.block_num = self.tik_instance.Scalar(Constant.UINT32, name="block_num")
        self.roi_loop1 = self.tik_instance.Scalar(Constant.UINT32, name="roi_loop1")
        self.roi_step1_l = self.tik_instance.Scalar(Constant.UINT32, name="roi_step1_l")
        self.roi_loop2 = self.tik_instance.Scalar(Constant.UINT32, name="roi_loop2")
        self.roi_step2_l = self.tik_instance.Scalar(Constant.UINT32, name="roi_step2_l")
        self.output_dim_align = self.tik_instance.Scalar(Constant.UINT32, name="output_dim_align")
        self.output_dim_align_c0 = self.tik_instance.Scalar(Constant.UINT32, name="output_dim_align_c0")
        self.output_dim = self.tik_instance.Scalar(Constant.UINT32, name="output_dim")
        self.group_size = self.tik_instance.Scalar(Constant.INT32, name="group_size")
        self.k2 = self.tik_instance.Scalar(Constant.UINT32, name="k2")
        self.spatial_scale = self.tik_instance.Scalar(self.dtype, name="spatial_scale")

        self.x_gm = self.tik_instance.Tensor(self.dtype, shape=(Constant.MAX_INT32, ),
                                             name="x_gm", scope=tbe_platform.scope_gm)
        self.rois_gm = self.tik_instance.Tensor(self.dtype, shape=(Constant.MAX_INT32, ),
                                                name="rois_gm", scope=tbe_platform.scope_gm)
        self.y_gm = self.tik_instance.Tensor(self.dtype, shape=(Constant.MAX_INT32, ),
                                             name="y_gm", scope=tbe_platform.scope_gm)
        self.tiling_gm = self.tik_instance.Tensor(Constant.UINT32,
                                                  (Constant.TILING_NUM, ),
                                                  scope=tik.scope_gm,
                                                  name="tiling_gm")

    def _load_rois_to_ub(self, rois_ub, rois_offset):
        """
        load rois data to ub from gm.

        Parameters
        ----------
        rois_ub: a tensor, which store rois data
        rois_offset: the roi offset of current loop in block_id aicore

        Returns
        -------
        None
        """
        burst_len = self.vec_elem_num * self.dsize // BLOCK_SIZE
        with self.tik_instance.for_range(0, Constant.DIGIT_5) as i:
            self.tik_instance.data_move(rois_ub[i, 0], self.rois_gm[rois_offset + i * self.roi_num_b],
                                        SID, BURST_1, burst_len, STRIDE_ZERO, STRIDE_ZERO)

    def _spatial_scale_rois(self, rois_ub, rois_floor_ub, rois_spatial_ub):
        """
        compute the width and height of rois and bin.

        Parameters
        ----------
        rois_ub: input rois data in ub, (5, self.vec_elem_num).
            batch_id,batch_id,batch_id...
            x1,x1,x1...
            y1,y1,y1...
            x2,x2,x2...
            y2,y2,y2...
        rois_floor_ub: store rois data of convert to s32
        rois_spatial_ub: store the width and height of rois and bin in ub

        Returns
        -------
        None
        """
        point_one_ub = self.tik_instance.Tensor(self.dtype, (self.vec_elem_num, ),
                                                name="point_one_ub",
                                                scope=tbe_platform.scope_ubuf)
        # rois_floor_ub[0]: batch id; rois_floor_ub[1-4]: roi coordinates
        # floor vconv from f16(or f32) to s32
        self.tik_instance.vconv(MASK64, 'round', rois_floor_ub, rois_ub,
                                self._get_rep_times(self.dtype) * Constant.DIGIT_5,
                                STRIDE_ONE, STRIDE_ONE, REP_STRIDE_EIGHT,
                                self._get_rep_stride(self.dtype))
        # s322f16: vconv.deq, or s322f32: vconv
        self.tik_instance.vconv(MASK64, '', rois_spatial_ub,
                                rois_floor_ub[1, 0],
                                self._get_rep_times(self.dtype) * Constant.DIGIT_4,
                                STRIDE_ONE, STRIDE_ONE,
                                self._get_rep_stride(self.dtype),
                                REP_STRIDE_EIGHT,
                                deqscale=self.deqscale)
        self.tik_instance.vadds(self.mask, rois_spatial_ub[2, 0],
                                rois_spatial_ub[2, 0], Constant.ONE_POINT, REPEAT_2,
                                STRIDE_ONE, STRIDE_ONE, REP_STRIDE_EIGHT,
                                REP_STRIDE_EIGHT)
        # multiply spatial
        self.tik_instance.vmuls(self.mask, rois_spatial_ub, rois_spatial_ub,
                                self.spatial_scale, REPEAT_4, STRIDE_ONE,
                                STRIDE_ONE, REP_STRIDE_EIGHT, REP_STRIDE_EIGHT)

        # roi width and height: roi_end_w-roi_start_w, roi_end_h-roi_start_h
        self.tik_instance.vsub(self.mask, rois_spatial_ub[4, 0],
                               rois_spatial_ub[2, 0], rois_spatial_ub,
                               REPEAT_2, STRIDE_ONE, STRIDE_ONE, STRIDE_ONE,
                               REP_STRIDE_EIGHT, REP_STRIDE_EIGHT,
                               REP_STRIDE_EIGHT)
        self.tik_instance.vector_dup(self.mask, point_one_ub, Constant.POINT_1,
                                     REPEAT_1, STRIDE_ONE, REP_STRIDE_EIGHT)
        self.tik_instance.vmax(self.mask, rois_spatial_ub[4, 0],
                               rois_spatial_ub[4, 0], point_one_ub, REPEAT_2,
                               STRIDE_ONE, STRIDE_ONE, STRIDE_ONE,
                               REP_STRIDE_EIGHT, REP_STRIDE_EIGHT, STRIDE_ZERO)

        pooled_k_recip = self.tik_instance.Scalar(self.dtype, name="pooled_k_recip")
        self._scalar_conv(pooled_k_recip, self.group_size)
        self.tik_instance.vector_dup(self.mask, point_one_ub, pooled_k_recip,
                                     REPEAT_1, STRIDE_ONE, REP_STRIDE_EIGHT)
        # bin width and height
        self._vdiv_fun(rois_spatial_ub[6, :], rois_spatial_ub[4, :], point_one_ub, REPEAT_1)
        self._vdiv_fun(rois_spatial_ub[7, :], rois_spatial_ub[5, :], point_one_ub, REPEAT_1)

    def _process_one_bin_2(self, stream):
        """
        process one bin of roi: inner_c1 == 1, or
                                (bin_all_dsize > self.ub_one_buf, or
                                 bin_load_stride > MAX_GAP_SIZE 65536)
        Parameters
        ----------
        stream: stream is a class

        Returns
        -------
        None
        """
        bursts_s = self.tik_instance.Scalar(Constant.INT32, name="bursts_s")
        bursts_s.set_as(stream.h_width)
        burst_len_s = self.tik_instance.Scalar(Constant.INT32, name="burst_len_s")
        burst_len_s.set_as(stream.w_width * Constant.C0 * self.dsize // BLOCK_SIZE)
        src_stride_s = self.tik_instance.Scalar(Constant.INT32, name="src_stride_s")
        src_stride_s.set_as((self.fm_w - stream.w_width) * Constant.C0 *
                            self.dsize // BLOCK_SIZE)
        ub_output_dim = stream.ub_output_dim
        output_dim_index = stream.output_dim_bias + stream.bin_i_offset
        bin_i_offset_c1 = output_dim_index // Constant.C0
        bin_i_offset_c0 = output_dim_index - bin_i_offset_c1 * Constant.C0

        load_count_ceil = _ceil_value(Constant.C0, self.vec_elem_num)
        load_count_align = load_count_ceil * self.vec_elem_num
        ub_bin_input_buf = self.tik_instance.Tensor(
            self.dtype, (self.ub_one_buf_elem, ), name="ub_bin_input_buf", scope=tbe_platform.scope_ubuf)
        ub_bin_output_buf = self.tik_instance.Tensor(
            self.dtype, (load_count_align, ), name="ub_bin_output_buf", scope=tbe_platform.scope_ubuf)
        self.tik_instance.vector_dup(self.mask, ub_bin_output_buf, 0,
                                     load_count_ceil, STRIDE_ONE,
                                     REP_STRIDE_EIGHT)
        # move feature map from gm to ub
        offset_x = (((stream.scalar_roi_batch_id*self.fm_c1+bin_i_offset_c1)*self.fm_h +
                    stream.h_start)*self.fm_w + stream.w_start)*Constant.C0 + bin_i_offset_c0
        self.tik_instance.data_move(ub_bin_input_buf,
                                    self.x_gm[offset_x], SID,
                                    bursts_s, burst_len_s, src_stride_s, STRIDE_ZERO)

        # avg pooling
        src0_rep_stride = Constant.C0 * self.dsize // BLOCK_SIZE

        with self.tik_instance.if_scope(stream.bin_area < Constant.DIGIT_256):
            self.tik_instance.vadd([0, 0x1], ub_bin_output_buf,
                                   ub_bin_input_buf, ub_bin_output_buf,
                                   stream.bin_area, STRIDE_ONE,
                                   STRIDE_ONE, STRIDE_ONE, STRIDE_ZERO,
                                   src0_rep_stride, STRIDE_ZERO)
        with self.tik_instance.else_scope():
            tail = stream.bin_area % Constant.DIGIT_255
            times = stream.bin_area // Constant.DIGIT_255
            with self.tik_instance.for_range(0, times) as times_i:
                self.tik_instance.vadd(
                    [0, 0x1], ub_bin_output_buf,
                    ub_bin_input_buf[Constant.C0 * Constant.DIGIT_255 * times_i],
                    ub_bin_output_buf, Constant.DIGIT_255, STRIDE_ONE, STRIDE_ONE,
                    STRIDE_ONE, STRIDE_ZERO, src0_rep_stride, STRIDE_ZERO)
            with self.tik_instance.if_scope(tail != 0):
                self.tik_instance.vadd(
                    [0, 0x1], ub_bin_output_buf,
                    ub_bin_input_buf[Constant.C0 * Constant.DIGIT_255 * times],
                    ub_bin_output_buf, tail, STRIDE_ONE, STRIDE_ONE,
                    STRIDE_ONE, STRIDE_ZERO, src0_rep_stride, STRIDE_ZERO)

        bin_area_fp_ub = stream.bin_area_fp_ub
        self._vdiv_fun(ub_bin_output_buf, ub_bin_output_buf, bin_area_fp_ub, load_count_ceil)
        output_val = self.tik_instance.Scalar(self.dtype, name="output_val")
        output_val.set_as(ub_bin_output_buf[0])
        mask = stream.output_dim_mask
        ub_output_dim[mask] = output_val

    def _process_one_bin_area_positive(self, stream):
        """
        process one bin of roi when bin area is positive.

        Parameters
        ----------
        stream: stream is a class
        Returns
        -------
        None
        """
        bin_area_fp_ub = self.tik_instance.Tensor(self.dtype, (self.vec_elem_num, ),
                                                  name="bin_area_fp_ub",
                                                  scope=tbe_platform.scope_ubuf)

        bin_area_int32 = self.tik_instance.Tensor(Constant.INT32,
                                                  (Constant.DIGIT_64 * self._get_rep_times(self.dtype), ),
                                                  name="bin_area_int32",
                                                  scope=tbe_platform.scope_ubuf)
        self.tik_instance.vector_dup(MASK64, bin_area_int32,
                                     stream.bin_area,
                                     self._get_rep_times(self.dtype),
                                     STRIDE_ONE,
                                     REP_STRIDE_EIGHT)
        # s322f16:vconv.deq, or s322f32:vconv
        self.tik_instance.vconv(MASK64,
                                '',
                                bin_area_fp_ub,
                                bin_area_int32,
                                self._get_rep_times(self.dtype),
                                STRIDE_ONE,
                                STRIDE_ONE,
                                self._get_rep_stride(self.dtype),
                                REP_STRIDE_EIGHT,
                                deqscale=self.deqscale)

        stream.bin_area_fp_ub = bin_area_fp_ub

    def _process_one_bin(self, stream):
        """
        process one bin of roi.

        Parameters
        ----------
        stream: stream is a class

        Returns
        -------
        None
        """
        output_dim_shape = (self.output_dim_align, )
        ub_output_dim = self.tik_instance.Tensor(self.dtype,
                                                 output_dim_shape,
                                                 name="ub_output_dim",
                                                 scope=tbe_platform.scope_ubuf)
        self.tik_instance.vector_dup(self.mask, ub_output_dim, 0.,
                                     repeat_times=self.output_dim_align // self.mask,
                                     dst_blk_stride=1, dst_rep_stride=Constant.DIGIT_8)
        stream.ub_output_dim = ub_output_dim

        with self.tik_instance.if_scope(stream.bin_area > 0):
            self._process_one_bin_area_positive(stream)
            with self.tik_instance.if_scope(self.output_dim <= 1):
                thread_num = 1
            with self.tik_instance.else_scope():
                thread_num = 2
            with self.tik_instance.for_range(0, self.output_dim,
                                             thread_num=thread_num) as out_dim:
                stream.output_dim_mask = out_dim
                stream.output_dim_bias = out_dim * self.k2

                self._process_one_bin_2(stream)

        burst_len = Constant.C0 * self.dsize // BLOCK_SIZE
        offset_y = (((stream.cur_roi_output_offset * self.y_c1 + 0) * self.y_h +
                    stream.row)*self.y_w + stream.col)*Constant.C0 + 0
        self.tik_instance.data_move(self.y_gm[offset_y], ub_output_dim, SID,
                                    _ceil_value(self.output_dim_align_c0, Constant.C0),
                                    burst_len,
                                    STRIDE_ZERO, (self.k2 - 1) * burst_len)

    def _process_one_roi(self, stream):
        """
        process one roi.

        Parameters
        ----------
        stream: stream is a class

        Returns
        -------
        None
        """
        with self.tik_instance.for_range(0, self.group_size) as row:
            stream.row = row
            # h coordinates of bin
            h_start = self.tik_instance.Scalar(Constant.INT32, name="h_start")
            h_start.set_as(stream.bin_start_h_floor[row])
            h_end = self.tik_instance.Scalar(Constant.INT32, name="h_end")
            h_end.set_as(stream.bin_end_h_ceil[row])
            stream.h_start = h_start
            stream.h_end = h_end
            h_width = self.tik_instance.Scalar(Constant.INT32, name="h_width")
            w_width = self.tik_instance.Scalar(Constant.INT32, name="w_width")
            bin_area = self.tik_instance.Scalar(Constant.INT32, name="bin_area")
            h_width.set_as(h_end - h_start)
            with self.tik_instance.if_scope(h_end <= h_start):
                h_width.set_as(0)
            stream.h_width = h_width
            with self.tik_instance.for_range(0, self.group_size) as col:
                stream.col = col
                # w coordinates of bin
                w_start = self.tik_instance.Scalar(Constant.INT32, name="w_start")
                w_start.set_as(stream.bin_start_w_floor[col])
                w_end = self.tik_instance.Scalar(Constant.INT32, name="w_end")
                w_end.set_as(stream.bin_end_w_ceil[col])
                stream.w_start = w_start
                stream.w_end = w_end

                bin_i_offset = self.tik_instance.Scalar(Constant.INT32, name="bin_i_offset")
                # bin_i offset of in roi, 0~(group_size^2-1)
                bin_i_offset.set_as(row * self.group_size + col)
                bin_i_offset.set_as(bin_i_offset % self.k2)
                stream.bin_i_offset = bin_i_offset

                w_width.set_as(w_end - w_start)
                with self.tik_instance.if_scope(w_end <= w_start):
                    w_width.set_as(0)
                bin_area.set_as(w_width * h_width)
                stream.w_width = w_width
                stream.bin_area = bin_area
                self._process_one_bin(stream)

    def _data_range_limit(self, minv, maxv, data):
        """
        data = min(max(data,minv),maxv)
        """
        self.tik_instance.vmax(MASK64, data, data, minv,
                               Constant.DIGIT_128 // MASK64,
                               STRIDE_ONE, STRIDE_ONE, STRIDE_ONE,
                               REP_STRIDE_EIGHT, REP_STRIDE_EIGHT,
                               STRIDE_ZERO)
        self.tik_instance.vmin(MASK64, data, data, maxv,
                               Constant.DIGIT_128 // MASK64,
                               STRIDE_ONE, STRIDE_ONE, STRIDE_ONE,
                               REP_STRIDE_EIGHT, REP_STRIDE_EIGHT,
                               STRIDE_ZERO)

    # 'pylint: disable=too-many-arguments
    def _a_mul_x_add_b(self, a, x, b, dst_float, dst_int, mode):
        """
        dst_float = a*x+b
        dst_int = mode(dst_float)

        Parameters
        ----------
        mode: a string, floor or ceil

        Returns
        -------
        None
        """
        self.tik_instance.vmuls(self.mask, dst_float,
                                a, x,
                                Constant.DIGIT_128 // self.vec_elem_num,
                                STRIDE_ONE, STRIDE_ONE,
                                REP_STRIDE_EIGHT, REP_STRIDE_EIGHT)
        self.tik_instance.vadds(self.mask, dst_float,
                                dst_float, b,
                                Constant.DIGIT_128 // self.vec_elem_num,
                                STRIDE_ONE, STRIDE_ONE,
                                REP_STRIDE_EIGHT, REP_STRIDE_EIGHT)
        self.tik_instance.vconv(MASK64, mode, dst_int,
                                dst_float,
                                Constant.DIGIT_128 // MASK64, STRIDE_ONE,
                                STRIDE_ONE, REP_STRIDE_EIGHT,
                                self._get_rep_stride(self.dtype))

    # 'pylint: disable=too-many-arguments
    def _a_mul_x_add_b_fp16(self, a, x, b, dst_float16, dst_float, dst_int, mode):
        """
        dst_float16 = a*x
        dst_float = dst_float16
        dst_float = dst_float + b
        dst_int = mode(dst_float)
        Parameters
        ----------
        mode: a string, 'floor' or 'ceil'

        Returns
        -------
        None
        """
        self.tik_instance.vmuls(self.mask, dst_float16,
                                a, x,
                                Constant.DIGIT_128 // self.vec_elem_num,
                                STRIDE_ONE, STRIDE_ONE,
                                REP_STRIDE_EIGHT, REP_STRIDE_EIGHT)
        self.tik_instance.vconv(MASK64, '', dst_float,
                                dst_float16, Constant.DIGIT_128 // MASK64,
                                STRIDE_ONE, STRIDE_ONE,
                                REP_STRIDE_EIGHT, REP_STRIDE_FOUR)
        self.tik_instance.vadds(MASK64, dst_float, dst_float, b,
                                Constant.DIGIT_128 // MASK64,
                                STRIDE_ONE,
                                STRIDE_ONE, REP_STRIDE_EIGHT,
                                REP_STRIDE_EIGHT)
        # floor vconv, from f16(or f32) to s32
        self.tik_instance.vconv(MASK64, mode, dst_int,
                                dst_float,
                                Constant.DIGIT_128 // MASK64,
                                STRIDE_ONE,
                                STRIDE_ONE, REP_STRIDE_EIGHT,
                                REP_STRIDE_EIGHT)

    # 'pylint: disable=too-many-arguments
    def _process_step1_roi(self, rois_floor_ub, rois_spatial_ub,
                           rois_num_offset, step_i_offset, step_i_num):
        """
        process roi_num roi cyclically, and process self.vec_elem_num roi each time.

        Parameters
        ----------
        rois_floor_ub: rois data, s32
        rois_spatial_ub: a tensor, the width and height of rois and bin
        step_i_offset: the roi offset of this loop in block_id aicore
        rois_num_offset: a Scalar, the offset in block_id aicore
        step_i_num: the number of rois of one loop in process

        Returns
        -------
        None
        """
        with self.tik_instance.for_range(0, step_i_num) as roi_i:
            stream = StreamParam()
            scalar_roi_batch_id = self.tik_instance.Scalar(Constant.INT32, name="scalar_roi_batch_id")
            scalar_roi_batch_id.set_as(rois_floor_ub[0, roi_i])
            stream.scalar_roi_batch_id = scalar_roi_batch_id

            cur_roi_output_offset = self.tik_instance.Scalar(Constant.INT32, name="cur_roi_output_offset")
            cur_roi_output_offset.set_as(rois_num_offset + step_i_offset + roi_i)
            stream.cur_roi_output_offset = cur_roi_output_offset

            scalar_bin_width = self.tik_instance.Scalar(self.dtype, name="scalar_bin_width")
            scalar_bin_width.set_as(rois_spatial_ub[6, roi_i])
            scalar_bin_height = self.tik_instance.Scalar(self.dtype, name="scalar_bin_height")
            scalar_bin_height.set_as(rois_spatial_ub[7, roi_i])

            bin_start_w_floor = self.tik_instance.Tensor(Constant.INT32, (Constant.DIGIT_128, ),
                                                         name="bin_start_w_floor",
                                                         scope=tbe_platform.scope_ubuf)
            bin_end_w_ceil = self.tik_instance.Tensor(Constant.INT32, (Constant.DIGIT_128, ),
                                                      name="bin_end_w_ceil",
                                                      scope=tbe_platform.scope_ubuf)
            bin_start_h_floor = self.tik_instance.Tensor(Constant.INT32, (Constant.DIGIT_128, ),
                                                         name="bin_start_h_floor",
                                                         scope=tbe_platform.scope_ubuf)
            bin_end_h_ceil = self.tik_instance.Tensor(Constant.INT32, (Constant.DIGIT_128, ),
                                                      name="bin_end_h_ceil",
                                                      scope=tbe_platform.scope_ubuf)

            scalar_roi_start_w = self.tik_instance.Scalar(Constant.FP32, name="scalar_roi_start_w")
            scalar_roi_start_h = self.tik_instance.Scalar(Constant.FP32, name="scalar_roi_start_h")
            if self.dtype == Constant.FP16:
                rois_spatial_fp32_ub = self.tik_instance.Tensor(
                    Constant.FP32, (Constant.DIGIT_8, Constant.DIGIT_128),
                    name="rois_spatial_fp32_ub",
                    scope=tbe_platform.scope_ubuf)
                self.tik_instance.vconv(MASK64, '', rois_spatial_fp32_ub,
                                        rois_spatial_ub,
                                        Constant.DIGIT_8 * Constant.DIGIT_128 // MASK64,
                                        STRIDE_ONE, STRIDE_ONE,
                                        REP_STRIDE_EIGHT, REP_STRIDE_FOUR)

                scalar_roi_start_w.set_as(rois_spatial_fp32_ub[0, roi_i])
                scalar_roi_start_h.set_as(rois_spatial_fp32_ub[1, roi_i])

                dst_float16 = self.tik_instance.Tensor(self.dtype, (Constant.DIGIT_128, ),
                                                       name="dst_float16",
                                                       scope=tbe_platform.scope_ubuf)
                dst_float = self.tik_instance.Tensor(Constant.FP32, (Constant.DIGIT_128, ),
                                                     name="dst_float",
                                                     scope=tbe_platform.scope_ubuf)
                # get start width floor
                self._a_mul_x_add_b_fp16(self.const_0_127_ub, scalar_bin_width, scalar_roi_start_w,
                                         dst_float16, dst_float, bin_start_w_floor, 'floor')

                # get end width ceil
                self._a_mul_x_add_b_fp16(self.const_1_128_ub, scalar_bin_width, scalar_roi_start_w,
                                         dst_float16, dst_float, bin_end_w_ceil, 'ceil')

                # scalar_roi_start_h + scalar_bin_height*(0...127)
                self._a_mul_x_add_b_fp16(self.const_0_127_ub, scalar_bin_height, scalar_roi_start_h,
                                         dst_float16, dst_float, bin_start_h_floor, 'floor')

                # scalar_roi_start_h add scalar_bin_height*(1...128)
                self._a_mul_x_add_b_fp16(self.const_1_128_ub, scalar_bin_height, scalar_roi_start_h,
                                         dst_float16, dst_float, bin_end_h_ceil, 'ceil')

            else:
                scalar_roi_start_w.set_as(rois_spatial_ub[0, roi_i])
                scalar_roi_start_h.set_as(rois_spatial_ub[1, roi_i])
                dst_float = self.tik_instance.Tensor(self.dtype, (Constant.DIGIT_128, ),
                                                     name="dst_float",
                                                     scope=tbe_platform.scope_ubuf)

                # scalar_roi_start_w + scalar_bin_width*(0...127)
                self._a_mul_x_add_b(self.const_0_127_ub, scalar_bin_width,
                                    scalar_roi_start_w, dst_float, bin_start_w_floor, 'floor')
                self._a_mul_x_add_b(self.const_1_128_ub, scalar_bin_width,
                                    scalar_roi_start_w, dst_float, bin_end_w_ceil, 'ceil')
                # scalar_roi_start_h + scalar_bin_height*(0...127)
                self._a_mul_x_add_b(self.const_0_127_ub, scalar_bin_height,
                                    scalar_roi_start_h, dst_float, bin_start_h_floor, 'floor')
                self._a_mul_x_add_b(self.const_1_128_ub, scalar_bin_height,
                                    scalar_roi_start_h, dst_float, bin_end_h_ceil, 'ceil')
            minv_ub = self.tik_instance.Tensor(Constant.INT32, (Constant.DIGIT_64,),
                                               name="minv_ub", scope=tbe_platform.scope_ubuf)
            maxv_ub = self.tik_instance.Tensor(Constant.INT32, (Constant.DIGIT_64,),
                                               name="maxv_ub", scope=tbe_platform.scope_ubuf)
            self.tik_instance.vector_dup(MASK64, minv_ub, 0, REPEAT_1,
                                         STRIDE_ONE, REP_STRIDE_EIGHT)
            self.tik_instance.vector_dup(MASK64, maxv_ub, self.fm_w, REPEAT_1,
                                         STRIDE_ONE, REP_STRIDE_EIGHT)
            self._data_range_limit(minv=minv_ub, maxv=maxv_ub, data=bin_start_w_floor)
            self._data_range_limit(minv=minv_ub, maxv=maxv_ub, data=bin_end_w_ceil)
            self.tik_instance.vector_dup(MASK64, maxv_ub, self.fm_h, REPEAT_1,
                                         STRIDE_ONE, REP_STRIDE_EIGHT)
            self._data_range_limit(minv=minv_ub, maxv=maxv_ub, data=bin_start_h_floor)
            self._data_range_limit(minv=minv_ub, maxv=maxv_ub, data=bin_end_h_ceil)

            stream.bin_start_h_floor = bin_start_h_floor
            stream.bin_end_h_ceil = bin_end_h_ceil
            stream.bin_start_w_floor = bin_start_w_floor
            stream.bin_end_w_ceil = bin_end_w_ceil

            self._process_one_roi(stream)

    def _process_rois(self, rois_num_offset, roi_loop, roi_step_l):
        """
        process roi_num roi cyclically, and process self.vec_elem_num roi each time.

        Parameters
        ----------
        rois_num_offset: the offset in block_id aicore
        roi_loop: loops of processing roi_num rois
        roi_step_l: the number of rois in last loop

        Returns
        -------
        None
        """
        with self.tik_instance.for_range(0, roi_loop) as inner_i:
            rois_ub = self.tik_instance.Tensor(self.dtype, (Constant.DIGIT_5, self.vec_elem_num),
                                               name="rois_ub",
                                               scope=tbe_platform.scope_ubuf)
            rois_offset = self.tik_instance.Scalar(Constant.UINT32, name="rois_offset")
            rois_offset.set_as(rois_num_offset + self.vec_elem_num * inner_i)
            # move rois data to ub from gm
            self._load_rois_to_ub(rois_ub, rois_offset)

            # calculate spatial scale rois
            rois_floor_ub = self.tik_instance.Tensor(Constant.INT32,
                                                     shape=(Constant.DIGIT_5, self.vec_elem_num),
                                                     name="rois_floor_ub",
                                                     scope=tbe_platform.scope_ubuf)
            rois_spatial_ub = self.tik_instance.Tensor(self.dtype,
                                                       shape=(Constant.DIGIT_8, self.vec_elem_num),
                                                       name="rois_spatial_ub",
                                                       scope=tbe_platform.scope_ubuf)
            self._spatial_scale_rois(rois_ub, rois_floor_ub, rois_spatial_ub)

            with self.tik_instance.if_scope(inner_i == (roi_loop - 1)):
                self._process_step1_roi(rois_floor_ub, rois_spatial_ub,
                                        rois_num_offset, self.vec_elem_num * inner_i,
                                        roi_step_l)
            with self.tik_instance.else_scope():
                self._process_step1_roi(rois_floor_ub, rois_spatial_ub,
                                        rois_num_offset, self.vec_elem_num * inner_i,
                                        self.vec_elem_num)

    def _process_rois_multi_batch(self, rois_num_offset, roi_loop,
                                  roi_step_l, batch_id):
        """
        process roi_num roi cyclically, and process self.vec_elem_num roi each time.

        Parameters
        ----------
        rois_num_offset: the offset in block_id aicore
        roi_loop: loops of processing roi_num rois
        roi_step_l: the number of rois in last loop
        batch_id: batch id

        Returns
        -------
        None
        """
        with self.tik_instance.for_range(0, roi_loop) as inner_i:
            rois_ub = self.tik_instance.Tensor(self.dtype, (Constant.DIGIT_5, self.vec_elem_num),
                                               name="rois_ub",
                                               scope=tbe_platform.scope_ubuf)
            # rois addr offset
            rois_offset = self.tik_instance.Scalar(Constant.INT32, name="rois_offset")
            rois_offset.set_as(rois_num_offset + batch_id *
                               (self.roi_num_b * Constant.DIGIT_5) +
                               self.vec_elem_num * inner_i)
            # move rois data to ub from gm
            self._load_rois_to_ub(rois_ub, rois_offset)

            # calculate spatial scale rois
            rois_floor_ub = self.tik_instance.Tensor(
                Constant.INT32, (Constant.DIGIT_5, self.vec_elem_num),
                name="rois_floor_ub",
                scope=tbe_platform.scope_ubuf)
            rois_spatial_ub = self.tik_instance.Tensor(
                self.dtype, (Constant.DIGIT_8, self.vec_elem_num),
                name="rois_spatial_ub",
                scope=tbe_platform.scope_ubuf)
            self._spatial_scale_rois(rois_ub, rois_floor_ub, rois_spatial_ub)

            with self.tik_instance.if_scope(inner_i == (roi_loop - 1)):
                self._process_step1_roi(rois_floor_ub, rois_spatial_ub,
                                        rois_num_offset + self.roi_num_b * batch_id,
                                        self.vec_elem_num * inner_i, roi_step_l)
            with self.tik_instance.else_scope():
                self._process_step1_roi(rois_floor_ub, rois_spatial_ub,
                                        rois_num_offset + self.roi_num_b * batch_id,
                                        self.vec_elem_num * inner_i, self.vec_elem_num)

    def _vdiv(self, dst, divisor, dividend, repeat):
        self.tik_instance.vdiv(self.mask, dst, divisor, dividend, repeat,
                               STRIDE_ONE, STRIDE_ONE, STRIDE_ONE,
                               REP_STRIDE_EIGHT, REP_STRIDE_EIGHT,
                               STRIDE_ZERO)

    def _newton_div_fp16(self, dst, divisor, dividend, repeat):
        with self.tik_instance.new_stmt_scope():
            divisor_f32 = self.tik_instance.Tensor(Constant.FP32,
                                                   shape=(self.vec_elem_num,),
                                                   name="divisor_f32",
                                                   scope=tbe_platform.scope_ubuf)
            dividend_f32 = self.tik_instance.Tensor(Constant.FP32,
                                                    shape=(self.vec_elem_num,),
                                                    name="dividend_f32",
                                                    scope=tbe_platform.scope_ubuf)
            rec_f32 = self.tik_instance.Tensor(Constant.FP32,
                                               shape=(self.vec_elem_num,),
                                               name="rec_f32",
                                               scope=tbe_platform.scope_ubuf)
            self.tik_instance.vconv(MASK64, '', divisor_f32, divisor,
                                    Constant.DIGIT_128 // MASK64, STRIDE_ONE, STRIDE_ONE,
                                    REP_STRIDE_EIGHT, REP_STRIDE_FOUR)
            self.tik_instance.vconv(MASK64, '', dividend_f32, dividend,
                                    Constant.DIGIT_128 // MASK64, STRIDE_ONE, STRIDE_ONE,
                                    REP_STRIDE_EIGHT, REP_STRIDE_FOUR)
            self.tik_instance.vrec(MASK64, rec_f32, dividend_f32,
                                   Constant.DIGIT_128 // MASK64, STRIDE_ONE, STRIDE_ONE,
                                   REP_STRIDE_EIGHT, REP_STRIDE_EIGHT)

            # Newton start
            self.tik_instance.vmul(MASK64, dividend_f32, dividend_f32, rec_f32,
                                   Constant.DIGIT_128 // MASK64, STRIDE_ONE, STRIDE_ONE,
                                   STRIDE_ONE, REP_STRIDE_EIGHT,
                                   REP_STRIDE_EIGHT, REP_STRIDE_EIGHT)
            self.tik_instance.vadds(MASK64, dividend_f32, dividend_f32, Constant.NEG_TWO,
                                    Constant.DIGIT_128 // MASK64, STRIDE_ONE, STRIDE_ONE,
                                    REP_STRIDE_EIGHT, REP_STRIDE_EIGHT)
            self.tik_instance.vmul(MASK64, dividend_f32, dividend_f32, rec_f32,
                                   Constant.DIGIT_128 // MASK64, STRIDE_ONE, STRIDE_ONE,
                                   STRIDE_ONE, REP_STRIDE_EIGHT,
                                   REP_STRIDE_EIGHT, REP_STRIDE_EIGHT)
            self.tik_instance.vmuls(MASK64, dividend_f32, dividend_f32, Constant.NEG_ONE,
                                    Constant.DIGIT_128 // MASK64, STRIDE_ONE, STRIDE_ONE,
                                    REP_STRIDE_EIGHT, REP_STRIDE_EIGHT)
            # Newton end
            self.tik_instance.vmul(MASK64, divisor_f32, divisor_f32, dividend_f32,
                                   Constant.DIGIT_128 // MASK64, STRIDE_ONE, STRIDE_ONE,
                                   STRIDE_ONE, REP_STRIDE_EIGHT,
                                   REP_STRIDE_EIGHT, STRIDE_ZERO)
            self.tik_instance.vconv(MASK64, '', dst, divisor_f32,
                                    Constant.DIGIT_128 // MASK64, STRIDE_ONE, STRIDE_ONE,
                                    REP_STRIDE_FOUR, REP_STRIDE_EIGHT)

    def _newton_div_fp32(self, dst, divisor, dividend, repeat):
        with self.tik_instance.new_stmt_scope():
            rec_f32 = self.tik_instance.Tensor(Constant.FP32,
                                               shape=(self.vec_elem_num,),
                                               name="rec_f32",
                                               scope=tbe_platform.scope_ubuf)
            dividend_copy = self.tik_instance.Tensor(self.dtype,
                                                     shape=(self.vec_elem_num,),
                                                     name="dividend_copy",
                                                     scope=tbe_platform.scope_ubuf)
            self.tik_instance.vrec(self.mask, rec_f32, dividend, repeat,
                                   STRIDE_ONE, STRIDE_ONE,
                                   REP_STRIDE_EIGHT, REP_STRIDE_EIGHT)
            self.tik_instance.data_move(dividend_copy, dividend, SID, BURST_1,
                                        self.vec_elem_num // self._get_block_elem_num(self.dtype),
                                        STRIDE_ZERO, STRIDE_ZERO)

            # Newton start
            self.tik_instance.vmul(self.mask, dividend_copy, dividend_copy, rec_f32,
                                   repeat, STRIDE_ONE, STRIDE_ONE,
                                   STRIDE_ONE, REP_STRIDE_EIGHT,
                                   REP_STRIDE_EIGHT, REP_STRIDE_EIGHT)
            self.tik_instance.vadds(self.mask, dividend_copy, dividend_copy, Constant.NEG_TWO,
                                    repeat, STRIDE_ONE, STRIDE_ONE,
                                    REP_STRIDE_EIGHT, REP_STRIDE_EIGHT)
            self.tik_instance.vmul(self.mask, dividend_copy, dividend_copy, rec_f32,
                                   repeat, STRIDE_ONE, STRIDE_ONE,
                                   STRIDE_ONE, REP_STRIDE_EIGHT,
                                   REP_STRIDE_EIGHT, REP_STRIDE_EIGHT)
            self.tik_instance.vmuls(self.mask, dividend_copy, dividend_copy, Constant.NEG_ONE,
                                    repeat, STRIDE_ONE, STRIDE_ONE,
                                    REP_STRIDE_EIGHT, REP_STRIDE_EIGHT)
            # Newton end
            self.tik_instance.vmul(self.mask, dst, divisor, dividend_copy,
                                   repeat, STRIDE_ONE, STRIDE_ONE,
                                   STRIDE_ONE, REP_STRIDE_EIGHT,
                                   REP_STRIDE_EIGHT, STRIDE_ZERO)


# 'pylint: disable=too-many-arguments, invalid-name, unused-argument
@register_operator("PSROIPoolingV2")
def psroipooling_v2(x_dict, rois_dict, y_dict, spatial_scale,
                    output_dim, group_size, kernel_name="PSROIPoolingV2"):
    """
    PSROIPoolingV2 interface.

    Parameters
    ----------
    x_dict: feature map size and data type, 5HD
    rois_dict: rois_dict size and data type, (batch, 5, rois_num), rois all
                nums is batch*rois_num
    y_dict: output size and data type, 5HD
    spatial_scale: spatial scale
    output_dim: number of output channels
    group_size: number of groups encoding position sensitive score maps
    kernel_name: kernel name of PSROIPoolingV2 op

    Returns
    -------
    tik_instance
    """
    psroi_instance = PSROIPoolingV2Class(x_dict, rois_dict, y_dict, kernel_name)
    return psroi_instance.compute()
