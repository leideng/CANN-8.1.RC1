#!/usr/bin/env python
# coding: utf-8
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
PSROIPoolingGradV2D
"""
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context


# 'pylint:disable=too-few-public-methods, invalid-name
class Constant(object):
    C0 = 16
    TILING_NUM = 64
    DIGIT_2 = 2
    DIGIT_4 = 4
    DIGIT_5 = 5
    DIGIT_8 = 8
    BLOCK_SIZE = 32
    DIGIT_64 = 64
    DIGIT_128 = 128
    RESERVE_NUM = 1024

    MAX_INT32 = (1 << 31) - 1
    POINT_1 = 0.1
    POINT_5 = 0.5
    UINT32 = "uint32"
    INT32 = "int32"
    FP32 = "float32"


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
        self.bin_all_dsize = None
        self.bin_c0_dsize = None


def _ceil_value(value, factor):
    """
    math: ceil(value/factor)
    """
    return (value + factor - 1) // factor


# 'pylint: disable=invalid-name, too-many-locals, too-many-arguments
# 'pylint: disable=too-many-instance-attributes, too-many-lines
class PSROIPoolingGradV2D(object):
    # 'pylint: disable= unused-argument
    def __init__(self, x_dict, rois_dict, y_dict, kernel_name):
        """
        constructor of PSROIPoolingGradV2DClass
        Parameters
        ----------
        x_dict: dict describes input fm, NC1HWC0
        rois_dict: dict describes input rois
        kernel_name: name of kernel
        -------
        """
        self.ub_size = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) - Constant.RESERVE_NUM
        self.tik_instance = tik.Tik()
        self.aicore_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)

        self.dtype = x_dict.get("dtype").lower()
        self.kernel_name = kernel_name
        self.dsize = Constant.DIGIT_4
        self.ub_one_buf = self.ub_size // Constant.DIGIT_4
        self.ub_one_buf_elem = self.ub_one_buf // self.dsize
        self.mask = Constant.DIGIT_64
        self.roi_step = Constant.DIGIT_64

        self.const_0_127_ub = None
        self.const_1_128_ub = None
        self._process_fun = None

        self.thread_num = None
        self._init_aicore_params()

    def compute(self):
        """
        op compute
        Returns self.tik_instance
        """
        self._get_tiling_params()

        with self.tik_instance.for_range(0,
                                         self.block_num,
                                         block_num=self.block_num) as block_id:
            # process of one aicore
            self._init_const_tensor()
            self._process_fun(block_id)

        tbe_context.get_context().add_compile_info(
            "vars", {"core_num": self.aicore_num})
        config = {"enable_const_fold": True}
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[self.x_gm, self.rois_gm],
                                   outputs=[self.y_gm],
                                   config=config,
                                   flowtable=[self.tiling_gm])
        return self.tik_instance

    def _init_aicore_params(self):
        self.x_gm = self.tik_instance.Tensor(self.dtype, shape=(Constant.MAX_INT32, ),
                                             name="x_gm", scope=tik.scope_gm)
        self.rois_gm = self.tik_instance.Tensor(self.dtype, shape=(Constant.MAX_INT32, ),
                                                name="rois_gm", scope=tik.scope_gm)
        self.y_gm = self.tik_instance.Tensor(self.dtype, shape=(Constant.MAX_INT32, ),
                                             name="y_gm", scope=tik.scope_gm,
                                             is_atomic_add=True)
        self.tiling_gm = self.tik_instance.Tensor(Constant.UINT32, (Constant.TILING_NUM, ),
                                                  scope=tik.scope_gm, name="tiling_gm")
        self.roi_num_b = self.tik_instance.Scalar(Constant.UINT32, name="roi_num_b")
        self.fm_batch = self.tik_instance.Scalar(Constant.UINT32, name="fm_batch")
        self.fm_c1 = self.tik_instance.Scalar(Constant.UINT32, name="fm_c1")
        self.fm_h = self.tik_instance.Scalar(Constant.INT32, name="fm_h")
        self.fm_w = self.tik_instance.Scalar(Constant.INT32, name="fm_w")
        self.fm_hw = self.tik_instance.Scalar(Constant.UINT32, name="fm_hw")
        self.x_c1 = self.tik_instance.Scalar(Constant.UINT32, name="x_c1")
        self.x_h = self.tik_instance.Scalar(Constant.UINT32, name="x_h")
        self.x_w = self.tik_instance.Scalar(Constant.UINT32, name="x_w")
        self.inner_c_size = self.tik_instance.Scalar(Constant.UINT32, name="inner_c_size")
        self.core_num_var = self.tik_instance.Scalar(Constant.UINT32, name="core_num_var")
        self.outer_loop = self.tik_instance.Scalar(Constant.UINT32, name="outer_loop")
        self.outer_tail = self.tik_instance.Scalar(Constant.UINT32, name="outer_tail")
        self.block_num = self.tik_instance.Scalar(Constant.UINT32, name="block_num")
        self.num1 = self.tik_instance.Scalar(Constant.UINT32, name="num1")
        self.num2 = self.tik_instance.Scalar(Constant.UINT32, name="num2")

        self.roi_loop1 = self.tik_instance.Scalar(Constant.UINT32, name="roi_loop1")
        self.roi_step1_l = self.tik_instance.Scalar(Constant.UINT32, name="roi_step1_l")
        self.roi_loop2 = self.tik_instance.Scalar(Constant.UINT32, name="roi_loop2")
        self.roi_step2_l = self.tik_instance.Scalar(Constant.UINT32, name="roi_step2_l")
        self.output_dim_align = self.tik_instance.Scalar(Constant.UINT32, name="output_dim_align")
        self.upload_repeat_times = self.tik_instance.Scalar(Constant.UINT32, name="upload_repeat_times")
        self.k2 = self.tik_instance.Scalar(Constant.UINT32, name="k2")
        self.output_dim = self.tik_instance.Scalar(Constant.UINT32, name="output_dim")
        self.group_size = self.tik_instance.Scalar(Constant.INT32, name="group_size")
        self.spatial_scale = self.tik_instance.Scalar(Constant.FP32, name="spatial_scale")

    def _init_const_tensor(self):
        self.const_0_127_ub = self.tik_instance.Tensor(self.dtype, (Constant.DIGIT_128, ),
                                                       name="const_0_127_ub",
                                                       scope=tbe_platform.scope_ubuf)
        with self.tik_instance.for_range(0, Constant.DIGIT_128) as i:
            self.const_0_127_ub[i].set_as(i)
        self.const_1_128_ub = self.tik_instance.Tensor(self.dtype, (Constant.DIGIT_128, ),
                                                       name="const_1_128_ub",
                                                       scope=tbe_platform.scope_ubuf)
        self.tik_instance.vadds(self.mask, self.const_1_128_ub, self.const_0_127_ub, 1.0,
                                Constant.DIGIT_2, 1, 1, Constant.DIGIT_8, Constant.DIGIT_8)

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

            self.tik_instance.data_move(tiling_ub,
                                        self.tiling_gm,
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
            self.inner_c_size.set_as(tiling_ub[6])
            self.x_c1.set_as(tiling_ub[7])
            self.x_h.set_as(tiling_ub[8])
            self.x_w.set_as(tiling_ub[9])
            self.core_num_var.set_as(tiling_ub[10])
            self.outer_loop.set_as(tiling_ub[11])
            self.outer_tail.set_as(tiling_ub[12])
            self.block_num.set_as(tiling_ub[13])
            self.num1.set_as(tiling_ub[14])
            self.num2.set_as(tiling_ub[15])
            self.roi_loop1.set_as(tiling_ub[16])
            self.roi_step1_l.set_as(tiling_ub[17])
            self.roi_loop2.set_as(tiling_ub[18])
            self.roi_step2_l.set_as(tiling_ub[19])
            self.output_dim_align.set_as(tiling_ub[20])
            self.upload_repeat_times.set_as(tiling_ub[21])
            self.k2.set_as(tiling_ub[22])
            self.output_dim.set_as(tiling_ub[23])
            self.group_size.set_as(tiling_ub[24])
            self.spatial_scale.set_as(tiling_ub[25])

        with self.tik_instance.if_scope(self.fm_batch == 1):
            self._process_fun = self._process_onebatch
        with self.tik_instance.else_scope():
            self._process_fun = self._process_mutilbatch

        with self.tik_instance.if_scope(self.output_dim <= 1):
            self.thread_num = 1
        with self.tik_instance.else_scope():
            self.thread_num = 2

    def _process_onebatch(self, block_id):
        # process roi nums: num1
        rois_num_offset = self.tik_instance.Scalar(Constant.UINT32,
                                                   name="rois_num_offset")
        with self.tik_instance.if_scope(block_id < self.outer_tail):
            # rois_num_offset is the offset in block_id aicore
            rois_num_offset.set_as(block_id * self.num1)
            self._process_rois(rois_num_offset, self.roi_loop1,
                               self.roi_step1_l)
        # process roi nums: num2
        with self.tik_instance.else_scope():
            with self.tik_instance.if_scope(self.outer_loop > 0):
                rois_num_offset.set_as(self.outer_tail * self.num1 +
                                       (block_id - self.outer_tail) *
                                       self.num2)
                self._process_rois(rois_num_offset, self.roi_loop2,
                                   self.roi_step2_l)

    def _process_mutilbatch(self, block_id):
        rois_num_offset = self.tik_instance.Scalar(Constant.UINT32,
                                                   name="rois_num_offset")
        # process roi nums: num1*fm_batch
        with self.tik_instance.if_scope(block_id < self.outer_tail):
            # rois_num_offset is the offset in block_id aicore
            with self.tik_instance.for_range(0, self.fm_batch) as batch_id:
                rois_num_offset.set_as(block_id * self.num1)
                self._process_rois_multi_batch(self.roi_step, rois_num_offset,
                                               self.roi_loop1,
                                               self.roi_step1_l, batch_id)
        # process roi nums: num2*fm_batch
        with self.tik_instance.else_scope():
            with self.tik_instance.if_scope(self.outer_loop > 0):
                with self.tik_instance.for_range(0, self.fm_batch) as batch_id:
                    rois_num_offset.set_as(self.outer_tail * self.num1 +
                                           (block_id - self.outer_tail) *
                                           self.num2)
                    self._process_rois_multi_batch(self.roi_step,
                                                   rois_num_offset,
                                                   self.roi_loop2,
                                                   self.roi_step2_l, batch_id)

    def _load_rois_to_ub(self, rois_ub, rois_offset, roi_step):
        """
        load rois data to ub from gm.

        Parameters
        ----------
        rois_ub: a tensor, which store rois data
        rois_offset: the roi offset of current loop in block_id aicore
        roi_step: the number of rois per loop in process, 64(fp32) 

        Returns
        -------
        None
        """
        burst_len = roi_step * self.dsize // Constant.BLOCK_SIZE
        with self.tik_instance.for_range(0, Constant.DIGIT_5) as i:
            self.tik_instance.data_move(rois_ub[i, 0],
                                        self.rois_gm[rois_offset +
                                                     i * self.roi_num_b],
                                        sid=0,
                                        nburst=1,
                                        burst=burst_len,
                                        src_stride=0,
                                        dst_stride=0)

    def _spatial_scale_rois(self, rois_ub, rois_round_ub, rois_spatial_ub,
                            roi_step):
        """
        compute the width and height of rois and bin.

        Parameters
        ----------
        rois_ub: input rois data in ub, (5, roi_step).
            batch_id,batch_id,batch_id...
            x1,x1,x1...
            y1,y1,y1...
            x2,x2,x2...
            y2,y2,y2...
        rois_round_ub: store rois data of convert to s32
        rois_spatial_ub: store the width and height of rois and bin in ub
        roi_step: the number of rois per loop in process, 64(fp32)

        Returns
        -------
        None
        """
        # rois_round_ub[0]: batch id; rois_round_ub[1-4]: roi coordinates
        self.tik_instance.vconv(self.mask, 'round', rois_round_ub, rois_ub,
                                Constant.DIGIT_5, 1, 1, Constant.DIGIT_8,
                                Constant.DIGIT_8)

        self.tik_instance.vconv(self.mask, '', rois_spatial_ub,
                                rois_round_ub[1, 0], Constant.DIGIT_4, 1, 1,
                                Constant.DIGIT_8, Constant.DIGIT_8)
        self.tik_instance.vadds(self.mask, rois_spatial_ub[2, 0],
                                rois_spatial_ub[2, 0], 1.0, Constant.DIGIT_2,
                                1, 1, Constant.DIGIT_8, Constant.DIGIT_8)
        # multiply spatial
        self.tik_instance.vmuls(self.mask,
                                rois_spatial_ub,
                                rois_spatial_ub,
                                self.spatial_scale,
                                repeat_times=Constant.DIGIT_4,
                                dst_blk_stride=1,
                                src_blk_stride=1,
                                dst_rep_stride=Constant.DIGIT_8,
                                src_rep_stride=Constant.DIGIT_8)
        # roi width and height: roi_end_w-roi_start_w, roi_end_h-roi_start_h
        self.tik_instance.vsub(self.mask, rois_spatial_ub[4, 0],
                               rois_spatial_ub[2, 0], rois_spatial_ub,
                               Constant.DIGIT_2, 1, 1, 1, Constant.DIGIT_8,
                               Constant.DIGIT_8, Constant.DIGIT_8)
        const_ub = self.tik_instance.Tensor(self.dtype, (roi_step, ),
                                            name="const_ub",
                                            scope=tbe_platform.scope_ubuf)
        self.tik_instance.vector_dup(self.mask, const_ub, Constant.POINT_1,
                                     repeat_times=1, dst_blk_stride=1,
                                     dst_rep_stride=Constant.DIGIT_8)
        self.tik_instance.vmax(self.mask,
                               rois_spatial_ub[4, 0],
                               rois_spatial_ub[4, 0],
                               const_ub,
                               repeat_times=Constant.DIGIT_2,
                               dst_blk_stride=1,
                               src0_blk_stride=1,
                               src1_blk_stride=1,
                               dst_rep_stride=Constant.DIGIT_8,
                               src0_rep_stride=Constant.DIGIT_8,
                               src1_rep_stride=0)

        fp_scalar = self.tik_instance.Scalar(self.dtype, name="fp_scalar", init_value=self.group_size)
        self.tik_instance.vector_dup(self.mask, const_ub, fp_scalar,
                                     repeat_times=1, dst_blk_stride=1,
                                     dst_rep_stride=Constant.DIGIT_8)
        self.tik_instance.vdiv(self.mask, rois_spatial_ub[6, :],
                               rois_spatial_ub[4, :], const_ub, 1, 1, 1,
                               1, Constant.DIGIT_8, Constant.DIGIT_8, 0)
        self.tik_instance.vdiv(self.mask, rois_spatial_ub[7, :],
                               rois_spatial_ub[5, :], const_ub, 1, 1, 1,
                               1, Constant.DIGIT_8, Constant.DIGIT_8, 0)

    def _process_one_area_whole(self, stream):

        burst_len_s = self.tik_instance.Scalar(Constant.UINT32,
                                               name="burst_len_s")
        burst_len_s.set_as(stream.w_width * Constant.C0 * self.dsize //
                           Constant.BLOCK_SIZE)
        burst_one_s = self.tik_instance.Scalar(Constant.UINT32,
                                               name="burst_one_s")
        burst_one_s.set_as(Constant.C0 * self.dsize // Constant.BLOCK_SIZE)
        dst_stride_s = self.tik_instance.Scalar(Constant.UINT32,
                                                name="dst_stride_s")
        dst_stride_s.set_as((self.fm_w - stream.w_width) * Constant.C0 *
                            self.dsize // Constant.BLOCK_SIZE)
        ub_output_dim = stream.ub_output_dim
        diff_value = self.tik_instance.Scalar(self.dtype, name="diff_value")

        with self.tik_instance.for_range(0, self.output_dim,
                                         thread_num=self.thread_num) as out_dim:
            ub_bin_input_buf = self.tik_instance.Tensor(
                self.dtype, (self.ub_one_buf_elem, ),
                name="ub_bin_input_buf",
                scope=tbe_platform.scope_ubuf)
            diff_value.set_as(ub_output_dim[out_dim])
            output_dim_index = out_dim * self.k2 + stream.bin_i_offset
            self.tik_instance.vector_dup(
                self.mask, ub_bin_input_buf, 0.,
                _ceil_value(stream.h_width * stream.w_width * Constant.C0,
                            self.mask), 1, Constant.DIGIT_8)
            bin_i_offset_c1 = output_dim_index // Constant.C0
            bin_i_offset_c0 = output_dim_index - bin_i_offset_c1 * Constant.C0

            # add diff_val
            self.tik_instance.vadds([0, 0x1], ub_bin_input_buf,
                                    ub_bin_input_buf, diff_value,
                                    stream.h_width * stream.w_width, 1, 1,
                                    burst_one_s, burst_one_s)
            y_offset = (((stream.scalar_roi_batch_id * self.fm_c1 + bin_i_offset_c1) *
                         self.fm_h + stream.h_start) * self.fm_w + stream.w_start) * Constant.C0 + bin_i_offset_c0
            self._download(src_ub=ub_bin_input_buf, dst_offset=y_offset,
                           nburst=stream.h_width, burst=burst_len_s, dst_stride=dst_stride_s)

    # 'pylint: disable=too-many-arguments
    def _download(self, src_ub, dst_offset, nburst, burst, dst_stride):
        self.tik_instance.set_atomic_add(1)
        self.tik_instance.data_move(self.y_gm[dst_offset],
                                    src_ub,
                                    sid=0,
                                    nburst=nburst,
                                    burst=burst,
                                    src_stride=0,
                                    dst_stride=dst_stride)
        self.tik_instance.set_atomic_add(0)

    def _process_one_area_every_height(self, stream):

        burst_len_s = self.tik_instance.Scalar(Constant.UINT32,
                                               name="burst_len_s")
        burst_len_s.set_as(stream.w_width * Constant.C0 * self.dsize //
                           Constant.BLOCK_SIZE)
        burst_one_s = self.tik_instance.Scalar(Constant.UINT32,
                                               name="burst_one_s")
        burst_one_s.set_as(Constant.C0 * self.dsize // Constant.BLOCK_SIZE)
        dst_stride_s = self.tik_instance.Scalar(Constant.UINT32,
                                                name="dst_stride_s")
        dst_stride_s.set_as((self.fm_w - stream.w_width) * Constant.C0 * self.dsize // Constant.BLOCK_SIZE)
        ub_output_dim = stream.ub_output_dim
        diff_value = self.tik_instance.Scalar(self.dtype, name="diff_value", init_value=0)

        with self.tik_instance.for_range(0, self.output_dim,
                                         thread_num=self.thread_num) as out_dim:
            ub_bin_input_buf = self.tik_instance.Tensor(
                self.dtype, (self.ub_one_buf_elem, ),
                name="ub_bin_input_buf",
                scope=tbe_platform.scope_ubuf)
            diff_value.set_as(ub_output_dim[out_dim])
            output_dim_index = out_dim * self.k2 + stream.bin_i_offset
            bin_i_offset_c1 = output_dim_index // Constant.C0
            bin_i_offset_c0 = output_dim_index - bin_i_offset_c1 * Constant.C0
            self.tik_instance.vector_dup(
                self.mask, ub_bin_input_buf, 0.,
                _ceil_value(stream.h_width * stream.w_width * Constant.C0,
                            self.mask), 1, Constant.DIGIT_8)
            with self.tik_instance.for_range(stream.h_start,
                                             stream.h_end) as height:
                # add diff_val
                self.tik_instance.vadds([0, 0x1], ub_bin_input_buf,
                                        ub_bin_input_buf, diff_value,
                                        stream.h_width * stream.w_width, 1, 1,
                                        burst_one_s, burst_one_s)
                y_offset = (((stream.scalar_roi_batch_id * self.fm_c1 + bin_i_offset_c1) * self.fm_h +
                            height) * self.fm_w + stream.w_start) * Constant.C0 + bin_i_offset_c0
                self._download(src_ub=ub_bin_input_buf, dst_offset=y_offset,
                               nburst=stream.h_width, burst=burst_len_s, dst_stride=dst_stride_s)

    def _process_one_bin_2(self, stream):
        """
        process one bin of roi: inner_c1 == 1, or
                                (bin_all_dsize > self.ub_one_buf, or
                                 bin_load_stride > 65536)
        Parameters
        ----------
        stream: param is a struct, contains multiple keys

        Returns
        -------
        None
        """
        with self.tik_instance.if_scope(stream.bin_area > 0):
            bin_size = self.tik_instance.Scalar(Constant.UINT32,
                                                name="bin_size")
            bin_size.set_as(stream.w_width * stream.h_width * Constant.C0)
            # bin_size is small than 1024
            with self.tik_instance.if_scope(bin_size < self.ub_one_buf_elem):
                self._process_one_area_whole(stream)
            with self.tik_instance.else_scope():
                self._process_one_area_every_height(stream)

    def _get_bin_area_inv(self, stream):
        bin_area_inv = self.tik_instance.Scalar(self.dtype,
                                                name="bin_area_inv")
        with self.tik_instance.new_stmt_scope():

            dst_scalar = self.tik_instance.Scalar(self.dtype,
                                                  name="dst_scalar",
                                                  init_value=0.0)
            self.tik_instance.scalar_conv('', dst_scalar, stream.bin_area)
            bin_area_inv.set_as(1.0 / dst_scalar)

        return bin_area_inv

    def _process_one_bin(self, stream):
        """
        process one bin of roi.

        Parameters
        ----------
        stream: param is a struct, contains multiple keys

        Returns
        -------
        None
        """
        ub_output_dim = self.tik_instance.Tensor(self.dtype,
                                                 (self.output_dim_align, ),
                                                 name="ub_output_dim",
                                                 scope=tbe_platform.scope_ubuf)
        self.tik_instance.vector_dup(self.mask, ub_output_dim, 0.,
                                     repeat_times=self.output_dim_align // self.mask,
                                     dst_blk_stride=1,
                                     dst_rep_stride=Constant.DIGIT_8)
        burst_len = Constant.C0 * self.dsize // Constant.BLOCK_SIZE

        # move feature map from gm to ub
        x_offset = (((stream.cur_roi_output_offset * self.x_c1) * self.x_h +
                     stream.row) * self.x_w + stream.col) * Constant.C0
        self.tik_instance.data_move(ub_output_dim,
                                    self.x_gm[x_offset],
                                    sid=0,
                                    nburst=self.upload_repeat_times,
                                    burst=burst_len,
                                    src_stride=(self.k2 - 1) * burst_len,
                                    dst_stride=0)
        with self.tik_instance.if_scope(stream.bin_area == 0):
            self.tik_instance.vmuls(Constant.C0, ub_output_dim, ub_output_dim,
                                    0., self.upload_repeat_times, 1, 1, burst_len,
                                    burst_len)
        with self.tik_instance.else_scope():
            bin_area_inv = self._get_bin_area_inv(stream)
            self.tik_instance.vmuls(Constant.C0, ub_output_dim, ub_output_dim,
                                    bin_area_inv, self.upload_repeat_times, 1, 1, burst_len,
                                    burst_len)
        stream.ub_output_dim = ub_output_dim

        self._process_one_bin_2(stream)

    def _process_one_roi(self, stream):
        """
        process one roi.

        Parameters
        ----------
        params: param is a dict, contains multiple keys

        Returns
        -------
        None
        """
        with self.tik_instance.for_range(0, self.group_size) as row:
            # h coordinates of bin
            h_start = self.tik_instance.Scalar(Constant.INT32, name="h_start")
            h_start.set_as(stream.bin_start_h_floor[row])
            h_end = self.tik_instance.Scalar(Constant.INT32, name="h_end")
            h_end.set_as(stream.bin_end_h_ceil[row])
            stream.row = row
            stream.h_start = h_start
            stream.h_end = h_end
            h_width = self.tik_instance.Scalar(Constant.INT32, name="h_width")
            h_width.set_as(h_end - h_start)
            with self.tik_instance.if_scope(h_end <= h_start):
                h_width.set_as(0)
            stream.h_width = h_width
            with self.tik_instance.for_range(0, self.group_size) as col:
                # w coordinates of bin
                w_start = self.tik_instance.Scalar(Constant.INT32, name="w_start")
                w_start.set_as(stream.bin_start_w_floor[col])
                w_end = self.tik_instance.Scalar(Constant.INT32, name="w_end")
                w_end.set_as(stream.bin_end_w_ceil[col])

                stream.col = col
                stream.w_start = w_start
                stream.w_end = w_end

                bin_i_offset = self.tik_instance.Scalar(Constant.INT32,
                                                        name="bin_i_offset")
                # bin_i offset of in roi, 0~(group_size^2-1)
                bin_i_offset.set_as(row * self.group_size + col)
                stream.bin_i_offset = bin_i_offset % self.k2
                w_width = self.tik_instance.Scalar(Constant.INT32, name="w_width")
                bin_area = self.tik_instance.Scalar(Constant.INT32, name="bin_area")
                w_width.set_as(w_end - w_start)
                with self.tik_instance.if_scope(w_end <= w_start):
                    w_width.set_as(0)
                bin_area.set_as(w_width * h_width)
                stream.w_width = w_width
                stream.bin_area = bin_area

                bin_all_dsize = self.tik_instance.Scalar(Constant.INT32, name="bin_all_dsize")
                bin_all_dsize.set_as(bin_area * self.inner_c_size)
                bin_c0_dsize = self.tik_instance.Scalar(Constant.INT32, name="bin_c0_dsize")
                bin_c0_dsize.set_as(bin_area * Constant.C0 * self.dsize)
                stream.bin_all_dsize = bin_all_dsize
                stream.bin_c0_dsize = bin_c0_dsize

                self._process_one_bin(stream)

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
        self.tik_instance.vmuls(self.mask, dst_float, a, x,
                                repeat_times=Constant.DIGIT_2, dst_blk_stride=1, src_blk_stride=1,
                                dst_rep_stride=Constant.DIGIT_8, src_rep_stride=Constant.DIGIT_8)

        self.tik_instance.vadds(self.mask, dst_float, dst_float, b,
                                repeat_times=Constant.DIGIT_2, dst_blk_stride=1, src_blk_stride=1,
                                dst_rep_stride=Constant.DIGIT_8, src_rep_stride=Constant.DIGIT_8)
        self.tik_instance.vconv(self.mask, mode, dst_int, dst_float,
                                repeat_times=Constant.DIGIT_2, dst_blk_stride=1, src_blk_stride=1,
                                dst_rep_stride=Constant.DIGIT_8, src_rep_stride=Constant.DIGIT_8)

    def _data_range_limit(self, minv, maxv, data):
        """
        data = min(max(data,minv),maxv)
        """
        self.tik_instance.vmax(self.mask, data, data, minv,
                               Constant.DIGIT_2,
                               1, 1, 1, Constant.DIGIT_8, Constant.DIGIT_8, 0)
        self.tik_instance.vmin(self.mask, data, data, maxv,
                               Constant.DIGIT_2,
                               1, 1, 1, Constant.DIGIT_8, Constant.DIGIT_8, 0)

    def _data_limit(self, stream):
        minv_ub = self.tik_instance.Tensor(Constant.INT32, (Constant.DIGIT_128, ),
                                           name="minv_ub", scope=tbe_platform.scope_ubuf)
        self.tik_instance.vector_dup(Constant.DIGIT_64, minv_ub, 0,
                                     repeat_times=Constant.DIGIT_2,
                                     dst_blk_stride=1,
                                     dst_rep_stride=Constant.DIGIT_8)
        maxv_ub = self.tik_instance.Tensor(Constant.INT32, (Constant.DIGIT_128, ),
                                           name="maxv_ub", scope=tbe_platform.scope_ubuf)
        self.tik_instance.vector_dup(Constant.DIGIT_64, maxv_ub, self.fm_w,
                                     repeat_times=Constant.DIGIT_2,
                                     dst_blk_stride=1, dst_rep_stride=Constant.DIGIT_8)
        self._data_range_limit(minv=minv_ub, maxv=maxv_ub, data=stream.bin_start_w_floor)
        self._data_range_limit(minv=minv_ub, maxv=maxv_ub, data=stream.bin_end_w_ceil)

        self.tik_instance.vector_dup(Constant.DIGIT_64, maxv_ub, self.fm_h,
                                     repeat_times=Constant.DIGIT_2,
                                     dst_blk_stride=1,
                                     dst_rep_stride=Constant.DIGIT_8)
        self._data_range_limit(minv=minv_ub, maxv=maxv_ub, data=stream.bin_start_h_floor)
        self._data_range_limit(minv=minv_ub, maxv=maxv_ub, data=stream.bin_end_h_ceil)

    def _process_step1_roi(self, rois_round_ub, rois_spatial_ub,
                           rois_num_offset, step_i_offset, step_i_num):
        """
        process roi_num roi cyclically, and process roi_step roi each time.
        Parameters
        ----------
        rois_round_ub: rois data, s32
        rois_spatial_ub: a tensor, the width and height of rois and bin
        step_i_offset: the roi offset of this loop in block_id aicore
        rois_num_offset: a Scalar, the offset in block_id aicore
        step_i_num: the number of rois of one loop in process

        Returns None
        """
        with self.tik_instance.for_range(0, step_i_num) as roi_i:
            stream = StreamParam()

            scalar_roi_batch_id = self.tik_instance.Scalar(Constant.INT32, name="scalar_roi_batch_id")
            scalar_roi_batch_id.set_as(rois_round_ub[0, roi_i])
            stream.scalar_roi_batch_id = scalar_roi_batch_id
            cur_roi_output_offset = self.tik_instance.Scalar(Constant.INT32, name="cur_roi_output_offset")
            cur_roi_output_offset.set_as(rois_num_offset + step_i_offset + roi_i)
            stream.cur_roi_output_offset = cur_roi_output_offset
            scalar_bin_width = self.tik_instance.Scalar(
                self.dtype, name="scalar_bin_width")
            scalar_bin_width.set_as(rois_spatial_ub[6, roi_i])
            scalar_bin_height = self.tik_instance.Scalar(
                self.dtype, name="scalar_bin_height")
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

            scalar_roi_start_w = self.tik_instance.Scalar(self.dtype, name="scalar_roi_start_w")
            scalar_roi_start_w.set_as(rois_spatial_ub[0, roi_i])
            scalar_roi_start_h = self.tik_instance.Scalar(self.dtype, name="scalar_roi_start_h")
            scalar_roi_start_h.set_as(rois_spatial_ub[1, roi_i])
            dst_float = self.tik_instance.Tensor(self.dtype, (Constant.DIGIT_128, ),
                                                 name="dst_float",
                                                 scope=tbe_platform.scope_ubuf)
            # scalar_roi_start_w + scalar_bin_width*(0...127)
            self._a_mul_x_add_b(self.const_0_127_ub, scalar_bin_width,
                                scalar_roi_start_w, dst_float, bin_start_w_floor, 'floor')
            # scalar_roi_start_w + scalar_bin_width*(1...128)
            self._a_mul_x_add_b(self.const_1_128_ub, scalar_bin_width,
                                scalar_roi_start_w, dst_float, bin_end_w_ceil, 'ceil')

            # scalar_roi_start_h + scalar_bin_height*(0...127)
            self._a_mul_x_add_b(self.const_0_127_ub, scalar_bin_height,
                                scalar_roi_start_h, dst_float, bin_start_h_floor, 'floor')
            self._a_mul_x_add_b(self.const_1_128_ub, scalar_bin_height,
                                scalar_roi_start_h, dst_float, bin_end_h_ceil, 'ceil')

            stream.bin_start_h_floor = bin_start_h_floor
            stream.bin_end_h_ceil = bin_end_h_ceil
            stream.bin_start_w_floor = bin_start_w_floor
            stream.bin_end_w_ceil = bin_end_w_ceil

            self._data_limit(stream)
            self._process_one_roi(stream)

    def _process_rois(self, rois_num_offset, roi_loop, roi_step_l):
        """
        process roi_num roi cyclically, and process roi_step roi each time.

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
            rois_ub = self.tik_instance.Tensor(
                self.dtype, (Constant.DIGIT_5, self.roi_step),
                name="rois_ub",
                scope=tbe_platform.scope_ubuf)
            rois_offset = self.tik_instance.Scalar(Constant.INT32,
                                                   "rois_offset")
            rois_offset.set_as(rois_num_offset + self.roi_step * inner_i)
            # move rois data to ub from gm
            self._load_rois_to_ub(rois_ub, rois_offset, self.roi_step)

            # calculate spatial scale rois
            rois_round_ub = self.tik_instance.Tensor(
                Constant.INT32, (Constant.DIGIT_5, self.roi_step),
                name="rois_round_ub",
                scope=tbe_platform.scope_ubuf)
            rois_spatial_ub = self.tik_instance.Tensor(
                self.dtype, (Constant.DIGIT_8, self.roi_step),
                name="rois_spatial_ub",
                scope=tbe_platform.scope_ubuf)
            self._spatial_scale_rois(rois_ub, rois_round_ub, rois_spatial_ub,
                                     self.roi_step)

            with self.tik_instance.if_scope(inner_i == (roi_loop - 1)):
                self._process_step1_roi(rois_round_ub, rois_spatial_ub,
                                        rois_num_offset,
                                        self.roi_step * inner_i, roi_step_l)
            with self.tik_instance.else_scope():
                self._process_step1_roi(rois_round_ub, rois_spatial_ub,
                                        rois_num_offset,
                                        self.roi_step * inner_i, self.roi_step)

    def _process_rois_multi_batch(self, roi_step, rois_num_offset, roi_loop,
                                  roi_step_l, batch_id):
        """
        process roi_num roi cyclically, and process roi_step roi each time.

        Parameters
        ----------
        roi_step: the number of rois per loop in process, 64(fp32)
        rois_num_offset: the offset in block_id aicore
        roi_loop: loops of processing roi_num rois
        roi_step_l: the number of rois in last loop
        batch_id: batch id

        Returns
        -------
        None
        """
        with self.tik_instance.for_range(0, roi_loop) as inner_i:
            rois_ub = self.tik_instance.Tensor(self.dtype,
                                               (Constant.DIGIT_5, roi_step),
                                               name="rois_ub",
                                               scope=tbe_platform.scope_ubuf)
            # rois addr offset
            rois_offset = self.tik_instance.Scalar(Constant.INT32,
                                                   name="rois_offset")
            rois_offset.set_as(rois_num_offset + batch_id *
                               (self.roi_num_b * Constant.DIGIT_5) +
                               roi_step * inner_i)
            # move rois data to ub from gm
            self._load_rois_to_ub(rois_ub, rois_offset, roi_step)

            # calculate spatial scale rois
            rois_round_ub = self.tik_instance.Tensor(Constant.INT32, (Constant.DIGIT_5, roi_step),
                                                     name="rois_round_ub",
                                                     scope=tbe_platform.scope_ubuf)
            rois_spatial_ub = self.tik_instance.Tensor(self.dtype, (Constant.DIGIT_8, roi_step),
                                                       name="rois_spatial_ub",
                                                       scope=tbe_platform.scope_ubuf)

            self._spatial_scale_rois(rois_ub, rois_round_ub, rois_spatial_ub, roi_step)
            with self.tik_instance.if_scope(inner_i == (roi_loop - 1)):
                self._process_step1_roi(rois_round_ub, rois_spatial_ub,
                                        rois_num_offset + self.roi_num_b * batch_id,
                                        roi_step * inner_i, roi_step_l)
            with self.tik_instance.else_scope():
                self._process_step1_roi(rois_round_ub, rois_spatial_ub,
                                        rois_num_offset + self.roi_num_b * batch_id,
                                        roi_step * inner_i, roi_step)


# 'pylint: disable= unused-argument
@register_operator("PSROIPoolingGradV2D")
def psroipooling_grad_v2d(x_dict,
                          rois_dict,
                          y_dict,
                          spatial_scale,
                          output_dim,
                          group_size,
                          input_size,
                          kernel_name="PSROIPoolingGradV2D"):
    """
    PSROIPoolingGradV2D interface.

    Parameters
    ----------
    x_dict: feature map size and data type, 5HD
    rois_dict: rois_dict size and data type, (batch, 5, rois_num), rois all
                nums is batch*rois_num
    y_dict: output size and data type, 5HD
    output_dim: number of output channels
    group_size: number of groups encoding position sensitive score maps
    spatial_scale: spatial scale
    input_size: grad input size (h, w)
    kernel_name: kernel name of PSROIPoolingGradV2D op

    Returns
    -------
    tik_instance
    """
    psroi_instance = PSROIPoolingGradV2D(x_dict, rois_dict, y_dict, kernel_name)
    return psroi_instance.compute()
