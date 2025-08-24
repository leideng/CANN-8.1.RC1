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
resize_bilinear_v2.py
"""
from impl.util import util_tik_comm_func
from impl.util.platform_adapter import register_operator
from impl.util.util_tik_comm_func import OpBase
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe_context
from tbe.common.platform import get_bit_len


# 'pylint:disable=too-few-public-methods,too-many-instance-attributes
class Constant(object):
    """
    The class for constant
    """
    # max uint16
    MAX_UINT16 = 2 ** 16 - 1
    # ting param num
    TILING_ARG_NUM = 16
    # reserved ub size
    RESERVED_UB_SIZE = 8 * 1024
    # the num of assist_gm size(0, 1, 2, ...., 255)
    ASSIST_NUM = 256


# 'pylint: disable=invalid-name,too-many-locals,too-many-arguments,too-many-statements,too-many-instance-attributes
class SyncResizeBilinearV2(OpBase):
    """
    Function: use to store SyncResizeBilinearV2 base parameters
    Modify: 2021-10-20
    """
    # 'pylint: disable=unused-argument
    def __init__(self, images, size, y, align_corners,
                 half_pixel_centers, kernel_name):

        OpBase.__init__(self)
        self.is_bilinear = True
        self.images_dtype = images.get("dtype").lower()
        self.output_dtype = y.get("dtype").lower()
        self.inner_dtype = "float32"
        self.size_dtype = size.get("dtype").lower()
        self.align_corners = align_corners
        self.half_pixel_centers = half_pixel_centers

        # check dtype
        para_check.check_dtype(self.size_dtype, ("int64", "int32"), param_name="size")
        para_check.check_dtype(self.images_dtype, ("float32", "float16"), param_name="images")

        self.kernel_name = kernel_name
        self.ub_size_bytes = self.ub_size_bytes - Constant.RESERVED_UB_SIZE

        self.block_num = 16 if self.inner_dtype in ("float16",) else 8
        self.vector_num = self.block_num * 8
        self.input_block_num = 16 if self.images_dtype in ("float16",) else 8
        self.output_block_num = 16 if self.output_dtype in ("float16",) else 8
        self.block_num = 16 if self.inner_dtype in ("float16",) else 8
        self.ub_max_num = self.ub_size_bytes // 32 // 2 * self.block_num
        self.inner_bytes_size = get_bit_len(self.inner_dtype) // 8
        self.input_bytes_size = get_bit_len(self.images_dtype) // 8

        self.images_shape_c0 = 16
        self.height_idx_segment_num = 512
        self.width_idx_segment_num = 512
        # init gm addr
        tiling_dict = {"dtype": "int64", "shape": (Constant.TILING_ARG_NUM,)}
        self.op_init_gm([images, size], [y], tiling_info=tiling_dict, is_fused_1d=True)
        self.images_gm, self.size_gm = self.input_gm_list
        self.out_gm = self.output_gm_list[0]

        # gen assist ub for [0, 1, 2, ...., 255]
        assist_value = list(range(Constant.ASSIST_NUM))
        self.assist_gm = self.tik_instance.Tensor("float32", (Constant.ASSIST_NUM,),
                                                  name="assist_gm",
                                                  scope=tik.scope_gm,
                                                  init_value=assist_value)

        self.stride_threshold = Constant.MAX_UINT16 if self.images_dtype in ("float16",) else Constant.MAX_UINT16 // 2
        self.dst_stride_threshold = Constant.MAX_UINT16 if self.output_dtype == "float16" else Constant.MAX_UINT16 // 2
        self.is_suport_vdiv = tbe_platform.api_check_support("tik.vdiv", "float32")
        self._init_tiling_params()

        # init scaler for each core
        # nc1 start addr offset for per core
        self.core_nc_start = self.tik_instance.Scalar("int64", name="core_nc_start")
        # h start addr offset for per core
        self.core_height_start = self.tik_instance.Scalar("int64", name="core_height_start")
        # w start addr offset for per core
        self.core_width_start = self.tik_instance.Scalar("int64", name="core_width_start")
        # nc1 process len for per core
        self.core_nc_num = self.tik_instance.Scalar("int64", name="core_nc_num")
        # h process len for per core
        self.core_height_num = self.tik_instance.Scalar("int64", name="core_height_num")
        # w process len for per core
        self.core_width_num = self.tik_instance.Scalar("int64", name="core_width_num")
        self.cut_width_num = None
        self.cut_height_num = None

        # init stride scalar flag
        self.scalar_is_src_stride = self.tik_instance.Scalar("int32", name="scalar_is_src_stride", init_value=1)
        self.scalar_is_dst_stride = self.tik_instance.Scalar("int32", name="scalar_is_dst_stride", init_value=1)

        # init ub
        self.height_idx_ub = None
        self.width_idx_ub = None
        self.idx_ub_fp32 = None
        self.idx_cb_fp32 = None
        self.image_out_ub = None
        self.image_in_cb_ping = None
        self.image_out_ub = None
        self.image_in_cb_ping = None

    def tiling_args(self):
        """
        init tiling_args
        """
        with self.tik_instance.new_stmt_scope():
            tiling_ub = self.tik_instance.Tensor("int64", (Constant.TILING_ARG_NUM,), name="tiling_ub",
                                                 scope=tik.scope_ubuf)
            src_start_w_int32 = self.tik_instance.Scalar("int32", name="src_start_w_int32")
            dst_start_w_int32 = self.tik_instance.Scalar("int32", name="dst_start_w_int32")
            self.tik_instance.data_move(tiling_ub, self.tiling_gm, 0, 1, (Constant.TILING_ARG_NUM + 3) // 4, 0, 0)
            self.tiling_key.set_as(tiling_ub[0])
            self.tiling_batch.set_as(tiling_ub[1])
            self.tiling_c1.set_as(tiling_ub[2])
            self.tiling_in_height.set_as(tiling_ub[3])
            self.tiling_in_width.set_as(tiling_ub[4])
            self.tiling_out_height.set_as(tiling_ub[5])
            self.tiling_out_width.set_as(tiling_ub[6])
            self.tiling_bc1_cut_num.set_as(tiling_ub[7])
            self.tiling_height_cut_num.set_as(tiling_ub[8])
            self.tiling_width_cut_num.set_as(tiling_ub[9])
            self.ori_h.set_as(tiling_ub[10])
            self.ori_w.set_as(tiling_ub[11])

            src_start_w_int32.set_as(tiling_ub[12])
            dst_start_w_int32.set_as(tiling_ub[13])
            self.tik_instance.scalar_conv('none', dst=self.src_start_w, src=src_start_w_int32)
            self.tik_instance.scalar_conv('none', dst=self.dst_start_w, src=dst_start_w_int32)
            # calcu stride scalar flag
            with self.tik_instance.if_scope(self.tiling_in_height * self.tiling_in_width > self.stride_threshold):
                self.scalar_is_src_stride.set_as(0)
            with self.tik_instance.if_scope(
                    self.tiling_out_height * self.tiling_out_width > self.dst_stride_threshold):
                self.scalar_is_dst_stride.set_as(0)

    def core_scedule_args(self, core_idx):
        """
        get runtime tiling parameters from tiling data with core_id

        need_input1: image info -->
                   tiling_batch*tiling_c1 tiling_in_height tiling_in_width tiling_out_height tiling_out_width
        need_input2: cut core info ---> tiling_bc1_cut_num tiling_height_cut_num tiling_width_cut_num
        output: the process info for each core -->
                   self.core_nc_start/self.core_nc_num
                   self.core_height_start/self.core_height_num
                   self.core_width_start/self.core_width_num

        proc:
            core_nc_num = (batch*c1 + bc1_cut_num - 1) // bc1_cut_num
            core_nc_start = (core_id // (height_cut_num * width_cut_num)) * core_nc_num
            core_height_num = (height + height_cut_num - 1) // height_cut_num
            core_height_start = ((core_id % (height_cut_num * width_cut_num)) // width_cut_num) * core_height_num
            core_width_num = (width + width_cut_num - 1) // width_cut_num
            core_width_start = ((core_id % (height_cut_num * width_cut_num)) // width_cut_num) * core_width_num

            for example:
                input info:
                    16, 2, 32, 32, 16 resize to 16, 2, 64, 64, 16     h from 32->64 w from 32->64
                cut info: tiling_bc1_cut_num, tiling_height_cut_num, tiling_width_cut_num
                    4, 4, 2

                core_nc_num = ceil(32, 4) = 8
                core_nc_start = (core_idx // (4*2)) * core_nc_num
                   ---> 0 <= core_idx < 8  core_nc_start = 0
                   ---> 8 <= core_idx < 16  core_nc_start = 8
                   ---> 16 <= core_idx < 24  core_nc_start = 16
                   ---> 24 <= core_idx < 32  core_nc_start = 24
        """
        # h process len for per core
        self.cut_height_num = self.tik_instance.Scalar("int64", name="cut_height_num")
        # w process len for per core
        self.cut_width_num = self.tik_instance.Scalar("int64", name="cut_width_num")
        self.cut_height_num.set_as(self.tiling_out_height)
        self.cut_width_num.set_as(self.tiling_out_width)
        with self.tik_instance.if_scope(self.tiling_key == 111000):
            # when tiling_key is 111000, will cut by input
            self.cut_height_num.set_as(self.tiling_in_height)
            self.cut_width_num.set_as(self.tiling_in_width)
        with self.tik_instance.if_scope(self.tiling_key == 101000):
            self.cut_width_num.set_as(self.tiling_in_width)

        # fix the core cut num
        # fix for height_cut_num
        self.tiling_height_cut_num.set_as(
            (self.cut_height_num + self.tiling_height_cut_num - 1) // self.tiling_height_cut_num)
        self.tiling_height_cut_num.set_as(
            (self.cut_height_num + self.tiling_height_cut_num - 1) // self.tiling_height_cut_num)
        # fix for width_cut_num
        self.tiling_width_cut_num.set_as(
            (self.cut_width_num + self.tiling_width_cut_num - 1) // self.tiling_width_cut_num)
        self.tiling_width_cut_num.set_as(
            (self.cut_width_num + self.tiling_width_cut_num - 1) // self.tiling_width_cut_num)
        # fix for nc_cut_num
        self.tiling_bc1_cut_num.set_as(
            (self.tiling_batch * self.tiling_c1 + self.tiling_bc1_cut_num - 1) // self.tiling_bc1_cut_num)
        self.tiling_bc1_cut_num.set_as(
            (self.tiling_batch * self.tiling_c1 + self.tiling_bc1_cut_num - 1) // self.tiling_bc1_cut_num)

        nc_segment = (self.tiling_batch * self.tiling_c1 + self.tiling_bc1_cut_num - 1) // self.tiling_bc1_cut_num
        h_segment = (self.cut_height_num + self.tiling_height_cut_num - 1) // self.tiling_height_cut_num
        w_segment = (self.cut_width_num + self.tiling_width_cut_num - 1) // self.tiling_width_cut_num
        self.core_nc_start.set_as((core_idx // (self.tiling_height_cut_num * self.tiling_width_cut_num)) * nc_segment)
        self.core_height_start.set_as(
            ((core_idx %
              (self.tiling_height_cut_num * self.tiling_width_cut_num)) // self.tiling_width_cut_num) * h_segment)
        self.core_width_start.set_as(
            ((core_idx %
              (self.tiling_height_cut_num * self.tiling_width_cut_num)) % self.tiling_width_cut_num) * w_segment)
        self.core_nc_num.set_as(nc_segment)
        self.core_height_num.set_as(h_segment)
        self.core_width_num.set_as(w_segment)
        with self.tik_instance.if_scope(self.tiling_key == 101000):
            # when tiling_key is 101000, w start will start from align_num*n
            align_num = self.tiling_out_width // self.tiling_in_width
            self.core_width_num.set_as(self.core_width_num * align_num)
            self.core_width_start.set_as(self.core_width_start * align_num)
            self.cut_width_num.set_as(self.tiling_in_width * align_num)

        with self.tik_instance.if_scope(self.core_nc_start + self.core_nc_num >= self.tiling_batch * self.tiling_c1):
            self.core_nc_num.set_as(self.tiling_batch * self.tiling_c1 - self.core_nc_start)
        with self.tik_instance.if_scope(self.core_height_start + self.core_height_num >= self.cut_height_num):
            self.core_height_num.set_as(self.cut_height_num - self.core_height_start)
        with self.tik_instance.if_scope(self.core_width_start + self.core_width_num >= self.cut_width_num):
            self.core_width_num.set_as(self.cut_width_num - self.core_width_start)
        core_used = self.tiling_width_cut_num * self.tiling_height_cut_num * self.tiling_bc1_cut_num
        with self.tik_instance.if_scope(core_idx >= core_used):
            self.core_nc_num.set_as(0)
            self.core_height_num.set_as(0)
            self.core_width_num.set_as(0)
        self.calculate_scale()

    def calculate_scale(self):
        """
        calculate scale user input h/w and output h/w
        if align_corners == True and out_size > 1:
            scale = (in_size - 1) / (out_size - 1)
        else:
            scale = in_size / out_size
        """
        with self.tik_instance.new_stmt_scope():
            height_input_fp32 = self.tik_instance.Tensor("float32", (self.block_num * 2,),
                                                         name="height_input_fp32",
                                                         scope=tik.scope_ubuf)
            width_input_fp32 = self.tik_instance.Tensor("float32", (self.block_num * 2,),
                                                        name="width_input_fp32",
                                                        scope=tik.scope_ubuf)
            height_input_int32 = self.tik_instance.Tensor("int32", (self.block_num * 2,),
                                                          name="height_input_int32",
                                                          scope=tik.scope_ubuf)
            width_input_int32 = self.tik_instance.Tensor("int32", (self.block_num * 2,),
                                                         name="width_input_int32",
                                                         scope=tik.scope_ubuf)
            height_output_fp32 = self.tik_instance.Tensor("float32", (self.block_num * 2,),
                                                          name="height_output_fp32",
                                                          scope=tik.scope_ubuf)
            width_output_fp32 = self.tik_instance.Tensor("float32", (self.block_num * 2,),
                                                         name="width_output_fp32",
                                                         scope=tik.scope_ubuf)

            scale_ori_h = self.tik_instance.Scalar("int32")
            scale_ori_w = self.tik_instance.Scalar("int32")
            scale_ori_h.set_as(self.ori_h)
            scale_ori_w.set_as(self.ori_w)

            size_ub = self.tik_instance.Tensor(self.size_dtype, [2], name="size_ub", scope=tik.scope_ubuf)
            self.tik_instance.data_move(size_ub, self.size_gm, 0, 1, 1, 0, 0)
            size_h = self.tik_instance.Scalar("int32")
            size_w = self.tik_instance.Scalar("int32")
            size_h.set_as(size_ub[0])
            size_w.set_as(size_ub[1])

            height_input_int32[0].set_as(scale_ori_h)
            width_input_int32[0].set_as(scale_ori_w)
            height_input_int32[self.block_num].set_as(size_h)
            width_input_int32[self.block_num].set_as(size_w)
            util_tik_comm_func.tik_func_vconv(self.tik_instance, height_input_fp32, height_input_int32, 1)
            util_tik_comm_func.tik_func_vconv(self.tik_instance, width_input_fp32, width_input_int32, 1)
            util_tik_comm_func.tik_func_vconv(self.tik_instance, height_output_fp32,
                                              height_input_int32[self.block_num:], 1)
            util_tik_comm_func.tik_func_vconv(self.tik_instance, width_output_fp32, width_input_int32[self.block_num:],
                                              1)

            with self.tik_instance.if_scope(tik.all(self.align_corners, size_h > 1)):
                self.tik_instance.vadds(1, height_output_fp32, height_output_fp32, -1.0, 1, 1, 1, 8, 8)
                self.tik_instance.vadds(1, height_input_fp32, height_input_fp32, -1.0, 1, 1, 1, 8, 8)

            if not self.is_suport_vdiv:
                self.tik_instance.vrec(1, height_output_fp32[self.block_num:], height_output_fp32, 1, 1, 1, 8, 8)
                _tik_fuc_vrec_newton(self.tik_instance,
                                     height_output_fp32[self.block_num:],
                                     height_output_fp32,
                                     1,
                                     block_num=self.block_num)
                self.tik_instance.vmul(1, height_input_fp32, height_input_fp32, height_output_fp32[self.block_num:], 1,
                                       1, 1, 1, 8, 8, 8)
            else:
                self.tik_instance.vdiv(1, height_input_fp32, height_input_fp32, height_output_fp32, 1, 1, 1, 1, 8, 8,
                                       8)
            self.resize_scale_h.set_as(height_input_fp32[0])

            with self.tik_instance.if_scope(tik.all(self.align_corners, size_w > 1)):
                self.tik_instance.vadds(1, width_output_fp32, width_output_fp32, -1.0, 1, 1, 1, 8, 8)
                self.tik_instance.vadds(1, width_input_fp32, width_input_fp32, -1.0, 1, 1, 1, 8, 8)
            if not self.is_suport_vdiv:
                self.tik_instance.vrec(1, width_output_fp32[self.block_num:], width_output_fp32, 1, 1, 1, 8, 8)
                _tik_fuc_vrec_newton(self.tik_instance,
                                     width_output_fp32[self.block_num:],
                                     width_output_fp32,
                                     1,
                                     block_num=self.block_num)
                self.tik_instance.vmul(1, width_input_fp32, width_input_fp32, width_output_fp32[self.block_num:], 1, 1,
                                       1, 1, 8, 8, 8)
            else:
                self.tik_instance.vdiv(1, width_input_fp32, width_input_fp32, width_output_fp32, 1, 1, 1, 1, 8, 8, 8)
            self.resize_scale_w.set_as(width_input_fp32[0])

    def scalar_vconv_int32_to_fp32(self, int32_value, float32_value):
        """
        vconv one scalar from int32 to fp32 usr vector
        """
        with self.tik_instance.new_stmt_scope():
            idx_int32_tmp = self.tik_instance.Tensor("int32", (64,), name="idx_int32_tmp", scope=tik.scope_ubuf)
            idx_fp32_tmp = self.tik_instance.Tensor("float32", (64,), name="idx_fp32_tmp", scope=tik.scope_ubuf)
            idx_int32_tmp[0].set_as(int32_value)
            util_tik_comm_func.tik_func_vconv(self.tik_instance, idx_fp32_tmp, idx_int32_tmp, 1)
            float32_value.set_as(idx_fp32_tmp[0])

    def calcu_out_in_idx(self,
                         scale,
                         des_idx_ub,
                         src_idx_fp_ub,
                         idx_num,
                         des_weight_fp32_ub_list=None,
                         max_idx_ub=None,
                         one_fp32_ub=None,
                         des_idx_ub_1=None,
                         max_idx_ub_int=None,
                         one_int32_ub=None,
                         mem_info=None,
                         src_start=0,
                         dst_start=0):
        """
        if the attr half_pixel_centers is true, will do vconv_f322s32f((idx + 0.5) * scale - 0.5)
        if the attr half_pixel_centers is false, will do vconv_f322s32f(idx * scale)
        """
        with self.tik_instance.new_stmt_scope():
            calcu_out_in_idx_tmp_ub = self.tik_instance.Tensor(src_idx_fp_ub.dtype,
                                                               src_idx_fp_ub.shape,
                                                               name="calcu_out_in_idx_tmp_ub",
                                                               scope=tik.scope_ubuf)
            vector_repeat_num = (idx_num + 63) // 64
            self.tik_instance.vadds(64, src_idx_fp_ub, src_idx_fp_ub, dst_start, vector_repeat_num, 1, 1,
                                    8, 8)
            if self.half_pixel_centers:
                # `calcu: (idx + 0.5) * scale - 0.5`
                self.tik_instance.vadds(64, calcu_out_in_idx_tmp_ub, src_idx_fp_ub, 0.5, vector_repeat_num, 1, 1, 8, 8)
                self.tik_instance.vmuls(64, calcu_out_in_idx_tmp_ub, calcu_out_in_idx_tmp_ub, scale, vector_repeat_num,
                                        1, 1, 8, 8)
                self.tik_instance.vadds(64, calcu_out_in_idx_tmp_ub, calcu_out_in_idx_tmp_ub, -0.5, vector_repeat_num,
                                        1, 1, 8, 8)
                if mem_info is not None:
                    # `when the fp32_point < 0, will modify to 0`
                    self.tik_instance.vmax(64, calcu_out_in_idx_tmp_ub, calcu_out_in_idx_tmp_ub,
                                           mem_info.get("zero").get("fp32"), vector_repeat_num,
                                           1, 1, 0, 8, 8, 0)
            else:
                # `calcu: idx * scale`
                self.tik_instance.vmuls(64, calcu_out_in_idx_tmp_ub, src_idx_fp_ub, scale, vector_repeat_num, 1, 1, 8,
                                        8)

            self.tik_instance.vadds(64, calcu_out_in_idx_tmp_ub, calcu_out_in_idx_tmp_ub, -src_start,
                                    vector_repeat_num, 1, 1, 8, 8)
            # do vmax for 0

            util_tik_comm_func.tik_func_vconv(self.tik_instance,
                                              des_idx_ub,
                                              calcu_out_in_idx_tmp_ub,
                                              idx_num,
                                              mode="floor")
            self.tik_instance.vadds(64, src_idx_fp_ub, src_idx_fp_ub, -dst_start, vector_repeat_num, 1, 1,
                                    8, 8)

            if des_idx_ub_1 is not None:
                util_tik_comm_func.tik_func_vcomple(self.tik_instance,
                                                    "vadd",
                                                    des_idx_ub_1,
                                                    des_idx_ub,
                                                    one_int32_ub,
                                                    idx_num,
                                                    src1_blk=0,
                                                    src1_rep=0)
                util_tik_comm_func.tik_func_vcomple(self.tik_instance,
                                                    "vmin",
                                                    des_idx_ub_1,
                                                    des_idx_ub_1,
                                                    max_idx_ub_int,
                                                    idx_num,
                                                    src1_blk=0,
                                                    src1_rep=0)

            if des_weight_fp32_ub_list is not None:
                des_weight_fp32_ub = des_weight_fp32_ub_list[0]
                calcu_out_in_idx_int_fp32 = self.tik_instance.Tensor(src_idx_fp_ub.dtype,
                                                                     src_idx_fp_ub.shape,
                                                                     name="calcu_out_in_idx_int_fp32",
                                                                     scope=tik.scope_ubuf)
                # `do fp32 vmin(calcu_out_in_idx_tmp_ub, max_h/max_w)`
                if max_idx_ub is not None:
                    util_tik_comm_func.tik_func_vcomple(self.tik_instance,
                                                        "vmin",
                                                        calcu_out_in_idx_tmp_ub,
                                                        calcu_out_in_idx_tmp_ub,
                                                        max_idx_ub,
                                                        idx_num,
                                                        src1_blk=0,
                                                        src1_rep=0)
                # will use vconv_f322s32f to cast to int32
                util_tik_comm_func.tik_func_vconv(self.tik_instance, calcu_out_in_idx_int_fp32, des_idx_ub, idx_num)

                util_tik_comm_func.tik_func_vcomple(self.tik_instance, "vsub", des_weight_fp32_ub,
                                                    calcu_out_in_idx_tmp_ub, calcu_out_in_idx_int_fp32, idx_num)
                if len(des_weight_fp32_ub_list) == 2:
                    des_weight_fp32_ub_2 = des_weight_fp32_ub_list[1]
                    # get 1-x
                    util_tik_comm_func.tik_func_vcomple(self.tik_instance,
                                                        "vsub",
                                                        des_weight_fp32_ub_2,
                                                        one_fp32_ub,
                                                        des_weight_fp32_ub,
                                                        idx_num,
                                                        src0_blk=0,
                                                        src0_rep=0)

    def calcu_out_in_idx_vbi(self, scale, src0_offset, src1, idx_ub_fp32, mem_info, idx_num, src_start=0, dst_start=0):
        """
        if the attr half_pixel_centers is true, will do vconv_f322s32f((idx + 0.5) * scale - 0.5)
        if the attr half_pixel_centers is false, will do vconv_f322s32f(idx * scale)
        """
        with self.tik_instance.new_stmt_scope():
            util_tik_comm_func.tik_func_vadds(self.tik_instance,
                                              idx_ub_fp32,
                                              idx_ub_fp32,
                                              dst_start,
                                              idx_num)
            if self.half_pixel_centers:
                # `calcu: (idx + 0.5) * scale - 0.5`
                util_tik_comm_func.tik_func_vadds(self.tik_instance,
                                                  src1,
                                                  idx_ub_fp32,
                                                  0.5,
                                                  idx_num,
                                                  dst_blk=4,
                                                  dst_rep=32)
                util_tik_comm_func.tik_func_vmuls(self.tik_instance,
                                                  src1,
                                                  src1,
                                                  scale,
                                                  idx_num,
                                                  src_blk=4,
                                                  dst_blk=4,
                                                  dst_rep=32,
                                                  src_rep=32)
                util_tik_comm_func.tik_func_vadds(self.tik_instance,
                                                  src1,
                                                  src1,
                                                  -0.5,
                                                  idx_num,
                                                  src_blk=4,
                                                  dst_blk=4,
                                                  dst_rep=32,
                                                  src_rep=32)
                # when the point < 0, will modify to 0
                util_tik_comm_func.tik_func_vcomple(self.tik_instance,
                                                    "vmax",
                                                    src1,
                                                    src1,
                                                    mem_info.get("zero").get("fp32"),
                                                    idx_num,
                                                    src0_blk=4,
                                                    src1_blk=0,
                                                    dst_blk=4,
                                                    src0_rep=32,
                                                    src1_rep=0,
                                                    dst_rep=32)
            else:
                # `calcu: idx * scale`
                util_tik_comm_func.tik_func_vmuls(self.tik_instance,
                                                  src1,
                                                  idx_ub_fp32,
                                                  scale,
                                                  idx_num,
                                                  dst_blk=4,
                                                  dst_rep=32)

            util_tik_comm_func.tik_func_vadds(self.tik_instance,
                                              src1,
                                              src1,
                                              -src_start,
                                              idx_num,
                                              src_blk=4,
                                              dst_blk=4,
                                              dst_rep=32,
                                              src_rep=32)
            util_tik_comm_func.tik_func_vconv(self.tik_instance, src0_offset, src1, idx_num * 4, mode="floor")
            util_tik_comm_func.tik_func_vadds(self.tik_instance,
                                              idx_ub_fp32,
                                              idx_ub_fp32,
                                              -dst_start,
                                              idx_num)

            util_tik_comm_func.tik_func_vcomple(self.tik_instance,
                                                "vadd",
                                                src0_offset[8:],
                                                src0_offset,
                                                mem_info.get("one").get("int32"),
                                                idx_num,
                                                src1_blk=0,
                                                src1_rep=0,
                                                src0_blk=4,
                                                dst_blk=4,
                                                src0_rep=32,
                                                dst_rep=32)
            util_tik_comm_func.tik_func_vcomple(self.tik_instance,
                                                "vmin",
                                                src0_offset,
                                                src0_offset,
                                                mem_info.get("in_width").get("int32"),
                                                idx_num * 4,
                                                src1_blk=0,
                                                src1_rep=0)
            util_tik_comm_func.tik_func_vcomple(self.tik_instance,
                                                "vmin",
                                                src1,
                                                src1,
                                                mem_info.get("in_width").get("fp32"),
                                                idx_num * 4,
                                                src1_blk=0,
                                                src1_rep=0)
            src1_tmp = self.tik_instance.Tensor(src1.dtype, src1.shape, name="src1_tmp", scope=tik.scope_ubuf)

            # get weight
            util_tik_comm_func.tik_func_vconv(self.tik_instance, src1_tmp, src0_offset, idx_num * 4)
            util_tik_comm_func.tik_func_vcomple(self.tik_instance,
                                                "vsub",
                                                src1,
                                                src1,
                                                src1_tmp,
                                                idx_num,
                                                src1_blk=4,
                                                src0_blk=4,
                                                dst_blk=4,
                                                src0_rep=32,
                                                src1_rep=32,
                                                dst_rep=32)
            util_tik_comm_func.tik_func_vcomple(self.tik_instance,
                                                "vsub",
                                                src1[8:],
                                                mem_info.get("one").get("fp32"),
                                                src1,
                                                idx_num,
                                                src0_blk=0,
                                                src0_rep=0,
                                                src1_blk=4,
                                                src1_rep=32,
                                                dst_blk=4,
                                                dst_rep=32)

    def _function_data_move(self, copy_line_num, copy_line_length, ub_info, scalar_is_src_stride):
        """
        function _function_data_move
        use to copy input to ub/l1
        """
        data_move_src_men, data_move_src_offset = ub_info[0]
        data_move_dst_mem, data_move_dst_offset = ub_info[1]
        data_move_burst_len = copy_line_length * self.images_shape_c0 // self.input_block_num
        with self.tik_instance.if_scope(scalar_is_src_stride == 0):
            with self.tik_instance.for_range(0, copy_line_num) as _segment_idx:
                data_move_dst_offset_new = \
                    data_move_dst_offset \
                    + copy_line_length * self.images_shape_c0 * _segment_idx
                data_move_src_offset_new = \
                    data_move_src_offset \
                    + _segment_idx * self.tiling_in_width * self.tiling_in_height * self.images_shape_c0
                data_move_burst_num = 1
                self.tik_instance.data_move(data_move_dst_mem[data_move_dst_offset_new:],
                                            data_move_src_men[data_move_src_offset_new:], 0, data_move_burst_num,
                                            data_move_burst_len, 0, 0)
        with self.tik_instance.else_scope():
            data_move_burst_num = copy_line_num
            data_move_src_stride = \
                (self.tiling_in_width * self.tiling_in_height - copy_line_length) \
                * self.images_shape_c0 // self.input_block_num
            self.tik_instance.data_move(data_move_dst_mem[data_move_dst_offset:],
                                        data_move_src_men[data_move_src_offset:], 0, data_move_burst_num,
                                        data_move_burst_len, data_move_src_stride, 0)

    def _function_data_move_out(self, copy_line_num, copy_line_length, ub_info, is_stride):
        """
        function _function_data_move_out
        use to copy output to gm
        """
        data_move_src_men, data_move_src_offset = ub_info[0]
        data_move_dst_mem, data_move_dst_offset = ub_info[1]
        data_move_burst_len = copy_line_length * self.images_shape_c0 // self.output_block_num

        with self.tik_instance.if_scope(is_stride == 0):
            with self.tik_instance.for_range(0, copy_line_num) as _segment_idx:
                data_move_dst_offset_new = \
                    data_move_dst_offset \
                    + _segment_idx * self.tiling_out_width * self.tiling_out_height * self.images_shape_c0
                data_move_src_offset_new = \
                    data_move_src_offset + copy_line_length * self.images_shape_c0 * _segment_idx
                data_move_burst_num = 1
                self.tik_instance.data_move(data_move_dst_mem[data_move_dst_offset_new:],
                                            data_move_src_men[data_move_src_offset_new:], 0, data_move_burst_num,
                                            data_move_burst_len, 0, 0)
        with self.tik_instance.else_scope():
            data_move_burst_num = copy_line_num
            data_move_dst_stride = \
                (self.tiling_out_width * self.tiling_out_height - copy_line_length) \
                * self.images_shape_c0 // self.output_block_num
            self.tik_instance.data_move(data_move_dst_mem[data_move_dst_offset:],
                                        data_move_src_men[data_move_src_offset:], 0, data_move_burst_num,
                                        data_move_burst_len, 0, data_move_dst_stride)

    def _function_default_apply_ub(self):
        """
        _function_default_apply_ub
        """
        # init tmp ub for int32/float32 h-1 w-1 0 1
        function_default_int32_ub = self.tik_instance.Tensor("int32", (8 * 8,),
                                                             name="function_default_int32_ub",
                                                             scope=tik.scope_ubuf)
        function_default_fp32_ub = self.tik_instance.Tensor("float32", (8 * 8,),
                                                            name="function_default_fp32_ub",
                                                            scope=tik.scope_ubuf)
        with self.tik_instance.new_stmt_scope():
            scalar_input = self.tik_instance.Scalar("int32", name="scalar_input")
            self.tik_instance.vector_dup(8, function_default_int32_ub[0], 0, 1, 1, 8)
            self.tik_instance.vector_dup(8, function_default_int32_ub[8], 1, 1, 1, 8)
            scalar_input.set_as(self.tiling_in_width - 1)
            self.tik_instance.vector_dup(8, function_default_int32_ub[16], scalar_input, 1, 1, 8)
            scalar_input.set_as(self.tiling_in_height - 1)
            self.tik_instance.vector_dup(8, function_default_int32_ub[24], scalar_input, 1, 1, 8)
            scalar_input.set_as(self.core_height_start)
            self.tik_instance.vector_dup(8, function_default_int32_ub[32], scalar_input, 1, 1, 8)
            scalar_input.set_as(self.core_width_start)
            self.tik_instance.vector_dup(8, function_default_int32_ub[40], scalar_input, 1, 1, 8)
            util_tik_comm_func.tik_func_vconv(self.tik_instance, function_default_fp32_ub, function_default_int32_ub,
                                              64)
        mem_default = {}
        mem_default["zero"] = {"int32": function_default_int32_ub[0:], "fp32": function_default_fp32_ub[0:]}
        mem_default["one"] = {"int32": function_default_int32_ub[8:], "fp32": function_default_fp32_ub[8:]}
        mem_default["in_width"] = {"int32": function_default_int32_ub[16:], "fp32": function_default_fp32_ub[16:]}
        mem_default["in_height"] = {"int32": function_default_int32_ub[24:], "fp32": function_default_fp32_ub[24:]}
        mem_default["height_start"] = {"int32": function_default_int32_ub[32:], "fp32": function_default_fp32_ub[32:]}
        mem_default["width_start"] = {"int32": function_default_int32_ub[40:], "fp32": function_default_fp32_ub[40:]}

        return mem_default

    # 'pylint: disable=unused-argument
    def _function_resize_with_l1_default(self,
                                         is_src_stride_copy=False,
                                         is_dst_stride_copy=False,
                                         is_w_algin=False,
                                         is_h_big_to_small=False):
        """
        _function_default, run this
        """
        tiling_inner_hw_num = 256
        self.height_idx_segment_num = 32
        self.width_idx_segment_num = 64
        max_nc = tiling_inner_hw_num // self.width_idx_segment_num
        if self.images_dtype != self.inner_dtype:
            max_nc = tiling_inner_hw_num // self.width_idx_segment_num // 2
        mem_info = self._function_default_apply_ub()

        self.idx_ub_fp32 = self.tik_instance.Tensor("float32", (self.width_idx_segment_num * 8,),
                                                    name="idx_ub_fp32",
                                                    scope=tik.scope_ubuf)
        self.width_idx_ub = self.tik_instance.Tensor("int32", (self.width_idx_segment_num * 8,),
                                                     name="width_idx",
                                                     scope=tik.scope_ubuf)
        width_weight_ub = self.tik_instance.Tensor("float32", (self.width_idx_segment_num * 8 * 2,),
                                                   name="width_weight_ub",
                                                   scope=tik.scope_ubuf)
        self.height_idx_ub = self.tik_instance.Tensor("int32", (64,), name="height_idx", scope=tik.scope_ubuf)
        height_weight_ub = self.tik_instance.Tensor("float32", (64,), name="height_weight_ub", scope=tik.scope_ubuf)

        scalar_in_height = self.tik_instance.Scalar("int64", name="scalar_in_height")
        scalar_in_height.set_as(self.tiling_in_height - 1)

        # gen 0 - self.width_idx_segment_num to l1 fp32
        with self.tik_instance.new_stmt_scope():
            self.tik_instance.vector_dup(8, self.idx_ub_fp32, 0.0, 1, 1, 8)
            with self.tik_instance.for_range(0, self.width_idx_segment_num - 1) as ub_idx:
                self.tik_instance.vadds(8, self.idx_ub_fp32[(ub_idx + 1) * 8], self.idx_ub_fp32[ub_idx * 8], 1.0, 1, 1,
                                        1, 8, 8)
        scalar_idx_fp32 = self.tik_instance.Scalar("float32", name="scalar_idx_fp32")
        # vconv start idx from int32 scalar to fp32 scalar
        self.scalar_vconv_int32_to_fp32(self.core_width_start, scalar_idx_fp32)
        # do vadds 0,1,2,3,4 + fp32_w_start_offset for per core
        self.tik_instance.vadds(64, self.idx_ub_fp32, self.idx_ub_fp32, scalar_idx_fp32,
                                (self.width_idx_segment_num * 8 + 63) // 64, 1, 1, 8, 8)

        # calcu is_src_stride_copy and is_dst_stride_copy use scalar
        scalar_is_src_stride = self.scalar_is_src_stride
        scalar_is_dst_stride = self.scalar_is_dst_stride
        # calcu is_src_stride_copy and is_dst_stride_copy use scalar end

        # init a scalar for w segment one time
        w_loop_segment = self.tik_instance.Scalar("int32",
                                                  name="w_loop_segment",
                                                  init_value=self.width_idx_segment_num)
        with self.tik_instance.if_scope(tik.all(self.core_width_num < w_loop_segment, self.core_width_num > 0)):
            w_loop_segment.set_as(self.core_width_num)

        if is_w_algin:
            # if width is input_w resize to n*input_w, one segment must be n algin
            # exp: 24 resize to 48, one segment of width must be 2*n
            with self.tik_instance.new_stmt_scope():
                algin_num_scalar = self.tik_instance.Scalar("int32", name="algin_num_scalar")
                algin_num_scalar.set_as(self.tiling_out_width // self.tiling_in_width)
                w_loop_segment.set_as(w_loop_segment // algin_num_scalar * algin_num_scalar)
        else:
            # when big w -> small w, the input will be overload
            w_input_output_rate = (self.tiling_in_width + self.tiling_out_width - 1) // self.tiling_out_width
            max_input_w = self.width_idx_segment_num * 4 * 4
            max_output_w = max_input_w // w_input_output_rate
            with self.tik_instance.if_scope(w_loop_segment > max_output_w):
                w_loop_segment.set_as(max_output_w)
            with self.tik_instance.if_scope(w_loop_segment == 0):
                w_loop_segment.set_as(1)

        w_loop_num = self.tik_instance.Scalar("int32", name="w_loop_num")
        w_tail_num = self.tik_instance.Scalar("int32", name="w_tail_num")
        nc_max_segment = self.tik_instance.Scalar("int32", name="nc_max_segment")
        nc_loop = self.tik_instance.Scalar("int32", name="nc_loop")
        nc_tail = self.tik_instance.Scalar("int32", name="nc_tail")

        w_loop_num.set_as(self.core_width_num // w_loop_segment)
        w_tail_num.set_as(self.core_width_num % w_loop_segment)
        nc_max_segment.set_as(self.ub_max_num // (w_loop_segment * self.images_shape_c0))
        with self.tik_instance.if_scope(tik.all(self.core_nc_num < nc_max_segment, self.core_nc_num > 0)):
            nc_max_segment.set_as(self.core_nc_num)
        with self.tik_instance.if_scope(nc_max_segment > max_nc):
            nc_max_segment.set_as(max_nc)
        nc_loop.set_as(self.core_nc_num // nc_max_segment)
        nc_tail.set_as(self.core_nc_num % nc_max_segment)
        nc_process_num_ub = self.tik_instance.Tensor("int32", (self.block_num,),
                                                     name="nc_process_num_ub",
                                                     scope=tik.scope_ubuf)
        nc_loop_num_ceil = self.tik_instance.Scalar("int32", name="nc_loop_num_ceil")
        nc_loop_num_ceil.set_as((self.core_nc_num + nc_max_segment - 1) // nc_max_segment)
        nc_loop_num_floor = nc_loop
        nc_process_num_ub[0].set_as(nc_max_segment)
        nc_process_num_ub[1].set_as(nc_tail)

        w_process_num_ub = self.tik_instance.Tensor("int32", (self.block_num,),
                                                    name="w_process_num_ub",
                                                    scope=tik.scope_ubuf)
        w_loop_num_ceil = self.tik_instance.Scalar("int32", name="w_loop_num_ceil")
        w_loop_num_ceil.set_as((self.core_width_num + w_loop_segment - 1) // w_loop_segment)
        w_loop_num_floor = w_loop_num
        w_process_num_ub[0].set_as(w_loop_segment)
        w_process_num_ub[1].set_as(w_tail_num)

        self.scalar_vconv_int32_to_fp32(w_loop_segment, scalar_idx_fp32)

        # for start_h idx and weight
        height_idx_ub_fp32_start = self.tik_instance.Tensor("float32", (64,),
                                                            name="height_idx_ub_fp32_start",
                                                            scope=tik.scope_ubuf)

        self.tik_instance.vadds(64, height_idx_ub_fp32_start, mem_info.get("height_start").get("fp32"),
                                0.0, 1, 1, 0, 8, 0)
        height_idx_ub_start = self.tik_instance.Tensor("int32", (64,), name="height_idx", scope=tik.scope_ubuf)
        height_weight_fp32_start = self.tik_instance.Tensor("float32", (128,),
                                                            name="height_weight_fp32_start",
                                                            scope=tik.scope_ubuf)
        self.calcu_out_in_idx(self.resize_scale_h,
                              height_idx_ub_start,
                              height_idx_ub_fp32_start,
                              64,
                              des_weight_fp32_ub_list=[height_weight_fp32_start],
                              mem_info=mem_info)

        is_need_copy_repeat = is_h_big_to_small

        def _run_w_loop_default(w_loop_idx, w_do_len, h_loop_offset, h_do_len):
            w_gm_offset = w_loop_idx * w_loop_segment + self.core_width_start
            self.calcu_out_in_idx(
                self.resize_scale_w,
                self.width_idx_ub,
                self.idx_ub_fp32,
                self.width_idx_segment_num * 8,
                des_weight_fp32_ub_list=[width_weight_ub, width_weight_ub[self.width_idx_segment_num * 8:]],
                max_idx_ub=mem_info.get("in_width").get("fp32"),
                one_fp32_ub=mem_info.get("one").get("fp32"),
                mem_info=mem_info,
                src_start=self.src_start_w,
                dst_start=self.dst_start_w)
            self.tik_instance.vadds(64, self.idx_ub_fp32, self.idx_ub_fp32, scalar_idx_fp32,
                                    (self.width_idx_segment_num * 8 + 63) // 64, 1, 1, 8, 8)

            scalar_w_start_idx = self.tik_instance.Scalar("int32", name="scalar_w_start_idx")
            scalar_w_end_idx = self.tik_instance.Scalar("int32", name="scalar_w_end_idx")
            input_w_len = self.tik_instance.Scalar("int32", name="input_w_len")

            scalar_w_start_idx.set_as(self.width_idx_ub[0])
            scalar_w_end_idx.set_as(self.width_idx_ub[(w_do_len - 1) * 8])
            range_burst_len = self.tik_instance.Scalar("int32",
                                                       name="range_burst_len",
                                                       init_value=(self.images_shape_c0 // self.input_block_num) * 2)

            with self.tik_instance.if_scope(scalar_w_end_idx < self.tiling_in_width - 1):
                input_w_len.set_as(scalar_w_end_idx - scalar_w_start_idx + 1 + self.is_bilinear)
            with self.tik_instance.else_scope():
                input_w_len.set_as(scalar_w_end_idx - scalar_w_start_idx + 1)
                with self.tik_instance.if_scope(input_w_len == 1):
                    range_burst_len.set_as(self.images_shape_c0 // self.input_block_num)

            # scalar for copy_l1_to_ub
            range_cbuf_burst_stride = self.tik_instance.Scalar("int32", name="range_cbuf_burst_stride")
            range_ub_burst_strde = self.tik_instance.Scalar("int32", name="range_ub_burst_strde")
            nc_offset = input_w_len * self.images_shape_c0
            range_cbuf_burst_stride.set_as(nc_offset // self.input_block_num - range_burst_len)
            range_ub_burst_strde.set_as(w_do_len * self.images_shape_c0 * 2 // self.input_block_num - range_burst_len)

            # w index to c0
            tmp_ub = self.tik_instance.Tensor("int32", (128,), name="tmp_ub", scope=tik.scope_ubuf)
            util_tik_comm_func.tik_func_vector(self.tik_instance, tmp_ub, scalar_w_start_idx, 64)
            util_tik_comm_func.tik_func_vcomple(self.tik_instance,
                                                "vsub",
                                                self.width_idx_ub,
                                                self.width_idx_ub,
                                                tmp_ub,
                                                self.width_idx_segment_num * 8,
                                                src1_rep=0)
            util_tik_comm_func.tik_func_vector(self.tik_instance, tmp_ub[64:], self.images_shape_c0, 64)
            util_tik_comm_func.tik_func_vcomple(self.tik_instance,
                                                "vmul",
                                                self.width_idx_ub,
                                                self.width_idx_ub,
                                                tmp_ub[64:],
                                                self.width_idx_segment_num * 8,
                                                src1_rep=0)

            # one segment h and one segment w
            def _do_single_nc(do_nc_num, _nc_loop_idx):
                # copy the h start info to ub before h loop
                height_idx_ub_fp32 = self.tik_instance.Tensor("float32", (64,),
                                                              name="height_idx_ub_fp32",
                                                              scope=tik.scope_ubuf)
                height_idx_ub = self.tik_instance.Tensor("int32", (64,), name="height_idx_ub", scope=tik.scope_ubuf)
                self.tik_instance.data_move(height_idx_ub_fp32, height_idx_ub_fp32_start, 0, 1, 8, 0, 0)
                self.tik_instance.data_move(height_idx_ub, height_idx_ub_start, 0, 1, 8, 0, 0)
                self.tik_instance.data_move(height_weight_ub, height_weight_fp32_start, 0, 1, 8, 0, 0)
                resize_tail_mask1 = self.tik_instance.Scalar("int64", name="resize_tail_mask1")
                resize_tail_mask2 = self.tik_instance.Scalar("int64", name="resize_tail_mask2")
                resize_tail_mask1.set_as((do_nc_num * w_do_len * self.images_shape_c0 * 2) % (self.block_num * 8))
                resize_tail_mask2.set_as((w_do_len * self.images_shape_c0) % (self.block_num * 8))
                scalar_in_h_idx_pre = self.tik_instance.Scalar("int32", name="scalar_in_h_idx_pre")
                scalar_in_h_idx_pre.set_as(-1)

                def _do_one_height(h_idx, input_l1, input_ub, inner_ub):
                    scalar_in_h_idx = self.tik_instance.Scalar("int32", name="scalar_in_h_idx")
                    scalar_in_h_idx.set_as(height_idx_ub[0])
                    scalar_in_h_weight = self.tik_instance.Scalar("float32", name="scalar_in_h_weight")
                    scalar_in_h_weight.set_as(height_weight_ub[0])

                    scalar_in_h_idx_bottom = self.tik_instance.Scalar("int64", name="scalar_in_h_idx_bottom")
                    scalar_in_h_idx_bottom.set_as(scalar_in_h_idx + 1)
                    self.tik_instance.scalar_min(scalar_in_h_idx_bottom, scalar_in_h_idx_bottom, scalar_in_height)

                    self.tik_instance.vadds(64, height_idx_ub_fp32, height_idx_ub_fp32, 1.0, 1, 1, 1, 8, 8)
                    self.calcu_out_in_idx(self.resize_scale_h,
                                          height_idx_ub,
                                          height_idx_ub_fp32,
                                          1,
                                          des_weight_fp32_ub_list=[height_weight_ub],
                                          mem_info=mem_info)
                    with self.tik_instance.if_scope(
                            tik.any(not is_need_copy_repeat, scalar_in_h_idx != scalar_in_h_idx_pre)):
                        with self.tik_instance.new_stmt_scope(disable_sync=True):
                            nc_gm_input_offset = \
                                (_nc_loop_idx * nc_max_segment + self.core_nc_start) * self.tiling_in_width \
                                * self.tiling_in_height * self.images_shape_c0
                            h_gm_input_offset = \
                                (scalar_in_h_idx * self.tiling_in_width + scalar_w_start_idx) * self.images_shape_c0
                            top_gm_offset = nc_gm_input_offset + h_gm_input_offset
                            # copy top value
                            ub_info = [[self.images_gm, top_gm_offset], [input_l1, 0]]
                            self._function_data_move(do_nc_num, input_w_len, ub_info, scalar_is_src_stride)

                            # copy bottom value
                            h_gm_input_offset = \
                                (scalar_in_h_idx_bottom * self.tiling_in_width + scalar_w_start_idx) \
                                * self.images_shape_c0
                            bottom_gm_offset = nc_gm_input_offset + h_gm_input_offset

                            ub_info = [[self.images_gm, bottom_gm_offset],
                                       [input_l1, do_nc_num * input_w_len * self.images_shape_c0]]
                            self._function_data_move(do_nc_num, input_w_len, ub_info, scalar_is_src_stride)

                        # rerange the input from l1 to ub
                        with self.tik_instance.new_stmt_scope(disable_sync=True):
                            if not is_w_algin:
                                with self.tik_instance.for_range(0, w_do_len) as w_idx:
                                    scalar_in_w_idx = self.tik_instance.Scalar("int32", name="scalar_in_w_idx")
                                    scalar_in_w_idx.set_as(self.width_idx_ub[w_idx * 8])
                                    burst_num = do_nc_num * 2

                                    self.tik_instance.data_move(input_ub[w_idx * self.images_shape_c0 * 2],
                                                                input_l1[scalar_in_w_idx], 0, burst_num,
                                                                range_burst_len, range_cbuf_burst_stride,
                                                                range_ub_burst_strde)
                            else:
                                pass

                        scalar_in_h_idx_pre.set_as(scalar_in_h_idx)

                        # do resize compute in input_ub
                        # 1. cast to fp32
                        # 2. do 'bottom = (bottom - top) * y_lerp + top'
                        # 3. do 'out = bottom_left + (bottom_right - bottom_left) * x_lerp'
                    total_num = do_nc_num * w_do_len * self.images_shape_c0 * 4
                    if self.inner_dtype != self.images_dtype:
                        self.tik_instance.vconv(64, "", inner_ub, input_ub, total_num // 64, 1, 1, 8, 4)
                    else:
                        if is_need_copy_repeat:
                            self.tik_instance.data_move(inner_ub, input_ub, 0, 1, total_num // self.block_num, 0, 0)
                    total_num = do_nc_num * w_do_len * self.images_shape_c0 * 2
                    input_bottom_ub = inner_ub[total_num:]
                    input_top_ub = inner_ub[0:]
                    with self.tik_instance.if_scope(total_num >= self.block_num * 8):
                        self.tik_instance.vsub(self.block_num * 8, input_bottom_ub, input_bottom_ub, input_top_ub,
                                               total_num // (self.block_num * 8), 1, 1, 1, 8, 8, 8)
                        self.tik_instance.vmuls(self.block_num * 8, input_bottom_ub, input_bottom_ub,
                                                scalar_in_h_weight, total_num // (self.block_num * 8), 1, 1, 8, 8)
                        self.tik_instance.vadd(self.block_num * 8, input_bottom_ub, input_bottom_ub, input_top_ub,
                                               total_num // (self.block_num * 8), 1, 1, 1, 8, 8, 8)
                    with self.tik_instance.if_scope(resize_tail_mask1 > 0):
                        offset = (total_num // (self.block_num * 8)) * self.block_num * 8
                        self.tik_instance.vsub(resize_tail_mask1, input_bottom_ub[offset:], input_bottom_ub[offset:],
                                               input_top_ub[offset:], 1, 1, 1, 1, 8, 8, 8)
                        self.tik_instance.vmuls(resize_tail_mask1, input_bottom_ub[offset:], input_bottom_ub[offset:],
                                                scalar_in_h_weight, 1, 1, 1, 8, 8)
                        self.tik_instance.vadd(resize_tail_mask1, input_bottom_ub[offset:], input_bottom_ub[offset:],
                                               input_top_ub[offset:], 1, 1, 1, 1, 8, 8, 8)

                    with self.tik_instance.for_range(0, do_nc_num) as nc_idx:
                        total_num = w_do_len * self.images_shape_c0
                        input_left = input_bottom_ub[nc_idx * w_do_len * self.images_shape_c0 * 2:]
                        input_right = \
                            input_bottom_ub[nc_idx * w_do_len * self.images_shape_c0 * 2 + self.images_shape_c0:]
                        output_ub = input_top_ub[nc_idx * w_do_len * self.images_shape_c0:]
                        with self.tik_instance.if_scope(total_num >= self.block_num * 8):
                            self.tik_instance.vsub(self.block_num * 8, output_ub, input_right, input_left,
                                                   total_num // (self.block_num * 8), 2, 4, 4, 16, 32, 32)
                            self.tik_instance.vsub(self.block_num * 8, output_ub[self.block_num:],
                                                   input_right[self.block_num:], input_left[self.block_num:],
                                                   total_num // (self.block_num * 8), 2, 4, 4, 16, 32, 32)
                            self.tik_instance.vmul(self.block_num * 8, output_ub[0:], output_ub[0:],
                                                   width_weight_ub[0:], total_num // (self.block_num * 8), 2, 2, 1, 16,
                                                   16, 8)
                            self.tik_instance.vmul(self.block_num * 8, output_ub[self.block_num:],
                                                   output_ub[self.block_num:], width_weight_ub[0:],
                                                   total_num // (self.block_num * 8), 2, 2, 1, 16, 16, 8)
                            self.tik_instance.vadd(self.block_num * 8, output_ub[0:], output_ub[0:], input_left[0:],
                                                   total_num // (self.block_num * 8), 2, 2, 4, 16, 16, 32)
                            self.tik_instance.vadd(self.block_num * 8, output_ub[self.block_num:],
                                                   output_ub[self.block_num:], input_left[self.block_num:],
                                                   total_num // (self.block_num * 8), 2, 2, 4, 16, 16, 32)
                        with self.tik_instance.if_scope(resize_tail_mask2 > 0):
                            repeat_offset = total_num // (self.block_num * 8) * (self.block_num * 8)
                            self.tik_instance.vsub(resize_tail_mask2, output_ub[repeat_offset * 2:],
                                                   input_right[repeat_offset * 4:], input_left[repeat_offset * 4:], 1,
                                                   2, 4, 4, 16, 32, 32)
                            self.tik_instance.vsub(resize_tail_mask2, output_ub[repeat_offset * 2 + self.block_num:],
                                                   input_right[repeat_offset * 4 + self.block_num:],
                                                   input_left[repeat_offset * 4 + self.block_num:], 1, 2, 4, 4, 16, 32,
                                                   32)
                            self.tik_instance.vmul(resize_tail_mask2, output_ub[repeat_offset * 2:],
                                                   output_ub[repeat_offset * 2:], width_weight_ub[0:], 1, 2, 2, 1, 16,
                                                   16, 8)
                            self.tik_instance.vmul(resize_tail_mask2, output_ub[repeat_offset * 2 + self.block_num:],
                                                   output_ub[repeat_offset * 2 + self.block_num:], width_weight_ub[0:],
                                                   1, 2, 2, 1, 16, 16, 8)
                            self.tik_instance.vadd(resize_tail_mask2, output_ub[repeat_offset * 2:],
                                                   output_ub[repeat_offset * 2:], input_left[repeat_offset * 4:], 1, 2,
                                                   2, 4, 16, 16, 32)
                            self.tik_instance.vadd(resize_tail_mask2, output_ub[repeat_offset * 2 + self.block_num:],
                                                   output_ub[repeat_offset * 2 + self.block_num:],
                                                   input_left[repeat_offset * 4 + self.block_num:], 1, 2, 2, 4, 16, 16,
                                                   32)

                    # copy output
                    if self.output_dtype != input_top_ub.dtype:
                        output_cast_ub = input_top_ub.reinterpret_cast_to(self.output_dtype)
                        output_num = do_nc_num * w_do_len * self.images_shape_c0
                        self.tik_instance.vconv(64, "", output_cast_ub, input_top_ub[0:], (output_num + 63) // 64, 1,
                                                1, 4, 8)
                        output_ub = output_cast_ub[0:]
                    else:
                        output_ub = input_top_ub[0:]
                    nc_gm_output_offset = \
                        (_nc_loop_idx * nc_max_segment + self.core_nc_start) * self.tiling_out_width \
                        * self.tiling_out_height * self.images_shape_c0
                    h_gm_output_offset = \
                        ((h_idx + h_loop_offset) * self.tiling_out_width + w_gm_offset) * self.images_shape_c0
                    output_offset = nc_gm_output_offset + h_gm_output_offset
                    ub_info = [[output_ub, 0], [self.out_gm, output_offset]]
                    with self.tik_instance.new_stmt_scope(disable_sync=True):
                        self._function_data_move_out(do_nc_num, w_do_len, ub_info, scalar_is_dst_stride)

                ub_max_num = self.width_idx_segment_num * max_nc * self.images_shape_c0 * 4
                if not is_need_copy_repeat:
                    image_in_cb_ping = self.tik_instance.Tensor(self.images_dtype, (ub_max_num * 4,),
                                                                name="image_in_cb_ping",
                                                                scope=tik.scope_cbuf)
                    image_out_ub_ping = self.tik_instance.Tensor(self.images_dtype, (ub_max_num,),
                                                                 name="image_out_ub_ping",
                                                                 scope=tik.scope_ubuf)
                    image_in_cb_pang = self.tik_instance.Tensor(self.images_dtype, (ub_max_num * 4,),
                                                                name="image_in_cb_pang",
                                                                scope=tik.scope_cbuf)
                    image_out_ub_pang = self.tik_instance.Tensor(self.images_dtype, (ub_max_num,),
                                                                 name="image_out_ub_pang",
                                                                 scope=tik.scope_ubuf)
                    resize_inner_ub_ping = image_out_ub_ping
                    resize_inner_ub_pang = image_out_ub_pang
                    if self.images_dtype != self.inner_dtype:
                        resize_inner_ub_ping = self.tik_instance.Tensor(self.inner_dtype, (ub_max_num,),
                                                                        name="resize_inner_ub_ping",
                                                                        scope=tik.scope_ubuf)
                        resize_inner_ub_pang = self.tik_instance.Tensor(self.inner_dtype, (ub_max_num,),
                                                                        name="resize_inner_ub_pang",
                                                                        scope=tik.scope_ubuf)
                else:
                    image_in_cb_ping = self.tik_instance.Tensor(self.images_dtype, (ub_max_num * 4,),
                                                                name="image_in_cb_ping",
                                                                scope=tik.scope_cbuf)
                    image_out_ub_ping = self.tik_instance.Tensor(self.images_dtype, (ub_max_num,),
                                                                 name="image_out_ub_ping",
                                                                 scope=tik.scope_ubuf)
                    image_in_cb_pang = self.tik_instance.Tensor(self.images_dtype, (ub_max_num * 4,),
                                                                name="image_in_cb_pang",
                                                                scope=tik.scope_cbuf)
                    image_out_ub_pang = image_out_ub_ping
                    if self.images_dtype != self.inner_dtype:
                        resize_inner_ub_ping = self.tik_instance.Tensor(self.inner_dtype, (ub_max_num,),
                                                                        name="resize_inner_ub_ping",
                                                                        scope=tik.scope_ubuf)
                        resize_inner_ub_pang = self.tik_instance.Tensor(self.inner_dtype, (ub_max_num,),
                                                                        name="resize_inner_ub_pang",
                                                                        scope=tik.scope_ubuf)
                    else:
                        resize_inner_ub_ping = self.tik_instance.Tensor(self.images_dtype, (ub_max_num,),
                                                                        name="image_out_ub_pang",
                                                                        scope=tik.scope_ubuf)
                        resize_inner_ub_pang = resize_inner_ub_ping

                with self.tik_instance.for_range(0, h_do_len // 2) as _h_idx:
                    _do_one_height(_h_idx * 2, image_in_cb_ping, image_out_ub_ping, resize_inner_ub_ping)
                    _do_one_height(_h_idx * 2 + 1, image_in_cb_pang, image_out_ub_pang, resize_inner_ub_pang)
                with self.tik_instance.if_scope(h_do_len % 2 == 1):
                    _do_one_height(h_do_len - 1, image_in_cb_ping, image_out_ub_ping, resize_inner_ub_ping)

            with self.tik_instance.for_range(0, nc_loop_num_ceil) as nc_loop_idx:
                nc_loop_do_num = self.tik_instance.Scalar("int32", name="nc_loop_do_num")
                nc_loop_do_num.set_as(nc_process_num_ub[nc_loop_idx // nc_loop_num_floor])
                _do_single_nc(nc_loop_do_num, nc_loop_idx)

        def _run_h_loop_default(h_loop_idx, h_do_len):
            h_gm_offset = h_loop_idx * self.height_idx_segment_num + self.core_height_start
            with self.tik_instance.for_range(0, w_loop_num_ceil) as w_loop_idx:
                w_loop_do_num = self.tik_instance.Scalar("int32", name="w_loop_do_num")
                w_loop_do_num.set_as(w_process_num_ub[w_loop_idx // w_loop_num_floor])
                _run_w_loop_default(w_loop_idx, w_loop_do_num, h_gm_offset, h_do_len)

        _run_h_loop_default(0, self.core_height_num)

    def _function_reisze_with_no_bilinear(self, is_equal=False):
        """
        _function_reisze_with_no_bilinear, do case(n,1,1,c  ->  n,h,w,c) and  (n,h,w,c  ->  n,h,w,c)
        n,h,w,c  ->  n,h,w,c == n*h*w,1,1,c -> n*h*w,1,1,c
        """
        core_height_num = 1
        core_width_num = 1
        core_height_start = 0
        core_width_start = 0
        tiling_in_height = 1
        tiling_in_width = 1
        tiling_out_height = self.tiling_out_height
        tiling_out_width = self.tiling_out_width
        if is_equal:
            tiling_out_height = 1
            tiling_out_width = 1

        max_number_in_ub_left = self.ub_max_num // 2
        self.width_idx_segment_num = 128
        size_h_n = tiling_out_height // tiling_in_height
        size_w_n = tiling_out_width // tiling_in_width
        output_w_size = core_width_num * size_w_n
        w_output_size_one_line = self.tik_instance.Scalar("int64",
                                                          name="output_w_size",
                                                          init_value=self.width_idx_segment_num)
        copy_w_algin_num_in_ub = self.tik_instance.Scalar("int64", name="copy_w_algin_num_in_ub", init_value=size_w_n)
        with self.tik_instance.if_scope(tik.all(output_w_size < self.width_idx_segment_num, self.core_width_num > 0)):
            w_output_size_one_line.set_as(output_w_size)

        w_output_size_one_line.set_as((w_output_size_one_line // size_w_n) * size_w_n)
        with self.tik_instance.if_scope(w_output_size_one_line == 0):
            w_output_size_one_line.set_as(self.width_idx_segment_num)
            copy_w_algin_num_in_ub.set_as(self.width_idx_segment_num)

        w_copy_num = self.tik_instance.Scalar("int64", name="w_copy_num")
        w_copy_tail = self.tik_instance.Scalar("int64", name="w_copy_tail")
        w_copy_num.set_as(output_w_size // w_output_size_one_line)
        w_copy_tail.set_as(output_w_size % w_output_size_one_line)

        nc_loop = self.tik_instance.Scalar("int32", name="nc_loop")
        nc_tail = self.tik_instance.Scalar("int32", name="nc_tail")
        nc_segment = self.tik_instance.Scalar("int32", name="nc_segment")
        nc_segment.set_as(max_number_in_ub_left // self.images_shape_c0 // w_output_size_one_line)
        nc_loop.set_as(self.core_nc_num // nc_segment)
        nc_tail.set_as(self.core_nc_num % nc_segment)

        def _run_w_loop_default(w_loop_idx, w_do_len, h_loop_offset, h_do_len):
            w_gm_offset = core_width_start + w_loop_idx
            input_w_len = w_do_len // size_w_n
            scalar_in_w_idx = w_gm_offset

            # one segment h and one segment w
            def _do_single_nc(do_nc_num, _nc_loop_idx, image_out_ub, image_input_ub):

                def _do_one_height(h_idx, output_ub, input_ub):
                    h_gm_offset = h_idx + h_loop_offset
                    scalar_in_h_idx = self.tik_instance.Scalar("int32", name="scalar_in_h_idx")
                    scalar_in_h_idx.set_as(h_gm_offset)
                    with self.tik_instance.new_stmt_scope(disable_sync=True):
                        data_move_cbuf_offset = 0
                        nc_gm_input_offset = \
                            (_nc_loop_idx * nc_segment + self.core_nc_start) \
                            * tiling_in_width * tiling_in_height
                        data_move_gm_offset = \
                            nc_gm_input_offset + scalar_in_h_idx * tiling_in_width + scalar_in_w_idx
                        data_move_burst_num = 1
                        data_move_burst_len = (input_w_len * self.images_shape_c0 // self.input_block_num) * do_nc_num
                        data_move_src_stride = 0
                        data_move_dst_stride = 0
                        self.tik_instance.data_move(input_ub[data_move_cbuf_offset],
                                                    self.images_gm[data_move_gm_offset * self.images_shape_c0], 0,
                                                    data_move_burst_num, data_move_burst_len, data_move_src_stride,
                                                    data_move_dst_stride)

                    # input_w_len
                    w_algin_num = copy_w_algin_num_in_ub
                    if self.output_dtype == self.images_dtype:
                        with self.tik_instance.new_stmt_scope(disable_sync=True):
                            with self.tik_instance.for_range(0, input_w_len) as w_input_idx:
                                # input dtype is equal output dtype, use datamove to rearange
                                burst_num = do_nc_num
                                burst_len = self.images_shape_c0 // self.output_block_num
                                src_burst_stride = \
                                    (input_w_len * self.images_shape_c0) // self.output_block_num - burst_len
                                dst_burst_stride = \
                                    (copy_w_algin_num_in_ub * self.images_shape_c0) // self.output_block_num \
                                    - burst_len

                                self.tik_instance.data_move(
                                    output_ub[w_input_idx * w_algin_num * self.images_shape_c0],
                                    input_ub[w_input_idx * self.images_shape_c0], 0, burst_num, burst_len,
                                    src_burst_stride, dst_burst_stride)
                                copy_output_ub = output_ub
                    else:
                        total_num = do_nc_num * input_w_len * self.images_shape_c0
                        self.tik_instance.vconv(64, "none", output_ub, input_ub, (total_num + 63) // 64, 1, 1, 8, 4)
                        if not is_equal:
                            copy_output_ub = input_ub.reinterpret_cast_to(self.output_dtype)
                            with self.tik_instance.new_stmt_scope(disable_sync=True):
                                with self.tik_instance.for_range(0, input_w_len) as w_input_idx:
                                    # input dtype is equal output dtype, use datamove to rearange
                                    burst_num = do_nc_num
                                    burst_len = self.images_shape_c0 // self.output_block_num
                                    src_burst_stride = \
                                        (input_w_len * self.images_shape_c0) // self.output_block_num - burst_len
                                    dst_burst_stride = \
                                        (copy_w_algin_num_in_ub * self.images_shape_c0) // self.output_block_num \
                                        - burst_len
                                    self.tik_instance.data_move(
                                        copy_output_ub[w_input_idx * w_algin_num * self.images_shape_c0],
                                        output_ub[w_input_idx * self.images_shape_c0], 0, burst_num, burst_len,
                                        src_burst_stride, dst_burst_stride)
                        else:
                            copy_output_ub = output_ub

                    # datamove to all
                    burst_num = do_nc_num * input_w_len
                    burst_len = self.images_shape_c0 // self.output_block_num
                    with self.tik_instance.new_stmt_scope(disable_sync=True):
                        with self.tik_instance.for_range(1, w_algin_num) as copy_num:
                            data_move_src_offset = 0
                            data_move_dst_offset = self.images_shape_c0 * copy_num
                            data_move_src_stride = (w_algin_num - 1) * self.images_shape_c0 // self.output_block_num
                            data_move_dst_stride = data_move_src_stride
                            self.tik_instance.data_move(copy_output_ub[data_move_dst_offset:],
                                                        copy_output_ub[data_move_src_offset:], 0, burst_num, burst_len,
                                                        data_move_src_stride, data_move_dst_stride)
                    with self.tik_instance.new_stmt_scope(disable_sync=True):
                        with self.tik_instance.for_range(0, size_h_n) as _nh_idx:
                            nc_gm_offset = \
                                (_nc_loop_idx * nc_segment + self.core_nc_start) \
                                * tiling_out_width * tiling_out_height
                            output_gm_offset = \
                                nc_gm_offset + h_gm_offset * size_h_n * tiling_out_width \
                                + w_gm_offset * size_w_n + _nh_idx * tiling_out_width
                            data_move_ub_offset = 0
                            if is_equal:
                                data_move_burst_num = 1
                                data_move_burst_len = \
                                    do_nc_num * w_do_len * size_w_n * self.images_shape_c0 // self.output_block_num
                                data_move_dst_stride = 0
                                self.tik_instance.data_move(self.out_gm[output_gm_offset * self.images_shape_c0:],
                                                            copy_output_ub[data_move_ub_offset:], 0,
                                                            data_move_burst_num, data_move_burst_len, 0,
                                                            data_move_dst_stride)
                            else:
                                with self.tik_instance.if_scope(self.scalar_is_dst_stride != 0):
                                    data_move_burst_num = do_nc_num
                                    with self.tik_instance.for_range(0, w_copy_num) as w_copy_idx:
                                        data_move_burst_len = \
                                            w_output_size_one_line * self.images_shape_c0 // self.output_block_num
                                        new_output_gm_offset = \
                                            output_gm_offset * self.images_shape_c0 \
                                            + w_copy_idx * w_output_size_one_line * self.images_shape_c0
                                        data_move_dst_stride = \
                                            (tiling_out_width * tiling_out_height - w_output_size_one_line) \
                                            * self.images_shape_c0 // self.output_block_num
                                        self.tik_instance.data_move(self.out_gm[new_output_gm_offset:],
                                                                    copy_output_ub[data_move_ub_offset:], 0,
                                                                    data_move_burst_num, data_move_burst_len, 0,
                                                                    data_move_dst_stride)
                                    with self.tik_instance.if_scope(w_copy_tail != 0):
                                        data_move_burst_len = \
                                            w_copy_tail * self.images_shape_c0 // self.output_block_num
                                        new_output_gm_offset = \
                                            output_gm_offset * self.images_shape_c0 \
                                            + w_copy_num * w_output_size_one_line * self.images_shape_c0
                                        data_move_dst_stride = \
                                            (tiling_out_width * tiling_out_height - w_copy_tail) \
                                            * self.images_shape_c0 // self.output_block_num
                                        self.tik_instance.data_move(self.out_gm[new_output_gm_offset:],
                                                                    copy_output_ub[data_move_ub_offset:], 0,
                                                                    data_move_burst_num, data_move_burst_len, 0,
                                                                    data_move_dst_stride)
                                with self.tik_instance.if_scope(self.scalar_is_dst_stride == 0):
                                    data_move_burst_num = 1
                                    with self.tik_instance.for_range(0, w_copy_num) as w_copy_idx:
                                        with self.tik_instance.for_range(0, do_nc_num) as nc_idx:
                                            data_move_burst_len = \
                                                w_output_size_one_line * self.images_shape_c0 // self.output_block_num
                                            new_output_gm_offset = \
                                                output_gm_offset * self.images_shape_c0 \
                                                + w_copy_idx * w_output_size_one_line * self.images_shape_c0 \
                                                + nc_idx * tiling_out_width * tiling_out_height * self.images_shape_c0
                                            data_move_dst_stride = 0
                                            data_move_ub_offset_new = \
                                                data_move_ub_offset \
                                                + nc_idx * w_output_size_one_line * self.images_shape_c0
                                            self.tik_instance.data_move(self.out_gm[new_output_gm_offset:],
                                                                        copy_output_ub[data_move_ub_offset_new:], 0,
                                                                        data_move_burst_num, data_move_burst_len, 0,
                                                                        data_move_dst_stride)
                                    with self.tik_instance.if_scope(w_copy_tail != 0):
                                        with self.tik_instance.for_range(0, do_nc_num) as nc_idx:
                                            data_move_burst_len = \
                                                w_copy_tail * self.images_shape_c0 // self.output_block_num
                                            new_output_gm_offset = \
                                                output_gm_offset * self.images_shape_c0 \
                                                + w_copy_num * w_output_size_one_line * self.images_shape_c0 \
                                                + nc_idx * tiling_out_width * tiling_out_height * self.images_shape_c0
                                            data_move_dst_stride = 0
                                            data_move_ub_offset_new = \
                                                data_move_ub_offset \
                                                + nc_idx * w_output_size_one_line * self.images_shape_c0
                                            self.tik_instance.data_move(self.out_gm[new_output_gm_offset:],
                                                                        copy_output_ub[data_move_ub_offset_new:], 0,
                                                                        data_move_burst_num, data_move_burst_len, 0,
                                                                        data_move_dst_stride)

                _do_one_height(h_do_len - 1, image_out_ub, image_input_ub)

            # apply meme for nc double buff
            image_out_ub_ping = self.tik_instance.Tensor(self.output_dtype, (max_number_in_ub_left,),
                                                         name="image_out_ub_ping",
                                                         scope=tik.scope_ubuf)
            if self.images_dtype == self.output_dtype:
                image_input_ub_ping = self.tik_instance.Tensor(self.images_dtype, (max_number_in_ub_left,),
                                                               name="image_input_ub_ping",
                                                               scope=tik.scope_ubuf)
            else:
                image_input_ub_ping = self.tik_instance.Tensor(self.images_dtype, (max_number_in_ub_left * 2,),
                                                               name="image_input_ub_ping",
                                                               scope=tik.scope_ubuf)

            image_out_ub_pang = self.tik_instance.Tensor(self.output_dtype, (max_number_in_ub_left,),
                                                         name="image_out_ub_pang",
                                                         scope=tik.scope_ubuf)
            if self.images_dtype == self.output_dtype:
                image_input_ub_pang = self.tik_instance.Tensor(self.images_dtype, (max_number_in_ub_left,),
                                                               name="image_input_ub_pang",
                                                               scope=tik.scope_ubuf)
            else:
                image_input_ub_pang = self.tik_instance.Tensor(self.images_dtype, (max_number_in_ub_left * 2,),
                                                               name="image_input_ub_pang",
                                                               scope=tik.scope_ubuf)

            with self.tik_instance.for_range(0, nc_loop // 2) as nc_loop_idx:
                _do_single_nc(nc_segment, nc_loop_idx * 2, image_out_ub_ping, image_input_ub_ping)
                _do_single_nc(nc_segment, nc_loop_idx * 2 + 1, image_out_ub_pang, image_input_ub_pang)
            with self.tik_instance.if_scope(nc_loop % 2 == 1):
                _do_single_nc(nc_segment, nc_loop - 1, image_out_ub_ping, image_input_ub_ping)
            with self.tik_instance.if_scope(nc_tail != 0):
                _do_single_nc(nc_tail, nc_loop, image_out_ub_pang, image_input_ub_pang)

        def _run_h_loop_default(h_loop_idx, h_do_len):
            # calcu h idx
            _run_w_loop_default(0, size_w_n, h_loop_idx, h_do_len)

        _run_h_loop_default(core_height_start, core_height_num)

    def _function_default_apply_ub_vbi(self, vbi_dtype="float32"):
        """
        _function_default_apply_ub_vbi
        """
        # init tmp ub for int32/float32 h-1 w-1 0 1
        function_default_int32_ub = self.tik_instance.Tensor("int32", (8 * 8,),
                                                             name="default_const_int32_ub",
                                                             scope=tik.scope_ubuf)
        function_default_fp32_ub = self.tik_instance.Tensor("float32", (8 * 8,),
                                                            name="default_const_fp32_ub",
                                                            scope=tik.scope_ubuf)
        with self.tik_instance.new_stmt_scope():
            scalar_input = self.tik_instance.Scalar("int32", name="scalar_input")
            self.tik_instance.vector_dup(8, function_default_int32_ub[0:], 0, 1, 1, 8)
            self.tik_instance.vector_dup(8, function_default_int32_ub[8:], 1, 1, 1, 8)
            scalar_input.set_as(self.tiling_in_width - 1)
            self.tik_instance.vector_dup(8, function_default_int32_ub[16:], scalar_input, 1, 1, 8)
            scalar_input.set_as(self.tiling_in_height - 1)
            self.tik_instance.vector_dup(8, function_default_int32_ub[24:], scalar_input, 1, 1, 8)
            scalar_input.set_as(self.core_height_start)
            self.tik_instance.vector_dup(8, function_default_int32_ub[32:], scalar_input, 1, 1, 8)
            # block byte
            self.tik_instance.vector_dup(8, function_default_int32_ub[40:], self.block_num * self.inner_bytes_size, 1,
                                         1, 8)
            # neg block byte
            self.tik_instance.vector_dup(8, function_default_int32_ub[48:],
                                         -1 * self.block_num * self.inner_bytes_size, 1, 1, 8)
            util_tik_comm_func.tik_func_vconv(self.tik_instance, function_default_fp32_ub, function_default_int32_ub,
                                              64)

        mem_default = {}
        mem_default["zero"] = {"int32": function_default_int32_ub[0:], "fp32": function_default_fp32_ub[0:]}
        mem_default["one"] = {"int32": function_default_int32_ub[8:], "fp32": function_default_fp32_ub[8:]}
        mem_default["in_width"] = {"int32": function_default_int32_ub[16:], "fp32": function_default_fp32_ub[16:]}
        mem_default["in_height"] = {"int32": function_default_int32_ub[24:], "fp32": function_default_fp32_ub[24:]}
        mem_default["height_start"] = {"int32": function_default_int32_ub[32:], "fp32": function_default_fp32_ub[32:]}
        mem_default["block_byte_pos"] = {
            "int32": function_default_int32_ub[40:],
            "fp32": function_default_fp32_ub[40:]
        }
        mem_default["block_byte_neg"] = {
            "int32": function_default_int32_ub[48:],
            "fp32": function_default_fp32_ub[48:]
        }

        # apply resize vbi ub
        mem_default["src0_offset"] = self.tik_instance.Tensor("int32", (self.width_idx_segment_num * 4,),
                                                              name="src0_offset_ub",
                                                              scope=tik.scope_ubuf)
        mem_default["src1"] = self.tik_instance.Tensor("float32", (self.width_idx_segment_num * 4,),
                                                       name="src1_ub",
                                                       scope=tik.scope_ubuf)

        mem_default["src1_vbi_ping"] = self.tik_instance.Tensor(vbi_dtype, (self.width_idx_segment_num * 4,),
                                                                name="src1_vbi_ping",
                                                                scope=tik.scope_ubuf)
        mem_default["src1_vbi_pang"] = self.tik_instance.Tensor(vbi_dtype, (self.width_idx_segment_num * 4,),
                                                                name="src1_vbi_pang",
                                                                scope=tik.scope_ubuf)
        if self.output_dtype != vbi_dtype:
            mem_default["output_vbi_ub_ping"] = self.tik_instance.Tensor(
                vbi_dtype, (self.width_idx_segment_num * self.images_shape_c0,),
                name="output_vbi_ub_ping",
                scope=tik.scope_ubuf)
            mem_default["output_vbi_ub_pang"] = mem_default.get("output_vbi_ub_ping")
        else:
            mem_default["output_vbi_ub_ping"] = self.tik_instance.Tensor(
                vbi_dtype, (self.width_idx_segment_num * self.images_shape_c0,),
                name="output_vbi_ub_ping",
                scope=tik.scope_ubuf)
            mem_default["output_vbi_ub_pang"] = self.tik_instance.Tensor(
                vbi_dtype, (self.width_idx_segment_num * self.images_shape_c0,),
                name="output_vbi_ub_pang",
                scope=tik.scope_ubuf)

        mem_default["src0_ping"] = self.tik_instance.Tensor(self.images_dtype,
                                                            (self.width_idx_segment_num * self.images_shape_c0 * 4,),
                                                            name="src0_ping",
                                                            scope=tik.scope_ubuf)
        mem_default["src0_pang"] = self.tik_instance.Tensor(self.images_dtype,
                                                            (self.width_idx_segment_num * self.images_shape_c0 * 4,),
                                                            name="src0_pang",
                                                            scope=tik.scope_ubuf)

        return mem_default

    # 'pylint: disable=unused-argument
    def _function_default_vbi_fp32(self,
                                   is_src_stride_copy=False,
                                   is_dst_stride_copy=False,
                                   is_w_algin=False,
                                   is_big_to_small=False):
        """
        _function_default, run this
        """
        self.height_idx_segment_num = 64
        self.width_idx_segment_num = 128
        mem_info = self._function_default_apply_ub_vbi()
        mem_info["src0_mid"] = None
        if self.inner_dtype != self.images_dtype:
            mem_info["src0_mid"] = \
                self.tik_instance.Tensor(self.inner_dtype, (self.width_idx_segment_num * self.images_shape_c0 * 4,),
                                         name="src0_mid", scope=tik.scope_ubuf)

        src0_offset = mem_info.get("src0_offset")
        src1 = mem_info.get("src1")

        self.idx_ub_fp32 = self.tik_instance.Tensor("float32", (self.width_idx_segment_num,),
                                                    name="idx_ub_fp32",
                                                    scope=tik.scope_ubuf)
        scalar_in_height = self.tik_instance.Scalar("int64", name="scalar_in_height")
        scalar_in_height.set_as(self.tiling_in_height - 1)

        # gen 0 - self.width_idx_segment_num to l1 fp32
        with self.tik_instance.new_stmt_scope():
            fill_index_in_ub(self.tik_instance, self.idx_ub_fp32, self.width_idx_segment_num)

        scalar_idx_fp32 = self.tik_instance.Scalar("float32", name="scalar_idx_fp32")
        # vconv start idx from int32 scalar to fp32 scalar
        self.scalar_vconv_int32_to_fp32(self.core_width_start, scalar_idx_fp32)
        # do vadds 0,1,2,3,4 + fp32_w_start_offset for per core
        self.tik_instance.vadds(64, self.idx_ub_fp32, self.idx_ub_fp32, scalar_idx_fp32,
                                (self.width_idx_segment_num + 63) // 64, 1, 1, 8, 8)

        # init a scalar for w segment one time
        w_loop_segment = self.tik_instance.Scalar("int32",
                                                  name="w_loop_segment",
                                                  init_value=self.width_idx_segment_num)

        with self.tik_instance.if_scope(tik.all(self.core_width_num < w_loop_segment, self.core_width_num > 0)):
            w_loop_segment.set_as(self.core_width_num)

        # when big w -> small w, the input will be overload
        w_input_output_rate = (self.tiling_in_width + self.tiling_out_width - 1) // self.tiling_out_width
        max_input_w = self.width_idx_segment_num * 2
        max_output_w = max_input_w // w_input_output_rate
        with self.tik_instance.if_scope(w_loop_segment > max_output_w):
            w_loop_segment.set_as(max_output_w)
        with self.tik_instance.if_scope(w_loop_segment == 0):
            w_loop_segment.set_as(1)

        w_loop_num = self.tik_instance.Scalar("int32", name="w_loop_num")
        w_tail_num = self.tik_instance.Scalar("int32", name="w_tail_num")
        nc_max_segment = self.tik_instance.Scalar("int32", name="nc_max_segment")
        nc_loop = self.tik_instance.Scalar("int32", name="nc_loop")
        nc_tail = self.tik_instance.Scalar("int32", name="nc_tail")

        w_loop_num.set_as(self.core_width_num // w_loop_segment)
        w_tail_num.set_as(self.core_width_num % w_loop_segment)
        nc_max_segment.set_as(self.ub_max_num // (w_loop_segment * self.images_shape_c0))
        with self.tik_instance.if_scope(tik.all(self.core_nc_num < nc_max_segment, self.core_nc_num > 0)):
            nc_max_segment.set_as(self.core_nc_num)
        nc_max_segment.set_as(1)
        nc_loop.set_as(self.core_nc_num // nc_max_segment)
        nc_tail.set_as(self.core_nc_num % nc_max_segment)
        nc_process_num_ub = self.tik_instance.Tensor("int32", (self.block_num,),
                                                     name="nc_process_num_ub",
                                                     scope=tik.scope_ubuf)
        nc_loop_num_ceil = self.tik_instance.Scalar("int32", name="nc_loop_num_ceil")
        nc_loop_num_ceil.set_as((self.core_nc_num + nc_max_segment - 1) // nc_max_segment)
        nc_loop_num_floor = nc_loop
        nc_process_num_ub[0].set_as(nc_max_segment)
        nc_process_num_ub[1].set_as(nc_tail)

        w_process_num_ub = self.tik_instance.Tensor("int32", (self.block_num,),
                                                    name="w_process_num_ub",
                                                    scope=tik.scope_ubuf)
        w_loop_num_ceil = self.tik_instance.Scalar("int32", name="w_loop_num_ceil")
        w_loop_num_ceil.set_as((self.core_width_num + w_loop_segment - 1) // w_loop_segment)
        w_loop_num_floor = w_loop_num
        w_process_num_ub[0].set_as(w_loop_segment)
        w_process_num_ub[1].set_as(w_tail_num)

        self.scalar_vconv_int32_to_fp32(w_loop_segment, scalar_idx_fp32)

        # for start_h idx and weight
        height_idx_ub_fp32_start = self.tik_instance.Tensor("float32", (64,),
                                                            name="height_idx_ub_fp32_start",
                                                            scope=tik.scope_ubuf)

        self.tik_instance.vadds(64, height_idx_ub_fp32_start, mem_info.get("height_start").get("fp32"),
                                0.0, 1, 1, 0, 8, 0)
        height_idx_ub_start = self.tik_instance.Tensor("int32", (64,), name="height_idx", scope=tik.scope_ubuf)
        height_weight_fp32_start = self.tik_instance.Tensor("float32", (64,),
                                                            name="height_weight_fp32_start",
                                                            scope=tik.scope_ubuf)
        self.calcu_out_in_idx(self.resize_scale_h,
                              height_idx_ub_start,
                              height_idx_ub_fp32_start,
                              8,
                              des_weight_fp32_ub_list=[height_weight_fp32_start, height_weight_fp32_start[8:]],
                              max_idx_ub=mem_info.get("in_height").get("fp32"),
                              max_idx_ub_int=mem_info.get("in_height").get("int32"),
                              one_fp32_ub=mem_info.get("one").get("fp32"),
                              des_idx_ub_1=height_idx_ub_start[8:],
                              one_int32_ub=mem_info.get("one").get("int32"),
                              mem_info=mem_info)

        def _run_w_loop_default(w_loop_idx, w_do_len, h_loop_offset, h_do_len):
            w_gm_offset = w_loop_idx * w_loop_segment + self.core_width_start
            self.calcu_out_in_idx_vbi(self.resize_scale_w, src0_offset, src1, self.idx_ub_fp32, mem_info,
                                      self.width_idx_segment_num, src_start=self.src_start_w,
                                      dst_start=self.dst_start_w)
            self.tik_instance.vadds(64, self.idx_ub_fp32, self.idx_ub_fp32, scalar_idx_fp32,
                                    (self.width_idx_segment_num + 63) // 64, 1, 1, 8, 8)

            scalar_w_start_idx = self.tik_instance.Scalar("int32", name="scalar_w_start_idx")
            scalar_w_end_idx = self.tik_instance.Scalar("int32", name="scalar_w_end_idx")
            input_w_len = self.tik_instance.Scalar("int32", name="input_w_len")

            scalar_w_start_idx.set_as(src0_offset[0])
            scalar_w_end_idx.set_as(src0_offset[(w_do_len - 1) // 8 * 32 + (w_do_len - 1) % 8])
            scalar_w_start_idx_neg = self.tik_instance.Scalar("int32", name="scalar_w_start_idx_neg")
            scalar_w_start_idx_neg.set_as(0 - scalar_w_start_idx)
            with self.tik_instance.if_scope(scalar_w_end_idx < self.tiling_in_width - 1):
                input_w_len.set_as(scalar_w_end_idx - scalar_w_start_idx + 1 + self.is_bilinear)
            with self.tik_instance.else_scope():
                input_w_len.set_as(scalar_w_end_idx - scalar_w_start_idx + 1)

            tmp_ub = self.tik_instance.Tensor("int32", (8,), name="tmp_ub", scope=tik.scope_ubuf)
            util_tik_comm_func.tik_func_vector(self.tik_instance, tmp_ub, scalar_w_start_idx_neg, 8)
            # for vbi src0_offset
            util_tik_comm_func.tik_func_vcomple(self.tik_instance,
                                                "vadd",
                                                src0_offset,
                                                src0_offset,
                                                tmp_ub,
                                                self.width_idx_segment_num * 4,
                                                src1_blk=0,
                                                src1_rep=0)

            util_tik_comm_func.tik_func_vector(self.tik_instance, tmp_ub, input_w_len, 8)
            util_tik_comm_func.tik_func_vcomple(self.tik_instance,
                                                "vadd",
                                                src0_offset[16:],
                                                src0_offset,
                                                tmp_ub,
                                                self.width_idx_segment_num,
                                                src1_blk=0,
                                                src1_rep=0,
                                                src0_blk=4,
                                                dst_blk=4,
                                                src0_rep=32,
                                                dst_rep=32)
            util_tik_comm_func.tik_func_vcomple(self.tik_instance,
                                                "vadd",
                                                src0_offset[24:],
                                                src0_offset[8:],
                                                tmp_ub,
                                                self.width_idx_segment_num,
                                                src1_blk=0,
                                                src1_rep=0,
                                                src0_blk=4,
                                                dst_blk=4,
                                                src0_rep=32,
                                                dst_rep=32)
            util_tik_comm_func.tik_func_vector(self.tik_instance, tmp_ub, self.images_shape_c0 * 4, 8)
            util_tik_comm_func.tik_func_vcomple(self.tik_instance,
                                                "vmul",
                                                src0_offset,
                                                src0_offset,
                                                tmp_ub,
                                                self.width_idx_segment_num * 4,
                                                src1_blk=0,
                                                src1_rep=0)

            # one segment h and one segment w
            def _do_single_nc(do_nc_num, _nc_loop_idx):
                # copy the h start info to ub before h loop
                height_idx_ub_fp32 = self.tik_instance.Tensor("float32", (64,),
                                                              name="height_idx_ub_fp32",
                                                              scope=tik.scope_ubuf)
                height_idx_ub = self.tik_instance.Tensor("int32", (64,), name="height_idx_ub", scope=tik.scope_ubuf)
                height_weight_ub = self.tik_instance.Tensor(self.inner_dtype, (64,),
                                                            name="height_weight_ub",
                                                            scope=tik.scope_ubuf)
                self.tik_instance.data_move(height_idx_ub_fp32, height_idx_ub_fp32_start, 0, 1, 8, 0, 0)
                self.tik_instance.data_move(height_idx_ub, height_idx_ub_start, 0, 1, 8, 0, 0)
                self.tik_instance.data_move(height_weight_ub, height_weight_fp32_start, 0, 1, 8, 0, 0)

                def _do_one_height(h_idx, ub_list):
                    src0_ub = ub_list[0]
                    output_ub = ub_list[1]
                    src1_ub = ub_list[2]
                    mid_ub = ub_list[3]
                    h_gm_offset = h_idx + h_loop_offset
                    scalar_in_h_idx = self.tik_instance.Scalar("int32", name="scalar_in_h_idx")
                    scalar_in_h_idx.set_as(height_idx_ub[0])
                    scalar_in_h_idx_1 = self.tik_instance.Scalar("int32", name="scalar_in_h_idx")
                    scalar_in_h_idx_1.set_as(height_idx_ub[8])
                    scalar_in_h_weight = self.tik_instance.Scalar("float32", name="scalar_in_h_weight")
                    scalar_in_h_weight.set_as(height_weight_ub[0])
                    scalar_in_h_weight_1 = self.tik_instance.Scalar("float32", name="scalar_in_h_weight_1")
                    scalar_in_h_weight_1.set_as(height_weight_ub[8])
                    # calcu next h idx in height_idx_ub_fp32
                    self.tik_instance.vadds(8, height_idx_ub_fp32, height_idx_ub_fp32, 1.0, 1, 1, 1, 8, 8)
                    self.calcu_out_in_idx(self.resize_scale_h,
                                          height_idx_ub,
                                          height_idx_ub_fp32,
                                          8,
                                          des_weight_fp32_ub_list=[height_weight_ub, height_weight_ub[8:]],
                                          max_idx_ub=mem_info.get("in_height").get("fp32"),
                                          max_idx_ub_int=mem_info.get("in_height").get("int32"),
                                          one_fp32_ub=mem_info.get("one").get("fp32"),
                                          des_idx_ub_1=height_idx_ub[8:],
                                          one_int32_ub=mem_info.get("one").get("int32"),
                                          mem_info=mem_info)

                    nc_gm_input_offset = \
                        (_nc_loop_idx * nc_max_segment + self.core_nc_start) * self.tiling_in_width \
                        * self.tiling_in_height * self.images_shape_c0

                    with self.tik_instance.new_stmt_scope(disable_sync=True):
                        h_gm_input_offset = \
                            (scalar_in_h_idx * self.tiling_in_width + scalar_w_start_idx) * self.images_shape_c0

                        top_gm_offset = nc_gm_input_offset + h_gm_input_offset
                        ub_info = [[self.images_gm, top_gm_offset], [src0_ub, 0]]
                        # copy top value
                        self._function_data_move(1, input_w_len, ub_info, 1)

                        # copy bottom value
                        h_gm_input_offset = \
                            (scalar_in_h_idx_1 * self.tiling_in_width + scalar_w_start_idx) * self.images_shape_c0
                        bottom_gm_offset = nc_gm_input_offset + h_gm_input_offset

                        ub_info = [[self.images_gm, bottom_gm_offset], [src0_ub, input_w_len * self.images_shape_c0]]
                        self._function_data_move(1, input_w_len, ub_info, 1)

                    if mid_ub is not None:
                        util_tik_comm_func.tik_func_vconv(self.tik_instance, mid_ub, src0_ub, src0_ub.shape[0])
                        src0_ub = mid_ub

                    vmuls_value = scalar_in_h_weight_1
                    # get (1 - x_lerp) * (1 - y lerp)
                    util_tik_comm_func.tik_func_vmuls(self.tik_instance,
                                                      src1_ub[0:],
                                                      src1[8:],
                                                      vmuls_value,
                                                      self.width_idx_segment_num,
                                                      src_blk=4,
                                                      dst_blk=4,
                                                      src_rep=32,
                                                      dst_rep=32)
                    # get x_lerp * (1 - y lerp)
                    util_tik_comm_func.tik_func_vmuls(self.tik_instance,
                                                      src1_ub[8:],
                                                      src1[0:],
                                                      vmuls_value,
                                                      self.width_idx_segment_num,
                                                      src_blk=4,
                                                      dst_blk=4,
                                                      src_rep=32,
                                                      dst_rep=32)
                    vmuls_value = scalar_in_h_weight
                    # get (1 - x_lerp) * y lerp
                    util_tik_comm_func.tik_func_vmuls(self.tik_instance,
                                                      src1_ub[2 * 8:],
                                                      src1[8:],
                                                      vmuls_value,
                                                      self.width_idx_segment_num,
                                                      src_blk=4,
                                                      dst_blk=4,
                                                      src_rep=32,
                                                      dst_rep=32)
                    # get x_lerp * y lerp
                    util_tik_comm_func.tik_func_vmuls(self.tik_instance,
                                                      src1_ub[3 * 8:],
                                                      src1[0:],
                                                      vmuls_value,
                                                      self.width_idx_segment_num,
                                                      src_blk=4,
                                                      dst_blk=4,
                                                      src_rep=32,
                                                      dst_rep=32)
                    # vbi can not support scalar repeat, now use max repeat to replace scalar (w_do_len + 7) // 8
                    self.tik_instance.vbi(64, output_ub, src0_ub, src1_ub, src0_offset, 2,
                                          self.width_idx_segment_num // 8, 4, 1, 8 * 8 * 2)
                    util_tik_comm_func.tik_func_vcomple(self.tik_instance,
                                                        "vadd",
                                                        src0_offset,
                                                        src0_offset,
                                                        mem_info.get("block_byte_pos").get("int32"),
                                                        self.width_idx_segment_num * 4,
                                                        src1_blk=0,
                                                        src1_rep=0)
                    # vbi can not support scalar repeat, now use max repeat to replace  scalar (w_do_len + 7) // 8
                    self.tik_instance.vbi(64, output_ub[8:], src0_ub, src1_ub, src0_offset, 2,
                                          self.width_idx_segment_num // 8, 4, 1, 8 * 8 * 2)
                    util_tik_comm_func.tik_func_vcomple(self.tik_instance,
                                                        "vadd",
                                                        src0_offset,
                                                        src0_offset,
                                                        mem_info.get("block_byte_neg").get("int32"),
                                                        self.width_idx_segment_num * 4,
                                                        src1_blk=0,
                                                        src1_rep=0)
                    if self.output_dtype != output_ub.dtype:
                        output_mid_ub = output_ub.reinterpret_cast_to(self.output_dtype)
                        output_num = do_nc_num * w_do_len * self.images_shape_c0
                        self.tik_instance.vconv(64, "", output_mid_ub, output_ub, (output_num + 63) // 64, 1, 1, 4, 8)
                        output_ub = output_mid_ub[0:]
                    nc_gm_offset = \
                        (_nc_loop_idx * nc_max_segment + self.core_nc_start) \
                        * self.tiling_out_height * self.tiling_out_width
                    output_gm_offset = nc_gm_offset + h_gm_offset * self.tiling_out_width + w_gm_offset
                    data_move_dst_mem = self.out_gm
                    data_move_src_men = output_ub
                    data_move_dst_offset_new = output_gm_offset * self.images_shape_c0
                    data_move_src_offset_new = 0
                    data_move_burst_num = do_nc_num
                    data_move_burst_len = w_do_len * self.images_shape_c0 // self.output_block_num
                    self.tik_instance.data_move(data_move_dst_mem[data_move_dst_offset_new:],
                                                data_move_src_men[data_move_src_offset_new:], 0, data_move_burst_num,
                                                data_move_burst_len, 0, 0)

                ub_list_ping = [
                    mem_info.get("src0_ping"), mem_info.get("output_vbi_ub_ping"), mem_info.get("src1_vbi_ping"),
                    mem_info.get("src0_mid")
                ]
                ub_list_pang = [
                    mem_info.get("src0_pang"), mem_info.get("output_vbi_ub_pang"), mem_info.get("src1_vbi_pang"),
                    mem_info.get("src0_mid")
                ]
                with self.tik_instance.for_range(0, h_do_len // 2) as _h_idx:
                    _do_one_height(_h_idx * 2 + 0, ub_list_ping)
                    _do_one_height(_h_idx * 2 + 1, ub_list_pang)
                with self.tik_instance.if_scope(h_do_len % 2 != 0):
                    _do_one_height(h_do_len - 1, ub_list_ping)

            with self.tik_instance.for_range(0, nc_loop_num_ceil) as nc_loop_idx:
                nc_loop_do_num = self.tik_instance.Scalar("int32", name="nc_loop_do_num")
                nc_loop_do_num.set_as(nc_process_num_ub[nc_loop_idx // nc_loop_num_floor])
                nc_loop_do_num = 1
                _do_single_nc(nc_loop_do_num, nc_loop_idx)

        def _run_h_loop_default(h_loop_idx, h_do_len):
            h_loop_segment_start = h_loop_idx * self.height_idx_segment_num + self.core_height_start
            h_gm_offset = h_loop_segment_start
            # calcu h idx

            with self.tik_instance.for_range(0, w_loop_num_ceil) as w_loop_idx:
                w_loop_do_num = self.tik_instance.Scalar("int32", name="w_loop_do_num")
                w_loop_do_num.set_as(w_process_num_ub[w_loop_idx // w_loop_num_floor])
                _run_w_loop_default(w_loop_idx, w_loop_do_num, h_gm_offset, h_do_len)

        _run_h_loop_default(0, self.core_height_num)

    # 'pylint: disable=unused-argument
    def _function_default_vbi_fp16(self,
                                   is_src_stride_copy=False,
                                   is_dst_stride_copy=False,
                                   is_w_algin=False,
                                   is_big_to_small=False):
        """
        _function_default, run this
        """
        self.height_idx_segment_num = 64
        self.width_idx_segment_num = 128
        mem_info = self._function_default_apply_ub_vbi(vbi_dtype="float16")
        output_ub_ping = None
        output_ub_pang = None
        if self.output_dtype != "float16":
            output_ub_ping = self.tik_instance.Tensor(self.output_dtype,
                                                      (self.width_idx_segment_num * self.images_shape_c0,),
                                                      name="output_ub_ping",
                                                      scope=tik.scope_ubuf)
            output_ub_pang = self.tik_instance.Tensor(self.output_dtype,
                                                      (self.width_idx_segment_num * self.images_shape_c0,),
                                                      name="output_ub_pang",
                                                      scope=tik.scope_ubuf)

        src0_offset = mem_info.get("src0_offset")
        src1 = mem_info.get("src1")
        src0_offset_vbi_ping = self.tik_instance.Tensor(src0_offset.dtype,
                                                        src0_offset.shape,
                                                        name="src0_offset_vbi_ping",
                                                        scope=tik.scope_ubuf)
        src1_vbi_fp32_ping = self.tik_instance.Tensor(src1.dtype,
                                                      src1.shape,
                                                      name="src1_vbi_fp32_ping",
                                                      scope=tik.scope_ubuf)

        self.idx_ub_fp32 = self.tik_instance.Tensor("float32", (self.width_idx_segment_num,),
                                                    name="idx_ub_fp32",
                                                    scope=tik.scope_ubuf)
        scalar_in_height = self.tik_instance.Scalar("int64", name="scalar_in_height")
        scalar_in_height.set_as(self.tiling_in_height - 1)

        # gen 0 - self.width_idx_segment_num to l1 fp32
        with self.tik_instance.new_stmt_scope():
            fill_index_in_ub(self.tik_instance, self.idx_ub_fp32, self.width_idx_segment_num)

        scalar_idx_fp32 = self.tik_instance.Scalar("float32", name="scalar_idx_fp32")
        # vconv start idx from int32 scalar to fp32 scalar
        self.scalar_vconv_int32_to_fp32(self.core_width_start, scalar_idx_fp32)
        # do vadds 0,1,2,3,4 + fp32_w_start_offset for per core
        self.tik_instance.vadds(64, self.idx_ub_fp32, self.idx_ub_fp32, scalar_idx_fp32,
                                (self.width_idx_segment_num + 63) // 64, 1, 1, 8, 8)

        # init a scalar for w segment one time
        w_loop_segment = self.tik_instance.Scalar("int32",
                                                  name="w_loop_segment",
                                                  init_value=self.width_idx_segment_num)

        with self.tik_instance.if_scope(tik.all(self.core_width_num < w_loop_segment, self.core_width_num > 0)):
            w_loop_segment.set_as(self.core_width_num)

        # when big w -> small w, the input will be overload
        w_input_output_rate = (self.tiling_in_width + self.tiling_out_width - 1) // self.tiling_out_width
        max_input_w = self.width_idx_segment_num * 2
        max_output_w = max_input_w // w_input_output_rate
        with self.tik_instance.if_scope(w_loop_segment > max_output_w):
            w_loop_segment.set_as(max_output_w)
        with self.tik_instance.if_scope(w_loop_segment == 0):
            w_loop_segment.set_as(1)

        w_loop_num = self.tik_instance.Scalar("int32", name="w_loop_num")
        w_tail_num = self.tik_instance.Scalar("int32", name="w_tail_num")
        nc_max_segment = self.tik_instance.Scalar("int32", name="nc_max_segment")
        nc_loop = self.tik_instance.Scalar("int32", name="nc_loop")
        nc_tail = self.tik_instance.Scalar("int32", name="nc_tail")

        w_loop_num.set_as(self.core_width_num // w_loop_segment)
        w_tail_num.set_as(self.core_width_num % w_loop_segment)
        nc_max_segment.set_as(self.ub_max_num // (w_loop_segment * self.images_shape_c0))
        with self.tik_instance.if_scope(tik.all(self.core_nc_num < nc_max_segment, self.core_nc_num > 0)):
            nc_max_segment.set_as(self.core_nc_num)
        nc_max_segment.set_as(1)
        nc_loop.set_as(self.core_nc_num // nc_max_segment)
        nc_tail.set_as(self.core_nc_num % nc_max_segment)
        nc_process_num_ub = self.tik_instance.Tensor("int32", (self.block_num,),
                                                     name="nc_process_num_ub",
                                                     scope=tik.scope_ubuf)
        nc_loop_num_ceil = self.tik_instance.Scalar("int32", name="nc_loop_num_ceil")
        nc_loop_num_ceil.set_as((self.core_nc_num + nc_max_segment - 1) // nc_max_segment)
        nc_loop_num_floor = nc_loop
        nc_process_num_ub[0].set_as(nc_max_segment)
        nc_process_num_ub[1].set_as(nc_tail)

        w_process_num_ub = self.tik_instance.Tensor("int32", (self.block_num,),
                                                    name="w_process_num_ub",
                                                    scope=tik.scope_ubuf)
        w_loop_num_ceil = self.tik_instance.Scalar("int32", name="w_loop_num_ceil")
        w_loop_num_ceil.set_as((self.core_width_num + w_loop_segment - 1) // w_loop_segment)
        w_loop_num_floor = w_loop_num
        w_process_num_ub[0].set_as(w_loop_segment)
        w_process_num_ub[1].set_as(w_tail_num)

        self.scalar_vconv_int32_to_fp32(w_loop_segment, scalar_idx_fp32)

        # for start_h idx and weight
        height_idx_ub_fp32_start = self.tik_instance.Tensor("float32", (64,),
                                                            name="height_idx_ub_fp32_start",
                                                            scope=tik.scope_ubuf)

        self.tik_instance.vadds(64, height_idx_ub_fp32_start, mem_info.get("height_start").get("fp32"),
                                0.0, 1, 1, 0, 8, 0)
        height_idx_ub_start = self.tik_instance.Tensor("int32", (64,), name="height_idx", scope=tik.scope_ubuf)
        height_weight_fp32_start = self.tik_instance.Tensor("float32", (64,),
                                                            name="height_weight_fp32_start",
                                                            scope=tik.scope_ubuf)
        self.calcu_out_in_idx(self.resize_scale_h,
                              height_idx_ub_start,
                              height_idx_ub_fp32_start,
                              8,
                              des_weight_fp32_ub_list=[height_weight_fp32_start, height_weight_fp32_start[8:]],
                              max_idx_ub=mem_info.get("in_height").get("fp32"),
                              max_idx_ub_int=mem_info.get("in_height").get("int32"),
                              one_fp32_ub=mem_info.get("one").get("fp32"),
                              des_idx_ub_1=height_idx_ub_start[8:],
                              one_int32_ub=mem_info.get("one").get("int32"),
                              mem_info=mem_info)

        def _run_w_loop_default(w_loop_idx, w_do_len, h_loop_offset, h_do_len):
            w_gm_offset = w_loop_idx * w_loop_segment + self.core_width_start
            self.calcu_out_in_idx_vbi(self.resize_scale_w, src0_offset, src1, self.idx_ub_fp32, mem_info,
                                      self.width_idx_segment_num, src_start=self.src_start_w,
                                      dst_start=self.dst_start_w)
            self.tik_instance.vadds(64, self.idx_ub_fp32, self.idx_ub_fp32, scalar_idx_fp32,
                                    (self.width_idx_segment_num + 63) // 64, 1, 1, 8, 8)

            scalar_w_start_idx = self.tik_instance.Scalar("int32", name="scalar_w_start_idx")
            scalar_w_end_idx = self.tik_instance.Scalar("int32", name="scalar_w_end_idx")
            input_w_len = self.tik_instance.Scalar("int32", name="input_w_len")

            scalar_w_start_idx.set_as(src0_offset[0])
            scalar_w_end_idx.set_as(src0_offset[(w_do_len - 1) // 8 * 32 + (w_do_len - 1) % 8])
            scalar_w_start_idx_neg = self.tik_instance.Scalar("int32", name="scalar_w_start_idx_neg")
            scalar_w_start_idx_neg.set_as(0 - scalar_w_start_idx)
            with self.tik_instance.if_scope(scalar_w_end_idx < self.tiling_in_width - 1):
                input_w_len.set_as(scalar_w_end_idx - scalar_w_start_idx + 1 + self.is_bilinear)
            with self.tik_instance.else_scope():
                input_w_len.set_as(scalar_w_end_idx - scalar_w_start_idx + 1)

            tmp_ub = self.tik_instance.Tensor("int32", (8,), name="tmp_ub", scope=tik.scope_ubuf)
            util_tik_comm_func.tik_func_vector(self.tik_instance, tmp_ub, scalar_w_start_idx_neg, 8)
            # for vbi src0_offset
            util_tik_comm_func.tik_func_vcomple(self.tik_instance,
                                                "vadd",
                                                src0_offset,
                                                src0_offset,
                                                tmp_ub,
                                                self.width_idx_segment_num * 4,
                                                src1_blk=0,
                                                src1_rep=0)

            util_tik_comm_func.tik_func_vector(self.tik_instance, tmp_ub, input_w_len, 8)
            util_tik_comm_func.tik_func_vcomple(self.tik_instance,
                                                "vadd",
                                                src0_offset[16:],
                                                src0_offset,
                                                tmp_ub,
                                                self.width_idx_segment_num,
                                                src1_blk=0,
                                                src1_rep=0,
                                                src0_blk=4,
                                                dst_blk=4,
                                                src0_rep=32,
                                                dst_rep=32)
            util_tik_comm_func.tik_func_vcomple(self.tik_instance,
                                                "vadd",
                                                src0_offset[24:],
                                                src0_offset[8:],
                                                tmp_ub,
                                                self.width_idx_segment_num,
                                                src1_blk=0,
                                                src1_rep=0,
                                                src0_blk=4,
                                                dst_blk=4,
                                                src0_rep=32,
                                                dst_rep=32)
            util_tik_comm_func.tik_func_vector(self.tik_instance, tmp_ub, self.images_shape_c0 * self.input_bytes_size,
                                               8)
            util_tik_comm_func.tik_func_vcomple(self.tik_instance,
                                                "vmul",
                                                src0_offset,
                                                src0_offset,
                                                tmp_ub,
                                                self.width_idx_segment_num * 4,
                                                src1_blk=0,
                                                src1_rep=0)

            # one segment h and one segment w
            def _do_single_nc(do_nc_num, _nc_loop_idx):
                # copy the h start info to ub before h loop
                height_idx_ub_fp32 = self.tik_instance.Tensor("float32", (64,),
                                                              name="height_idx_ub_fp32",
                                                              scope=tik.scope_ubuf)
                height_idx_ub = self.tik_instance.Tensor("int32", (64,), name="height_idx_ub", scope=tik.scope_ubuf)
                height_weight_ub = self.tik_instance.Tensor(self.inner_dtype, (64,),
                                                            name="height_weight_ub",
                                                            scope=tik.scope_ubuf)
                self.tik_instance.data_move(height_idx_ub_fp32, height_idx_ub_fp32_start, 0, 1, 8, 0, 0)
                self.tik_instance.data_move(height_idx_ub, height_idx_ub_start, 0, 1, 8, 0, 0)
                self.tik_instance.data_move(height_weight_ub, height_weight_fp32_start, 0, 1, 8, 0, 0)

                def _do_one_height(h_idx, ub_list):
                    src0_ub = ub_list[0]
                    output_vbi_ub = ub_list[1]
                    src1_ub = src1_vbi_fp32_ping
                    src1_vbi_ub = ub_list[2]
                    src0_vbi_offset = ub_list[3]
                    mid_ub = ub_list[4]
                    h_gm_offset = h_idx + h_loop_offset
                    scalar_in_h_idx = self.tik_instance.Scalar("int32", name="scalar_in_h_idx")
                    scalar_in_h_idx.set_as(height_idx_ub[0])
                    scalar_in_h_idx_1 = self.tik_instance.Scalar("int32", name="scalar_in_h_idx")
                    scalar_in_h_idx_1.set_as(height_idx_ub[8])
                    scalar_in_h_weight = self.tik_instance.Scalar("float32", name="scalar_in_h_weight")
                    scalar_in_h_weight.set_as(height_weight_ub[0])
                    scalar_in_h_weight_1 = self.tik_instance.Scalar("float32", name="scalar_in_h_weight_1")
                    scalar_in_h_weight_1.set_as(height_weight_ub[8])
                    # calcu next h idx in height_idx_ub_fp32
                    self.tik_instance.data_move(src0_vbi_offset, src0_offset, 0, 1,
                                                self.width_idx_segment_num * 4 // 8, 0, 0)
                    self.tik_instance.vadds(8, height_idx_ub_fp32, height_idx_ub_fp32, 1.0, 1, 1, 1, 8, 8)
                    self.calcu_out_in_idx(self.resize_scale_h,
                                          height_idx_ub,
                                          height_idx_ub_fp32,
                                          8,
                                          des_weight_fp32_ub_list=[height_weight_ub, height_weight_ub[8:]],
                                          max_idx_ub=mem_info.get("in_height").get("fp32"),
                                          max_idx_ub_int=mem_info.get("in_height").get("int32"),
                                          one_fp32_ub=mem_info.get("one").get("fp32"),
                                          des_idx_ub_1=height_idx_ub[8:],
                                          one_int32_ub=mem_info.get("one").get("int32"),
                                          mem_info=mem_info)

                    nc_gm_input_offset = \
                        (_nc_loop_idx * nc_max_segment + self.core_nc_start) * self.tiling_in_width \
                        * self.tiling_in_height * self.images_shape_c0

                    with self.tik_instance.new_stmt_scope(disable_sync=True):
                        h_gm_input_offset = \
                            (scalar_in_h_idx * self.tiling_in_width + scalar_w_start_idx) * self.images_shape_c0

                        top_gm_offset = nc_gm_input_offset + h_gm_input_offset
                        ub_info = [[self.images_gm, top_gm_offset], [src0_ub, 0]]
                        # copy top value
                        self._function_data_move(1, input_w_len, ub_info, 1)

                        # copy bottom value
                        h_gm_input_offset = \
                            (scalar_in_h_idx_1 * self.tiling_in_width + scalar_w_start_idx) * self.images_shape_c0
                        bottom_gm_offset = nc_gm_input_offset + h_gm_input_offset

                        ub_info = [[self.images_gm, bottom_gm_offset], [src0_ub, input_w_len * self.images_shape_c0]]
                        self._function_data_move(1, input_w_len, ub_info, 1)

                    vmuls_value = scalar_in_h_weight_1
                    # get (1 - x_lerp) * (1 - y lerp)
                    util_tik_comm_func.tik_func_vmuls(self.tik_instance,
                                                      src1_ub[0:],
                                                      src1[8:],
                                                      vmuls_value,
                                                      self.width_idx_segment_num,
                                                      src_blk=4,
                                                      dst_blk=4,
                                                      src_rep=32,
                                                      dst_rep=32)
                    # get x_lerp * (1 - y lerp)
                    util_tik_comm_func.tik_func_vmuls(self.tik_instance,
                                                      src1_ub[8:],
                                                      src1[0:],
                                                      vmuls_value,
                                                      self.width_idx_segment_num,
                                                      src_blk=4,
                                                      dst_blk=4,
                                                      src_rep=32,
                                                      dst_rep=32)
                    vmuls_value = scalar_in_h_weight
                    # get (1 - x_lerp) * y lerp
                    util_tik_comm_func.tik_func_vmuls(self.tik_instance,
                                                      src1_ub[2 * 8:],
                                                      src1[8:],
                                                      vmuls_value,
                                                      self.width_idx_segment_num,
                                                      src_blk=4,
                                                      dst_blk=4,
                                                      src_rep=32,
                                                      dst_rep=32)
                    # get x_lerp * y lerp
                    util_tik_comm_func.tik_func_vmuls(self.tik_instance,
                                                      src1_ub[3 * 8:],
                                                      src1[0:],
                                                      vmuls_value,
                                                      self.width_idx_segment_num,
                                                      src_blk=4,
                                                      dst_blk=4,
                                                      src_rep=32,
                                                      dst_rep=32)

                    util_tik_comm_func.tik_func_vconv(self.tik_instance, src1_vbi_ub, src1_ub, src1_ub.shape[0])

                    # vbi can not support scalar repeat, now use max repeat to replace scalar (w_do_len + 7) // 8
                    self.tik_instance.vbi(128, output_vbi_ub, src0_ub, src1_vbi_ub, src0_vbi_offset, 1,
                                          (w_do_len + 7) // 8, 4, 1, 128)
                    if mid_ub is not None:
                        total_output_num = w_do_len * self.images_shape_c0
                        self.tik_instance.vconv(64, "", mid_ub, output_vbi_ub, (total_output_num + 63) // 64, 1, 1, 8,
                                                4)
                        output_vbi_ub = mid_ub

                    nc_gm_offset = \
                        (_nc_loop_idx * nc_max_segment + self.core_nc_start) \
                        * self.tiling_out_height * self.tiling_out_width
                    output_gm_offset = nc_gm_offset + h_gm_offset * self.tiling_out_width + w_gm_offset
                    data_move_dst_mem = self.out_gm
                    data_move_src_men = output_vbi_ub
                    data_move_dst_offset_new = output_gm_offset * self.images_shape_c0
                    data_move_src_offset_new = 0
                    data_move_burst_num = do_nc_num
                    data_move_burst_len = w_do_len * self.images_shape_c0 // self.output_block_num
                    self.tik_instance.data_move(data_move_dst_mem[data_move_dst_offset_new:],
                                                data_move_src_men[data_move_src_offset_new:], 0, data_move_burst_num,
                                                data_move_burst_len, 0, 0)

                ub_list_ping = [
                    mem_info.get("src0_ping"), mem_info.get("output_vbi_ub_ping"), mem_info.get("src1_vbi_ping"),
                    src0_offset_vbi_ping, output_ub_ping
                ]
                ub_list_pang = [
                    mem_info.get("src0_pang"), mem_info.get("output_vbi_ub_pang"), mem_info.get("src1_vbi_pang"),
                    src0_offset_vbi_ping, output_ub_pang
                ]
                with self.tik_instance.for_range(0, h_do_len // 2) as _h_idx:
                    _do_one_height(_h_idx * 2 + 0, ub_list_ping)
                    _do_one_height(_h_idx * 2 + 1, ub_list_pang)
                with self.tik_instance.if_scope(h_do_len % 2 != 0):
                    _do_one_height(h_do_len - 1, ub_list_ping)

            with self.tik_instance.for_range(0, nc_loop_num_ceil) as nc_loop_idx:
                nc_loop_do_num = self.tik_instance.Scalar("int32", name="nc_loop_do_num")
                nc_loop_do_num.set_as(nc_process_num_ub[nc_loop_idx // nc_loop_num_floor])
                nc_loop_do_num = 1
                _do_single_nc(nc_loop_do_num, nc_loop_idx)

        def _run_h_loop_default(h_loop_idx, h_do_len):
            h_loop_segment_start = h_loop_idx * self.height_idx_segment_num + self.core_height_start
            h_gm_offset = h_loop_segment_start
            # calcu h idx

            with self.tik_instance.for_range(0, w_loop_num_ceil) as w_loop_idx:
                w_loop_do_num = self.tik_instance.Scalar("int32", name="w_loop_do_num")
                w_loop_do_num.set_as(w_process_num_ub[w_loop_idx // w_loop_num_floor])
                _run_w_loop_default(w_loop_idx, w_loop_do_num, h_gm_offset, h_do_len)

        _run_h_loop_default(0, self.core_height_num)

    def _tiling_compute_default(self):
        """
        default function
        """
        check_vcopy_supported = tbe_platform.api_check_support("tik.vcopy")
        check_vbi_supported = tbe_platform.api_check_support("tik.vbi", "float32")
        if check_vcopy_supported and check_vbi_supported:
            # for function tiling 100000 vbi fp32
            self._function_default_vbi_fp32()
        elif tbe_platform.api_check_support("tik.vbi", "float16") and self.images_dtype == "float16":
            # for function tiling 100000 vbi fp16
            self._function_default_vbi_fp16()
        else:
            # for others function tiling 100000 with l1
            with self.tik_instance.if_scope(self.tiling_out_height // self.tiling_in_height < 2):
                self._function_resize_with_l1_default(is_h_big_to_small=False)
            with self.tik_instance.if_scope(self.tiling_out_height // self.tiling_in_height >= 2):
                self._function_resize_with_l1_default(is_h_big_to_small=True)

    def _tiling_compute_with_no_bilinear(self):
        """
        _tiling_compute_with_no_bilinear,
        do case  (n,1,1,c  ->  n,h,w,c) and  (n,h,w,c  ->  n,h,w,c)
        and the case: n,h,w,c  ->  n,h,w,c == n*h*w,1,1,c -> n*h*w,1,1,c
        """
        with self.tik_instance.if_scope(self.tiling_out_height + self.tiling_out_width > 2):
            # process case: input h = 1 and input w = 1
            self._function_reisze_with_no_bilinear(is_equal=False)
        with self.tik_instance.if_scope(self.tiling_out_height + self.tiling_out_width == 2):
            # process case: input h = output h and input w = output w
            self._function_reisze_with_no_bilinear(is_equal=True)

    def _function_reisze_with_nc_process(self):
        """
        _function_reisze_with_nc_process
        """
        self.height_idx_segment_num = 256
        self.width_idx_segment_num = 256
        max_nc = 128
        mem_info = self._function_default_apply_ub()
        idx_ub_fp32 = self.tik_instance.Tensor("float32", (self.width_idx_segment_num,),
                                               name="idx_ub_fp32",
                                               scope=tik.scope_ubuf)
        width_idx_ub = self.tik_instance.Tensor("int32", (self.width_idx_segment_num * 2,),
                                                name="width_idx",
                                                scope=tik.scope_ubuf)
        width_weight_ub = self.tik_instance.Tensor("float32", (self.width_idx_segment_num,),
                                                   name="width_weight_ub",
                                                   scope=tik.scope_ubuf)
        height_idx_ub = self.tik_instance.Tensor("int32", (self.height_idx_segment_num * 2,),
                                                 name="height_idx",
                                                 scope=tik.scope_ubuf)
        height_weight_ub = self.tik_instance.Tensor("float32", (self.height_idx_segment_num,),
                                                    name="height_weight_ub",
                                                    scope=tik.scope_ubuf)

        scalar_in_height = self.tik_instance.Scalar("int64", name="scalar_in_height")
        scalar_in_height.set_as(self.tiling_in_height - 1)

        self.tik_instance.data_move(idx_ub_fp32, self.assist_gm, 0, 1,
                                    (self.width_idx_segment_num + self.block_num - 1) // self.block_num, 0, 0)

        with self.tik_instance.new_stmt_scope():
            idx_ub_fp32_tmp = self.tik_instance.Tensor("float32", (self.width_idx_segment_num,),
                                                       name="idx_ub_fp32_tmp",
                                                       scope=tik.scope_ubuf)
            # do vadds 0,1,2,3,4 + fp32_w_start_offset for per core
            util_tik_comm_func.tik_func_vcomple(self.tik_instance,
                                                "vadd",
                                                idx_ub_fp32_tmp,
                                                idx_ub_fp32,
                                                mem_info.get("width_start").get("fp32"),
                                                self.width_idx_segment_num,
                                                src1_blk=0,
                                                src1_rep=0)
            self.calcu_out_in_idx(self.resize_scale_w,
                                  width_idx_ub,
                                  idx_ub_fp32_tmp,
                                  self.width_idx_segment_num,
                                  des_weight_fp32_ub_list=[width_weight_ub],
                                  des_idx_ub_1=width_idx_ub[self.height_idx_segment_num:],
                                  max_idx_ub=mem_info.get("in_width").get("fp32"),
                                  max_idx_ub_int=mem_info.get("in_width").get("int32"),
                                  one_fp32_ub=mem_info.get("one").get("fp32"),
                                  one_int32_ub=mem_info.get("one").get("int32"),
                                  mem_info=mem_info,
                                  src_start=self.src_start_w,
                                  dst_start=self.dst_start_w)
            # do vadds 0,1,2,3,4 + fp32_w_start_offset for per core
            util_tik_comm_func.tik_func_vcomple(self.tik_instance,
                                                "vadd",
                                                idx_ub_fp32_tmp,
                                                idx_ub_fp32,
                                                mem_info.get("height_start").get("fp32"),
                                                self.width_idx_segment_num,
                                                src1_blk=0,
                                                src1_rep=0)
            self.calcu_out_in_idx(self.resize_scale_h,
                                  height_idx_ub,
                                  idx_ub_fp32_tmp,
                                  self.height_idx_segment_num,
                                  des_weight_fp32_ub_list=[height_weight_ub],
                                  des_idx_ub_1=height_idx_ub[self.height_idx_segment_num:],
                                  max_idx_ub=mem_info.get("in_height").get("fp32"),
                                  max_idx_ub_int=mem_info.get("in_height").get("int32"),
                                  one_fp32_ub=mem_info.get("one").get("fp32"),
                                  one_int32_ub=mem_info.get("one").get("int32"),
                                  mem_info=mem_info)

        nc_max_segment = self.tik_instance.Scalar("int32", name="nc_max_segment")
        nc_max_segment.set_as(max_nc)
        nc_loop = self.tik_instance.Scalar("int32", name="nc_loop")
        nc_tail = self.tik_instance.Scalar("int32", name="nc_tail")

        with self.tik_instance.if_scope(tik.all(self.core_nc_num < nc_max_segment, self.core_nc_num > 0)):
            nc_max_segment.set_as(self.core_nc_num)

        nc_loop.set_as(self.core_nc_num // nc_max_segment)
        nc_tail.set_as(self.core_nc_num % nc_max_segment)
        nc_process_num_ub = self.tik_instance.Tensor("int32", (self.block_num,),
                                                     name="nc_process_num_ub",
                                                     scope=tik.scope_ubuf)
        nc_loop_num_ceil = self.tik_instance.Scalar("int32", name="nc_loop_num_ceil")
        nc_loop_num_ceil.set_as((self.core_nc_num + nc_max_segment - 1) // nc_max_segment)
        nc_loop_num_floor = nc_loop
        nc_process_num_ub[0].set_as(nc_max_segment)
        nc_process_num_ub[1].set_as(nc_tail)

        def _do_resize_with_nc(w_index, h_index, nc_index, do_nc_num, ub_list):
            input_top, input_bottom = ub_list[0]
            _, _, output_h_ub, output_last = ub_list
            input_w0_index = self.tik_instance.Scalar("int32", name="input_w0_index")
            input_w1_index = self.tik_instance.Scalar("int32", name="input_w1_index")
            input_h0_index = self.tik_instance.Scalar("int32", name="input_h0_index")
            input_h1_index = self.tik_instance.Scalar("int32", name="input_h1_index")
            input_w_weight = self.tik_instance.Scalar("float32", name="input_w_weight")
            input_h_weight = self.tik_instance.Scalar("float32", name="input_h_weight")

            input_w0_index.set_as(width_idx_ub[w_index])
            input_w1_index.set_as(width_idx_ub[self.width_idx_segment_num + w_index])
            input_h0_index.set_as(height_idx_ub[h_index])
            input_h1_index.set_as(height_idx_ub[self.height_idx_segment_num + h_index])
            input_w_weight.set_as(width_weight_ub[w_index])
            input_h_weight.set_as(height_weight_ub[h_index])
            # step 1 copy top
            top_left_gm_offset = \
                nc_index * self.tiling_in_width * self.tiling_in_height \
                + input_h0_index * self.tiling_in_width \
                + input_w0_index
            ub_info = [[self.images_gm, top_left_gm_offset * self.images_shape_c0], [input_top, 0]]
            self._function_data_move(do_nc_num, input_w1_index - input_w0_index + 1, ub_info, 1)

            # step 3 copy x0y1 x1y1
            bottom_left_gm_offset = \
                nc_index * self.tiling_in_width * self.tiling_in_height \
                + input_h1_index * self.tiling_in_width \
                + input_w0_index
            ub_info = [[self.images_gm, bottom_left_gm_offset * self.images_shape_c0], [input_bottom, 0]]
            self._function_data_move(do_nc_num, input_w1_index - input_w0_index + 1, ub_info, 1)

            if ub_list[1] is not None:
                input_ori_ub_fp32_top = ub_list[1][0]
                input_ori_ub_fp32_bottom = ub_list[1][1]
                input_num = do_nc_num * self.images_shape_c0 * 2
                self.tik_instance.vconv(self.vector_num, "", input_ori_ub_fp32_top, input_top,
                                        (input_num + self.vector_num - 1) // self.vector_num, 1, 1, 8, 4)
                self.tik_instance.vconv(self.vector_num, "", input_ori_ub_fp32_bottom, input_bottom,
                                        (input_num + self.vector_num - 1) // self.vector_num, 1, 1, 8, 4)
                input_bottom = input_ori_ub_fp32_bottom
                input_top = input_ori_ub_fp32_top

            input_num = do_nc_num * self.images_shape_c0 * 2
            # `calcu: top + (bottom - top) * input_h_weight`
            self.tik_instance.vsub(self.vector_num, input_bottom, input_bottom, input_top,
                                   (input_num + self.vector_num - 1) // self.vector_num, 1, 1, 1, 8, 8, 8)
            self.tik_instance.vmuls(self.vector_num, output_h_ub, input_bottom, input_h_weight,
                                    (input_num + self.vector_num - 1) // self.vector_num, 1, 1, 8, 8)
            self.tik_instance.vadd(self.vector_num, output_h_ub, input_top, output_h_ub,
                                   (input_num + self.vector_num - 1) // self.vector_num, 1, 1, 1, 8, 8, 8)

            # `calcu: left + (right - left) * input_w_weight`
            input_num = do_nc_num * self.images_shape_c0
            with self.tik_instance.if_scope(input_w1_index != input_w0_index):
                self.tik_instance.vsub(self.images_shape_c0, output_last, output_h_ub[self.images_shape_c0:],
                                       output_h_ub, do_nc_num, 1, 1, 1, 2, 4, 4)
                self.tik_instance.vmuls(self.vector_num, output_last, output_last, input_w_weight,
                                        (input_num + self.vector_num - 1) // self.vector_num, 1, 1, 8, 8)
                self.tik_instance.vadd(self.images_shape_c0, output_last, output_last, output_h_ub, do_nc_num, 1, 1, 1,
                                       2, 2, 4)
            with self.tik_instance.if_scope(input_w1_index == input_w0_index):
                self.tik_instance.vmuls(self.vector_num, output_last, output_h_ub, 1.0,
                                        (input_num + self.vector_num - 1) // self.vector_num, 1, 1, 8, 8)

            # step 6 copy output
            if self.inner_dtype != self.output_dtype:
                # when the output is fp16, will cast inner_dtype to output_dtype
                output_cast_ub = output_h_ub.reinterpret_cast_to(self.output_dtype)
                self.tik_instance.vconv(self.vector_num, "", output_cast_ub, output_last,
                                        (input_num + self.vector_num - 1) // self.vector_num, 1, 1, 4, 8)
                output_last = output_cast_ub
            output_gm_offset = \
                nc_index * self.tiling_out_width * self.tiling_out_height \
                + (h_index + self.core_height_start) * self.tiling_out_width \
                + w_index + self.core_width_start

            data_move_burst_num = do_nc_num
            data_move_burst_len = self.images_shape_c0 // self.output_block_num
            data_move_src_stride = 0
            data_move_dst_stride = \
                (self.tiling_out_width * self.tiling_out_height - 1) \
                * self.images_shape_c0 // self.output_block_num
            self.tik_instance.data_move(self.out_gm[output_gm_offset * self.images_shape_c0:], output_last, 0,
                                        data_move_burst_num, data_move_burst_len, data_move_src_stride,
                                        data_move_dst_stride)

        image_in_top_ping = self.tik_instance.Tensor(self.images_dtype, (2 * max_nc * self.images_shape_c0,),
                                                     name="image_in_top_ping",
                                                     scope=tik.scope_ubuf)
        image_in_top_pang = self.tik_instance.Tensor(self.images_dtype, (2 * max_nc * self.images_shape_c0,),
                                                     name="image_in_top_pang",
                                                     scope=tik.scope_ubuf)
        image_in_bottom_ping = self.tik_instance.Tensor(self.images_dtype, (2 * max_nc * self.images_shape_c0,),
                                                        name="image_in_bottom_ping",
                                                        scope=tik.scope_ubuf)
        image_in_bottom_pang = self.tik_instance.Tensor(self.images_dtype, (2 * max_nc * self.images_shape_c0,),
                                                        name="image_in_bottom_pang",
                                                        scope=tik.scope_ubuf)
        image_output_top_bottom_ping = self.tik_instance.Tensor(self.inner_dtype, (2 * max_nc * self.images_shape_c0,),
                                                                name="image_output_top_bottom_ping",
                                                                scope=tik.scope_ubuf)
        image_output_top_bottom_pang = self.tik_instance.Tensor(self.inner_dtype, (2 * max_nc * self.images_shape_c0,),
                                                                name="image_output_top_bottom_pang",
                                                                scope=tik.scope_ubuf)
        image_output_ping = self.tik_instance.Tensor(self.inner_dtype, (max_nc * self.images_shape_c0,),
                                                     name="image_output_ping",
                                                     scope=tik.scope_ubuf)
        image_output_pang = self.tik_instance.Tensor(self.inner_dtype, (max_nc * self.images_shape_c0,),
                                                     name="image_output_pang",
                                                     scope=tik.scope_ubuf)
        if self.inner_dtype != self.images_dtype:
            image_top_ub_fp32_ping = self.tik_instance.Tensor(self.inner_dtype, (2 * max_nc * self.images_shape_c0,),
                                                              name="image_top_ub_fp32_ping",
                                                              scope=tik.scope_ubuf)
            image_bottom_ub_fp32_ping = self.tik_instance.Tensor(self.inner_dtype,
                                                                 (2 * max_nc * self.images_shape_c0,),
                                                                 name="image_bottom_ub_fp32_ping",
                                                                 scope=tik.scope_ubuf)
            image_top_ub_fp32_pang = self.tik_instance.Tensor(self.inner_dtype, (2 * max_nc * self.images_shape_c0,),
                                                              name="image_top_ub_fp32_pang",
                                                              scope=tik.scope_ubuf)
            image_bottom_ub_fp32_pang = self.tik_instance.Tensor(self.inner_dtype,
                                                                 (2 * max_nc * self.images_shape_c0,),
                                                                 name="image_bottom_ub_fp32_pang",
                                                                 scope=tik.scope_ubuf)
            ping_fp32_list = [image_top_ub_fp32_ping, image_bottom_ub_fp32_ping]
            pang_fp32_list = [image_top_ub_fp32_pang, image_bottom_ub_fp32_pang]
        else:
            ping_fp32_list = None
            pang_fp32_list = None

        input_list_ping = [image_in_top_ping, image_in_bottom_ping]
        ping_ub_list = [input_list_ping, ping_fp32_list, image_output_top_bottom_ping, image_output_ping]

        input_list_pang = [image_in_top_pang, image_in_bottom_pang]
        pang_ub_list = [input_list_pang, pang_fp32_list, image_output_top_bottom_pang, image_output_pang]
        with self.tik_instance.for_range(0, nc_loop_num_ceil) as nc_loop_idx:
            nc_idx = nc_loop_idx * nc_max_segment + self.core_nc_start
            nc_num = self.tik_instance.Scalar("int32", name="nc_num")
            nc_num.set_as(nc_process_num_ub[nc_loop_idx // nc_loop_num_floor])
            with self.tik_instance.for_range(0, (self.core_width_num * self.core_height_num + 1) // 2) as total_idx:
                w_idx = (total_idx * 2 + 0) % self.core_width_num
                h_idx = (total_idx * 2 + 0) // self.core_width_num
                _do_resize_with_nc(w_idx, h_idx, nc_idx, nc_num, ping_ub_list)
                with self.tik_instance.if_scope(total_idx * 2 + 1 < self.core_width_num * self.core_height_num):
                    w_idx = (total_idx * 2 + 1) % self.core_width_num
                    h_idx = (total_idx * 2 + 1) // self.core_width_num
                    _do_resize_with_nc(w_idx, h_idx, nc_idx, nc_num, pang_ub_list)

    def resize_bilinear_v2_operator(self):
        """
        resize_bilinear_v2_operator
        """
        # regist compute base on tiling_key
        self.regist_compute(100110, self._function_reisze_with_nc_process)
        self.regist_compute(999999, self._tiling_compute_with_no_bilinear)
        self.regist_compute(100000, self._tiling_compute_default)
        # run all regist compute base tiling key
        self.op_run_compute()
        tbe_context.get_context().add_compile_info("global_variable_link", True)
        tbe_context.get_context().add_compile_info(
            "vars", {
                "ub_size": self.ub_size_bytes,
                "core_num": self.core_nums,
                "max_w_len": self.ub_max_num // self.images_shape_c0,
                "align_corners": int(self.align_corners),
                "half_pixel_centers": int(self.half_pixel_centers)
            })
        # Build CCE
        self.op_build_cce()

        return self.tik_instance
    
    def _init_tiling_params(self):
        self.resize_scale_h = self.tik_instance.Scalar("float32", name="resize_scale_h")
        self.resize_scale_w = self.tik_instance.Scalar("float32", name="resize_scale_w")
        self.scalar_idx_fp32 = self.tik_instance.Scalar("float32", name="scalar_idx_fp32")
        self.tiling_key = self.tik_instance.Scalar("int64", name="tiling_key")
        self.tiling_batch = self.tik_instance.Scalar("int64", name="tiling_batch")
        self.tiling_c1 = self.tik_instance.Scalar("int64", name="tiling_c1")
        self.tiling_in_height = self.tik_instance.Scalar("int64", name="tiling_in_height")
        self.tiling_in_width = self.tik_instance.Scalar("int64", name="tiling_in_width")
        self.tiling_out_height = self.tik_instance.Scalar("int64", name="tiling_out_height")
        self.tiling_out_width = self.tik_instance.Scalar("int64", name="tiling_out_width")
        self.tiling_bc1_cut_num = self.tik_instance.Scalar("int64", name="tiling_bc1_cut_num")
        self.tiling_height_cut_num = self.tik_instance.Scalar("int64", name="tiling_height_cut_num")
        self.tiling_width_cut_num = self.tik_instance.Scalar("int64", name="tiling_width_cut_num")
        self.ori_h = self.tik_instance.Scalar("int64", name="ori_h")
        self.ori_w = self.tik_instance.Scalar("int64", name="ori_w")
        self.src_start_w = self.tik_instance.Scalar("float32", name="src_start_w")
        self.dst_start_w = self.tik_instance.Scalar("float32", name="dst_start_w")


def _tik_fuc_vrec_newton(tik_instance, vrec_ub, origin_ub, do_len, newton_iteration=6, block_num=16):
    """
    only do newton for vrec result

    Parameters
    ----------
    tik_instance: class
        tik_instance
    vrec_ub: ub
        the result of vrec
    origin_ub: ub
        the origin input for vrec
    do_len: int
        vrec num
    newton_iteration: int
        do newton iteration
    block_num: int
        num in one block

    Returns
    -------
    None
    """
    with tik_instance.new_stmt_scope():
        vrec_newton_1 = tik_instance.Tensor(vrec_ub.dtype, (((do_len + block_num - 1) // block_num) * block_num,),
                                            name="vrec_newton_1",
                                            scope=tik.scope_ubuf)
        vrec_newton_2 = tik_instance.Tensor(vrec_ub.dtype, (((do_len + block_num - 1) // block_num) * block_num,),
                                            name="vrec_newton_2",
                                            scope=tik.scope_ubuf)

        def _one_newton():
            tik_instance.vmul(1, vrec_newton_1, vrec_ub, origin_ub, 1, 1, 1, 1, 8, 8, 8)
            tik_instance.vmuls(1, vrec_newton_2, vrec_newton_1, -1, 1, 1, 1, 8, 8)
            tik_instance.vadds(1, vrec_newton_1, vrec_newton_2, 2, 1, 1, 1, 8, 8)
            tik_instance.vmul(1, vrec_ub, vrec_newton_1, vrec_ub, 1, 1, 1, 1, 8, 8, 8)

        for _ in range(newton_iteration):
            _one_newton()


def fill_index_in_ub(tik_instance, idx_ub, idx_num, vector_num=64):
    """
    fill 0,1,2  .... (idx_num -1) in idx_ub
    when the idx_num is less than 16, fill it one by one
    when the type is not int32, will fill in int32 ub and cast to idx_ub dtype
    when the type is int32, will fill in int32 one by one
    """
    # when the idx_num is less than 16, fill it one by one
    _idx_scalar = tik_instance.Scalar(dtype=idx_ub.dtype)

    vector_num_ub = tik_instance.Tensor(idx_ub.dtype, (vector_num,), name="vector_num_ub", scope=tik.scope_ubuf)
    for _idx in range(vector_num // 8):
        _idx_scalar.set_as(_idx)
        idx_ub[_idx].set_as(_idx_scalar)
    tik_instance.vector_dup(vector_num, vector_num_ub, vector_num // 8, 1, 1, 8)
    with tik_instance.for_range(1, 8) as add_idx:
        add_offset = add_idx * vector_num // 8
        tik_instance.vadd(vector_num // 8, idx_ub[add_offset:], vector_num_ub, idx_ub[add_offset - (vector_num // 8):],
                          1, 1, 1, 1, 8, 0, 8)

    tik_instance.vector_dup(vector_num, vector_num_ub, vector_num, 1, 1, 8)
    idx_vector_num = (idx_num + vector_num - 1) // vector_num
    with tik_instance.for_range(1, idx_vector_num) as add_idx:
        add_offset = add_idx * vector_num
        tik_instance.vadd(vector_num, idx_ub[add_offset:], vector_num_ub, idx_ub[add_offset - vector_num:], 1, 1, 1, 1,
                          8, 0, 8)


@register_operator("SyncResizeBilinearV2")
# 'pylint: disable=unused-argument,too-many-arguments
def sync_resize_bilinear_v2(images,
                            size,
                            y,
                            ori_image_size=None,
                            split_size=None,
                            src_start_w=None,
                            dst_start_w=None,
                            align_corners=False,
                            half_pixel_centers=False,
                            kernel_name="resize_bilinear_v2"):
    """Resize `images` to `size` using bilinear interpolation.

    Parameters
    ----------
    images: dict
        the dict of input, include shape of input_tensor which layout
        only support 5HD and dtype supports 'float16', 'float32'
    size: dict
        the dict of input, the height and width of output tensor
        only support 5HD and dtype supports 'float16', 'float32'
    y: dict
        the dict of output, include shape of input_tensor which layout
        only support 5HD and dtype supports 'float16', 'float32'
    ori_image_size : list
    split_size: list
    src_start_w: int
    dst_start_w: int
    align_corners: bool
        whether align_corners
    half_pixel_centers: bool
        whether half_pixel_centers
    kernel_name: str
        cce kernel name, default value is `resize_bilinear_v2`

    Returns
    -------
    tik_instance
    """
    obj = SyncResizeBilinearV2(images, size, y, align_corners, half_pixel_centers, kernel_name)

    instance = obj.resize_bilinear_v2_operator()
    return instance
