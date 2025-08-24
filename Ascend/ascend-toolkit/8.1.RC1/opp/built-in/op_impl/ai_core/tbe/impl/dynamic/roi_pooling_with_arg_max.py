#!/usr/bin/python
# -*- coding: utf-8 -*-
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
dynamic roi_pool
"""
import numpy as np
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from impl import constant_util as constant

# 'pylint: disable=too-few-public-methods


class Constant(object):
    """
    The class for constant
    """
    STRIDE_EIGHT = 8
    # the number of blocks skipped per repeat
    STRIDE_FOUR = 4
    # the number of blocks skipped per repeat
    STRIDE_ONE = 1
    # the number of blocks per transposition
    MAX_INT32 = 2 ** 31 - 1
    # max int32
    # ting param num
    TILING_ARG_NUM = 32
    RESERVED_UB = 20480
    NUMBER_SIXTEEN = 16
    MIN_FLOAT = -3402823424
    MAX_FLOAT = 3402823424
    C0_NUM = 16
    MASK_NUM_FP32 = 64
    MASK_NUM_FP16 = 128
    MAX_ROI = 2000
    INT16_BYTE = 2
    INT32_BYTE = 4
    MAX_BIN = 30
    ROI_SHAPE = 5


# 'pylint: disable=useless-object-inheritance,too-many-instance-attributes,too-many-statements
class RoiPool(object):

    # 'pylint: disable=too-many-arguments,invalid-name,too-many-statements,too-many-locals,unused-argument
    def __init__(self, x, rois, y, argmax, pooled_h, pooled_w, spatial_scale, kernel_name):
        """
        Init BoundingBoxDecode base parameters

        Parameters
        ----------
        x : dict
            shape and dtype of feature_map
        rois : dict
            shape and dtype of input rois (N,5)
        pooled_h : int
            the height of pooling
        pooled_w : int
            the width of pooling
        spatial_scale : float
            the spatial_scale of rois
        kernel_name : str
            kernel name, default value is "roi_pool"

        Returns
        -------
        None
        """
        byte_size = 8
        block_number_fp16 = 32
        self.tik_instance = tik.Tik()

        self.x_dtype = x.get("dtype").lower()
        self.rois_dtype = rois.get("dtype").lower()
        self.kernel_name = kernel_name
        self.compute_assist()
        self.rois_dtype_bytes_size = tbe_platform.get_bit_len(self.rois_dtype) // byte_size
        self.x_dtype_bytes_size = tbe_platform.get_bit_len(self.x_dtype) // byte_size
        self.rois_data_each_block = constant.BLOCK_SIZE // self.rois_dtype_bytes_size
        self.x_data_each_block = constant.BLOCK_SIZE // self.x_dtype_bytes_size
        self.int16_perblock = constant.BLOCK_SIZE // Constant.INT16_BYTE  
        self.int32_perblock = constant.BLOCK_SIZE // Constant.INT32_BYTE
        self.core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.each_repeat_block_number = block_number_fp16
        self.used_ub = tik.Dprofile().get_unified_buffer_size() - Constant.RESERVED_UB
        self.available_ub_size = 64
        self.xmask_num = 64 if self.x_dtype == "float32" else 128
        Constant.MIN_FLOAT = -3402823424 if self.x_dtype == "float32" else -65504
        Constant.MAX_FLOAT = 3402823424 if self.x_dtype == "float32" else 65504
        self.support_fp32 = tbe_platform.api_check_support("tik.vec_reduce_max", "float32")
        self.is_support_data_move_pad = tbe_platform.api_check_support("tik.data_move_pad", self.rois_dtype)
        self.allow_dtype = self.x_dtype
        if not self.support_fp32:
            self.allow_dtype = "float16"
            self.xmask_num = 128
            Constant.MIN_FLOAT = -65504
            Constant.MAX_FLOAT = 65504
        else:
            self.allow_dtype = self.x_dtype
        self.mask_num = 64 if self.x_dtype == "float32" else 128
        
        self.roi_batch_id = self.tik_instance.Tensor("float32", (self.available_ub_size, 5),
                                                     name="roi_batch_id", scope=tik.scope_ubuf)
        self.rois_ub_front = self.tik_instance.Tensor("float32", (self.available_ub_size, 5),
                                                      name="rois_ub_front", scope=tik.scope_ubuf)
        self.rois_ub_tail = self.tik_instance.Tensor("float32", (self.available_ub_size, 5),
                                                     name="rois_ub_tail", scope=tik.scope_ubuf)
        self.rois_ub_width = self.tik_instance.Tensor("float32", (self.available_ub_size, 5),
                                                      name="rois_ub_width", scope=tik.scope_ubuf)
        self.rois_ub_height = self.tik_instance.Tensor("float32", (self.available_ub_size, 5),
                                                       name="rois_ub_height", scope=tik.scope_ubuf)
        self.bin_size_height = self.tik_instance.Tensor("float32", (self.available_ub_size, 5),
                                                        name="bin_size_height", scope=tik.scope_ubuf)
        self.bin_size_width = self.tik_instance.Tensor("float32", (self.available_ub_size, 5),
                                                       name="bin_size_width", scope=tik.scope_ubuf)     
        # init gm data
        self.x_gm = self.tik_instance.Tensor(self.x_dtype, [Constant.MAX_INT32],
                                             name="x_gm", scope=tik.scope_gm)
        self.rois_gm = self.tik_instance.Tensor(self.rois_dtype, [Constant.MAX_INT32],
                                                name="rois_gm", scope=tik.scope_gm)
        self.roi_actual_num_gm = self.tik_instance.Tensor("int32", [Constant.MAX_INT32],
                                                          name="roi_actual_num_gm", scope=tik.scope_gm)
        self.y_out_gm = self.tik_instance.Tensor(self.x_dtype, [Constant.MAX_INT32],
                                                 name="y_out_gm", scope=tik.scope_gm,
                                                 is_atomic_add=True)
        self.index_out_gm = self.tik_instance.Tensor("int32", [Constant.MAX_INT32],
                                                     name="index_out_gm", scope=tik.scope_gm,
                                                     is_atomic_add=True, init_value=-1)
        self.tiling_gm = self.tik_instance.Tensor("int32", (Constant.TILING_ARG_NUM,),
                                                  name="tiling_gm", scope=tik.scope_gm)
        # init tiling data
        self.box_num = self.tik_instance.Scalar("int32", name="box_num")
        self.fm_batch = self.tik_instance.Scalar("int32", name="fm_batch")
        self.channels = self.tik_instance.Scalar("int32", name="channels")
        self.fm_hight = self.tik_instance.Scalar("int32", name="fm_hight")
        self.fm_width = self.tik_instance.Scalar("int32", name="fm_width")
        self.c1_num = self.tik_instance.Scalar("int32", name="c1_num")
        self.core_data = self.tik_instance.Scalar("int32", name="core_data")
        self.core_used = self.tik_instance.Scalar("int32", name="core_used")
        self.copy_loop = self.tik_instance.Scalar("int32", name="copy_loop")
        self.copy_tail = self.tik_instance.Scalar("int32", name="copy_tail")
        self.last_copy_loop = self.tik_instance.Scalar("int32", name="last_copy_loop")
        self.last_copy_tail = self.tik_instance.Scalar("int32", name="last_copy_tail")
        self.tiling_core_num = self.tik_instance.Scalar(dtype="int32", name="tiling_core_num")
        self.pooled_h = self.tik_instance.Scalar(dtype="int32", name="pooled_h")
        self.pooled_w = self.tik_instance.Scalar(dtype="int32", name="pooled_w")
        self.spatial_scale = self.tik_instance.Scalar(dtype="float32", name="spatial_scale")
        
        self.indices_assist_ub_fp16 = self.tik_instance.Tensor("float16",
                                                               (Constant.MAX_BIN * Constant.MAX_BIN, Constant.C0_NUM),
                                                               name="indices_assist_ub_fp16",
                                                               scope=tik.scope_ubuf)
        self.indices_assist_ub2_fp16 = self.tik_instance.Tensor("float16",
                                                                (Constant.MAX_BIN * Constant.MAX_BIN, Constant.C0_NUM),
                                                                name="indices_assist_ub2_fp16",
                                                                scope=tik.scope_ubuf)
    
    def compute_assist(self):
        """
        func compute_assist
        """
        assist_data_c0 = list(range(Constant.C0_NUM))
        assist_data_c2 = []
        for i in range(Constant.MAX_ROI):
            assist_data_c2.append(assist_data_c0)
        self.assist_gm2 = self.tik_instance.Tensor("float32", (Constant.MAX_ROI, Constant.C0_NUM), name="assist_gm2",
                                                   scope=tik.scope_gm, init_value=assist_data_c2)
        assist_data_form = list(range(Constant.MAX_ROI))
        assist_data_T = []
        for i in range(Constant.C0_NUM):
            assist_data_T.append(assist_data_form)
        assist_data_array = np.asarray(assist_data_T).T
        assist_data = assist_data_array.tolist()
        self.assist_gm = self.tik_instance.Tensor("float32", (Constant.MAX_ROI, Constant.C0_NUM), name="assist_gm",
                                                  scope=tik.scope_gm, init_value=assist_data)
        
    def calculate_pooling_per_width(self, bin_w_now, bin_h_now, batch_id, bin_x1_now, bin_y1_now, masknum, address):
        with self.tik_instance.new_stmt_scope():
            max_number = self.tik_instance.Tensor(self.x_dtype, (self.c1_num, Constant.C0_NUM),
                                                  name="max_number", scope=tik.scope_ubuf)
            max_number_idx = self.tik_instance.Tensor("int32", (self.c1_num, Constant.C0_NUM),
                                                      name="max_number_idx", scope=tik.scope_ubuf)
            temp_argmax = self.tik_instance.Tensor("int32", (Constant.C0_NUM,),
                                                   name="temp_argmax", scope=tik.scope_ubuf)
            dup_repeat = (self.c1_num * Constant.C0_NUM + Constant.MASK_NUM_FP32 - 1) // Constant.MASK_NUM_FP32
            self.tik_instance.vector_dup(Constant.MASK_NUM_FP32, max_number_idx, -1, dup_repeat, 1, 8)
            self.tik_instance.vector_dup(16, temp_argmax, -1, 1, 1, 0)
            
            bin_h_now_fp32 = self.tik_instance.Scalar("float32")
            self.tik_instance.scalar_conv('', bin_h_now_fp32, bin_h_now)
            bin_w_now_fp32 = self.tik_instance.Scalar("float32")
            self.tik_instance.scalar_conv('', bin_w_now_fp32, bin_w_now)
            fm_width_fp32 = self.tik_instance.Scalar("float32")
            self.tik_instance.scalar_conv('', fm_width_fp32, self.fm_width)
            bin_y1_now_fp32 = self.tik_instance.Scalar("float32")
            self.tik_instance.scalar_conv('', bin_y1_now_fp32, bin_y1_now)
            bin_x1_now_fp32 = self.tik_instance.Scalar("float32")
            self.tik_instance.scalar_conv('', bin_x1_now_fp32, bin_x1_now)

            bin_w_gap = self.fm_width - bin_w_now
            x_loop_address = batch_id * self.channels * self.fm_hight * self.fm_width
            burst_y = (self.channels + self.x_data_each_block - 1) // self.x_data_each_block
            burst_argmax = (self.channels + self.int32_perblock - 1) // self.int32_perblock
            
            temp_x_ub = self.tik_instance.Tensor(self.allow_dtype, (bin_h_now * bin_w_now, Constant.C0_NUM),
                                                 name="temp_x_ub", scope=tik.scope_ubuf)
            temp_max = self.tik_instance.Tensor(self.allow_dtype, (self.mask_num,),
                                                name="temp_max", scope=tik.scope_ubuf)
            with self.tik_instance.for_range(0, self.c1_num) as c1_idx:
                x_loop_address = x_loop_address + c1_idx * self.fm_width * self.fm_hight * Constant.C0_NUM + \
                                      bin_y1_now * self.fm_width * Constant.C0_NUM + bin_x1_now * Constant.C0_NUM
                
                if self.support_fp32 is False:
                    self.tik_instance.data_move(temp_x_ub, self.x_gm[x_loop_address],
                                                constant.SID, bin_h_now, bin_w_now * self.x_dtype_bytes_size // 2,
                                                bin_w_gap * self.x_dtype_bytes_size // 2, 0)
                    self.tik_instance.h_reduce_max(temp_max, temp_x_ub, 0)
                    # caculate argmax per c0s
                    self.tik_instance.h_reduce_argmax(temp_argmax, temp_x_ub, 0)
                else:
                    with self.tik_instance.if_scope(bin_h_now * bin_w_now == 1):
                        self.tik_instance.data_move(temp_max, self.x_gm[x_loop_address],
                                                    constant.SID, bin_h_now, bin_w_now * self.x_dtype_bytes_size // 2,
                                                    bin_w_gap * self.x_dtype_bytes_size // 2, 0)
                        self.tik_instance.data_move(temp_x_ub, self.x_gm[x_loop_address],
                                                    constant.SID, bin_h_now, bin_w_now * self.x_dtype_bytes_size // 2,
                                                    bin_w_gap * self.x_dtype_bytes_size // 2, 0)
                    with self.tik_instance.else_scope():
                        self.tik_instance.data_move(temp_x_ub, self.x_gm[x_loop_address],
                                                    constant.SID, bin_h_now, bin_w_now * self.x_dtype_bytes_size // 2,
                                                    bin_w_gap * self.x_dtype_bytes_size // 2, 0)
                    
                        self.tik_instance.vec_max(Constant.C0_NUM, temp_max,
                                                  temp_x_ub, temp_x_ub[Constant.C0_NUM], 1, 8, 8, 8)
                        with self.tik_instance.for_range(1, bin_h_now * bin_w_now) as idx:
                            self.tik_instance.vec_max(Constant.C0_NUM, temp_max, temp_max,
                                                      temp_x_ub[idx * Constant.C0_NUM], 1, 8, 8, 8)
                    cmp_mask_ub_temp = self.tik_instance.Tensor("uint16", (1,),
                                                                name="cmp_mask_ub_temp", scope=tik.scope_ubuf)
                    cmp_mask_ub = self.tik_instance.Tensor("uint16", (bin_h_now * bin_w_now,),
                                                           name="cmp_mask_ub", scope=tik.scope_ubuf)
                    
                    temp_argmax_num = self.tik_instance.Tensor("float16", (Constant.C0_NUM,),
                                                               name="temp_argmax_num", scope=tik.scope_ubuf)   
                    temp_argmax_idx = self.tik_instance.Tensor("float16", (Constant.C0_NUM,),
                                                               name="temp_argmax_idx", scope=tik.scope_ubuf)
                    temp_argmax_form2 = self.tik_instance.Tensor("int32", (Constant.C0_NUM,),
                                                                 name="temp_argmax_form2", scope=tik.scope_ubuf)
                    
                    with self.tik_instance.for_range(0, bin_h_now * bin_w_now) as idx:
                        self.tik_instance.vcmpv_ge(cmp_mask_ub_temp, temp_x_ub[idx * Constant.C0_NUM],
                                                   temp_max, 1, 1, 1, 8, 8)
                        cmp_mask_ub[idx].set_as(cmp_mask_ub_temp[0])
                    
                    self.tik_instance.vreducev2(bin_h_now * bin_w_now * Constant.C0_NUM, temp_argmax_num,
                                                self.indices_assist_ub_fp16, cmp_mask_ub,
                                                (bin_h_now * bin_w_now * Constant.C0_NUM +\
                                                 Constant.MASK_NUM_FP16 - 1) // Constant.MASK_NUM_FP16,
                                                1, 8, 1, None, "counter")
                    self.tik_instance.vec_conv(Constant.C0_NUM, 'round', temp_argmax, temp_argmax_num, 1, 8, 4)
                # caculate argmax in fm
                temp_argmax_fp32_conv = self.tik_instance.Tensor("float32", (Constant.C0_NUM,),
                                                                 name="temp_argmax_fp32_conv", scope=tik.scope_ubuf)
                temp_max_y1_per_c = self.tik_instance.Tensor("float32", (Constant.C0_NUM,),
                                                             name="temp_max_y1_per_c", scope=tik.scope_ubuf)
                temp_max_x1_per_c = self.tik_instance.Tensor("float32", (Constant.C0_NUM,),
                                                             name="temp_max_x1_per_c", scope=tik.scope_ubuf)
                temp_div = self.tik_instance.Tensor("float32", (Constant.C0_NUM,),
                                                    name="temp_div", scope=tik.scope_ubuf)
                self.tik_instance.vec_conv(Constant.C0_NUM, '', temp_argmax_fp32_conv, temp_argmax, 1,
                                           Constant.STRIDE_EIGHT, Constant.STRIDE_EIGHT)

                # caculate bin_y1_per_c address// bin_w_now
                self.tik_instance.vec_dup(Constant.C0_NUM, temp_div, bin_w_now_fp32, 1, 8)
                self.tik_instance.vdiv(Constant.C0_NUM, temp_max_y1_per_c, temp_argmax_fp32_conv,
                                       temp_div, 1, 1, 1, 1, 8, 8, 8)
                self.tik_instance.vec_conv(Constant.C0_NUM, "floor", temp_argmax, temp_max_y1_per_c, 1, 8, 8)
                self.tik_instance.vec_conv(Constant.C0_NUM, "", temp_max_y1_per_c, temp_argmax, 1, 8, 8)

                # caculate bin_x1_per_c address - (address// bin_w_now) * bin_w_now
                self.tik_instance.vec_muls(Constant.C0_NUM, temp_max_x1_per_c, temp_max_y1_per_c,
                                           bin_w_now_fp32, 1, 8, 8)
                self.tik_instance.vec_sub(Constant.C0_NUM, temp_max_x1_per_c, temp_argmax_fp32_conv, temp_max_x1_per_c,
                                          1, 8, 8, 8)

                # caculate argmax (temp_max_y1_per_c + bin_y1_now) * fm_width + temp_max_x2_per_c + bin_x1_now
                self.tik_instance.vec_adds(Constant.C0_NUM, temp_max_y1_per_c,
                                           temp_max_y1_per_c, bin_y1_now_fp32, 1, 8, 8)
                self.tik_instance.vec_muls(Constant.C0_NUM, temp_max_y1_per_c,
                                           temp_max_y1_per_c, fm_width_fp32, 1, 8, 8)
                self.tik_instance.vec_add(Constant.C0_NUM, temp_max_y1_per_c,
                                          temp_max_y1_per_c, temp_max_x1_per_c, 1, 8, 8, 8)
                self.tik_instance.vec_adds(Constant.C0_NUM, temp_max_y1_per_c,
                                           temp_max_y1_per_c, bin_x1_now_fp32, 1, 8, 8)
                self.tik_instance.vec_conv(Constant.C0_NUM, "round", temp_argmax, temp_max_y1_per_c, 1, 8, 8)
                
                if self.support_fp32 is False:
                    with self.tik_instance.for_range(0, Constant.C0_NUM) as i:
                        max_number[c1_idx, i].set_as(temp_max[i])
                        max_number_idx[c1_idx, i].set_as(temp_argmax[i])
                else:
                    self.tik_instance.vreducev2(bin_h_now * bin_w_now * Constant.C0_NUM, temp_argmax_idx,
                                                self.indices_assist_ub2_fp16, cmp_mask_ub,
                                                (bin_h_now * bin_w_now * Constant.C0_NUM +\
                                                 Constant.MASK_NUM_FP16 - 1) // Constant.MASK_NUM_FP16,
                                                1, 8, 1, None, "counter")
                    self.tik_instance.vec_conv(Constant.C0_NUM, 'round', temp_argmax_form2, temp_argmax_idx, 1, 8, 4)
                        
                    argmax_idx = self.tik_instance.Scalar("int32")
                    with self.tik_instance.for_range(0, Constant.C0_NUM) as i:
                        argmax_idx.set_as(temp_argmax_form2[i])
                        max_number[c1_idx, i].set_as(temp_max[i])
                        max_number_idx[c1_idx * Constant.C0_NUM + argmax_idx].set_as(temp_argmax[i])
            self.tik_instance.data_move(self.y_out_gm[address], max_number, constant.SID, constant.DEFAULT_NBURST,
                                        burst_y, constant.STRIDE_ZERO, constant.STRIDE_ZERO)
            self.tik_instance.data_move(self.index_out_gm[address], max_number_idx, constant.SID,
                                        constant.DEFAULT_NBURST, burst_argmax,
                                        constant.STRIDE_ZERO, constant.STRIDE_ZERO)

    def calculate_pooling_width(self, pw, repeat_time, batch_idx, bin_h_now,
                                batch_id, bin_y1_now, masknum, address):
        rois_ub_front_scalar = self.tik_instance.Scalar(dtype="float32")
        rois_ub_front_scalar.set_as(self.rois_ub_front[batch_idx * 5])

        bin_x1 = self.tik_instance.Scalar(dtype="float32")
        bin_x1.set_as(self.bin_size_width[batch_idx * 5])
        bin_x1.set_as(bin_x1 * pw)
        bin_x1.set_as(bin_x1 + rois_ub_front_scalar)
        bin_x1_now = self.tik_instance.Scalar(dtype="int32")
        self.tik_instance.scalar_conv('floor', bin_x1_now, bin_x1)

        bin_x2 = self.tik_instance.Scalar(dtype="float32")
        bin_x2.set_as(self.bin_size_width[batch_idx * 5])
        bin_x2.set_as(bin_x2 * (pw + 1))
        bin_x2.set_as(bin_x2 + rois_ub_front_scalar)
        bin_x2_now = self.tik_instance.Scalar(dtype="int32")
        self.tik_instance.scalar_conv('ceil', bin_x2_now, bin_x2)

        with self.tik_instance.if_scope(bin_x1_now < 0):
            bin_x1_now.set_as(0)
        with self.tik_instance.if_scope(bin_x1_now > self.fm_width):
            bin_x1_now.set_as(self.fm_width)
        with self.tik_instance.if_scope(bin_x2_now < 0):
            bin_x2_now.set_as(0)
        with self.tik_instance.if_scope(bin_x2_now > self.fm_width):
            bin_x2_now.set_as(self.fm_width)

        bin_w_now = self.tik_instance.Scalar(dtype="int32")
        bin_w_now.set_as(bin_x2_now - bin_x1_now)

        with self.tik_instance.if_scope(bin_w_now > 0):
            self.calculate_pooling_per_width(
                bin_w_now, bin_h_now, batch_id, bin_x1_now, bin_y1_now, masknum, address)

    def calculate_pooling_result(self, batch_id, batch_idx, repeat_time, loop_input_result, masknum):

        roi_width_now = self.tik_instance.Scalar(dtype="float32")
        roi_width_now.set_as(self.rois_ub_width[batch_idx * 5])

        roi_hight_now = self.tik_instance.Scalar(dtype="float32")
        roi_hight_now.set_as(self.rois_ub_height[batch_idx * 5 + 1])
        
        with self.tik_instance.if_scope(tik.all(roi_width_now > 0, roi_hight_now > 0)):
            with self.tik_instance.for_range(0, self.pooled_h) as ph:
                rois_ub_front_scalar_y = self.tik_instance.Scalar(dtype="float32")
                rois_ub_front_scalar_y.set_as(self.rois_ub_front[batch_idx * 5 + 1])
                
                bin_y1 = self.tik_instance.Scalar(dtype="float32")
                bin_y1.set_as(self.bin_size_height[batch_idx * 5 + 1])
                bin_y1.set_as(bin_y1 * ph)
                bin_y1.set_as(bin_y1 + rois_ub_front_scalar_y)
                bin_y1_now = self.tik_instance.Scalar(dtype="int32")
                self.tik_instance.scalar_conv('floor', bin_y1_now, bin_y1)

                bin_y2 = self.tik_instance.Scalar(dtype="float32")
                bin_y2.set_as(self.bin_size_height[batch_idx * 5 + 1])
                bin_y2.set_as(bin_y2 * (ph+1))
                bin_y2.set_as(bin_y2 + rois_ub_front_scalar_y)
                bin_y2_now = self.tik_instance.Scalar(dtype="int32")
                self.tik_instance.scalar_conv('ceil', bin_y2_now, bin_y2)

                with self.tik_instance.if_scope(bin_y1_now < 0):
                    bin_y1_now.set_as(0)
                with self.tik_instance.if_scope(bin_y1_now > self.fm_hight):
                    bin_y1_now.set_as(self.fm_hight)
                with self.tik_instance.if_scope(bin_y2_now < 0):
                    bin_y2_now.set_as(0)
                with self.tik_instance.if_scope(bin_y2_now > self.fm_hight):
                    bin_y2_now.set_as(self.fm_hight)

                bin_h_now = self.tik_instance.Scalar(dtype="int32")
                bin_h_now.set_as(bin_y2_now - bin_y1_now)
                
                with self.tik_instance.if_scope(bin_h_now > 0):
                    with self.tik_instance.for_range(0, self.pooled_w) as pw:
                        address = loop_input_result + batch_idx * self.channels * self.pooled_h * self.pooled_w +\
                                  ph * self.pooled_w * self.channels + pw * self.channels
                        self.calculate_pooling_width(pw, repeat_time, batch_idx, bin_h_now,
                                                     batch_id, bin_y1_now, masknum, address)
            
    def calculation_process(self, loop_input_result, repeat_time, batch_number, mask_num):
        with self.tik_instance.for_range(0, batch_number) as batch_idx:
            batch_id = self.tik_instance.Scalar(dtype="float32")
            batch_id.set_as(self.roi_batch_id[batch_idx * 5])
            dst_scalar2 = self.tik_instance.Scalar(dtype="int32")
            self.tik_instance.scalar_conv("round", dst_scalar2, batch_id)
            self.calculate_pooling_result(
                dst_scalar2, batch_idx, repeat_time, loop_input_result, mask_num)

    def calculate_rois_size(self, loop_input, burst, repeat_times_now, ub_num):
        with self.tik_instance.new_stmt_scope():
            if self.rois_dtype == "float16":
                roi_batch_id_beforeconv = self.tik_instance.Tensor(self.rois_dtype, (self.available_ub_size, 5),
                                                                   name="roi_batch_id_berforeconv",
                                                                   scope=tik.scope_ubuf)
                roi_ub_front_beforeconv = self.tik_instance.Tensor(self.rois_dtype, (self.available_ub_size, 5),
                                                                   name="roi_ub_front_berforeconv",
                                                                   scope=tik.scope_ubuf)
                roi_ub_tail_beforeconv = self.tik_instance.Tensor(self.rois_dtype, (self.available_ub_size, 5),
                                                                  name="roi_ub_tail_berforeconv", scope=tik.scope_ubuf)
                # move rois height and width and batch_id to ub
                self.tik_instance.data_move(roi_batch_id_beforeconv, self.rois_gm[loop_input],
                                            constant.SID, constant.DEFAULT_NBURST,
                                            burst, constant.STRIDE_ZERO,
                                            constant.STRIDE_ZERO)

                self.tik_instance.data_move(roi_ub_front_beforeconv, self.rois_gm[loop_input + 1],
                                            constant.SID, constant.DEFAULT_NBURST,
                                            burst, constant.STRIDE_ZERO,
                                            constant.STRIDE_ZERO)

                if self.is_support_data_move_pad:
                    self.tik_instance.data_move_pad(roi_ub_tail_beforeconv, self.rois_gm[loop_input + 3],
                                                constant.DEFAULT_NBURST,
                                                ub_num * Constant.ROI_SHAPE * self.rois_dtype_bytes_size, 
                                                constant.STRIDE_ZERO, constant.STRIDE_ZERO, 
                                                right_padding=constant.INT_DEFAULT_ZERO, 
                                                left_padding=constant.INT_DEFAULT_ZERO, 
                                                padding_value=None)
                else:
                    self.tik_instance.data_move(roi_ub_tail_beforeconv, self.rois_gm[loop_input + 3],
                                            constant.SID, constant.DEFAULT_NBURST,
                                            burst, constant.STRIDE_ZERO,
                                            constant.STRIDE_ZERO)
                with self.tik_instance.for_range(0, repeat_times_now) as repeat_idx:
                    self.tik_instance.vec_conv(
                        64, '', self.roi_batch_id[repeat_idx * 64],
                        roi_batch_id_beforeconv[repeat_idx * 64], 1, 4, 8)
                    self.tik_instance.vec_conv(
                        64, '', self.rois_ub_front[repeat_idx * 64],
                        roi_ub_front_beforeconv[repeat_idx * 64], 1, 4, 8)
                    self.tik_instance.vec_conv(
                        64, '', self.rois_ub_tail[repeat_idx * 64],
                        roi_ub_tail_beforeconv[repeat_idx * 64], 1, 4, 8)
            else:
                # move rois height and width and batch_id to ub
                self.tik_instance.data_move(self.roi_batch_id, self.rois_gm[loop_input],
                                            constant.SID, constant.DEFAULT_NBURST,
                                            burst, constant.STRIDE_ZERO,
                                            constant.STRIDE_ZERO)

                self.tik_instance.data_move(self.rois_ub_front, self.rois_gm[loop_input + 1],
                                            constant.SID, constant.DEFAULT_NBURST,
                                            burst, constant.STRIDE_ZERO,
                                            constant.STRIDE_ZERO)

                if self.is_support_data_move_pad:
                    self.tik_instance.data_move_pad(self.rois_ub_tail, self.rois_gm[loop_input + 3],
                                                constant.DEFAULT_NBURST,
                                                ub_num * Constant.ROI_SHAPE * self.rois_dtype_bytes_size, 
                                                constant.STRIDE_ZERO, constant.STRIDE_ZERO, 
                                                right_padding=constant.INT_DEFAULT_ZERO, 
                                                left_padding=constant.INT_DEFAULT_ZERO, 
                                                padding_value=None)
                else:
                    self.tik_instance.data_move(self.rois_ub_tail, self.rois_gm[loop_input + 3],
                                            constant.SID, constant.DEFAULT_NBURST,
                                            burst, constant.STRIDE_ZERO,
                                            constant.STRIDE_ZERO)
            self.tik_instance.vec_muls(
                64, self.rois_ub_front, self.rois_ub_front, self.spatial_scale, repeat_times_now, 8, 8)

            self.tik_instance.vec_adds(
                64, self.rois_ub_tail, self.rois_ub_tail, 1.0, repeat_times_now, 8, 8)
            self.tik_instance.vec_muls(
                64, self.rois_ub_tail, self.rois_ub_tail, self.spatial_scale, repeat_times_now, 8, 8)

            # calculate rois height and width
            # rois_ub_width [width,height,xxx,xxx,xxx] 
            # rois_ub_height [height,xxx,xxx,xxx,width]
            self.tik_instance.vec_sub(
                64, self.rois_ub_width, self.rois_ub_tail, self.rois_ub_front, repeat_times_now, 8, 8, 8)
            self.tik_instance.vec_sub(
                64, self.rois_ub_height, self.rois_ub_tail, self.rois_ub_front, repeat_times_now, 8, 8, 8)
        
            temp_ub = self.tik_instance.Tensor("float32", (self.available_ub_size, 5),
                                               name="temp_ub", scope=tik.scope_ubuf)
            pool_h_fp32 = self.tik_instance.Scalar(dtype="float32")
            self.tik_instance.scalar_conv('', pool_h_fp32, self.pooled_h)
            pool_w_fp32 = self.tik_instance.Scalar(dtype="float32")
            self.tik_instance.scalar_conv('', pool_w_fp32, self.pooled_w)
            # calculate bin height and width
            # bin_size_height [bin_size_h,xxx,xxx,xxx,xxx]
            # bin_size_width [bin_size_w,xxx,xxx,xxx,xxx]
            self.tik_instance.vec_dup(
                64, temp_ub, pool_h_fp32, repeat_times_now, 8)
            self.tik_instance.vdiv(64, self.bin_size_height, self.rois_ub_height,
                                   temp_ub, repeat_times_now, 1, 1, 1, 8, 8, 8)
            self.tik_instance.vec_dup(
                64, temp_ub, pool_w_fp32, repeat_times_now, 8)
            self.tik_instance.vdiv(64, self.bin_size_width, self.rois_ub_width,
                                   temp_ub, repeat_times_now, 1, 1, 1, 8, 8, 8)
    
    def copy_only(self, core_index, loop_num, tail_num):
        mask64 = 64
        mask128 = 128
        mask_num = mask128 if self.x_dtype == 'float16' else mask64
        repeat_times = (self.available_ub_size * 5 + 64 - 1) // 64
        
        with self.tik_instance.new_stmt_scope():
            temp_x_ub = self.tik_instance.Tensor("float32", (Constant.MAX_BIN * Constant.MAX_BIN, Constant.C0_NUM),
                                                 name="temp_x_ub", scope=tik.scope_ubuf)
            burst_indices = (Constant.MAX_BIN * Constant.MAX_BIN * Constant.C0_NUM +\
                             self.int32_perblock - 1) // self.int32_perblock
            self.tik_instance.data_move(temp_x_ub, self.assist_gm, constant.SID,
                                        constant.DEFAULT_NBURST, burst_indices, constant.STRIDE_ZERO,
                                        constant.STRIDE_ZERO)
            self.tik_instance.vec_conv(Constant.MASK_NUM_FP32, '', self.indices_assist_ub_fp16, temp_x_ub,
                                       (Constant.MAX_BIN * Constant.MAX_BIN * Constant.C0_NUM +\
                                        Constant.MASK_NUM_FP32 - 1) // Constant.MASK_NUM_FP32, 4, 8)

            self.tik_instance.data_move(temp_x_ub, self.assist_gm2, constant.SID,
                                        constant.DEFAULT_NBURST, burst_indices,
                                        constant.STRIDE_ZERO, constant.STRIDE_ZERO)
            self.tik_instance.vec_conv(Constant.MASK_NUM_FP32, '', self.indices_assist_ub2_fp16, temp_x_ub,
                                       (Constant.MAX_BIN * Constant.MAX_BIN * Constant.C0_NUM +\
                                        Constant.MASK_NUM_FP32 - 1) // Constant.MASK_NUM_FP32, 4, 8)
 
        with self.tik_instance.for_range(0, loop_num) as loop_idx:
            loop_input = core_index * self.core_data * 5 + loop_idx * self.available_ub_size * 5
            burst = (self.available_ub_size * 5 + self.rois_data_each_block - 1) // self.rois_data_each_block
            self.calculate_rois_size(
                loop_input, burst, repeat_times, self.available_ub_size)
            # multi kernel is divided by the number of rois
            # but batch_id corresponding to roi is inconsistent, each roi shall be calculated independently
            loop_input_result = core_index * self.core_data * self.channels * self.pooled_h * self.pooled_w +\
                loop_idx * self.available_ub_size * self.channels * self.pooled_h * self.pooled_w
            self.calculation_process(
                loop_input_result, repeat_times, self.available_ub_size, mask_num)

        with self.tik_instance.if_scope(tail_num > 0):
            with self.tik_instance.if_scope((tail_num * 5) < 64):
                repeat_times2 = 1
            with self.tik_instance.else_scope():
                repeat_times2 = (tail_num * 5 + 64 - 1) // 64
            loop_input = core_index * self.core_data * 5 + loop_num * self.available_ub_size * 5
            burst = (tail_num * 5 + self.rois_data_each_block - 1) // self.rois_data_each_block
            self.calculate_rois_size(
                loop_input, burst, repeat_times2, tail_num)

            loop_input_result = core_index * self.core_data * self.channels * self.pooled_h * self.pooled_w +\
                loop_num * self.available_ub_size * self.channels * self.pooled_h * self.pooled_w
            self.calculation_process(
                loop_input_result, repeat_times2, tail_num, mask_num)
    
    def get_tiling_args(self, tiling_ub):
        """
        get runtime tiling params from tiling

        Parameters
        ----------

        Returns
        -------
        None
        """
        # read tiling int32 scalar
        self.box_num.set_as(tiling_ub[0])
        self.fm_batch.set_as(tiling_ub[1])
        self.channels.set_as(tiling_ub[2])
        self.fm_hight.set_as(tiling_ub[3])
        self.fm_width.set_as(tiling_ub[4])
        self.c1_num.set_as(tiling_ub[5])
        self.core_data.set_as(tiling_ub[6])
        self.core_used.set_as(tiling_ub[7])
        self.copy_loop.set_as(tiling_ub[8])
        self.copy_tail.set_as(tiling_ub[9])
        self.last_copy_loop.set_as(tiling_ub[10])
        self.last_copy_tail.set_as(tiling_ub[11])
        self.tiling_core_num.set_as(tiling_ub[12])
        self.pooled_h.set_as(tiling_ub[13])
        self.pooled_w.set_as(tiling_ub[14])
        self.spatial_scale.set_as(tiling_ub[15])

    def tik_instance_function(self):
        """

        the entry of bounding_box_decode calculation

        Parameters
        ----------

        Returns
        -------
        None
        """
        with self.tik_instance.new_stmt_scope():
            tiling_ub = self.tik_instance.Tensor("int32", (Constant.TILING_ARG_NUM,),
                                                 name="tiling_ub", scope=tik.scope_ubuf)
            self.tik_instance.data_move(
                tiling_ub, self.tiling_gm, 0, 1, 4, 0, 0)
            self.get_tiling_args(tiling_ub)
         
        with self.tik_instance.for_range(0, self.tiling_core_num, block_num=self.tiling_core_num) as core_index:
            with self.tik_instance.if_scope(core_index < (self.core_used - 1)):
                self.copy_only(core_index, self.copy_loop, self.copy_tail)
            with self.tik_instance.elif_scope(core_index == (self.core_used - 1)):
                self.copy_only(core_index, self.last_copy_loop, self.last_copy_tail)
        opt_config = {"enable_const_fold": True}
        tbe_context.get_context().add_compile_info("vars",
                                                   {"core_num": self.core_num,
                                                    "rois_data_each_block": self.rois_data_each_block,
                                                    "each_repeat_block_number": self.each_repeat_block_number,
                                                    "ub_max_size": self.available_ub_size})
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[self.x_gm, self.rois_gm, self.roi_actual_num_gm],
                                   outputs=[self.y_out_gm, self.index_out_gm],
                                   flowtable=[self.tiling_gm], config=opt_config)
        return self.tik_instance


# 'pylint: disable=unused-argument, too-many-locals, too-many-lines, too-many-arguments
@register_operator("RoiPoolingWithArgMax")
@para_check.check_op_params(para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT,
                            para_check.OPTION_INPUT,
                            para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_INT,
                            para_check.OPTION_ATTR_INT,
                            para_check.OPTION_ATTR_FLOAT,
                            para_check.OPTION_ATTR_FLOAT,
                            para_check.OPTION_ATTR_INT,
                            para_check.KERNEL_NAME)
def roi_pooling_with_arg_max(x, rois, roi_actual_num, y, argmax, pooled_h, pooled_w, spatial_scale_h,
                             spatial_scale_w, pool_channel, kernel_name="roi_pooling_with_arg_max"):
    """
    calculating data

    Parameters
    ----------
    x : dict
        shape and dtype of feature_map
    rois : dict
        shape and dtype of input rois (N,5)
    y : dict
        shape and dtype of output (n, c, pooled_h, pooled_w)
    pooled_h : int
        the height of pooling
    pooled_w : int
        the width of pooling
    spatial_scale : float
        the spatial_scale of rois
    kernel_name : str
        kernel name, default value is "roi_pool"

    Returns
    -------
    None
    """
    x_dtype = x.get("dtype").lower()
    rois_dtype = rois.get("dtype").lower()
    para_check.check_dtype(x_dtype, ["float16", "float32"])
    para_check.check_dtype(rois_dtype, ["float16", "float32"])

    roi_pool_instance = RoiPool(
        x, rois, y, argmax, pooled_h, pooled_w, spatial_scale_h, kernel_name)
    instance = roi_pool_instance.tik_instance_function()
    return instance
