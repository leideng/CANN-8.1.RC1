#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2024. All rights reserved.
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
dynamic check_valid
"""
from impl.util.util_common import is_unknown_rank_input
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util import util_tik_comm_func
from impl import common_util
from impl import constant_util as constant
from te.utils.error_manager import error_manager_vector
import te.platform as tbe_platform


class Constant:
    #the num of ub that we need
    NUM_CUT_UB = 10
    SCALAR_MAX_FP16 = 2**16 - 1
    SCALAR_MIN_FP16 = 2**(-24)
    IMG_META_INPUT_NUM = 16
    TILING_ARG_NUM = 8
    ANCHOR_SIZE = 4


class CheckValid(object):

    def __init__(self, bbox_tensor, img_metas, valid_tensor, kernel_name):
        self.bbox_dtype = bbox_tensor.get("dtype").lower()
        self.img_metas_dtype = img_metas.get("dtype").lower()
        self.bbox_dtype_size = common_util.get_data_size(self.bbox_dtype)
        self.valid_dtype_size = 1
        self.__default_rows_per_job = 128 if self.bbox_dtype == "float16" else 64
        self.job_buf_row = 128 if self.bbox_dtype == "float16" else 64
        self.data_btype = 2 if self.bbox_dtype == "float16" else 4
        self.kernel_name = kernel_name
        self.tik_instance = tik.Tik()
        self.tik_profiling = tik.Dprofile()
        #get ai_core num
        self.ai_core_num = self.tik_profiling.get_aicore_num()
        #get ub size
        self.ub_size_bytes = self.tik_profiling.get_unified_buffer_size()
        self.img_meta_need_byte = Constant.IMG_META_INPUT_NUM * self.bbox_dtype_size
        self.ub_tensor_size = (self.ub_size_bytes -
                               self.img_meta_need_byte) // Constant.NUM_CUT_UB // self.bbox_dtype_size
        self.each_block_num = constant.BLOCK_SIZE // self.bbox_dtype_size

        # buffer for threshold extract
        self.img_metas_gm = self.tik_instance.Tensor(self.img_metas_dtype, (16,),
                                                     name="img_metas_gm",
                                                     scope=tik.scope_gm)
        self.img_metas_ub = self.tik_instance.Tensor(self.img_metas_dtype, (16,),
                                                     name="img_metas_ub",
                                                     scope=tik.scope_ubuf)
        self.threshold_h = self.tik_instance.Scalar(self.img_metas_dtype, "threshold_h")
        self.threshold_w = self.tik_instance.Scalar(self.img_metas_dtype, "threshold_w")
        self.__get_threshold()

        self.core_num_var = self.tik_instance.Scalar(name="core_num_var", init_value=self.ai_core_num)

        #set bbox gm tensor
        self.bbox_gm = self.tik_instance.Tensor(self.bbox_dtype, (Constant.SCALAR_MAX_FP16,),
                                                name="bbox_gm",
                                                scope=tik.scope_gm)
        self.date_ret_valid_gm = self.tik_instance.Tensor("int8", (Constant.SCALAR_MAX_FP16,),
                                                          name="date_ret_valid_gm",
                                                          scope=tik.scope_gm)
        self.tiling_gm = self.tik_instance.Tensor("int64", (Constant.TILING_ARG_NUM,),
                                                  name="tiling_gm",
                                                  scope=tik.scope_gm)
        self._init_tiling_scalars()
        self._cum_compute_tiling()
        self.last_align_job_head = self.tik_instance.Scalar("int64", name="last_align_job_head", init_value=0)
        self.padded_bytes = self.tik_instance.Scalar("int64", name="padded_bytes", init_value=0)
        self.last_job_row_aligned = self.__calc_last_job_row()
        self._process_elem_count = 128 if self.bbox_dtype == "float16" else 64
        self.quad_flag_ub = None
        self.quad_flags_sum_ub = None
        self.quad_threshold_ub = None
        self.quad_threshold_ub_fp16 = None
        self.ones_ub = None
        self.zeros_ub = None
        self.data_ret_int8_ub = None
        self.data_ret_mask_ub = None
        self.data_ret_ub = None
        self.ret_unfold_half_ub = None
        self.bbox_tensor_ub = None

        self.data_move_pad_support = tbe_platform.api_check_support("tik.data_move_pad")

    def set_running_core_num(self, tiling_core_num):
        self.core_num_var.set_as(tiling_core_num)

    def check_valid_compute(self):
        with self.tik_instance.for_range(0, self.core_num_var, block_num=self.core_num_var) as num_core_i:
            self._apply_ub_tensor()
            self._handle_out_loop(num_core_i)
        opt_config = {"out_of_bound_sync_check": True, "enable_const_fold": True}
        tbe_context.get_context().add_compile_info("vars", {
            "core_num": self.ai_core_num,
            "ub_size": self.ub_size_bytes
        })
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[self.bbox_gm, self.img_metas_gm],
                                   outputs=[self.date_ret_valid_gm],
                                   flowtable=[self.tiling_gm],
                                   config=opt_config)
        return self.tik_instance
    
    def _apply_ub_tensor(self):
        self.quad_flag_ub = self.tik_instance.Tensor(self.bbox_dtype, (self.job_buf_row * Constant.ANCHOR_SIZE,),
                                                     name="quad_flag_ub",
                                                     scope=tik.scope_ubuf)
        self.quad_flags_sum_ub = self.tik_instance.Tensor(self.bbox_dtype, (self.job_buf_row * Constant.ANCHOR_SIZE,),
                                                          name="quad_flags_sum_ub",
                                                          scope=tik.scope_ubuf)
        self.quad_threshold_ub = self.tik_instance.Tensor(self.bbox_dtype, (self.job_buf_row * Constant.ANCHOR_SIZE,),
                                                          name="quad_threshold_ub",
                                                          scope=tik.scope_ubuf)
        self.quad_threshold_ub_fp16 = self.tik_instance.Tensor("float16", (self.job_buf_row * Constant.ANCHOR_SIZE,),
                                                               name="quad_threshold_ub",
                                                               scope=tik.scope_ubuf)
        self.ones_ub = self.tik_instance.Tensor(self.bbox_dtype, (self.job_buf_row, Constant.ANCHOR_SIZE),
                                                name="ones_ub",
                                                scope=tik.scope_ubuf)
        self.zeros_ub = self.tik_instance.Tensor(self.bbox_dtype, (self.job_buf_row, Constant.ANCHOR_SIZE),
                                                 name="zeros_ub",
                                                 scope=tik.scope_ubuf)
        with self.tik_instance.if_scope(self.job_buf_row * Constant.ANCHOR_SIZE // self.job_buf_row < 1):
            _repeat_time = 1
        with self.tik_instance.else_scope():
            _repeat_time = self.job_buf_row * Constant.ANCHOR_SIZE // self.job_buf_row
        self.tik_instance.vector_dup(self._process_elem_count, self.ones_ub, 1, _repeat_time, 1, 8)
        self.tik_instance.vector_dup(self._process_elem_count, self.zeros_ub, 0, _repeat_time, 1, 8)

        self.data_ret_int8_ub = self.tik_instance.Tensor("int8", (self.job_buf_row, 1),
                                                         name="data_ret_int8_ub",
                                                         scope=tik.scope_ubuf)
        self.data_ret_mask_ub = self.tik_instance.Tensor(
            "uint16", (self.job_buf_row * Constant.ANCHOR_SIZE * self.data_btype // 32,),
            name="data_ret_mask_ub",
            scope=tik.scope_ubuf)
        self.data_ret_ub = self.tik_instance.Tensor(self.bbox_dtype, (self.job_buf_row, 1),
                                                    name="data_ret_ub",
                                                    scope=tik.scope_ubuf)
        self.ret_unfold_half_ub = self.tik_instance.Tensor(self.bbox_dtype, (self.job_buf_row * 2,),
                                                           name="ret_unfold_half_ub",
                                                           scope=tik.scope_ubuf)
        self.bbox_tensor_ub = self.tik_instance.Tensor(self.bbox_dtype, (self.job_buf_row, Constant.ANCHOR_SIZE),
                                                       name="bbox_tensor_ub",
                                                       scope=tik.scope_ubuf)

    def _handle_out_loop(self, index_core_i):
        with self.tik_instance.for_range(0, self.tiling_loop_per_core) as num_core_j:
            with self.tik_instance.if_scope(self.tiling_loop_per_core * index_core_i + \
                                            num_core_j == self.tiling_job_num - 1):
                self._handle_one_core(self.tiling_loop_per_core * index_core_i + num_core_j, True)
            with self.tik_instance.else_scope():
                self._handle_one_core(self.tiling_loop_per_core * index_core_i + num_core_j)
            # last schedule
        with self.tik_instance.if_scope(index_core_i < self.tiling_core_left):
            with self.tik_instance.if_scope(index_core_i == self.tiling_core_left - 1):
                # last core of whole task
                self._handle_one_core(self.tiling_loop_per_core * self.core_num_var + index_core_i, True)
            with self.tik_instance.else_scope():
                self._handle_one_core(self.tiling_loop_per_core * self.core_num_var + index_core_i)

    def _init_tiling_scalars(self):
        """
        The function of init tiling args
        """
        self.tiling_loop_per_core = self.tik_instance.Scalar("int64", name="tiling_loop_per_core", init_value=0)
        self.tiling_core_left = self.tik_instance.Scalar("int64", name="tiling_core_left", init_value=0)
        self.tiling_over_row = self.tik_instance.Scalar("int64", name="tiling_over_row", init_value=0)
        self.tiling_job_num = self.tik_instance.Scalar("int64", name="tiling_job_num", init_value=0)
        self.tiling_bbox_num = self.tik_instance.Scalar("int64", name="tiling_bbox_num", init_value=0)
        self.tiling_valid_num = self.tik_instance.Scalar("int64", name="tiling_valid_num", init_value=0)

    def _cum_compute_tiling(self):
        """
        The function of tiling
        """
        self.tiling_ub = self.tik_instance.Tensor("int64", (8,), name="tiling_ub", scope=tik.scope_ubuf)
        self.tik_instance.data_move(self.tiling_ub, self.tiling_gm, 0, 1, 2, 0, 0)
        self.tiling_loop_per_core.set_as(self.tiling_ub[0])
        self.tiling_core_left.set_as(self.tiling_ub[1])
        self.tiling_over_row.set_as(self.tiling_ub[2])
        self.tiling_job_num.set_as(self.tiling_ub[3])
        self.tiling_bbox_num.set_as(self.tiling_ub[4])
        self.tiling_valid_num.set_as(self.tiling_ub[5])
        self.set_running_core_num(self.tiling_ub[6])

    def _clear_quad_flags(self):
        """
        clear_quad_flags

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        with self.tik_instance.if_scope(self.job_buf_row * Constant.ANCHOR_SIZE // self.job_buf_row < 1):
            _repeat_time = 1
        with self.tik_instance.else_scope():
            _repeat_time = self.job_buf_row * Constant.ANCHOR_SIZE // self.job_buf_row

        self.tik_instance.vector_dup(self._process_elem_count, self.quad_flag_ub, 0, _repeat_time, 1, 8)
        self.tik_instance.vector_dup(self._process_elem_count, self.quad_flags_sum_ub, 0, _repeat_time, 1, 8)

    def _calc_col_ge_flag(self):
        with self.tik_instance.if_scope(self.job_buf_row * Constant.ANCHOR_SIZE // self.job_buf_row < 1):
            _repeat_time = 1
        with self.tik_instance.else_scope():
            _repeat_time = self.job_buf_row * Constant.ANCHOR_SIZE // self.job_buf_row
        _deal_elem_num = 128 if self.bbox_dtype == "float16" else 64
        rep_offset_fp16 = _deal_elem_num * self.data_btype // 32
        with self.tik_instance.if_scope(self.bbox_dtype == "float16"):
            self._calc_col_ge_flag_fp16(_repeat_time, _deal_elem_num, rep_offset_fp16)
        with self.tik_instance.if_scope(self.bbox_dtype == "float32"):
            self._calc_col_ge_flag_fp32(_repeat_time, _deal_elem_num, rep_offset_fp16)

    def _calc_col_ge_flag_fp16(self, _repeat_time, _deal_elem_num, rep_offset_fp16):
        # operation on succession
        float16_minimum = 2**(-24)
        float16_scalar = 2**12
        self.tik_instance.vector_dup(_deal_elem_num, self.zeros_ub, 0, _repeat_time, 1, 8)

        self.tik_instance.vmax(_deal_elem_num, self.ones_ub, self.bbox_tensor_ub, self.zeros_ub, _repeat_time, 1, 1, 1,
                               rep_offset_fp16, rep_offset_fp16, rep_offset_fp16)

        # 'zeros_ub' as x - max(0 ,x)
        self.tik_instance.vsub(_deal_elem_num, self.zeros_ub, self.bbox_tensor_ub, self.ones_ub, _repeat_time, 1, 1, 1,
                               rep_offset_fp16, rep_offset_fp16, rep_offset_fp16)

        self.tik_instance.vabs(_deal_elem_num, self.ones_ub, self.zeros_ub, _repeat_time, 1, 1, rep_offset_fp16,
                               rep_offset_fp16)

        self.tik_instance.vector_dup(_deal_elem_num, self.quad_threshold_ub, float16_minimum, _repeat_time, 1,
                                     rep_offset_fp16)

        self.tik_instance.vmin(_deal_elem_num, self.zeros_ub, self.ones_ub, self.quad_threshold_ub, _repeat_time, 1, 1,
                               1, rep_offset_fp16, rep_offset_fp16, rep_offset_fp16)

        self.tik_instance.vector_dup(_deal_elem_num, self.quad_threshold_ub, float16_scalar, _repeat_time, 1,
                                     rep_offset_fp16)

        self.tik_instance.vmul(_deal_elem_num, self.ones_ub, self.quad_threshold_ub, self.zeros_ub, _repeat_time, 1, 1,
                               1, rep_offset_fp16, rep_offset_fp16, rep_offset_fp16)

        self.tik_instance.vmul(_deal_elem_num, self.zeros_ub, self.quad_threshold_ub, self.ones_ub, _repeat_time, 1, 1,
                               1, rep_offset_fp16, rep_offset_fp16, rep_offset_fp16)

        self.tik_instance.vector_dup(_deal_elem_num, self.quad_threshold_ub, 1, _repeat_time, 1, rep_offset_fp16)

        self.tik_instance.vsub(_deal_elem_num, self.ones_ub, self.zeros_ub, self.quad_threshold_ub, _repeat_time, 1, 1,
                               1, rep_offset_fp16, rep_offset_fp16, rep_offset_fp16)

        # abs
        self.tik_instance.vabs(_deal_elem_num, self.quad_flag_ub, self.ones_ub, _repeat_time, 1, 1, rep_offset_fp16,
                               rep_offset_fp16)

        # step 4, add once unfold flag into sum flags
        self.__add_col_flag_to_sum()

    def _calc_col_ge_flag_fp32(self, _repeat_time, _deal_elem_num, rep_offset_fp16):
        # operation on succession
        float32_minimum = 2**(-126)
        float32_scalar1 = 2**44
        float32_scalar2 = 2**38
        self.tik_instance.vector_dup(_deal_elem_num, self.zeros_ub, 0, _repeat_time, 1, 8)

        self.tik_instance.vmax(_deal_elem_num, self.ones_ub, self.bbox_tensor_ub, self.zeros_ub, _repeat_time, 1, 1, 1,
                               rep_offset_fp16, rep_offset_fp16, rep_offset_fp16)

        # 'zeros_ub' as x - max(0 ,x)
        self.tik_instance.vsub(_deal_elem_num, self.zeros_ub, self.bbox_tensor_ub, self.ones_ub, _repeat_time, 1, 1, 1,
                               rep_offset_fp16, rep_offset_fp16, rep_offset_fp16)
        # |x - max(0 ,x)|
        self.tik_instance.vabs(_deal_elem_num, self.ones_ub, self.zeros_ub, _repeat_time, 1, 1, rep_offset_fp16,
                               rep_offset_fp16)
        
        self.tik_instance.vector_dup(_deal_elem_num, self.quad_threshold_ub, float32_minimum, _repeat_time, 1,
                                     rep_offset_fp16)
        # min(|x - max(0 ,x)|, float32_minimum)
        self.tik_instance.vmin(_deal_elem_num, self.zeros_ub, self.ones_ub, self.quad_threshold_ub, _repeat_time, 1, 1,
                               1, rep_offset_fp16, rep_offset_fp16, rep_offset_fp16)
        # min(|x - max(0 ,x)|, float32_minimum) * 2**44 * 2**44 * 2**38
        self.tik_instance.vector_dup(_deal_elem_num, self.quad_threshold_ub, float32_scalar1, _repeat_time, 1,
                                     rep_offset_fp16)

        self.tik_instance.vmul(_deal_elem_num, self.ones_ub, self.quad_threshold_ub, self.zeros_ub, _repeat_time, 1, 1,
                               1, rep_offset_fp16, rep_offset_fp16, rep_offset_fp16)

        self.tik_instance.vmul(_deal_elem_num, self.zeros_ub, self.quad_threshold_ub, self.ones_ub, _repeat_time, 1, 1,
                               1, rep_offset_fp16, rep_offset_fp16, rep_offset_fp16)
        
        self.tik_instance.vector_dup(_deal_elem_num, self.quad_threshold_ub, float32_scalar2, _repeat_time, 1,
                                     rep_offset_fp16)
        self.tik_instance.vmul(_deal_elem_num, self.ones_ub, self.quad_threshold_ub, self.zeros_ub, _repeat_time, 1, 1,
                               1, rep_offset_fp16, rep_offset_fp16, rep_offset_fp16)

        self.tik_instance.vector_dup(_deal_elem_num, self.quad_threshold_ub, 1, _repeat_time, 1, rep_offset_fp16)
        # （min(|x - max(0 ,x)|, float32_minimum) * 2**44 * 2**44 * 2**38） - 1
        self.tik_instance.vsub(_deal_elem_num, self.zeros_ub, self.ones_ub, self.quad_threshold_ub, _repeat_time, 1, 1,
                               1, rep_offset_fp16, rep_offset_fp16, rep_offset_fp16)

        # abs
        self.tik_instance.vabs(_deal_elem_num, self.quad_flag_ub, self.zeros_ub, _repeat_time, 1, 1, rep_offset_fp16,
                               rep_offset_fp16)

        # step 4, add once unfold flag into sum flags
        self.__add_col_flag_to_sum()

    def _calc_col_lt_flag(self):

        """
        calculate for each column, and add this column'result into
        quad_flags_sum
        """
        # set the mask to make sure that which to calc and compare
        # input1:
        #      x1, y1, x2, y2
        #      0.5, 0.2, 0.4, 0.5
        # input2:
        #      x, y
        #      0.3, 0.2
        # we only wanna sequence data x1,x2 cmp to x
        # and y1,y2 cmp to y
        # so set the mask to get this goal
        with self.tik_instance.if_scope(self.job_buf_row * Constant.ANCHOR_SIZE // self.job_buf_row < 1):
            _repeat_time = 1
        with self.tik_instance.else_scope():
            _repeat_time = self.job_buf_row * Constant.ANCHOR_SIZE // self.job_buf_row
        _deal_elem_num = 128 if self.bbox_dtype == "float16" else 64

        rep_offset_fp16 = _deal_elem_num * self.data_btype // 32
        with self.tik_instance.if_scope(self.bbox_dtype == "float16"):
            self._calc_col_lt_flag_fp16(_repeat_time, _deal_elem_num, rep_offset_fp16)
        with self.tik_instance.if_scope(self.bbox_dtype == "float32"):
            self._calc_col_lt_flag_fp32(_repeat_time, _deal_elem_num, rep_offset_fp16)
    
    def _calc_col_lt_flag_fp16(self, _repeat_time, _deal_elem_num, rep_offset_fp16):
        float16_minimum = 2**(-24)
        float16_scalar = 2**12
        _mask64 = 0xAAAAAAAAAAAAAAAA
        _maskh = _mask64 if self.bbox_dtype == "float16" else 0
        self.tik_instance.vector_dup([_maskh, _mask64], self.quad_threshold_ub, self.threshold_h, _repeat_time, 1,
                                     rep_offset_fp16)

        _mask64 = 0x5555555555555555
        _maskw = _mask64 if self.bbox_dtype == "float16" else 0
        self.tik_instance.vector_dup([_maskw, _mask64], self.quad_threshold_ub, self.threshold_w, _repeat_time, 1,
                                     rep_offset_fp16)

        self.tik_instance.vsub(_deal_elem_num, self.zeros_ub, self.quad_threshold_ub, self.bbox_tensor_ub, _repeat_time,
                               1, 1, 1, rep_offset_fp16, rep_offset_fp16, rep_offset_fp16)

        self.tik_instance.vector_dup(_deal_elem_num, self.quad_threshold_ub, float16_minimum, _repeat_time, 1, 8)

        self.tik_instance.vmin(_deal_elem_num, self.ones_ub, self.zeros_ub, self.quad_threshold_ub, _repeat_time, 1, 1,
                               1, rep_offset_fp16, rep_offset_fp16, rep_offset_fp16)

        self.tik_instance.vector_dup(_deal_elem_num, self.quad_threshold_ub, 0, _repeat_time, 1, rep_offset_fp16)
        self.tik_instance.vmax(_deal_elem_num, self.zeros_ub, self.ones_ub, self.quad_threshold_ub, _repeat_time, 1, 1,
                               1, rep_offset_fp16, rep_offset_fp16, rep_offset_fp16)

        # mul 2 times
        self.tik_instance.vector_dup(_deal_elem_num, self.quad_threshold_ub, float16_scalar, _repeat_time, 1,
                                     rep_offset_fp16)

        self.tik_instance.vmul(_deal_elem_num, self.ones_ub, self.quad_threshold_ub, self.zeros_ub, _repeat_time, 1, 1,
                               1, rep_offset_fp16, rep_offset_fp16, rep_offset_fp16)

        self.tik_instance.vmul(_deal_elem_num, self.quad_flag_ub, self.quad_threshold_ub, self.ones_ub, _repeat_time, 1,
                               1, 1, rep_offset_fp16, rep_offset_fp16, rep_offset_fp16)

        # step 4, add once unfold flag into sum flags
        self.__add_col_flag_to_sum()

    def _calc_col_lt_flag_fp32(self, _repeat_time, _deal_elem_num, rep_offset_fp16):
        """
        calculate for each column, and add this column'result into
        quad_flags_sum
        """
        float32_minimum = 2**(-126)
        float32_scalar1 = 2**44
        float32_scalar2 = 2**38
        _mask64 = 0xAAAAAAAAAAAAAAAA
        _maskh = _mask64 if self.bbox_dtype == "float16" else 0
        self.tik_instance.vector_dup([_maskh, _mask64], self.quad_threshold_ub, self.threshold_h, _repeat_time, 1,
                                     rep_offset_fp16)

        _mask64 = 0x5555555555555555
        _maskw = _mask64 if self.bbox_dtype == "float16" else 0
        self.tik_instance.vector_dup([_maskw, _mask64], self.quad_threshold_ub, self.threshold_w, _repeat_time, 1,
                                     rep_offset_fp16)

        self.tik_instance.vsub(_deal_elem_num, self.zeros_ub, self.quad_threshold_ub, self.bbox_tensor_ub, _repeat_time,
                               1, 1, 1, rep_offset_fp16, rep_offset_fp16, rep_offset_fp16)

        self.tik_instance.vector_dup(_deal_elem_num, self.quad_threshold_ub, float32_minimum, _repeat_time, 1, 8)

        self.tik_instance.vmin(_deal_elem_num, self.ones_ub, self.zeros_ub, self.quad_threshold_ub, _repeat_time, 1, 1,
                               1, rep_offset_fp16, rep_offset_fp16, rep_offset_fp16)

        self.tik_instance.vector_dup(_deal_elem_num, self.quad_threshold_ub, 0, _repeat_time, 1, rep_offset_fp16)
        self.tik_instance.vmax(_deal_elem_num, self.zeros_ub, self.ones_ub, self.quad_threshold_ub, _repeat_time, 1, 1,
                               1, rep_offset_fp16, rep_offset_fp16, rep_offset_fp16)

        # mul 3 times
        self.tik_instance.vector_dup(_deal_elem_num, self.quad_threshold_ub, float32_scalar1, _repeat_time, 1,
                                     rep_offset_fp16)

        self.tik_instance.vmul(_deal_elem_num, self.ones_ub, self.quad_threshold_ub, self.zeros_ub, _repeat_time, 1, 1,
                               1, rep_offset_fp16, rep_offset_fp16, rep_offset_fp16)

        self.tik_instance.vmul(_deal_elem_num, self.zeros_ub, self.quad_threshold_ub, self.ones_ub, _repeat_time, 1,
                               1, 1, rep_offset_fp16, rep_offset_fp16, rep_offset_fp16)

        self.tik_instance.vector_dup(_deal_elem_num, self.quad_threshold_ub, float32_scalar2, _repeat_time, 1,
                                     rep_offset_fp16)
        
        self.tik_instance.vmul(_deal_elem_num, self.quad_flag_ub, self.quad_threshold_ub, self.zeros_ub, _repeat_time,
                               1, 1, 1, rep_offset_fp16, rep_offset_fp16, rep_offset_fp16)
        # step 4, add once unfold flag into sum flags
        self.__add_col_flag_to_sum()

    def _merge_successive_four_elem_to_one_val(self):
        """
        merge successive four elements elements into once.
        """
        with self.tik_instance.if_scope(self.job_buf_row * Constant.ANCHOR_SIZE // self.job_buf_row < 1):
            _repeat_time = 1
        with self.tik_instance.else_scope():
            _repeat_time = self.job_buf_row * Constant.ANCHOR_SIZE // self.job_buf_row
        rep_offset_fp16 = self._process_elem_count * self.data_btype // 32

        self.tik_instance.vcpadd(
            self._process_elem_count,
            self.ret_unfold_half_ub,
            self.quad_flags_sum_ub,
            _repeat_time,  # repeat
            1,
            1,
            rep_offset_fp16)

        with self.tik_instance.if_scope(_repeat_time // 2 < 1):
            _repeat_times = 1
        with self.tik_instance.else_scope():
            _repeat_times = _repeat_time // 2

        self.tik_instance.vcpadd(self._process_elem_count, self.data_ret_ub, self.ret_unfold_half_ub, _repeat_times, 1,
                                 1, rep_offset_fp16)

    def _transform_to_one_or_zero(self):
        # 128 as the maximum elements could been processed in once command
        with self.tik_instance.if_scope(self.job_buf_row // self.job_buf_row < 1):
            _repeat_time = 1
        with self.tik_instance.else_scope():
            _repeat_time = self.job_buf_row // self.job_buf_row
        rep_offset_fp16 = self._process_elem_count * self.data_btype // 32
        rep_offset_int8 = self._process_elem_count * 1 // 32

        self.tik_instance.vmuls(
            self._process_elem_count,
            self.quad_threshold_ub,
            self.data_ret_ub,
            0.067,  # scalar, 0.25&floor, or 0.15 & round
            _repeat_time,  # repeat
            1,
            1,
            rep_offset_fp16,
            rep_offset_fp16)

        # (0,1,2,3)/4 --> 0/1
        # RuntimeError: v100 mini doesn't support float16 to int8 with floor mode.
        # mini support 'none' only, ei, round mode, 0.15*3-->0; 0.15*4 --> 1
        # cloud support the floor mode
        if self.bbox_dtype == "float32":
            self.tik_instance.vconv(self._process_elem_count, 'none', self.quad_threshold_ub_fp16,
                                    self.quad_threshold_ub, _repeat_time, 1, 1, rep_offset_fp16, rep_offset_fp16)
            self.tik_instance.vconv(self._process_elem_count, 'none', self.data_ret_int8_ub,
                                    self.quad_threshold_ub_fp16, _repeat_time, 1, 1, rep_offset_int8, rep_offset_fp16)
        else:
            self.tik_instance.vconv(self._process_elem_count, 'none', self.data_ret_int8_ub, self.quad_threshold_ub,
                                    _repeat_time, 1, 1, rep_offset_int8, rep_offset_fp16)

    def _get_last_job_header(self, head_type):
        """
        get_last_job_header

        Parameters
        ----------
        head_type : str
            head_type
        Returns
        -------
        result : int
            last_align_job_head
        """
        if head_type == "bbox":
            _unit = 4
            element_total_num = self.tiling_bbox_num
        elif head_type == "valid":
            _unit = 1
            element_total_num = self.tiling_valid_num
        else:
            error_manager_vector.raise_err_input_value_invalid("check_valid", "head_type", "bbox or valid", head_type)
        with self.tik_instance.if_scope(self.tiling_job_num > 1):
            self.last_align_job_head.set_as(-self.__default_rows_per_job * _unit + element_total_num)
        with self.tik_instance.else_scope():
            self.last_align_job_head.set_as(-self.last_job_row_aligned * _unit + element_total_num)

        # if only one job, we should cut in at the start.
        with self.tik_instance.if_scope(self.last_align_job_head < 0):
            self.last_align_job_head.set_as(0)

    def _handle_one_core(self, job_index, inverted=False):
        """
        entrance for each core

        Parameters
        ----------
        job_index : int
            job_index
        inverted : bool
            inverted

        Returns
        -------
        resut : instance
            tik_instance
        """
        self._clear_quad_flags()  # must clear it in each core
        self._move_bbox_to_ub(job_index, inverted)
        with self.tik_instance.for_range(0, 2) as n_col:
            with self.tik_instance.if_scope(n_col == 0):
                self._calc_col_ge_flag()

            with self.tik_instance.else_scope():
                self._calc_col_lt_flag()

        self._merge_successive_four_elem_to_one_val()
        self._transform_to_one_or_zero()

        self.__move_once_job_ret_to_gm(job_index, inverted)

    def __calc_last_job_row(self):
        """
        __calc_last_job_row

        Parameters
        ----------
        None

        Returns
        -------
        result : last_job_bytes // (self.bbox_dtype_size * self.bbox_shape[1])
        """

        # if last job is equal to the default job row.
        last_job_row = self.tik_instance.Scalar("int64", name="last_job_row", init_value=0)
        with self.tik_instance.if_scope(self.tiling_over_row < 1):
            last_job_row.set_as(self.__default_rows_per_job)
        with self.tik_instance.else_scope():
            over_bytes = self.tiling_over_row * Constant.ANCHOR_SIZE * self.bbox_dtype_size

            _align_size = 32
            align_32bytes_count = self.tik_instance.Scalar("int64", name="align_32bytes_count", init_value=0)
            align_32bytes_count.set_as(over_bytes // _align_size)

            with self.tik_instance.if_scope(over_bytes % _align_size > 0):
                align_32bytes_count.set_as(align_32bytes_count + 1)

            last_job_bytes = _align_size * align_32bytes_count
            # total less than 32B
            self.padded_bytes.set_as(last_job_bytes - over_bytes)
            last_job_row.set_as(last_job_bytes // (self.bbox_dtype_size * Constant.ANCHOR_SIZE))
        return last_job_row

    def __add_col_flag_to_sum(self):
        """
        add_col_flag_to_sum.
        """
        with self.tik_instance.if_scope(self.job_buf_row * Constant.ANCHOR_SIZE // self.job_buf_row < 1):
            _repeat_time = 1
        with self.tik_instance.else_scope():
            _repeat_time = self.job_buf_row * Constant.ANCHOR_SIZE // self.job_buf_row
        rep_offset_fp16 = self._process_elem_count * self.data_btype // 32

        self.tik_instance.vadd(
            self._process_elem_count,
            self.quad_flags_sum_ub,
            self.quad_flag_ub,
            self.quad_flags_sum_ub,
            _repeat_time,  # repeat
            1,
            1,
            1,
            rep_offset_fp16,
            rep_offset_fp16,
            rep_offset_fp16)

    def __move_once_job_ret_to_gm(self, job_index=0, inverted=False):
        """
        __move_once_job_ret_to_gm

        Parameters
        ----------
        job_index : int
            job_index
        inverted : bool
            inverted

        Returns
        -------
        None
        """
        # int8 as result
        _burst_int8 = self.tik_instance.Scalar("int64", name="_burst_int8", init_value=0)
        _burst_int8_byte = self.tik_instance.Scalar("int64", name="_burst_int8_byte", init_value=0)
        with self.tik_instance.if_scope(inverted is False):
            with self.tik_instance.if_scope(self.job_buf_row // 32 < 1):
                _burst_int8.set_as(1)
            with self.tik_instance.else_scope():
                _burst_int8.set_as(self.job_buf_row // 32)

            self.tik_instance.data_move(self.date_ret_valid_gm[self.job_buf_row * job_index], self.data_ret_int8_ub, 0,
                                        1, _burst_int8, 0, 0)
        with self.tik_instance.else_scope():
            self._get_last_job_header("valid")
            with self.tik_instance.if_scope(self.tiling_job_num > 1):
                with self.tik_instance.if_scope(self.job_buf_row * self.valid_dtype_size // 32 < 1):
                    _burst_int8.set_as(1)
                with self.tik_instance.else_scope():
                    _burst_int8.set_as(self.job_buf_row * self.valid_dtype_size // 32)
                self.tik_instance.data_move(self.date_ret_valid_gm[self.last_align_job_head], 
                                            self.data_ret_int8_ub, 0, 1,
                                            _burst_int8, 0, 0)
            with self.tik_instance.else_scope():
                with self.tik_instance.if_scope(self.last_job_row_aligned * self.valid_dtype_size // 32 < 1):
                    _burst_int8.set_as(1)
                with self.tik_instance.else_scope():
                    _burst_int8.set_as(self.last_job_row_aligned * self.valid_dtype_size // 32)
                    # owing to the output type is int8, different with input fp16
                    #  | * * * * * * * * * *(32B)
                    #                * * * * * * * * *(32B) |
                    # small shape, when last job output value non-align
                    _over_in_single_job = self.last_job_row_aligned * \
                                        self.valid_dtype_size % 32
                    with self.tik_instance.if_scope(_over_in_single_job > 0):
                        _burst_int8.set_as(_burst_int8 + 1)  # hard over write with one more  block
                _burst_int8_byte.set_as(self.__default_rows_per_job)
                with self.tik_instance.if_scope(self.tiling_over_row > 0):
                    _burst_int8_byte.set_as(self.tiling_over_row * self.valid_dtype_size)
                if self.data_move_pad_support:
                    self.tik_instance.data_move_pad(self.date_ret_valid_gm[self.last_align_job_head], 
                                                    self.data_ret_int8_ub, 1,
                                                    _burst_int8_byte, 0, 0)
                else:
                    self.tik_instance.data_move(self.date_ret_valid_gm[self.last_align_job_head], 
                                                self.data_ret_int8_ub, 0, 1,
                                                _burst_int8, 0, 0)

    def __get_threshold(self):
        """
        __get_threshold

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.tik_instance.data_move(self.img_metas_ub, self.img_metas_gm, 0, 1, 1, 0, 0)
        img_meta_tmp_ub = self.tik_instance.Tensor(self.img_metas_dtype, (16,),
                                                   name="img_meta_tmp_ub",
                                                   scope=tik.scope_ubuf)
        scalar_ratio = self.tik_instance.Scalar(self.img_metas_dtype, "scalar_ratio")
        scalar_ratio.set_as(self.img_metas_ub[2])
        #img_metas(1*scalar_ratio,
        #          2*scalar_ratio,
        #          0.2 * scalar_ratio
        #          4 * scalar_ratio
        #          5 * scalar_ratio
        #          6 * scalar_ratio
        #          ......)
        #we only need front two
        self.tik_instance.vmuls(16, img_meta_tmp_ub, self.img_metas_ub, scalar_ratio, 1, 1, 1, 8, 8)
        self.threshold_h.set_as(img_meta_tmp_ub[0])
        self.threshold_w.set_as(img_meta_tmp_ub[1])

    def _move_bbox_to_ub(self, job_index, inverted=False):
        """
        __move_job_bbox_to_ub

        Parameters
        ----------
        job_index : int
            job_index
        inverted : bool
            inverted

        Returns
        -------
        None
        """
        # fp32--/8--4B. fp16-/16--2B, int8-/32--1B
        _burst_fp16 = self.tik_instance.Scalar("int64", name="_burst_fp16", init_value=0)
        with self.tik_instance.if_scope(inverted is False):
            with self.tik_instance.if_scope(self.job_buf_row * Constant.ANCHOR_SIZE * self.data_btype // 32 < 1):
                _burst_fp16.set_as(1)
            with self.tik_instance.else_scope():
                _burst_fp16.set_as(self.job_buf_row * Constant.ANCHOR_SIZE * self.data_btype // 32)
            self.tik_instance.data_move(self.bbox_tensor_ub,
                                        self.bbox_gm[self.job_buf_row * job_index * Constant.ANCHOR_SIZE], 0, 1,
                                        _burst_fp16, 0, 0)
        with self.tik_instance.else_scope():
            self._get_last_job_header("bbox")
            with self.tik_instance.if_scope(self.tiling_job_num > 1):
                with self.tik_instance.if_scope(self.job_buf_row * Constant.ANCHOR_SIZE * self.data_btype // 32 < 1):
                    _burst_fp16.set_as(1)
                with self.tik_instance.else_scope():
                    _burst_fp16.set_as(self.job_buf_row * Constant.ANCHOR_SIZE * self.data_btype // 32)
                self.tik_instance.data_move(self.bbox_tensor_ub, 
                                            self.bbox_gm[self.last_align_job_head], 
                                            0, 1, _burst_fp16, 0, 0)
            with self.tik_instance.else_scope():
                with self.tik_instance.if_scope(
                        self.last_job_row_aligned * Constant.ANCHOR_SIZE * self.data_btype // 32 < 1):
                    _burst_fp16.set_as(1)
                with self.tik_instance.else_scope():
                    _burst_fp16.set_as(self.last_job_row_aligned * Constant.ANCHOR_SIZE * self.data_btype // 32)
                _burst_fp16_byte = self.last_job_row_aligned \
                    * Constant.ANCHOR_SIZE * self.data_btype - self.padded_bytes
                if self.data_move_pad_support:
                    self.tik_instance.data_move_pad(self.bbox_tensor_ub, 
                                                    self.bbox_gm[self.last_align_job_head], 
                                                    1, _burst_fp16_byte, 0, 0)
                else:
                    self.tik_instance.data_move(self.bbox_tensor_ub, 
                                                self.bbox_gm[self.last_align_job_head], 
                                                0, 1, _burst_fp16, 0, 0)


@register_operator("CheckValid")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def check_valid(bbox_tensor, img_metas, valid_tensor, kernel_name="check_valid"):
    """
    algorithm: check_valid
    calculating check_valid(bbox_tensor, img_metas):
    returns a valid tensor consists of 0/1.
    If the input x is [1,2,3,4],the img_metas tensor is
    (height, width),and the output is 0 or 1 to decide whether 
    the input x in feature map(img_metas)
    For example:
    bbox_tensor : [1, 2, 3, 4]
    img_metas:[5, 5]
    res :  [1]
    Parameters
    ----------
    bbox_tensor: dict
        dict with keys(shape and dtype) of bbox_tensor, and the dtype of bbox_tensor must
        be in [float16]
    img_metas : dict
        dict with keys(shape and dtype) of bbox_tensor, and the dtype of bbox_tensor must
        be in [float16]
    valid_tensor: list
        list contains 0/1,and the shape is same as bbox_tensor[0]
    kernel_name: str
        kernel name, default value is "check_valid"

    Returns
    -------
    None
    """
    bbox_shape = bbox_tensor.get("shape")
    bbox_dtype = bbox_tensor.get("dtype").lower()
    bbox_dtype_check_list = [
        "float16",
        "float32",
    ]
    para_check.check_dtype(bbox_dtype, bbox_dtype_check_list, param_name="bbox_tensor")
    img_metas_dtype = img_metas.get("dtype").lower()

    if img_metas_dtype != bbox_dtype:
        error_detail = "dtype of img_metas and bbox_tensor should be same"
        error_manager_vector.raise_err_two_input_dtype_invalid(kernel_name, "img_metas", \
                                                               "bbox_tensor", error_detail)
    dim_cnt = 2
    dim_second_size = 4
    is_unknown_rank = is_unknown_rank_input(bbox_tensor)
    if len(bbox_shape) != dim_cnt and not is_unknown_rank:
        error_detail = "the length of bbox_tensor'shape must be 2, while it is: %d" % len(bbox_shape)
        error_manager_vector.raise_err_input_shape_invalid(kernel_name, "bbox_tensor", error_detail)
    if bbox_shape[-1] != dim_second_size and not is_unknown_rank:
        error_detail = "the second dim of bbox_tensor must be 4, while it's: %d" % bbox_shape[-1]
        error_manager_vector.raise_err_input_shape_invalid(kernel_name, "bbox_tensor", error_detail)
    cvd = CheckValid(bbox_tensor, img_metas, valid_tensor, kernel_name)
    tik_instance = cvd.check_valid_compute()
    return tik_instance