# -*- coding:utf-8 -*-
"""
Copyright (c) Huawei Technologies Co., Ltd. 2021-2023. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.
You may not use this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

resize_nearest_neighbor_v2_grad
"""
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from impl.util import util_tik_comm_func
from impl.util.platform_adapter import error_manager_vector
from tbe.common.platform import get_bit_len


# 'pylint:disable=too-few-public-methods,too-many-instance-attributes
class Constant:
    """
    The class for constant
    """
    # max uint16
    MAX_UINT16 = 2 ** 16 - 1
    # max int64
    MAX_INT64 = 2 ** 63 - 1
    # tiling param num
    TILING_ARG_NUM = 12
    # reserved ub size
    RESERVED_UB_SIZE = 8 * 1024
    # The max repeat_times of vadd instruction
    VADD_MAX_REPEAT_TIMES = 255
    # max w_loop_segment
    W_LOOP_SEGMENT_MAX = 128

    BLOCK_BYTES = 32
    B64_PER_BLOCK = 4
    B16_BYTES = 2
    B32_BYTES = 4


# 'pylint: disable=too-many-instance-attributes,too-many-arguments,unused-argument
# 'pylint: disable=too-many-locals,too-many-statements,unused-argument,invalid-name
class ResizeNearestNeighborV2Grad:
    """
    Function: use to store ResizeNearestNeighborV2Grad base parameters
    Modify: 2021-03-10
    """
    grad_type_mapping = {
        "float32" : "float32",
        "bfloat16" : "float32",
    }

    @staticmethod
    def _get_ceil_int(int1, int2):
        return (int1 + int2 - 1) // int2

    def _init_base_param(self):
        self.ai_core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.ub_size_bytes = (tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) - Constant.RESERVED_UB_SIZE)
        self.ub_block_size = int(tbe_platform.get_soc_spec("ubblock_size"))
        self.grad_new_type_size = get_bit_len(self.grads_new_dtype) // 8
        self.block_num = self.ub_block_size // self.grad_new_type_size
        self.b16_block_num = 16
        self.vector_num = self.block_num * 8
        self.ub_max_num = self.ub_size_bytes // self.ub_block_size // 2 * self.block_num


    def _init_gm(self):
        self.tiling_gm = self.tik_instance.Tensor("int64", (Constant.TILING_ARG_NUM,),
                                                  name="tiling_gm", scope=tik.scope_gm)

        self.grads_gm = self.tik_instance.Tensor(self.grads_dtype, [Constant.MAX_INT64],
                                                 name="grads_gm", scope=tik.scope_gm)
        self.size_gm = self.tik_instance.Tensor(self.size_dtype, (2,),
                                                name="size_gm", scope=tik.scope_gm)
        self.out_gm = self.tik_instance.Tensor(self.grads_dtype, [Constant.MAX_INT64],
                                               name="out_gm", scope=tik.scope_gm, is_atomic_add=True)

    def _init_workspace(self):
        self.is_bfp16 = self.grads_dtype in ("bfloat16", )
        if self.is_bfp16:
            self.out_workspace = self.tik_instance.Tensor("float32", (Constant.MAX_INT64, ),
                                                          name="out_workspace",
                                                          scope=tik.scope_gm,
                                                          is_workspace=True,
                                                          is_atomic_add=True)
            self.sync_workspace = self.tik_instance.Tensor("int64", (self.ai_core_num * Constant.B64_PER_BLOCK, ),
                                                           name="sync_workspace",
                                                           scope=tik.scope_gm,
                                                           is_workspace=True,
                                                           is_atomic_add=True)

    def _init_tiling_data(self):
        # init tiling data
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
        self.tiling_core_num = self.tik_instance.Scalar("int64", name="tiling_core_num", init_value=self.ai_core_num)

    def _init_core_param(self):
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
        # h process len for per core
        self.cut_height_num = self.tik_instance.Scalar("int64", name="cut_height_num")
        # w process len for per core
        self.cut_width_num = self.tik_instance.Scalar("int64", name="cut_width_num")

    def __init__(self, grads, size, y, align_corners, half_pixel_centers, kernel_name):
        self.tik_instance = tik.Tik()
        self.grads_dtype = grads.get("dtype").lower()
        self.size_dtype = size.get("dtype").lower()
        self.grads_format = grads.get("format").lower()
        self.align_corners = align_corners
        self.half_pixel_centers = half_pixel_centers
        self.kernel_name = kernel_name
        self.grads_new_dtype = self.grad_type_mapping.get(self.grads_dtype)

        self.grads_shape_c0 = 16
        self.height_idx_segment_num = 512
        self.width_idx_segment_num = 512

        self.stride_threshold = Constant.MAX_UINT16 if self.grads_dtype in ("float16", ) else Constant.MAX_UINT16 // 2
        self.is_suport_vdiv = tbe_platform.api_check_support("tik.vdiv", "float32")
        self.is_move_pad_support = tbe_platform.api_check_support("tik.data_move_pad")

        # init ub
        self.height_idx = None
        self.width_idx = None
        self.input_idx = None
        self.grad_in_ub = None
        self.grad_out_ub = None
        self.grad_in_gm_ping = None
        self.grad_in_gm_ping = None
        self.mid_ub_ping = None
        self.mid_ub_pang = None
        self.grad_out_ub_ping = None
        self.grad_out_ub_pang = None
        self.input_offset = None

        self._init_base_param()
        self._init_gm()
        self._init_workspace()
        self._init_tiling_data()
        self._init_core_param()

    def scalar_vconv_int32_to_fp32(self, int32_value, float32_value):
        """
        vconv one scalar from int32 to fp32 usr vector
        """
        with self.tik_instance.new_stmt_scope():
            idx_int32_tmp = self.tik_instance.Tensor("int32", (64,),
                                                     name="idx_int32_tmp", scope=tik.scope_ubuf)
            idx_fp32_tmp = self.tik_instance.Tensor("float32", (64,),
                                                    name="idx_fp32_tmp", scope=tik.scope_ubuf)
            idx_int32_tmp[0].set_as(int32_value)
            util_tik_comm_func.tik_func_vconv(self.tik_instance, idx_fp32_tmp, idx_int32_tmp, 1)
            float32_value.set_as(idx_fp32_tmp[0])

    def calc_out_idx(self, scale, des_idx_ub, src_idx_ub, idx_num):
        """
        if not self.align_corners and self.half_pixel_centers:
            vconv_f322s32f((idx + 0.5) * scale)
        if not self.align_corners and not self.half_pixel_centers:
            vconv_f322s32f(idx * scale)
        if self.align_corners and not self.half_pixel_centers:
            vconv_f322s32r(idx * scale)
        if self.align_corners and self.half_pixel_centers:
            vconv_f322s32r((idx + 0.5) * scale)
        """
        with self.tik_instance.new_stmt_scope():
            calcu_out_in_idx_tmp_ub = self.tik_instance.Tensor(src_idx_ub.dtype, src_idx_ub.shape,
                                                               name="calcu_out_in_idx_tmp_ub", scope=tik.scope_ubuf)
            vector_repeat_num = ResizeNearestNeighborV2Grad._get_ceil_int(idx_num, 64)
            if self.half_pixel_centers:
                # `calcu: (idx + 0.5) * scale`
                self.tik_instance.vadds(64, calcu_out_in_idx_tmp_ub, src_idx_ub, 0.5,
                                        vector_repeat_num, 1, 1, 8, 8)
                self.tik_instance.vmuls(64, calcu_out_in_idx_tmp_ub, calcu_out_in_idx_tmp_ub, scale,
                                        vector_repeat_num, 1, 1, 8, 8)
            else:
                # `calcu: idx * scale`
                self.tik_instance.vmuls(64, calcu_out_in_idx_tmp_ub, src_idx_ub, scale,
                                        vector_repeat_num, 1, 1, 8, 8)
            if self.align_corners:
                # fix bug
                conv_mode = "away-zero" if tbe_platform.api_check_support("tik.data_move_pad") is True else "round"
                # will use vconv_f322s32r to cast to int32
                util_tik_comm_func.tik_func_vconv(self.tik_instance, des_idx_ub, calcu_out_in_idx_tmp_ub,
                                                  vector_repeat_num * 64, mode=conv_mode)
            else:
                # will use vconv_f322s32f to cast to int32
                util_tik_comm_func.tik_func_vconv(self.tik_instance, des_idx_ub, calcu_out_in_idx_tmp_ub,
                                                  vector_repeat_num * 64, mode="floor")

    def calculate_scale(self):
        """
        calculate scale user input h/w and output h/w
        """
        with self.tik_instance.new_stmt_scope():
            height_input_fp32 = self.tik_instance.Tensor("float32", (self.block_num * 2,),
                                                         name="height_input_fp32", scope=tik.scope_ubuf)
            width_input_fp32 = self.tik_instance.Tensor("float32", (self.block_num * 2,),
                                                        name="width_input_fp32", scope=tik.scope_ubuf)
            height_input_int32 = self.tik_instance.Tensor("int32", (self.block_num * 2,),
                                                          name="height_input_int32", scope=tik.scope_ubuf)
            width_input_int32 = self.tik_instance.Tensor("int32", (self.block_num * 2,),
                                                         name="width_input_int32", scope=tik.scope_ubuf)
            height_output_fp32 = self.tik_instance.Tensor("float32", (self.block_num * 2,),
                                                          name="height_output_fp32", scope=tik.scope_ubuf)
            width_output_fp32 = self.tik_instance.Tensor("float32", (self.block_num * 2,),
                                                         name="width_output_fp32", scope=tik.scope_ubuf)

            height_input_int32[0].set_as(self.tiling_out_height)
            width_input_int32[0].set_as(self.tiling_out_width)
            height_input_int32[self.block_num].set_as(self.tiling_in_height)
            width_input_int32[self.block_num].set_as(self.tiling_in_width)

            util_tik_comm_func.tik_func_vconv(self.tik_instance, height_input_fp32, height_input_int32, 1)
            util_tik_comm_func.tik_func_vconv(self.tik_instance, width_input_fp32, width_input_int32, 1)
            util_tik_comm_func.tik_func_vconv(self.tik_instance, height_output_fp32,
                                              height_input_int32[self.block_num:], 1)
            util_tik_comm_func.tik_func_vconv(self.tik_instance, width_output_fp32,
                                              width_input_int32[self.block_num:], 1)

            h_input_fp32 = self.tik_instance.Scalar("float32", "tmp_input_h", height_input_fp32[0])
            h_output_fp32 = self.tik_instance.Scalar("float32", "tmp_output_h", height_output_fp32[0])
            with self.tik_instance.if_scope(tik.all(self.align_corners, self.tiling_in_height > 1)):
                h_input_fp32.set_as(h_input_fp32 - 1.0)
                h_output_fp32.set_as(h_output_fp32 - 1.0)
            self.resize_scale_h.set_as(h_input_fp32 / h_output_fp32)

            w_input_fp32 = self.tik_instance.Scalar("float32", "tmp_input_w", width_input_fp32[0])
            w_output_fp32 = self.tik_instance.Scalar("float32", "tmp_output_w", width_output_fp32[0])
            with self.tik_instance.if_scope(tik.all(self.align_corners, self.tiling_in_width > 1)):
                w_input_fp32.set_as(w_input_fp32 - 1.0)
                w_output_fp32.set_as(w_output_fp32 - 1.0)
            self.resize_scale_w.set_as(w_input_fp32 / w_output_fp32)

    def resize_nearest_neighbor_v2_grad_operator(self):
        """
        resize_nearest_neighbor_v2_grad_operator
        """
        self._do_resize()
        opt_config = {"out_of_bound_sync_check": True,
                      "enable_const_fold": True
                     }

        tbe_context.get_context().add_compile_info("vars", {"ub_size": self.ub_size_bytes,
                                                            "core_num": self.ai_core_num,
                                                            "max_w_len": self.ub_max_num // self.grads_shape_c0,
                                                            "align_corners": int(self.align_corners),
                                                            "half_pixel_centers": int(self.half_pixel_centers)})

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=(self.grads_gm, self.size_gm),
                                   outputs=(self.out_gm,),
                                   flowtable=(self.tiling_gm,), config=opt_config)

        return self.tik_instance

    def _tiling_args(self, tiling_ub):
        """
        get runtime tiling parameters from tiling
        """
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
        self.tiling_core_num.set_as(tiling_ub[10])

    def _core_scalar_args_nchw_1(self, _core_idx):
        ori_block_num = self.ub_block_size // (get_bit_len(self.grads_dtype) // 8)
        input_size = self.tiling_batch * self.tiling_c1 * self.tiling_in_height * self.tiling_in_width
        input_blocks = ResizeNearestNeighborV2Grad._get_ceil_int(input_size, ori_block_num)
        nc_segment = ResizeNearestNeighborV2Grad._get_ceil_int(input_blocks, self.tiling_bc1_cut_num)
        self.core_nc_start.set_as((_core_idx % (self.tiling_bc1_cut_num)) * nc_segment * ori_block_num)
        self.core_nc_num.set_as(nc_segment * ori_block_num)
        with self.tik_instance.if_scope(self.core_nc_start + self.core_nc_num > input_size):
            self.core_nc_num.set_as(input_size - self.core_nc_start)

        with self.tik_instance.if_scope(_core_idx >= self.tiling_bc1_cut_num):
            self.core_nc_num.set_as(0)

    def _core_scalar_args_nchw_3(self, _core_idx):
        ori_block_num = self.ub_block_size // (get_bit_len(self.grads_dtype) // 8)
        output_size = self.tiling_batch * self.tiling_c1
        input_hw = self.tiling_in_height * self.tiling_in_width
        output_blocks = ResizeNearestNeighborV2Grad._get_ceil_int(output_size, ori_block_num)
        nc_segment = ResizeNearestNeighborV2Grad._get_ceil_int(output_blocks, self.tiling_bc1_cut_num)
        h_segment = ResizeNearestNeighborV2Grad._get_ceil_int(input_hw, self.tiling_height_cut_num)
        h_segment = ResizeNearestNeighborV2Grad._get_ceil_int(h_segment, self.vector_num) * self.vector_num
        self.core_nc_start.set_as((_core_idx % (self.tiling_bc1_cut_num)) * nc_segment * ori_block_num)
        self.core_height_start.set_as((_core_idx // self.tiling_bc1_cut_num) % self.tiling_height_cut_num * h_segment)
        self.core_nc_num.set_as(nc_segment * ori_block_num)
        self.core_height_num.set_as(h_segment)

        with self.tik_instance.if_scope(self.core_nc_start + self.core_nc_num > output_size):
            self.core_nc_num.set_as(output_size - self.core_nc_start)
        with self.tik_instance.if_scope(self.core_height_start + self.core_height_num > input_hw):
            self.core_height_num.set_as(input_hw - self.core_height_start)

        with self.tik_instance.if_scope(self.core_nc_num < 0):
            self.core_nc_num.set_as(0)
        with self.tik_instance.if_scope(self.core_height_num < 0):
            self.core_height_num.set_as(0)

        with self.tik_instance.if_scope(_core_idx >= self.tiling_bc1_cut_num * self.tiling_height_cut_num):
            self.core_nc_num.set_as(0)
            self.core_height_num.set_as(0)

    def _core_scalar_args_nc1hwc0(self, _core_idx):
        """
        get runtime tiling parameters from tiling
        """
        self.cut_height_num.set_as(self.tiling_in_height)
        self.cut_width_num.set_as(self.tiling_in_width)

        nc_segment = ResizeNearestNeighborV2Grad._get_ceil_int(self.tiling_batch * self.tiling_c1,
                                                               self.tiling_bc1_cut_num)
        h_segment = ResizeNearestNeighborV2Grad._get_ceil_int(self.cut_height_num, self.tiling_height_cut_num)
        w_segment = ResizeNearestNeighborV2Grad._get_ceil_int(self.cut_width_num, self.tiling_width_cut_num)
        self.core_nc_start.set_as(
            (_core_idx // (self.tiling_height_cut_num * self.tiling_width_cut_num)) * nc_segment)
        self.core_height_start.set_as(
            ((_core_idx % (self.tiling_height_cut_num * self.tiling_width_cut_num))
             // self.tiling_width_cut_num) * h_segment)
        self.core_width_start.set_as(
            ((_core_idx % (self.tiling_height_cut_num * self.tiling_width_cut_num))
             % self.tiling_width_cut_num) * w_segment)
        self.core_nc_num.set_as(nc_segment)
        self.core_height_num.set_as(h_segment)
        self.core_width_num.set_as(w_segment)

        with self.tik_instance.if_scope(
                self.core_nc_start + self.core_nc_num >= self.tiling_batch * self.tiling_c1):
            self.core_nc_num.set_as(self.tiling_batch * self.tiling_c1 - self.core_nc_start)
        with self.tik_instance.if_scope(
                self.core_height_start + self.core_height_num >= self.cut_height_num):
            self.core_height_num.set_as(self.cut_height_num - self.core_height_start)
        with self.tik_instance.if_scope(
                self.core_width_start + self.core_width_num >= self.cut_width_num):
            self.core_width_num.set_as(self.cut_width_num - self.core_width_start)

        with self.tik_instance.if_scope(self.core_nc_num < 0):
            self.core_nc_num.set_as(0)
        with self.tik_instance.if_scope(self.core_height_num < 0):
            self.core_height_num.set_as(0)
        with self.tik_instance.if_scope(self.core_width_num < 0):
            self.core_width_num.set_as(0)

        core_used = self.tiling_width_cut_num * self.tiling_height_cut_num * self.tiling_bc1_cut_num
        with self.tik_instance.if_scope(_core_idx >= core_used):
            self.core_nc_num.set_as(0)
            self.core_height_num.set_as(0)
            self.core_width_num.set_as(0)

    def _init_ub_tensor_for_idx(self, height_idx_len=0, width_idx_len=0, is_db=True):
        """
        compute the ub size of tensors
        """
        height_idx_len = self.height_idx_segment_num if height_idx_len == 0 else height_idx_len
        width_idx_len = self.width_idx_segment_num if width_idx_len == 0 else width_idx_len
        idx_max_len = max(height_idx_len, width_idx_len)
        self.height_idx = self.tik_instance.Tensor("int32", (height_idx_len,),
                                                      name="height_idx", scope=tik.scope_ubuf)
        self.width_idx = self.tik_instance.Tensor("int32", (width_idx_len,),
                                                     name="width_idx", scope=tik.scope_ubuf)

        self.input_idx = self.tik_instance.Tensor("float32", (idx_max_len,),
                                                  name="input_idx", scope=tik.scope_ubuf)
        avail_bytes = self.ub_size_bytes - (height_idx_len + width_idx_len + idx_max_len) * 4
        # ping-pang
        if self.is_bfp16 is True:
            # two parts as one for bfloat16 convert to float32
            ub_tensor_cnt = 1 + 2
            ub_tensor_cnt = ub_tensor_cnt * 2 if is_db else ub_tensor_cnt
            avail_block = avail_bytes // Constant.BLOCK_BYTES // ub_tensor_cnt
            self.ub_max_num = avail_block * self.block_num // self.b16_block_num * self.b16_block_num
        else:
            ub_tensor_cnt = 1
            ub_tensor_cnt = ub_tensor_cnt * 2 if is_db else ub_tensor_cnt
            avail_block = avail_bytes // Constant.BLOCK_BYTES // ub_tensor_cnt
            self.ub_max_num = avail_block * self.block_num

    def _init_ub_tensor_for_grads(self, mode="all"):
        """
        _init_ub_tensor_for_grads
        """
        if mode in ("all",):
            self.grad_out_ub = self.tik_instance.Tensor(self.grads_dtype, (self.ub_max_num,),
                                                        name="grad_out_ub", scope=tik.scope_ubuf)
            self.grad_in_gm_ping = self.tik_instance.Tensor(self.grads_dtype, (self.ub_max_num,),
                                                            name="grad_in_gm_ping", scope=tik.scope_gm)
        if mode in ("ub",):
            self.grad_out_ub = self.tik_instance.Tensor(self.grads_dtype, (self.ub_max_num,),
                                                        name="grad_out_ub", scope=tik.scope_ubuf)

    def _fill_index(self, idx_ub, idx_num, vector_num=64):
        """
        fill 0,1,2  .... (idx_num -1) in idx_ub
        when the idx_num is less than 16, fill it one by one
        when the type is not int32, will fill in int32 ub and cast to idx_ub dtype
        when the type is int32, will fill in int32 one by one
        """
        # when the idx_num is less than 16, fill it one by one
        vector_num_ub = self.tik_instance.Tensor(idx_ub.dtype, (vector_num,),
                                                 name="vector_num_ub", scope=tik.scope_ubuf)
        for _idx in range(vector_num // 8):
            idx_ub[_idx].set_as(_idx)

        self.tik_instance.vector_dup(vector_num, vector_num_ub, vector_num // 8, 1, 1, 8)

        with self.tik_instance.for_range(1, 8) as add_idx:
            add_offset = add_idx * vector_num // 8
            self.tik_instance.vadd(vector_num // 8, idx_ub[add_offset:], vector_num_ub,
                                   idx_ub[add_offset - (vector_num // 8):],
                                   1, 1, 1, 1, 8, 0, 8)

        self.tik_instance.vector_dup(vector_num, vector_num_ub, vector_num, 1, 1, 8)

        idx_vector_num = ResizeNearestNeighborV2Grad._get_ceil_int(idx_num, vector_num)
        with self.tik_instance.for_range(1, idx_vector_num) as add_idx:
            add_offset = add_idx * vector_num
            self.tik_instance.vadd(vector_num, idx_ub[add_offset:], vector_num_ub, idx_ub[add_offset - vector_num:],
                                   1, 1, 1, 1, 8, 0, 8)

    def _workspace_process(self, _core_idx, c0_len):
        """
        workspace process for bfloat16
        """
        total_elems = (self.tiling_batch * self.tiling_c1 * self.tiling_out_height * self.tiling_out_width * c0_len)
        total_lp_cnt = util_tik_comm_func.ceil_div(total_elems, self.ub_max_num)
        lp_per_core = util_tik_comm_func.ceil_div(total_lp_cnt, self.tiling_core_num)
        act_core_num = util_tik_comm_func.ceil_div(total_lp_cnt, lp_per_core)
        lp_last_core = self.tik_instance.Scalar("int32", "lp_last_core")
        with self.tik_instance.if_scope(total_lp_cnt % act_core_num == 0):
            lp_last_core.set_as(lp_per_core)
        with self.tik_instance.else_scope():
            lp_last_core.set_as(total_lp_cnt - lp_per_core * (act_core_num - 1))
        data_last_lp = self.tik_instance.Scalar("int32", "data_last_lp")
        with self.tik_instance.if_scope(total_elems % self.ub_max_num == 0):
            data_last_lp.set_as(self.ub_max_num)
        with self.tik_instance.else_scope():
            data_last_lp.set_as(total_elems % self.ub_max_num)

        def _workspace_to_gm(lp_cnt, data_len, offset=0):
            with self.tik_instance.for_range(0, lp_cnt) as lp_idx:
                self.tik_instance.data_move_pad(
                    out_ub_fp32, self.out_workspace[(_core_idx * lp_per_core + offset + lp_idx) * self.ub_max_num], 1,
                    data_len * Constant.B32_BYTES, 0, 0)
                util_tik_comm_func.tik_func_vconv(self.tik_instance, out_ub_bfp16, out_ub_fp32, self.ub_max_num,
                                                  "round")
                out_gm = self.out_gm.reinterpret_cast_to("float16")
                out_ub = out_ub_bfp16.reinterpret_cast_to("float16")
                self.tik_instance.data_move_pad(out_gm[(_core_idx * lp_per_core + offset + lp_idx) * self.ub_max_num],
                                                out_ub, 1, data_len * Constant.B16_BYTES, 0, 0)

        with self.tik_instance.new_stmt_scope():
            out_ub_fp32 = self.tik_instance.Tensor("float32", (self.ub_max_num, ), tik.scope_ubuf, "out_ub_fp32")
            out_ub_bfp16 = self.tik_instance.Tensor(self.grads_dtype, (self.ub_max_num, ), tik.scope_ubuf,
                                                    "out_ub_bfp16")

            with self.tik_instance.if_scope(_core_idx < act_core_num - 1):
                _workspace_to_gm(lp_per_core, self.ub_max_num)
            with self.tik_instance.elif_scope(_core_idx == act_core_num - 1):
                _workspace_to_gm(lp_last_core - 1, self.ub_max_num)
                offset = lp_last_core - 1
                util_tik_comm_func.tik_func_vector(self.tik_instance, out_ub_fp32, 0.0, self.ub_max_num)
                _workspace_to_gm(1, data_last_lp, offset)

    def _set_width_segment(self, w_loop_segment, is_w_align, is_big_to_small):
        if is_big_to_small:
            if is_w_align:
                with self.tik_instance.new_stmt_scope():
                    align_num_scalar = self.tik_instance.Scalar("int32", name="align_num_scalar")
                    align_num_scalar.set_as(self.tiling_in_width // self.tiling_out_width)
                    w_loop_segment.set_as(w_loop_segment // align_num_scalar * align_num_scalar)
                    with self.tik_instance.if_scope(w_loop_segment == Constant.W_LOOP_SEGMENT_MAX):
                        w_loop_segment.set_as(w_loop_segment - align_num_scalar)
            else:
                w_loop_segment.set_as(127)
        else:
            if is_w_align:
                # if width is input_w resize to n*input_w, one segment must be n align
                # exp: 24 resize to 48, one segment of width must be 2*n
                with self.tik_instance.new_stmt_scope():
                    align_num_scalar = self.tik_instance.Scalar("int32", name="align_num_scalar")
                    align_num_scalar.set_as(self.tiling_out_width // self.tiling_in_width)
                    w_loop_segment.set_as(w_loop_segment // align_num_scalar * align_num_scalar)

    def _copy_per_height_data_in(self, is_src_stride, nc_max_segment, nc_num, nc_idx, h_gm_offset, w_gm_offset,
                                 w_len, block_num, output_ub):
        with self.tik_instance.if_scope(is_src_stride == 0):
            with self.tik_instance.new_stmt_scope(disable_sync=True):
                with self.tik_instance.for_range(0, nc_num) as _segment_idx:
                    data_move_ubuf_offset = (w_len * self.grads_shape_c0) * _segment_idx
                    nc_gm_input_offset = ((nc_idx * nc_max_segment + self.core_nc_start + _segment_idx) *
                                          self.tiling_in_width * self.tiling_in_height)
                    data_move_gm_offset = (nc_gm_input_offset + h_gm_offset * self.tiling_in_width + w_gm_offset)
                    self.tik_instance.data_move(output_ub[data_move_ubuf_offset],
                                                self.grads_gm[data_move_gm_offset * self.grads_shape_c0], 0,
                                                1, w_len * self.grads_shape_c0 // block_num, 0, 0)
        with self.tik_instance.else_scope():
            data_move_ubuf_offset = 0
            nc_gm_input_offset = ((nc_idx * nc_max_segment + self.core_nc_start) *
                                  self.tiling_in_width * self.tiling_in_height)
            data_move_gm_offset = nc_gm_input_offset + h_gm_offset * self.tiling_in_width + w_gm_offset
            data_move_burst_num = nc_num
            data_move_burst_len = w_len * self.grads_shape_c0 // block_num
            data_move_src_stride = ((self.tiling_in_width * self.tiling_in_height - w_len) *
                                    self.grads_shape_c0 // block_num)
            self.tik_instance.data_move(output_ub[data_move_ubuf_offset],
                                        self.grads_gm[data_move_gm_offset * self.grads_shape_c0], 0,
                                        data_move_burst_num, data_move_burst_len, data_move_src_stride,
                                        0)
            
    def _copy_big_to_small_not_w_align_data_out(self, is_dst_stride, nc_max_segment, nc_num, nc_idx, w_len, in_h_idx,
                                                output_ub):
        out_gm = self.out_workspace if self.is_bfp16 is True else self.out_gm
        burst_len = self.grads_shape_c0 // self.block_num
        scalar_out_w_idx = self.tik_instance.Scalar("int32", name="scalar_out_w_idx", init_value=0)

        with self.tik_instance.if_scope(w_len == 1):
            scalar_in_w_idx = self.tik_instance.Scalar("int32", name="scalar_in_w_idx")
            scalar_in_w_idx.set_as(self.width_idx[0])
            with self.tik_instance.if_scope(is_dst_stride == 0):
                with self.tik_instance.for_range(0, nc_num) as _segment_idx:
                    burst_num = 1
                    nc_ubuf_offset = w_len * _segment_idx
                    nc_gm_offset = ((nc_idx * nc_max_segment + self.core_nc_start + _segment_idx) *
                                    self.tiling_out_width * self.tiling_out_height)
                    output_gm_offset = (nc_gm_offset + in_h_idx * self.tiling_out_width + scalar_in_w_idx)
                    self.tik_instance.data_move(out_gm[output_gm_offset * self.grads_shape_c0],
                                                output_ub[nc_ubuf_offset * self.grads_shape_c0], 0,
                                                burst_num, burst_len, 0, 0)
            with self.tik_instance.else_scope():
                nc_gm_offset = ((nc_idx * nc_max_segment + self.core_nc_start) *
                                self.tiling_out_width * self.tiling_out_height)
                output_gm_offset = (nc_gm_offset + in_h_idx * self.tiling_out_width + scalar_in_w_idx)
                burst_num = nc_num
                ubuf_burst_stride = 0
                gm_out_burst_stride = (self.tiling_out_width * self.tiling_out_height -
                                        1) * self.grads_shape_c0 // self.block_num
                self.tik_instance.data_move(out_gm[output_gm_offset * self.grads_shape_c0],
                                            output_ub[0], 0, burst_num, burst_len,
                                            ubuf_burst_stride, gm_out_burst_stride)
        with self.tik_instance.else_scope():
            with self.tik_instance.for_range(0, w_len - 1) as w_idx:
                scalar_in_w_idx = self.tik_instance.Scalar("int32", name="scalar_in_w_idx")
                scalar_in_w_idx_next = self.tik_instance.Scalar("int32", name="scalar_in_w_idx_next")
                scalar_in_w_idx.set_as(self.width_idx[w_idx])
                scalar_in_w_idx_next.set_as(self.width_idx[w_idx + 1])

                with self.tik_instance.if_scope(scalar_in_w_idx == scalar_in_w_idx_next):
                    rep_stride = w_len * self.grads_shape_c0 // self.block_num
                    self.tik_instance.vadd(self.grads_shape_c0,
                                           output_ub[scalar_out_w_idx * self.grads_shape_c0],
                                           output_ub[scalar_out_w_idx * self.grads_shape_c0],
                                           output_ub[(w_idx + 1) * self.grads_shape_c0], nc_num,
                                           1, 1, 1, rep_stride, rep_stride, rep_stride)
                    with self.tik_instance.if_scope(w_idx == w_len - 2):
                        with self.tik_instance.if_scope(is_dst_stride == 0):
                            with self.tik_instance.for_range(0, nc_num) as _segment_idx:
                                burst_num = 1
                                nc_ubuf_offset = w_len * _segment_idx + scalar_out_w_idx
                                nc_gm_offset = ((nc_idx * nc_max_segment + self.core_nc_start + _segment_idx) *
                                                self.tiling_out_width * self.tiling_out_height)
                                output_gm_offset = (nc_gm_offset + in_h_idx * self.tiling_out_width + scalar_in_w_idx)
                                self.tik_instance.data_move(out_gm[output_gm_offset * self.grads_shape_c0],
                                                            output_ub[nc_ubuf_offset * self.grads_shape_c0], 0,
                                                            burst_num, burst_len, 0, 0)
                        with self.tik_instance.else_scope():
                            nc_gm_offset = ((nc_idx * nc_max_segment + self.core_nc_start) *
                                            self.tiling_out_width * self.tiling_out_height)
                            output_gm_offset = (nc_gm_offset + in_h_idx * self.tiling_out_width + scalar_in_w_idx)
                            burst_num = nc_num
                            ubuf_burst_stride = ((w_len - 1) * self.grads_shape_c0 // self.block_num)
                            gm_out_burst_stride = ((self.tiling_out_width * self.tiling_out_height - 1) *
                                                   self.grads_shape_c0 // self.block_num)
                            self.tik_instance.data_move(out_gm[output_gm_offset * self.grads_shape_c0],
                                                        output_ub[scalar_out_w_idx * self.grads_shape_c0], 0, burst_num,
                                                        burst_len, ubuf_burst_stride, gm_out_burst_stride)
                with self.tik_instance.else_scope():
                    with self.tik_instance.if_scope(is_dst_stride == 0):
                        with self.tik_instance.for_range(0, nc_num) as _segment_idx:
                            burst_num = 1
                            nc_ubuf_offset = w_len * _segment_idx + scalar_out_w_idx
                            nc_gm_offset = ((nc_idx * nc_max_segment + self.core_nc_start + _segment_idx) *
                                            self.tiling_out_width * self.tiling_out_height)
                            output_gm_offset = (nc_gm_offset + in_h_idx * self.tiling_out_width + scalar_in_w_idx)

                            self.tik_instance.data_move(out_gm[output_gm_offset * self.grads_shape_c0],
                                                        output_ub[nc_ubuf_offset * self.grads_shape_c0], 0, burst_num,
                                                        burst_len, 0, 0)
                        scalar_out_w_idx.set_as(w_idx + 1)
                        with self.tik_instance.if_scope(w_idx == w_len - 2):
                            with self.tik_instance.for_range(0, nc_num) as _segment_idx:
                                burst_num = 1
                                nc_ubuf_offset = w_len * _segment_idx + scalar_out_w_idx
                                nc_gm_offset = ((nc_idx * nc_max_segment + self.core_nc_start + _segment_idx) *
                                                self.tiling_out_width * self.tiling_out_height)
                                output_gm_offset = (nc_gm_offset + in_h_idx * self.tiling_out_width +
                                                    scalar_in_w_idx_next)
                                self.tik_instance.data_move(out_gm[output_gm_offset * self.grads_shape_c0],
                                                            output_ub[nc_ubuf_offset * self.grads_shape_c0], 0,
                                                            burst_num, burst_len, 0, 0)
                    with self.tik_instance.else_scope():
                        nc_gm_offset = ((nc_idx * nc_max_segment + self.core_nc_start) *
                                        self.tiling_out_width * self.tiling_out_height)
                        output_gm_offset = (nc_gm_offset + in_h_idx * self.tiling_out_width + scalar_in_w_idx)
                        burst_num = nc_num
                        ubuf_burst_stride = (w_len - 1) * self.grads_shape_c0 // self.block_num
                        gm_out_burst_stride = ((self.tiling_out_width * self.tiling_out_height - 1) *
                                               self.grads_shape_c0 // self.block_num)
                        self.tik_instance.data_move(out_gm[output_gm_offset * self.grads_shape_c0],
                                                    output_ub[scalar_out_w_idx * self.grads_shape_c0], 0, burst_num,
                                                    burst_len, ubuf_burst_stride, gm_out_burst_stride)
                        scalar_out_w_idx.set_as(w_idx + 1)
                        with self.tik_instance.if_scope(w_idx == w_len - 2):
                            output_gm_offset = (nc_gm_offset + in_h_idx * self.tiling_out_width + scalar_in_w_idx_next)
                            self.tik_instance.data_move(out_gm[output_gm_offset * self.grads_shape_c0],
                                                        output_ub[scalar_out_w_idx * self.grads_shape_c0], 0, burst_num,
                                                        burst_len, ubuf_burst_stride, gm_out_burst_stride)


    def _copy_big_to_small_w_align_data_out(self, is_dst_stride, nc_max_segment, nc_num, nc_idx, w_len, in_h_idx,
                                            w_gm_offset, output_ub):
        out_gm = self.out_workspace if self.is_bfp16 is True else self.out_gm 
        burst_len = self.grads_shape_c0 // self.block_num
        w_align_num = self.tiling_in_width // self.tiling_out_width
        repeat_times = w_len // w_align_num
        with self.tik_instance.for_range(0, repeat_times) as repeat_idx:
            # The range of rep_stride is [0, 255] in vadd instruction
            output_ub_offset = repeat_idx * w_align_num
            rep_stride = w_len * self.grads_shape_c0 // self.block_num
            with self.tik_instance.for_range(0, w_align_num - 1) as w_align_idx:
                self.tik_instance.vadd(self.grads_shape_c0, output_ub[output_ub_offset * self.grads_shape_c0],
                                       output_ub[output_ub_offset * self.grads_shape_c0],
                                       output_ub[(output_ub_offset + w_align_idx + 1) * self.grads_shape_c0],
                                       nc_num, 1, 1, 1, rep_stride, rep_stride, rep_stride)

            with self.tik_instance.if_scope(is_dst_stride == 0):
                with self.tik_instance.for_range(0, nc_num) as _segment_idx:
                    nc_ubuf_offset = w_len * _segment_idx + repeat_idx * w_align_num
                    nc_gm_offset = ((nc_idx * nc_max_segment + self.core_nc_start + _segment_idx) *
                                    self.tiling_out_width * self.tiling_out_height)
                    output_gm_offset = (nc_gm_offset + in_h_idx * self.tiling_out_width +
                                        w_gm_offset // w_align_num + repeat_idx)
                    burst_num = 1
                    self.tik_instance.data_move(out_gm[output_gm_offset * self.grads_shape_c0],
                                                output_ub[nc_ubuf_offset * self.grads_shape_c0], 0,
                                                burst_num, burst_len, 0, 0)
            with self.tik_instance.else_scope():
                nc_ubuf_offset = repeat_idx * w_align_num
                nc_gm_offset = ((nc_idx * nc_max_segment + self.core_nc_start) *
                                self.tiling_out_width * self.tiling_out_height)
                output_gm_offset = (nc_gm_offset + in_h_idx * self.tiling_out_width +
                                    w_gm_offset // w_align_num + repeat_idx)
                burst_num = nc_num
                ubuf_burst_stride = (w_len - 1) * self.grads_shape_c0 // self.block_num
                gm_out_burst_stride = ((self.tiling_out_width * self.tiling_out_height - 1) *
                                       self.grads_shape_c0 // self.block_num)
                self.tik_instance.data_move(out_gm[output_gm_offset * self.grads_shape_c0],
                                            output_ub[nc_ubuf_offset * self.grads_shape_c0], 0,
                                            burst_num, burst_len, ubuf_burst_stride,
                                            gm_out_burst_stride)
                
    def _copy_not_big_to_small_not_w_align_data_out(self, is_dst_stride, nc_max_segment, nc_num, nc_idx, w_len,
                                                    in_h_idx, output_ub):
        out_gm = self.out_workspace if self.is_bfp16 is True else self.out_gm
        scalar_in_w_idx = self.tik_instance.Scalar("int32", name="scalar_in_w_idx")
        with self.tik_instance.if_scope(is_dst_stride == 0):
            with self.tik_instance.for_range(0, w_len) as w_idx:
                scalar_in_w_idx.set_as(self.width_idx[w_idx])
                burst_num = 1
                burst_len = self.grads_shape_c0 // self.block_num
                with self.tik_instance.for_range(0, nc_num) as _segment_idx:
                    nc_ubuf_offset = w_len * _segment_idx + w_idx
                    nc_gm_offset = ((nc_idx * nc_max_segment + self.core_nc_start + _segment_idx) *
                                    self.tiling_out_width * self.tiling_out_height)
                    output_gm_offset = (nc_gm_offset + in_h_idx * self.tiling_out_width + scalar_in_w_idx)
                    self.tik_instance.data_move(out_gm[output_gm_offset * self.grads_shape_c0],
                                                output_ub[nc_ubuf_offset * self.grads_shape_c0], 0,
                                                burst_num, burst_len, 0, 0)
        with self.tik_instance.else_scope():
            with self.tik_instance.for_range(0, w_len) as w_idx:
                scalar_in_w_idx.set_as(self.width_idx[w_idx])
                nc_ubuf_offset = w_len * self.grads_shape_c0
                burst_num = nc_num
                burst_len = self.grads_shape_c0 // self.block_num
                ubuf_burst_stride = nc_ubuf_offset // self.block_num - burst_len
                gm_out_burst_stride = (
                    self.tiling_out_width * self.tiling_out_height * self.grads_shape_c0 // self.block_num - burst_len)
                nc_gm_offset = ((nc_idx * nc_max_segment + self.core_nc_start) *
                                self.tiling_out_width * self.tiling_out_height)
                output_gm_offset = (nc_gm_offset + in_h_idx * self.tiling_out_width + scalar_in_w_idx)
                self.tik_instance.data_move(out_gm[output_gm_offset * self.grads_shape_c0],
                                            output_ub[w_idx * self.grads_shape_c0], 0, burst_num,
                                            burst_len, ubuf_burst_stride, gm_out_burst_stride)
                
    def _copy_not_big_to_small_w_align_data_out(self, is_dst_stride, nc_max_segment, nc_num, nc_idx, w_len,
                                                in_h_idx, w_gm_offset, output_ub):
        out_gm = self.out_workspace if self.is_bfp16 is True else self.out_gm
        w_align_num = self.tiling_out_width // self.tiling_in_width

        with self.tik_instance.if_scope(w_align_num == 1):
            with self.tik_instance.if_scope(is_dst_stride == 0):
                with self.tik_instance.for_range(0, nc_num) as _segment_idx:
                    burst_num = 1
                    burst_len = w_len * self.grads_shape_c0 // self.block_num
                    nc_ubuf_offset = w_len * _segment_idx
                    nc_gm_offset = ((nc_idx * nc_max_segment + self.core_nc_start + _segment_idx) *
                                    self.tiling_out_width * self.tiling_out_height)
                    output_gm_offset = (nc_gm_offset + in_h_idx * self.tiling_out_width + w_gm_offset)
                    self.tik_instance.data_move(out_gm[output_gm_offset * self.grads_shape_c0],
                                                output_ub[nc_ubuf_offset * self.grads_shape_c0], 0,
                                                burst_num, burst_len, 0, 0)
            with self.tik_instance.else_scope():
                burst_num = nc_num
                burst_len = w_len * self.grads_shape_c0 // self.block_num
                nc_ubuf_offset = 0
                nc_gm_offset = ((nc_idx * nc_max_segment + self.core_nc_start) *
                                self.tiling_out_width * self.tiling_out_height)
                output_gm_offset = (nc_gm_offset + in_h_idx * self.tiling_out_width + w_gm_offset)
                ubuf_burst_stride = 0
                gm_out_burst_stride = ((self.tiling_out_width * self.tiling_out_height - w_len) *
                                       self.grads_shape_c0 // self.block_num)
                self.tik_instance.data_move(out_gm[output_gm_offset * self.grads_shape_c0],
                                            output_ub[nc_ubuf_offset * self.grads_shape_c0], 0,
                                            burst_num, burst_len, ubuf_burst_stride,
                                            gm_out_burst_stride)
        with self.tik_instance.else_scope():
            with self.tik_instance.if_scope(is_dst_stride == 0):
                with self.tik_instance.for_range(0, w_len) as w_input_idx:
                    with self.tik_instance.for_range(0, nc_num) as _segment_idx:
                        burst_num = 1
                        burst_len = self.grads_shape_c0 // self.block_num
                        nc_ubuf_offset = w_len * _segment_idx + w_input_idx
                        nc_gm_offset = ((nc_idx * nc_max_segment + self.core_nc_start + _segment_idx) *
                                        self.tiling_out_width * self.tiling_out_height)
                        output_gm_offset = (nc_gm_offset + in_h_idx * self.tiling_out_width +
                                            (self.core_width_start + w_input_idx) * w_align_num)
                        self.tik_instance.data_move(out_gm[output_gm_offset * self.grads_shape_c0],
                                                    output_ub[nc_ubuf_offset * self.grads_shape_c0],
                                                    0, burst_num, burst_len, 0, 0)
            with self.tik_instance.else_scope():
                with self.tik_instance.for_range(0, w_len) as w_input_idx:
                    burst_num = nc_num
                    burst_len = self.grads_shape_c0 // self.block_num
                    nc_ubuf_offset = w_input_idx
                    ubuf_burst_stride = w_len * self.grads_shape_c0 // self.block_num - burst_len
                    gm_out_burst_stride = (self.tiling_out_width * self.tiling_out_height *
                                           self.grads_shape_c0 // self.block_num - burst_len)

                    nc_gm_offset = ((nc_idx * nc_max_segment + self.core_nc_start) *
                                    self.tiling_out_width * self.tiling_out_height)
                    output_gm_offset = (nc_gm_offset + in_h_idx * self.tiling_out_width +
                                        (self.core_width_start + w_input_idx) * w_align_num)
                    self.tik_instance.data_move(out_gm[output_gm_offset * self.grads_shape_c0],
                                                output_ub[nc_ubuf_offset * self.grads_shape_c0], 0,
                                                burst_num, burst_len, ubuf_burst_stride,
                                                gm_out_burst_stride)
        
    def _function_default(self, _core_idx, is_src_stride_copy=False, is_dst_stride_copy=False,
                          is_w_align=False, is_big_to_small=False):
        """
        _function_default, run this
        """
        self.tik_instance.set_atomic_add(1)

        self.height_idx_segment_num = 64
        self.width_idx_segment_num = 128
        # cut by output h and output w
        self._init_ub_tensor_for_idx()

        # gen 0-511 to ub fp32
        with self.tik_instance.new_stmt_scope():
            self._fill_index(self.input_idx, self.width_idx_segment_num)

        # calcu is_src_stride_copy and is_dst_stride_copy use scalar
        scalar_is_src_stride = self.tik_instance.Scalar("int32", name="scalar_is_src_stride", init_value=1)
        scalar_is_dst_stride = self.tik_instance.Scalar("int32", name="scalar_is_dst_stride", init_value=1)

        with self.tik_instance.if_scope(self.tiling_in_height * self.tiling_in_width > self.stride_threshold):
            scalar_is_src_stride.set_as(0)
        with self.tik_instance.if_scope(self.tiling_out_height * self.tiling_out_width > self.stride_threshold):
            scalar_is_dst_stride.set_as(0)

        # init a scalar for w segment one time
        w_loop_segment = self.tik_instance.Scalar("int32", name="w_loop_segment", init_value=self.width_idx_segment_num)
        self._set_width_segment(w_loop_segment, is_w_align, is_big_to_small)

        with self.tik_instance.if_scope(tik.all(self.core_width_num < w_loop_segment, self.core_width_num > 0)):
            w_loop_segment.set_as(self.core_width_num)

        w_loop_num = self.core_width_num // w_loop_segment
        w_tail_num = self.core_width_num % w_loop_segment
        nc_max_segment = self.tik_instance.Scalar("int64", name="nc_max_segment")
        nc_max_segment.set_as(self.ub_max_num // (w_loop_segment * self.grads_shape_c0))
        if is_big_to_small:
            self.tik_instance.scalar_min(nc_max_segment, nc_max_segment, Constant.VADD_MAX_REPEAT_TIMES)
        nc_loop = self.core_nc_num // nc_max_segment
        nc_tail = self.core_nc_num % nc_max_segment

        scalar_idx_fp32 = self.tik_instance.Scalar("float32", name="scalar_idx_fp32")
        # vconv start idx from int32 scalar to fp32 scalar
        self.scalar_vconv_int32_to_fp32(self.core_width_start, scalar_idx_fp32)
        # do vadds 0,1,2,3,4 + fp32_scalar
        self.tik_instance.vadds(64, self.input_idx, self.input_idx, scalar_idx_fp32,
                                (w_loop_segment + 63) // 64, 1, 1, 8, 8)
        self.scalar_vconv_int32_to_fp32(w_loop_segment, scalar_idx_fp32)

        def _run_w_loop_default(w_loop_idx, w_do_len, h_loop_offset, h_do_len):
            w_gm_offset = w_loop_idx * w_loop_segment + self.core_width_start
            self.calc_out_idx(self.resize_scale_w, self.width_idx, self.input_idx, self.width_idx_segment_num)
            width_repeat = ResizeNearestNeighborV2Grad._get_ceil_int(self.width_idx_segment_num, 64)
            self.tik_instance.vadds(64, self.input_idx, self.input_idx, scalar_idx_fp32, width_repeat, 1, 1, 8, 8)

            # one segment h and one segment w
            def _do_single_nc(do_nc_num, _nc_loop_idx):
                def _do_one_height(h_idx, output_ub, mid_ub):
                    h_gm_offset = h_idx + h_loop_offset
                    scalar_in_h_idx = self.tik_instance.Scalar("int32", name="scalar_in_h_idx")
                    height_idx = self.tik_instance.Tensor("int32", (64,), name="height_idx", scope=tik.scope_ubuf)
                    height_idx_ub_fp32 = self.tik_instance.Tensor("float32", (64,),
                                                                  name="height_idx_ub_fp32", scope=tik.scope_ubuf)
                    self.tik_instance.vector_dup(64, height_idx_ub_fp32, 0, 1, 1, 8)
                    util_tik_comm_func.tik_func_vector(self.tik_instance, height_idx,
                                                       h_idx + self.core_height_start, 64)
                    util_tik_comm_func.tik_func_vconv(self.tik_instance, height_idx_ub_fp32, height_idx, 64)

                    self.calc_out_idx(self.resize_scale_h, height_idx, height_idx_ub_fp32, 1)
                    scalar_in_h_idx.set_as(height_idx[0])

                    if self.is_bfp16:
                        self._copy_per_height_data_in(scalar_is_src_stride, nc_max_segment, do_nc_num, _nc_loop_idx,
                                                      h_gm_offset, w_gm_offset, w_do_len, self.b16_block_num, mid_ub)
                    else:
                        self._copy_per_height_data_in(scalar_is_src_stride, nc_max_segment, do_nc_num, _nc_loop_idx,
                                                      h_gm_offset, w_gm_offset, w_do_len, self.block_num, output_ub)

                    if self.is_bfp16 is True:
                        util_tik_comm_func.tik_func_vconv(self.tik_instance, output_ub, mid_ub, self.ub_max_num)

                    # When Win > Wout, set is_big_to_small as "True", default set it as "False"
                    if is_big_to_small:
                        # When the result of Win/Wout is integer and less than 120, set is_w_align as "True", default
                        # set it as "False"
                        if not is_w_align:
                            self._copy_big_to_small_not_w_align_data_out(scalar_is_dst_stride, nc_max_segment,
                                                                         do_nc_num, _nc_loop_idx, w_do_len,
                                                                         scalar_in_h_idx, output_ub)
                        else:
                            self._copy_big_to_small_w_align_data_out(scalar_is_dst_stride, nc_max_segment,
                                                                     do_nc_num, _nc_loop_idx, w_do_len,
                                                                     scalar_in_h_idx, w_gm_offset, output_ub)
                    else:
                        # When the result of Wout/Win is integer, set is_w_align as "True", default set it as "False"
                        if not is_w_align:
                            self._copy_not_big_to_small_not_w_align_data_out(scalar_is_dst_stride, nc_max_segment,
                                                                             do_nc_num, _nc_loop_idx, w_do_len,
                                                                             scalar_in_h_idx, output_ub)
                        else:
                            self._copy_not_big_to_small_w_align_data_out(scalar_is_dst_stride, nc_max_segment,
                                                                         do_nc_num, _nc_loop_idx, w_do_len,
                                                                         scalar_in_h_idx, w_gm_offset, output_ub)

                if self.is_bfp16 is False:
                    self.grad_out_ub_ping = self.tik_instance.Tensor(self.grads_dtype, (self.ub_max_num, ),
                                                                     name="grad_out_ub_ping",
                                                                     scope=tik.scope_ubuf)
                    self.grad_out_ub_pang = self.tik_instance.Tensor(self.grads_dtype, (self.ub_max_num, ),
                                                                     name="grad_out_ub_pang",
                                                                     scope=tik.scope_ubuf)
                else:
                    self.grad_out_ub_ping = self.tik_instance.Tensor("float32", (self.ub_max_num, ),
                                                                     name="grad_out_ub_ping",
                                                                     scope=tik.scope_ubuf)
                    self.grad_out_ub_pang = self.tik_instance.Tensor("float32", (self.ub_max_num, ),
                                                                     name="grad_out_ub_pang",
                                                                     scope=tik.scope_ubuf)
                    self.mid_ub_ping = self.tik_instance.Tensor(self.grads_dtype, (self.ub_max_num, ),
                                                                name="mid_ub_ping",
                                                                scope=tik.scope_ubuf)
                    self.mid_ub_pang = self.tik_instance.Tensor(self.grads_dtype, (self.ub_max_num, ),
                                                                name="mid_ub_pang",
                                                                scope=tik.scope_ubuf)
                    # to avoid overflow when data convert
                    ping_fp16 = self.mid_ub_ping.reinterpret_cast_to("float16")
                    pang_fp16 = self.mid_ub_pang.reinterpret_cast_to("float16")
                    util_tik_comm_func.tik_func_vector(self.tik_instance, ping_fp16, 0.0, self.ub_max_num)
                    util_tik_comm_func.tik_func_vector(self.tik_instance, pang_fp16, 0.0, self.ub_max_num)

                with self.tik_instance.for_range(0, h_do_len // 2) as _h_idx:
                    _do_one_height(_h_idx * 2, self.grad_out_ub_ping, self.mid_ub_ping)
                    _do_one_height(_h_idx * 2 + 1, self.grad_out_ub_pang, self.mid_ub_pang)
                with self.tik_instance.if_scope(h_do_len % 2 == 1):
                    _do_one_height(h_do_len - 1, self.grad_out_ub_ping, self.mid_ub_ping)

            with self.tik_instance.for_range(0, nc_loop) as nc_loop_idx:
                _do_single_nc(nc_max_segment, nc_loop_idx)
            with self.tik_instance.if_scope(nc_tail != 0):
                _do_single_nc(nc_tail, nc_loop)

        def _run_h_loop_default(h_loop_idx, h_do_len):
            h_loop_segment_start = h_loop_idx * self.height_idx_segment_num + self.core_height_start
            h_gm_offset = h_loop_segment_start
            # calcu h idx

            with self.tik_instance.for_range(0, w_loop_num) as w_loop_idx:
                _run_w_loop_default(w_loop_idx, w_loop_segment, h_gm_offset, h_do_len)
            with self.tik_instance.if_scope(w_tail_num != 0):
                _run_w_loop_default(w_loop_num, w_tail_num, h_gm_offset, h_do_len)

        _run_h_loop_default(0, self.core_height_num)

        self.tik_instance.set_atomic_add(0)

        if self.is_bfp16 is True:
            with self.tik_instance.if_scope(self.tiling_core_num > 1):
                self.tik_instance.block_barrier(self.sync_workspace)
            self._workspace_process(_core_idx, self.grads_shape_c0)

    def _function_hw_to_nhnw_resize(self):
        """
        _function_hw_to_nhnw_resize, when `tiling key = 111000, run this`
        """
        # h boardcast base input_h cut
        size_h_n = self.tiling_out_height // self.tiling_in_height
        size_w_n = self.tiling_out_width // self.tiling_in_width
        output_w_size = self.core_width_num * size_w_n
        w_output_size_one_line = self.tik_instance.Scalar("int64", name="input_w_size", init_value=0)
        w_output_size_one_line.set_as(output_w_size)

        block_num = self.block_num
        if self.is_bfp16 is True:
            block_num = self.b16_block_num

        with self.tik_instance.if_scope(
                tik.all(self.ub_max_num < output_w_size * self.grads_shape_c0,
                        self.core_width_num > 0)):
            w_output_size_one_line.set_as((self.ub_max_num // self.grads_shape_c0 // size_w_n) * size_w_n)
        with self.tik_instance.if_scope(w_output_size_one_line == 0):
            w_output_size_one_line.set_as((self.ub_max_num // self.grads_shape_c0 // size_w_n) * size_w_n)
        _w_loop_num = self.core_width_num // (w_output_size_one_line // size_w_n)
        _w_tail_num = self.core_width_num % (w_output_size_one_line // size_w_n)
        _segment_h_num = self.ub_max_num // self.grads_shape_c0 // w_output_size_one_line
        _h_loop_num = self.core_height_num // _segment_h_num
        _h_tail_num = self.core_height_num % _segment_h_num

        def _run_h_loop(h_loop_idx, h_do_len, w_start_offset, w_do_len, nc_idx):
            h_segment_start = h_loop_idx * _segment_h_num + self.core_height_start
            nc_segment_start = nc_idx + self.core_nc_start
            self._init_ub_tensor_for_grads("ub")

            # copy h * w input to ub
            data_move_gm_offset = (nc_segment_start * self.tiling_in_height * self.tiling_in_width +
                                   h_segment_start * self.tiling_in_width + w_start_offset)
            data_move_burst_num = h_do_len
            data_move_burst_len = w_do_len * self.grads_shape_c0 // block_num
            data_move_src_stride = (self.tiling_in_width - w_do_len) * self.grads_shape_c0 // block_num
            data_move_dst_stride = 0
            self.tik_instance.data_move(self.grad_out_ub, self.grads_gm[data_move_gm_offset * self.grads_shape_c0], 0,
                                        data_move_burst_num, data_move_burst_len, data_move_src_stride,
                                        data_move_dst_stride)

            with self.tik_instance.for_range(0, h_do_len) as _h_idx:
                # copy output according to h one by one
                data_move_src_offset = _h_idx * w_do_len * self.grads_shape_c0
                data_move_dst_offset = (nc_segment_start * self.tiling_out_height * self.tiling_out_width +
                                        h_segment_start * size_h_n * self.tiling_out_width + w_start_offset * size_w_n +
                                        _h_idx * size_h_n * self.tiling_out_width)
                data_move_burst_num = w_do_len
                data_move_burst_len = self.grads_shape_c0 // block_num
                data_move_src_stride = 0
                data_move_dst_stride = (size_w_n - 1) * self.grads_shape_c0 // block_num
                self.tik_instance.data_move(self.out_gm[data_move_dst_offset * self.grads_shape_c0:],
                                            self.grad_out_ub[data_move_src_offset:], 0, data_move_burst_num,
                                            data_move_burst_len, data_move_src_stride, data_move_dst_stride)

        def _run_w_loop(w_loop_idx, input_w_len):
            w_segment_start = w_loop_idx * (w_output_size_one_line // size_w_n) + self.core_width_start
            with self.tik_instance.for_range(0, self.core_nc_num) as nc_idx:
                with self.tik_instance.for_range(0, _h_loop_num, thread_num=2) as _h_loop_idx:
                    _run_h_loop(_h_loop_idx, _segment_h_num, w_segment_start, input_w_len, nc_idx)
                with self.tik_instance.if_scope(_h_tail_num != 0):
                    _run_h_loop(_h_loop_num, _h_tail_num, w_segment_start, input_w_len, nc_idx)

        with self.tik_instance.for_range(0, _w_loop_num) as _w_loop_idx:
            _run_w_loop(_w_loop_idx, w_output_size_one_line // size_w_n)
        with self.tik_instance.if_scope(_w_tail_num != 0):
            _run_w_loop(_w_loop_num, _w_tail_num)

    def _do_resize_base_tiling_key(self, _core_idx):
        """
        tiling braches
        """
        # calcu scale for h and w
        self.calculate_scale()

        # tiling_key format: 000000
        # 1. Reserved, default 1
        # 2. h align flag, 0: `h -> x.x*h, 1: h -> nh, 2: nh -> h, 3: h = h`
        # 3. w align flag, 0: `w -> x.x*w, 1: w -> nw, 2: nw -> w, 3: w = w`
        # 4. src stride flag, 0: can not copy with stride 1: can copy with stride
        # 5. des stride flag, 0: can not copy with stride 1: can copy with stride
        # 6. Reserved, default 0

        with self.tik_instance.if_scope(self.tiling_key == 100000):
            with self.tik_instance.new_stmt_scope():
                self._function_default(_core_idx, is_src_stride_copy=False, is_dst_stride_copy=False, is_w_align=False)
        with self.tik_instance.if_scope(self.tiling_key == 101000):
            with self.tik_instance.new_stmt_scope():
                self._function_default(_core_idx, is_src_stride_copy=False, is_dst_stride_copy=False, is_w_align=True)
        with self.tik_instance.if_scope(self.tiling_key == 100010):
            with self.tik_instance.new_stmt_scope():
                self._function_default(_core_idx,
                                       is_src_stride_copy=False,
                                       is_dst_stride_copy=False,
                                       is_w_align=False,
                                       is_big_to_small=True)
        with self.tik_instance.if_scope(self.tiling_key == 100110):
            with self.tik_instance.new_stmt_scope():
                self._function_default(_core_idx,
                                       is_src_stride_copy=False,
                                       is_dst_stride_copy=False,
                                       is_w_align=True,
                                       is_big_to_small=True)
        with self.tik_instance.if_scope(self.tiling_key == 111000):
            # tiling_key is 111000, mean: h,w resize to nh, mw
            with self.tik_instance.new_stmt_scope():
                self._function_hw_to_nhnw_resize()

    def _do_data_move_base(self, dst_param, src_param, repeat_param, move_dir):
        dst, dst_gap = dst_param
        src, src_gap = src_param
        repeat_times, step_len = repeat_param
        dtype_size = get_bit_len(src.dtype) // 8
        block_num = self.ub_block_size // dtype_size

        if self.is_move_pad_support:
            if move_dir == 0:
                self.tik_instance.data_move_pad(dst, src, repeat_times, step_len * dtype_size,
                                                (dst_gap - step_len) // block_num, (src_gap - step_len) * dtype_size)
            else:
                self.tik_instance.data_move_pad(dst, src, repeat_times, step_len * dtype_size,
                                                (dst_gap - step_len) * dtype_size, (src_gap - step_len) // block_num)
        else:
            nburst_len = ResizeNearestNeighborV2Grad._get_ceil_int(step_len, block_num)
            with self.tik_instance.if_scope(tik.all(dst_gap % block_num == 0, src_gap % block_num == 0)):
                self.tik_instance.data_move(dst, src, 0, repeat_times, nburst_len,
                                            (src_gap - step_len) // block_num,
                                            (dst_gap - step_len) // block_num)
            with self.tik_instance.else_scope():
                with self.tik_instance.for_range(0, repeat_times) as idx:
                    self.tik_instance.data_move(dst[idx * dst_gap], src[idx * src_gap], 0, 1, nburst_len, 0, 0)

    def _do_data_move_turbo(self, param, repeat_param, max_num, move_dir):
        '''
        dst, src
        dst_gap, src_gap, the second first - the first tail element num
        '''
        dst, dst_gap, src, src_gap = param
        repeat_times, step_len = repeat_param
        src_dtype = src.dtype
        dst_dtype = dst.dtype
        if src_dtype != dst_dtype:
            mid_ub = self.tik_instance.Tensor(self.grads_dtype, (max_num,), name="mid_ub", scope=tik.scope_ubuf)
            mid_ub_cast = mid_ub.reinterpret_cast_to("float16")
            util_tik_comm_func.tik_func_vector(self.tik_instance, mid_ub_cast, 0.0, max_num)
            if move_dir == 0:
                self._do_data_move_base([mid_ub, dst_gap], [src, src_gap], repeat_param, move_dir)
                util_tik_comm_func.tik_func_vconv(self.tik_instance, dst, mid_ub, max_num)
            elif move_dir == 1:
                util_tik_comm_func.tik_func_vconv(self.tik_instance, mid_ub, src, max_num, mode="round")
                self._do_data_move_base([dst, dst_gap], [mid_ub, src_gap], repeat_param, move_dir)
        else:
            self._do_data_move_base([dst, dst_gap], [src, src_gap], repeat_param, move_dir)

    def _do_nchw_pure(self):
        ori_block_num = self.ub_block_size // (get_bit_len(self.grads_dtype) // 8)
        max_num = self.ub_size_bytes // self.ub_block_size * ori_block_num
        self.grad_out_ub = self.tik_instance.Tensor(self.grads_dtype, (max_num,),
                                                    name="grad_out_ub", scope=tik.scope_ubuf)
        loop_times = self.core_nc_num // max_num
        loop_tail = self.core_nc_num % max_num

        def run_one_nc(idx, segment):
            gm_offset_in = self.core_nc_start + idx * max_num
            gm_offset_out = self.core_nc_start + idx * max_num
            self._do_data_move_base([self.grad_out_ub, segment], [self.grads_gm[gm_offset_in], segment],
                                    [1, segment], 0)
            self._do_data_move_base([self.out_gm[gm_offset_out], segment], [self.grad_out_ub, segment], [1, segment], 1)

        with self.tik_instance.for_range(0, loop_times) as loop_idx:
            run_one_nc(loop_idx, max_num)
        with self.tik_instance.if_scope(loop_tail != 0):
            run_one_nc(loop_times, loop_tail)

    def _do_nchw_one_2_n(self):
        output_size = self.tiling_out_width * self.tiling_out_height
        block_num = self.b16_block_num if self.is_bfp16 else self.block_num
        vector_num = block_num * block_num
        vec_repeat = vector_num // self.vector_num
        """
        |valid data(1Block)|Invalid data(7Block)|valid data(1Block)|Invalid data(7Block)|····|
        |----------------------------------max num-------------------------------------------|
        """
        self._init_ub_tensor_for_idx(vector_num, vector_num)
        valid_data_num = self.ub_max_num // block_num // block_num * block_num
        self.grad_in_ub = self.tik_instance.Tensor(self.grads_new_dtype, (self.ub_max_num, ),
                                                    name="grad_in_ub", scope=tik.scope_ubuf)
        self.grad_out_ub = self.tik_instance.Tensor(self.grads_new_dtype, (self.ub_max_num, ),
                                                    name="grad_out_ub", scope=tik.scope_ubuf)
        util_tik_comm_func.tik_func_vector(self.tik_instance, self.grad_in_ub, 0.0, self.ub_max_num)

        def _caculate_output_idx(out_height, out_width):
            '''
            this scenario input_idx is only 0
            '''
            util_tik_comm_func.tik_func_vector(self.tik_instance, self.input_idx, 0.0, vector_num)
            self.calc_out_idx(self.resize_scale_w, self.width_idx, self.input_idx, 64)
            self.calc_out_idx(self.resize_scale_h, self.height_idx, self.input_idx, 64)
            out_height.set_as(self.height_idx[0])
            out_width.set_as(self.width_idx[0])
        
        def _run_nc_loop_small_hw(idx, segment):
            gm_in_offset = self.core_nc_start + idx * valid_data_num
            gm_out_offset = (self.core_nc_start + idx * valid_data_num) * self.tiling_out_width * self.tiling_out_height
            repeat_times = ResizeNearestNeighborV2Grad._get_ceil_int(segment, block_num)
            self._do_data_move_turbo([self.grad_in_ub, vector_num, self.grads_gm[gm_in_offset], block_num],
                                     [repeat_times, block_num], self.ub_max_num, 0)
            with self.tik_instance.for_range(0, repeat_times) as repeat_idx:
                offset = self.tik_instance.Scalar("uint32", name="offset")
                offset.set_as(repeat_idx * vector_num)
                self.tik_instance.vgather(64, self.grad_out_ub[offset], self.grad_in_ub[offset], self.input_offset,
                                          vec_repeat, 8)
            self._do_data_move_turbo(
                [self.out_gm[gm_out_offset], block_num * self.tiling_out_height * self.tiling_out_width,
                 self.grad_out_ub, vector_num],
                [repeat_times, block_num * self.tiling_out_height * self.tiling_out_width], self.ub_max_num, 1)
            
        def _run_nc_loop_large_hw(idx, segment, out_height, out_width):
            gm_in_offset = self.core_nc_start + idx * valid_data_num
            gm_out_offset = ((self.core_nc_start + idx * valid_data_num) *
                             self.tiling_out_width * self.tiling_out_height + out_height * self.tiling_out_width +
                             out_width)
            repeat_times = ResizeNearestNeighborV2Grad._get_ceil_int(segment, block_num)
            self._do_data_move_turbo([self.grad_in_ub, block_num * block_num, self.grads_gm[gm_in_offset], block_num],
                                     [repeat_times, block_num], self.ub_max_num, 0)
            with self.tik_instance.for_range(0, repeat_times) as repeat_idx:
                offset = self.tik_instance.Scalar("uint32", name="offset")
                offset.set_as(repeat_idx * block_num * block_num)
                self.tik_instance.vgather(64, self.grad_out_ub[offset], self.grad_in_ub[offset], self.input_offset,
                                          vec_repeat, 8)
            self._do_data_move_turbo([self.out_gm[gm_out_offset], self.tiling_out_height * self.tiling_out_width,
                                      self.grad_out_ub, block_num],
                                     [segment, block_num], self.ub_max_num, 1)

        #gen offset
        out_height = self.tik_instance.Scalar("int32", name="out_height")
        out_width = self.tik_instance.Scalar("int32", name="out_width")
        _caculate_output_idx(out_height, out_width)
        self.input_offset = self.tik_instance.Tensor("int32", (vector_num,), name="input_offset", scope=tik.scope_ubuf)
        with self.tik_instance.new_stmt_scope():
            self._fill_index(self.input_idx, vector_num)
            self.tik_instance.vmuls(64, self.input_idx, self.input_idx, self.grad_new_type_size, vec_repeat, 1, 1, 8, 8)
            util_tik_comm_func.tik_func_vconv(self.tik_instance, self.input_offset, self.input_idx,
                                              vector_num, mode="round")
        
        with self.tik_instance.if_scope(output_size < block_num):
            #move in
            out_offset = self.tik_instance.Scalar("int32", name="out_offset")
            self.tik_instance.vector_dup(block_num, self.input_offset, block_num * self.grad_new_type_size, 1, 1, 8)
            for i in range(block_num):
                out_offset.set_as(i * output_size + out_height * self.tiling_out_width + out_width)
                self.input_offset[out_offset] = i * self.grad_new_type_size

            loop_times = self.core_nc_num // valid_data_num
            loop_tail = self.core_nc_num % valid_data_num
            with self.tik_instance.for_range(0, loop_times) as loop_idx:
                _run_nc_loop_small_hw(loop_idx, valid_data_num)
            with self.tik_instance.if_scope(loop_tail != 0):
                _run_nc_loop_small_hw(loop_times, loop_tail)
        with self.tik_instance.else_scope():
            out_offset = self.tik_instance.Scalar("int32", name="out_offset")
            self.tik_instance.vector_dup(block_num, self.input_offset, block_num * self.grad_new_type_size, 1, 1, 8)
            for i in range(block_num):
                out_offset.set_as(i * block_num)
                self.input_offset[out_offset] = i * self.grad_new_type_size

            loop_times = self.core_nc_num // valid_data_num
            loop_tail = self.core_nc_num % valid_data_num
            with self.tik_instance.for_range(0, loop_times) as loop_idx:
                _run_nc_loop_large_hw(loop_idx, valid_data_num, out_height, out_width)
            with self.tik_instance.if_scope(loop_tail != 0):
                _run_nc_loop_large_hw(loop_times, loop_tail, out_height, out_width)

    def _do_nchw_n_2_one(self, _core_idx):
        self.tik_instance.set_atomic_add(1)
        self._init_ub_tensor_for_idx(16, 16, False)
        reduce_out_num = util_tik_comm_func.ceil_align(self.ub_max_num // self.vector_num, self.vector_num)
        self.grad_in_ub = self.tik_instance.Tensor(self.grads_new_dtype, (self.ub_max_num, ),
                                                    name="grad_in_ub", scope=tik.scope_ubuf)
        self.grad_out_ub = self.tik_instance.Tensor(self.grads_new_dtype, (reduce_out_num, ),
                                                    name="grad_out_ub", scope=tik.scope_ubuf)

        ori_block_num = self.ub_block_size // (get_bit_len(self.grads_dtype) // 8)
        max_nc_num = util_tik_comm_func.floor_align(self.ub_max_num // self.vector_num, ori_block_num)
        nc_step = self.tik_instance.Scalar("int32", name="nc_step", init_value=self.core_nc_num)
        with self.tik_instance.if_scope(nc_step >= max_nc_num):
            nc_step.set_as(max_nc_num)
        with self.tik_instance.if_scope(nc_step == 0):
            nc_step.set_as(ori_block_num)
        with self.tik_instance.if_scope(nc_step > ori_block_num):
            nc_step.set_as(ori_block_num)

        max_height_num = self.ub_max_num // nc_step
        height_step = self.tik_instance.Scalar("int32", name="height_step")
        height_step.set_as(util_tik_comm_func.floor_align(max_height_num, self.vector_num))

        with self.tik_instance.if_scope(height_step > self.core_height_num):
            height_step.set_as(
                ResizeNearestNeighborV2Grad._get_ceil_int(self.core_height_num, self.vector_num) * self.vector_num)
        with self.tik_instance.if_scope(height_step > self.vector_num * self.vector_num):
            height_step.set_as(self.vector_num * self.vector_num)
        with self.tik_instance.if_scope(height_step == 0):
            height_step.set_as(self.vector_num)

        nc_loop_time = self.core_nc_num // nc_step
        nc_loop_tail = self.core_nc_num % nc_step

        def reduce_sum_nc_loop(src_ub, dst_ub, repeat, segment_len):
            segment_loop = segment_len // self.vector_num
            segment_tail = segment_len % self.vector_num
            second_calc_num = self.tik_instance.Scalar("int32", name="second_calc_num", init_value=segment_loop)
            with self.tik_instance.for_range(0, repeat) as repeat_idx:
                with self.tik_instance.if_scope(segment_loop != 0):
                    self.tik_instance.vcadd(self.vector_num, src_ub[repeat_idx * self.vector_num],
                                            src_ub[repeat_idx * height_step], segment_loop, 1, 1, 8)
                with self.tik_instance.if_scope(segment_tail != 0):
                    second_calc_num.set_as(segment_loop + 1)
                    self.tik_instance.vcadd(segment_tail, src_ub[repeat_idx * self.vector_num + segment_loop],
                                        src_ub[repeat_idx * height_step + segment_loop * self.vector_num], 1, 1, 1, 8)
                self.tik_instance.vcadd(second_calc_num, dst_ub, src_ub, repeat, 1, 1, 8)

        def run_one_nc_loop(nc_idx, nc_segment):
            hw_loop_time = self.core_height_num // height_step
            hw_loop_tail = self.core_height_num % height_step
            gm_nc_offset = self.core_nc_start + nc_idx * nc_step

            def run_one_hw_loop(hw_idx, hw_segment):
                util_tik_comm_func.tik_func_vector(self.tik_instance, self.grad_out_ub, 0.0,
                                                   reduce_out_num)
                gm_in_offset = (gm_nc_offset * self.tiling_in_height * self.tiling_in_width +
                                self.core_height_start + hw_idx * height_step)
                gm_out_offset = gm_nc_offset
                self._do_data_move_turbo([self.grad_in_ub, height_step,
                                          self.grads_gm[gm_in_offset], self.tiling_in_height * self.tiling_in_width],
                                          [nc_segment, hw_segment], self.ub_max_num, 0)
                reduce_sum_nc_loop(self.grad_in_ub, self.grad_out_ub, nc_segment, hw_segment)
                out_gm = self.out_workspace if self.is_bfp16 is True else self.out_gm
                self._do_data_move_turbo([out_gm[gm_out_offset], nc_segment, self.grad_out_ub, nc_segment],
                                     [1, nc_segment], self.ub_max_num // self.vector_num, 1)
                
            with self.tik_instance.for_range(0, hw_loop_time) as hw_loop_idx:
                run_one_hw_loop(hw_loop_idx, height_step)
            with self.tik_instance.if_scope(hw_loop_tail != 0):
                run_one_hw_loop(hw_loop_time, hw_loop_tail)

        with self.tik_instance.for_range(0, nc_loop_time) as nc_idx:
            run_one_nc_loop(nc_idx, nc_step)
        with self.tik_instance.if_scope(nc_loop_tail != 0):
            run_one_nc_loop(nc_loop_time, nc_loop_tail)
        self.tik_instance.set_atomic_add(0)

        if self.is_bfp16 is True:
            with self.tik_instance.if_scope(self.tiling_core_num > 1):
                self.tik_instance.block_barrier(self.sync_workspace)
            self._workspace_process(_core_idx, 1)

    def _do_resize_nchw_tiling_key(self, _core_idx):
        with self.tik_instance.if_scope(self.tiling_key == 1):
            self._core_scalar_args_nchw_1(_core_idx)
            self._do_nchw_pure()
        with self.tik_instance.if_scope(self.tiling_key == 2):
            self.calculate_scale()
            self._core_scalar_args_nchw_1(_core_idx)
            self._do_nchw_one_2_n()
        with self.tik_instance.if_scope(self.tiling_key == 3):
            self._core_scalar_args_nchw_3(_core_idx)
            self._do_nchw_n_2_one(_core_idx)


    def _do_resize(self):
        """
        main process of _do_resize
        """
        with self.tik_instance.new_stmt_scope():
            tiling_ub = self.tik_instance.Tensor("int64", (Constant.TILING_ARG_NUM,),
                                                 name="tiling_ub", scope=tik.scope_ubuf)
            self.tik_instance.data_move(tiling_ub, self.tiling_gm, 0, 1,
                                        Constant.TILING_ARG_NUM // Constant.B64_PER_BLOCK, 0, 0)
            self._tiling_args(tiling_ub)

        if self.grads_format == "nc1hwc0":
            with self.tik_instance.for_range(0, self.tiling_core_num, block_num=self.tiling_core_num) as _core_idx:
                self._core_scalar_args_nc1hwc0(_core_idx)
                self._do_resize_base_tiling_key(_core_idx)
        elif self.grads_format == "nchw":
            with self.tik_instance.for_range(0, self.tiling_core_num, block_num=self.tiling_core_num) as _core_idx:
                self._do_resize_nchw_tiling_key(_core_idx)


@register_operator("ResizeNearestNeighborV2Grad")
@para_check.check_op_params(para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_BOOL,
                            para_check.OPTION_ATTR_BOOL,
                            para_check.KERNEL_NAME)
def resize_nearest_neighbor_v2_grad(grads, size, y, align_corners=False, half_pixel_centers=False,
                                    kernel_name="resize_nearest_neighbor_grad"):
    """Resize `grads` to `size` using nearest neighbor interpolation.

    Parameters
    ----------
    grads: dict
        the dict of input, include shape of input_tensor which layout
        only support 5HD and dtype supports 'float32'
    size: list
        the list of input, the height and width of output tensor
        only support 5HD and dtype supports 'int32', 'int64'
    y: dict
        the dict of output, include shape of input_tensor which layout
        only support 5HD and dtype supports 'float32'
    align_corners: bool
        whether align_corners
    half_pixel_centers: bool
        whether half_pixel_centers
    kernel_name: str
        cce kernel name, default value is `resize_nearest_neighbor_grad`

    Returns
    -------
    tik_instance
    """
    grads_dtype = grads.get("dtype").lower()
    size_dtype = size.get("dtype").lower()
    para_check.check_dtype(size_dtype, ("int64", "int32"), param_name="size")
    para_check.check_dtype(grads_dtype, ("float32", "bfloat16"), param_name="grads")
    if align_corners is True and half_pixel_centers is True :
        rule_desc = "if half_pixel_centers is True, align_corners must be False"
        param_value = "%s and %s" % (align_corners, half_pixel_centers)
        error_manager_vector.raise_err_check_params_rules(kernel_name, rule_desc, \
                                                            "align_corners and half_pixel_centers", param_value)

    obj = ResizeNearestNeighborV2Grad(grads, size, y, align_corners, half_pixel_centers,
                                      kernel_name)
    return obj.resize_nearest_neighbor_v2_grad_operator()
