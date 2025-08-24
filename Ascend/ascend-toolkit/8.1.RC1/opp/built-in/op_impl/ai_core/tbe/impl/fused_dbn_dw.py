#!/usr/bin/env python
# -*- coding:utf-8 -*-
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
fused_dbn_dw
"""

from tbe.common.utils import shape_util
from impl.util import util_deconv_comm
from impl.util.platform_adapter import error_manager
from impl.util.platform_adapter import error_manager_cube
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tik


class Constant:
    """
    The class for constant.
    """
    BLOCK_DIM = 32
    BATCH_256 = 256
    BATCH_PER_CORE = 8


class Dbn2Conv2dBackpropFilter:
    """
    class of fused_dbn2_dw
    """
    def __init__(self, tik_instance, fmap_x, grads_input, dbn_x_input, diff_scale_input,
                 diff_offset_input, scale_input, batch_mean_input, batch_variance_input,
                 epsilon_input, dbn_res, dw_res, para_dict):
        self.tik_instance = tik_instance
        self.fmap = fmap_x
        self.grads = grads_input
        self.dbn_x = dbn_x_input
        self.diff_scale = diff_scale_input
        self.diff_offset = diff_offset_input
        self.scale = scale_input
        self.batch_mean = batch_mean_input
        self.batch_variance = batch_variance_input
        self.epsilon = epsilon_input

        self.dbn_res = dbn_res
        self.dw_res = dw_res
        self.block = 16

        self.strides = list(para_dict.get("strides", [1, 1]))
        self.padding = para_dict.get("padding", [0, 0, 0, 0])
        self.dilations = para_dict.get("dilations", [1, 1, 1, 1])
        self.groups = para_dict.get("groups", 1)

        self.kernel_name = para_dict.get("kernel_name", "conv2d_backprop_filter_cce")

        self.fmap_dtype = fmap_x.dtype
        self.grads_dtype = grads_input.dtype
        self.dbn_x_dtype = dbn_x_input.dtype
        self.diff_scale_dtype = diff_scale_input.dtype
        self.diff_offset_dtype = diff_offset_input.dtype
        self.scale_dtype = scale_input.dtype
        self.batch_mean_dtype = batch_mean_input.dtype
        self.batch_variance_dtype = batch_variance_input.dtype

        self.shape_x_5hd = list(self.fmap.shape)
        self.shape_grads_5hd = self.grads.shape
        self.dedy_channel_wise = self.diff_scale.shape

        self.shape_4d_x = para_dict.get("shape_4d_x")
        self.shape_4d_dedy = para_dict.get("shape_4d_dedy")
        self.shape_4d_filters = para_dict.get("shape_4d_filters")

    def dbn2_dw_compute(self):
        """
        fused_compute
        """

        def _choose_mode(mn_context, split_dict, block_idx):
            if split_dict.get("mode") == "common" or split_dict.get("mode") == "3a_1":
                self._outer_loop_mpp_fun_common(mn_context, split_dict, block_idx)
            if split_dict.get("mode") == "big_x":
                self._outer_loop_mpp_fun_big_x(mn_context, split_dict, block_idx)
            if split_dict.get("mode") == "conv1":
                self._outer_loop_mpp_fun_conv1(mn_context, split_dict, block_idx)

        with self.tik_instance.for_range(0, Constant.BLOCK_DIM, block_num=Constant.BLOCK_DIM) as block_idx:
            diff_scale_ub = self.tik_instance.Tensor(self.diff_scale_dtype, self.dedy_channel_wise,
                                                     name="diff_scale_ub", scope=tik.scope_ubuf)

            diff_offset_ub = self.tik_instance.Tensor(self.diff_offset_dtype, self.dedy_channel_wise,
                                                      name="diff_offset_ub", scope=tik.scope_ubuf)
            scale_ub = self.tik_instance.Tensor(self.scale_dtype, self.dedy_channel_wise,
                                                name="scale_ub", scope=tik.scope_ubuf)

            batch_mean_ub = self.tik_instance.Tensor(self.batch_mean_dtype, self.dedy_channel_wise,
                                                     name="batch_mean_ub", scope=tik.scope_ubuf)
            batch_variance_ub = self.tik_instance.Tensor(self.batch_variance_dtype, self.dedy_channel_wise,
                                                         name="batch_variance_ub", scope=tik.scope_ubuf)
            mn_context = {
                "diff_scale_ub": diff_scale_ub,
                "diff_offset_ub": diff_offset_ub,
                "scale_ub": scale_ub,
                "batch_mean_ub": batch_mean_ub,
                "batch_variance_ub": batch_variance_ub,
            }
            self._load_dbn_abc_all(mn_context)
            self._cal_dbn_abc(mn_context)
            split_dict = self._choose_split_dict()
            batch_num = self.shape_4d_dedy[0]
            if batch_num == Constant.BATCH_256:
                if split_dict.get("batch_status") == "inner":
                    self._outer_loop_mpp_fun_big_xy(mn_context, split_dict, block_idx)
                else:
                    with self.tik_instance.for_range(0, Constant.BATCH_PER_CORE) as loop_batch:
                        batch_dim = block_idx * Constant.BATCH_PER_CORE + loop_batch
                        _choose_mode(mn_context, split_dict, batch_dim)
            else:
                _choose_mode(mn_context, split_dict, block_idx)

    def _choose_split_dict(self):
        """
        get split_dict according to shape
        case_dedyh_fmap_h_dedychannel_fmapchannel
        """
        case_56_56_256_64_split_dict = {
            "batch_status": "outer",
            "mode": "common",
            "dedy_wigth": 56,
            "dedy_height": 56,
            "dedy_hw": 3136,
            "cal_hw": 2 * 2 * 56,
            "cal_M": 4 * 16,
            "cal_N": 4 * 16,
            "loop_M": 4,
            "loop_N": 1,
            "loop_K": 14,
            "cub_N": 1
        }

        case_28_28_512_128_split = {
            "batch_status": "outer",
            "mode": "common",
            "dedy_wigth": 28,
            "dedy_height": 28,
            "dedy_hw": 784,
            "cal_hw": 7 * 16,
            "cal_M": 64,
            "cal_N": 64,
            "loop_M": 8,
            "loop_N": 2,
            "loop_K": 7,
            "cub_N": 1
        }

        case_56_56_64_64_split_dict = {
            "batch_status": "outer",
            "mode": "big_x",
            "dedy_wigth": 56,
            "dedy_height": 56,
            "dedy_hw": 3136,
            "cal_hw": 7 * 16,
            "cal_M": 64,
            "cal_N": 64,
            "loop_M": 1,
            "loop_N": 1,
            "loop_K": 28,
            "cub_N": 1
        }

        case_28_56_512_256_split_dict = {
            "batch_status": "outer",
            "mode": "common",
            "dedy_wigth": 28,
            "dedy_height": 28,
            "dedy_hw": 784,
            "cal_hw": 7 * 16,
            "cal_M": 128,
            "cal_N": 64,
            "loop_M": 4,
            "loop_N": 4,
            "loop_K": 7,
            "cub_N": 1
        }

        case_56_56_64_256_split_dict = {
            "batch_status": "outer",
            "mode": "big_x",
            "dedy_wigth": 56,
            "dedy_height": 56,
            "dedy_hw": 3136,
            "cal_hw": 2 * 56,
            "cal_M": 64,
            "cal_N": 128,
            "loop_M": 1,
            "loop_N": 2,
            "loop_K": 28,
            "cub_N": 2
        }

        case_56_56_64_64_3_split_dict = {
            "batch_status": "outer",
            "mode": "big_x",
            "dedy_wigth": 56,
            "dedy_height": 56,
            "dedy_hw": 3136,
            "cal_hw": 7 * 16,
            "cal_M": 64,
            "cal_N": 9 * 16,
            "loop_M": 1,
            "loop_N": 4,
            "loop_K": 28,
            "cub_N": 4,
        }

        case_28_28_128_512_split_dict = {
            "batch_status": "inner",
            "mode": "big_x",
            "dedy_wigth": 28,
            "dedy_height": 28,
            "dedy_hw": 784,
            "cal_hw": 7 * 16,
            "cal_M": 128,
            "cal_N": 128,
            "loop_M": 1,
            "loop_N": 4,
            "loop_K": 7,
            "cub_N": 8
        }

        case_56_56_128_256_split_dict = {
            "batch_status": "outer",
            "mode": "big_x",
            "dedy_wigth": 56,
            "dedy_height": 56,
            "dedy_hw": 784,
            "cal_hw": 7 * 16,
            "cal_M": 128,
            "cal_N": 128,
            "loop_M": 1,
            "loop_N": 2,
            "loop_K": 28,
            "cub_N": 4
        }

        case_112_224_64_3_7_2_split_dict = {
            "batch_status": "outer",
            "mode": "conv1",
            "dedy_wigth": 112,
            "dedy_height": 112,
            "dedy_hw": 12544,
            "cal_hw": 16 * 14,
            "cal_M": 64,
            "cal_N": 16 * 49,
            "loop_M": 1,
            "loop_N": 1,
            "loop_K": 28,
            "loop_K_BL1": 2,
            "cub_N": 7
        }

        case_14_14_1024_256_split_dict = {
            "batch_status": "inner",
            "mode": "big_x",
            "dedy_wigth": 14,
            "dedy_height": 14,
            "dedy_hw": 196,
            "cal_hw": 14 * 14,
            "cal_M": 64,
            "cal_N": 16 * 4,
            "loop_M": 16,
            "loop_N": 4,
            "loop_K": 1,
            "loop_K_BL1": 1,
            "cub_N": 1
        }


        split_dict_all = {
            "case_56_56_256_64": case_56_56_256_64_split_dict,
            "case_28_28_512_128": case_28_28_512_128_split,
            "case_56_56_64_64": case_56_56_64_64_split_dict,
            "case_28_56_512_256": case_28_56_512_256_split_dict,
            "case_56_56_64_256": case_56_56_64_256_split_dict,
            "case_56_56_64_64_3": case_56_56_64_64_3_split_dict,
            "case_28_28_128_512": case_28_28_128_512_split_dict,
            "case_56_56_128_256": case_56_56_128_256_split_dict,
            "case_112_224_64_3_7_2": case_112_224_64_3_7_2_split_dict,
            "case_14_14_1024_256": case_14_14_1024_256_split_dict,
        }
        dedy_c, dedy_h = self.shape_4d_dedy[1:3]
        fmap_c, fmap_h = self.shape_4d_x[1:3]
        filter_h = self.shape_4d_filters[2]

        if self.shape_4d_filters[2] > 1:
            if self.strides[0] > 1:
                case_name = "case_{}_{}_{}_{}_{}_{}".format(dedy_h, fmap_h, dedy_c, fmap_c, filter_h, self.strides[0])
            else:
                case_name = "case_{}_{}_{}_{}_{}".format(dedy_h, fmap_h, dedy_c, fmap_c, filter_h)
        else:
            case_name = "case_{}_{}_{}_{}".format(dedy_h, fmap_h, dedy_c, fmap_c)
        if not split_dict_all.get(case_name):
            args_dict = {"errCode": "E60108", "reason": "dbn2_dw_fusion don't support this case, please check"}
            raise RuntimeError(args_dict, error_manager.get_error_message(args_dict))
        else:
            return split_dict_all.get(case_name)

    def _load_dbn_abc_all(self, mn_context):
        """
        copy all channel_wise tensor from gm to ub.
        """
        diff_scale_ub = mn_context.get("diff_scale_ub")
        diff_offset_ub = mn_context.get("diff_offset_ub")
        scale_ub = mn_context.get("scale_ub")
        batch_mean_ub = mn_context.get("batch_mean_ub")
        batch_variance_ub = mn_context.get("batch_variance_ub")
        one_burst = 8
        burst_len = self.dedy_channel_wise[1] * self.dedy_channel_wise[-1] // one_burst
        self.tik_instance.data_move(diff_scale_ub, self.diff_scale, sid=0, nburst=1,
                                    burst=burst_len, src_stride=0, dst_stride=0)
        self.tik_instance.data_move(diff_offset_ub, self.diff_offset, sid=0, nburst=1,
                                    burst=burst_len, src_stride=0, dst_stride=0)
        self.tik_instance.data_move(scale_ub, self.scale, sid=0, nburst=1,
                                    burst=burst_len, src_stride=0, dst_stride=0)
        self.tik_instance.data_move(batch_mean_ub, self.batch_mean, sid=0, nburst=1,
                                    burst=burst_len, src_stride=0, dst_stride=0)
        self.tik_instance.data_move(batch_variance_ub, self.batch_variance, sid=0,
                                    nburst=1, burst=burst_len, src_stride=0, dst_stride=0)

    def _cal_dbn_abc(self, mn_context):
        """
        channel_wise tensor calculation:
        `A = 1/sqrt(variance + episilion) * int_reduce * diff_scale`
        `B = int_reduce * (diff_offset - 1/sqrt(variance + episilion) * diff_scale * mean)`
        `C = scale/sqrt(variance + episilion)`
        diff_scale: dgamma
        scale: gamma
        diff_offset: dbeta

        scale_ub: C
        batch_mean: B
        diff_scale: A
        """
        diff_scale_ub = mn_context.get("diff_scale_ub")
        diff_offset_ub = mn_context.get("diff_offset_ub")
        scale_ub = mn_context.get("scale_ub")
        batch_mean_ub = mn_context.get("batch_mean_ub")
        batch_variance_ub = mn_context.get("batch_variance_ub")
        mask = 64
        one_repeat = 64
        repeat_times = self.dedy_channel_wise[1] * self.dedy_channel_wise[-1] // one_repeat
        inv_reduce = 1.0 / (self.shape_grads_5hd[0] * self.shape_grads_5hd[2] * self.shape_grads_5hd[3])
        inv_one_ub = self.tik_instance.Tensor(self.scale_dtype, self.dedy_channel_wise,
                                              name="inv_one_ub", scope=tik.scope_ubuf)
        # mask,dst,scalar, repeat_times, dst_blk_stride, dst_rep_stride
        dma_stride = 8
        self.tik_instance.vector_dup(mask, inv_one_ub, 1.0, repeat_times, 1, dma_stride)
        # mask,dst,scalar, repeat_times, dst_blk_stride, str_blk_stride, dst_rep_stride, src_rep_stride
        self.tik_instance.vadds(mask, batch_variance_ub, batch_variance_ub, self.epsilon,
                                repeat_times, 1, 1, dma_stride, dma_stride)
        self.tik_instance.vsqrt(mask, batch_variance_ub, batch_variance_ub, repeat_times,
                                1, 1, dma_stride, dma_stride)
        self.tik_instance.vdiv(mask, batch_variance_ub, inv_one_ub, batch_variance_ub, repeat_times,
                               1, 1, 1, dma_stride, dma_stride, dma_stride)
        self.tik_instance.vmuls(mask, diff_scale_ub, diff_scale_ub, inv_reduce, repeat_times,
                                1, 1, dma_stride, dma_stride)
        self.tik_instance.vmul(mask, diff_scale_ub, diff_scale_ub, batch_variance_ub, repeat_times,
                               1, 1, 1, dma_stride, dma_stride, dma_stride)
        self.tik_instance.vmuls(mask, batch_mean_ub, batch_mean_ub, -1, repeat_times,
                                1, 1, dma_stride, dma_stride)
        self.tik_instance.vmul(mask, batch_mean_ub, diff_scale_ub, batch_mean_ub, repeat_times,
                               1, 1, 1, dma_stride, dma_stride, dma_stride)
        self.tik_instance.vmul(mask, scale_ub, scale_ub, batch_variance_ub, repeat_times,
                               1, 1, 1, dma_stride, dma_stride, dma_stride)
        self.tik_instance.vmuls(mask, diff_offset_ub, diff_offset_ub, inv_reduce, repeat_times,
                                1, 1, dma_stride, dma_stride)
        self.tik_instance.vadd(mask, batch_mean_ub, diff_offset_ub, batch_mean_ub, repeat_times,
                               1, 1, 1, dma_stride, dma_stride, dma_stride)

        self.tik_instance.vmuls(mask, diff_scale_ub, diff_scale_ub, -1, repeat_times,
                                1, 1, dma_stride, dma_stride)
        self.tik_instance.vmuls(mask, batch_mean_ub, batch_mean_ub, -1, repeat_times,
                                1, 1, dma_stride, dma_stride)

    def _align(self, a, b):
        return (a + b - 1) // b * b

    def _cal_dbn_fm_fun(self, loop_m, loop_k, mn_context, split_dict, block_idx):
        sn_loop = split_dict.get("cal_M") // self.block
        cal_hw = split_dict.get("cal_hw")
        mode = split_dict.get("mode")
        dedy_h, dedy_w = self.shape_4d_dedy[2:]
        gm_stride = dedy_h * dedy_w - cal_hw
        dedy_l1_shape = [split_dict.get("cal_M") // self.block,
                         (cal_hw + self.block - 1) // self.block * self.block, self.block]
        dedy_l1 = self.tik_instance.Tensor("float16", dedy_l1_shape, name="dedy_l1", scope=tik.scope_cbuf)
        mn_context["dedy_l1"] = dedy_l1
        one_repeat = 64
        mask = 64
        block_fp32 = 8
        cal_hw_align = self._align(split_dict.get("cal_hw"), block_fp32)
        dbn_dy_shape1 = (1, sn_loop, cal_hw_align, self.block)
        src_stride = cal_hw_align - cal_hw
        pub_dy_temp1 = self.tik_instance.Tensor("float16", dbn_dy_shape1, name="pub_dy_temp1", scope=tik.scope_ubuf)
        with self.tik_instance.for_range(0, sn_loop, thread_num=min(2, sn_loop)) as loop_sn:
            diff_scale_ub = mn_context.get("diff_scale_ub")
            batch_mean_ub = mn_context.get("batch_mean_ub")
            scale_ub = mn_context.get("scale_ub")
            cal_hw = split_dict.get("cal_hw")
            repeat_times = (cal_hw * self.block + one_repeat - 1) // one_repeat

            c_idx = loop_m * sn_loop + loop_sn
            w_idx = loop_k * cal_hw % self.shape_grads_5hd[3]
            h_idx = loop_k * cal_hw // self.shape_grads_5hd[3]
            dbn_dy_shape = (1, 1, (split_dict.get("cal_hw") + block_fp32 - 1) // block_fp32 * block_fp32, self.block)

            pub_dy = self.tik_instance.Tensor("float32", dbn_dy_shape, name="pub_dy", scope=tik.scope_ubuf)
            pub_x = self.tik_instance.Tensor("float32", dbn_dy_shape, name="pub_x", scope=tik.scope_ubuf)
            pub_x_temp = self.tik_instance.Tensor("float16", dbn_dy_shape, name="pub_x_temp", scope=tik.scope_ubuf)
            pub_dy_temp = self.tik_instance.Tensor("float16", dbn_dy_shape, name="pub_dy_temp", scope=tik.scope_ubuf)

            dy_addr_offset = self.grads[block_idx, c_idx, h_idx, w_idx, 0]
            dbn_x_addr_offset = self.dbn_x[block_idx, c_idx, h_idx, w_idx, 0]
            self.tik_instance.data_move(pub_dy_temp, dy_addr_offset, 0, 1, cal_hw, 0, 0)
            self.tik_instance.data_move(pub_x_temp, dbn_x_addr_offset, 0, 1, cal_hw, 0, 0)
            # parameters are (mask round_mode dst src repeat_times, dst_blk_stride,
            # str_blk_stride, dst_rep_stride, src_rep_stride)
            self.tik_instance.vconv(mask, "", pub_x, pub_x_temp, repeat_times, 1, 1, 8, 4)
            self.tik_instance.vconv(mask, "", pub_dy, pub_dy_temp, repeat_times, 1, 1, 8, 4)
            # coef1
            # c0 is 16 need 2 block, the first block is the first vmadd the second block use thw second vmadd
            # one repeat is 8 block, pub_x's k is multiplys of 8
            # parameters are (mask dst src1  src2 repeat_times, dst_blk_stride, str1_blk_stride, str2_blk_stride,
            #  dst_rep_stride, src1_rep_stride, src2_rep_stride)
            self.tik_instance.vmadd(mask, pub_x, diff_scale_ub[0, c_idx, 0, 0, 0],
                                    batch_mean_ub[0, c_idx, 0, 0, 0], (repeat_times + 1) // 2, 2, 0, 0, 16, 0, 0)
            self.tik_instance.vmadd(mask, pub_x[0, 0, 0, 8], diff_scale_ub[0, c_idx, 0, 0, 8],
                                    batch_mean_ub[0, c_idx, 0, 0, 8], (repeat_times + 1) // 2, 2, 0, 0, 16, 0, 0)
            # dy - coef1
            self.tik_instance.vadd(mask, pub_x, pub_x, pub_dy, repeat_times, 1, 1, 1, 8, 8, 8)
            #
            self.tik_instance.vmul(mask, pub_dy, scale_ub[0, c_idx, 0, 0, 0], pub_x,
                                   (repeat_times + 1) // 2, 2, 0, 2, 16, 0, 16)
            self.tik_instance.vmul(mask, pub_dy[0, 0, 0, 8], scale_ub[0, c_idx, 0, 0, 8],
                                   pub_x[0, 0, 0, 8], (repeat_times + 1) // 2, 2, 0, 2, 16, 0, 16)
            self.tik_instance.vconv(mask, "", pub_dy_temp1[0, loop_sn, 0, 0], pub_dy, repeat_times, 1, 1, 4, 8)
            # copy dy to L1
            dbn_dedy_l1 = dedy_l1[loop_sn, 0, 0]
            self.tik_instance.data_move(dbn_dedy_l1, pub_dy_temp1[0, loop_sn, 0, 0], 0, 1, cal_hw, 0, 0)
            if mode != "conv1":
                dedy_l0a0 = mn_context.get("dedy_l0A0")
                self.tik_instance.load2dv1(dedy_l0a0[loop_sn, 0, 0, 0], dbn_dedy_l1, 0,
                                           (cal_hw + self.block - 1) // self.block, 1, 0, True)
            # copy data to dbn_res
        res_dbn = self.dbn_res[block_idx, loop_m * sn_loop, h_idx, w_idx, 0]
        self.tik_instance.data_move(res_dbn, pub_dy_temp1, 0, sn_loop, cal_hw, src_stride, gm_stride)

    def _loopk_mpp(self, loop_k, loop_n, mn_context, split_dict):
        """load3d and mmad"""
        fmap_l1 = mn_context.get("fmap_l1")
        fmap_h, fmap_w = self.shape_x_5hd[2:4]
        filter_h, filter_w = self.shape_4d_filters[2:]
        cal_n = split_dict.get("cal_N")
        cal_hw = split_dict.get("cal_hw")
        cal_m = split_dict.get("cal_M")

        def _update_filter_hw_position():
            with self.tik_instance.if_scope(filter_h * filter_w == 1):
                c1_index.set_as(c1_index + 1)
                feach_filter_h.set_as(0)
                feach_filter_w.set_as(0)
            with self.tik_instance.else_scope():
                feach_filter_w.set_as(feach_filter_w + 1)
                with self.tik_instance.if_scope(feach_filter_w >= filter_w):
                    feach_filter_h.set_as(feach_filter_h + 1)
                    with self.tik_instance.if_scope(feach_filter_h >= filter_h):
                        c1_index.set_as(c1_index + 1)
                        feach_filter_h.set_as(0)
                    feach_filter_w.set_as(0)

        fmap_l0_shape = [(split_dict.get("cal_hw") + self.block - 1) // self.block, cal_n // self.block,
                         self.block, self.block]
        fmap_l0b0 = self.tik_instance.Tensor("float16", fmap_l0_shape, name="fmap_l0B0", scope=tik.scope_cb)
        dw_l0c = mn_context.get("dw_l0C")
        dedy_l0a0 = mn_context.get("dedy_l0A0")

        c1_index = self.tik_instance.Scalar("int32", "c1_index", loop_n * cal_n // self.block // (filter_h * filter_w))
        k_dim = self.tik_instance.Scalar("int32", "k_dim", loop_n * cal_n // self.block % (filter_h * filter_w))
        dw_l0c_addr = dw_l0c[loop_n * cal_n // self.block, 0, 0, 0]
        left_top_h = loop_k * cal_hw // (self.shape_4d_dedy[3]) * self.strides[0] - self.padding[0]
        left_top_w = loop_k * cal_hw % (self.shape_4d_dedy[3]) * self.strides[1] - self.padding[2]
        padu, padd, padl, padr = self.padding
        stride_h, stride_w = self.strides
        feach_filter_h = self.tik_instance.Scalar("int32", "feach_filter_h", k_dim // filter_w)
        feach_filter_w = self.tik_instance.Scalar("int32", "feach_filter_w", k_dim % filter_w)
        fmap_l1 = mn_context.get("fmap_l1")
        with self.tik_instance.for_range(0, cal_n // self.block) as c_i:
            self.tik_instance.load3dv1(fmap_l0b0[0, c_i, 0, 0], fmap_l1, [padl, padr, padu, padd], fmap_h, fmap_w,
                                       c1_index, feach_filter_w, feach_filter_h, left_top_w, left_top_h,
                                       stride_w, stride_h, filter_w, filter_h, 1, 1,
                                       cal_n // self.block, 1, (cal_hw // self.block))
            _update_filter_hw_position()

        with self.tik_instance.if_scope(loop_k == 0):
            self.tik_instance.mmad(dw_l0c_addr, dedy_l0a0, fmap_l0b0, cal_m, cal_hw, cal_n, 0)
        with self.tik_instance.else_scope():
            self.tik_instance.mmad(dw_l0c_addr, dedy_l0a0, fmap_l0b0, cal_m, cal_hw, cal_n, 1)

    def _ub_mpp_fun(self, loop_n, loop_m, split_dict, mn_context, n_loop):
        """
        copy data from cc to gm

        Parameters
        ------------

        loop_n: value of loop_N.

        loop_m: value of loop_M.

        split_dict: dict, split information.

        mn_context: dict.

        n_loop: value of CUB_N.
        """
        cal_m = split_dict.get("cal_M")
        dw_l0c = mn_context.get("dw_l0C")
        dw_cub = mn_context.get("dw_cub")
        mode = split_dict.get("mode")
        if mode in ("common", "conv1"):
            shape_n = self.shape_4d_filters[2] * self.shape_4d_filters[3] * self.shape_x_5hd[1] // n_loop
            dw_res = self.dw_res[loop_n * shape_n, loop_m * cal_m // self.block, 0, 0]
            dw_l0c = dw_l0c[loop_n * shape_n, 0, 0, 0]
        elif mode == "big_x":
            shape_n = self.shape_4d_filters[2] * self.shape_4d_filters[3] * self.shape_x_5hd[1] // n_loop
            dw_res = self.dw_res[loop_n * shape_n, loop_m * cal_m // self.block, 0, 0]
            dw_l0c = dw_l0c[loop_n * shape_n, 0, 0, 0]
        self.tik_instance.data_move(dw_cub, dw_l0c, 0, 1, shape_n * cal_m // self.block, 0, 0)
        self.tik_instance.set_atomic_add(1)
        self.tik_instance.data_move(dw_res, dw_cub, 0, shape_n, 2 * cal_m, 0, 2 * (self.shape_4d_dedy[1] - cal_m))
        self.tik_instance.set_atomic_add(0)

    def _outer_loop_mpp_fun_common(self, mn_context, split_dict, block_idx):
        """common model"""
        m_loop = split_dict.get("loop_M")
        n_loop = split_dict.get("loop_N")
        k_loop = split_dict.get("loop_K")
        cub_n = split_dict.get("cub_N")
        # fmap is full loaded in L1
        if self.strides[0] == 2 and self.shape_4d_filters[2] == 1:
            # stride is 2 kernel is 1, copy gm to L1 can jump in h dim.
            fmap_l1_shape = [1, self.shape_x_5hd[1], self.shape_4d_dedy[2], self.shape_x_5hd[3], self.shape_x_5hd[4]]
            l1_fmap = self.tik_instance.Tensor("float16", fmap_l1_shape, name="fmap_l1", scope=tik.scope_cbuf)
            self.tik_instance.data_move(l1_fmap, self.fmap[block_idx, 0, 0, 0, 0], 0,
                                        self.shape_x_5hd[1] * fmap_l1_shape[2],
                                        self.shape_x_5hd[3], self.shape_x_5hd[3], 0)
            self.strides[0] = 1
            self.shape_x_5hd[2] = self.shape_4d_dedy[2]
        else:
            fmap_l1_shape = [1, self.shape_x_5hd[1], self.shape_x_5hd[2], self.shape_x_5hd[3], self.shape_x_5hd[4]]
            l1_fmap = self.tik_instance.Tensor("float16", fmap_l1_shape, name="fmap_l1", scope=tik.scope_cbuf)
            self.tik_instance.data_move(l1_fmap, self.fmap[block_idx, 0, 0, 0, 0], 0, self.shape_x_5hd[1],
                                        self.shape_x_5hd[2] * self.shape_x_5hd[3], 0, 0)

        mn_context["fmap_l1"] = l1_fmap
        shape_n = self.shape_4d_filters[2] * self.shape_4d_filters[3] * self.shape_x_5hd[1]
        l0c_shape = [shape_n, split_dict.get("cal_M") // self.block, self.block, self.block]
        dw_l0c = self.tik_instance.Tensor("float32", l0c_shape, name="dw_l0C", scope=tik.scope_cc)
        cub_shape = [shape_n // cub_n, split_dict.get("cal_M") // self.block, self.block, self.block]
        dw_cub = self.tik_instance.Tensor("float32", cub_shape, name="dw_cub", scope=tik.scope_ubuf)
        mn_context["dw_l0C"] = dw_l0c
        mn_context["dw_cub"] = dw_cub
        with self.tik_instance.for_range(0, m_loop, thread_num=min(2, m_loop)) as loop_m:
            pipe_k_loop = k_loop
            k_thread_num = 2
            if k_loop % 2 != 0:
                if k_loop == 1:
                    k_thread_num = 1
                else:
                    pipe_k_loop = k_loop - 1
            # if k % 2 is 1, split the cycle of k_dim with double buffer and the last once
            with self.tik_instance.for_range(0, pipe_k_loop, thread_num=k_thread_num) as loop_k:
                # double buffer part
                dedy_l0_shape = [split_dict.get("cal_M") // self.block,
                                 (split_dict.get("cal_hw") + self.block - 1) // self.block, self.block, self.block]
                dedy_l0a0 = self.tik_instance.Tensor("float16", dedy_l0_shape, name="dedy_l0A0", scope=tik.scope_ca)
                mn_context["dedy_l0A0"] = dedy_l0a0
                # vector in ub
                self._cal_dbn_fm_fun(loop_m, loop_k, mn_context, split_dict, block_idx)
                with self.tik_instance.for_range(0, n_loop, thread_num=min(n_loop, 2)) as loop_n:
                    self._loopk_mpp(loop_k, loop_n, mn_context, split_dict)
            # the last k
            if pipe_k_loop != k_loop:
                dedy_l0_shape = [split_dict.get("cal_M") // self.block,
                                 (split_dict.get("cal_hw") + self.block - 1) // self.block, self.block, self.block]
                dedy_l0a0 = self.tik_instance.Tensor("float16", dedy_l0_shape, name="dedy_l0A0", scope=tik.scope_ca)
                mn_context["dedy_l0A0"] = dedy_l0a0
                self._cal_dbn_fm_fun(loop_m, pipe_k_loop, mn_context, split_dict, block_idx)
                with self.tik_instance.for_range(0, n_loop, thread_num=min(n_loop, 2)) as loop_n:
                    self._loopk_mpp(pipe_k_loop, loop_n, mn_context, split_dict)
            with self.tik_instance.for_range(0, cub_n) as loop_n:
                self._ub_mpp_fun(loop_n, loop_m, split_dict, mn_context, cub_n)

    def _loopk_mpp_big_x(self, loop_k, loop_n, mn_context, split_dict, loop_batch=0):
        """load3d and mad for big_x model which fmap is load """
        _, fmap_w = self.shape_x_5hd[2:4]
        filter_h, filter_w = self.shape_4d_filters[2:]
        cal_n = split_dict.get("cal_N")
        cal_hw = split_dict.get("cal_hw")
        cal_m = split_dict.get("cal_M")
        fmap_l0b0 = mn_context.get("fmap_l0B0")
        dw_l0c = mn_context.get("dw_l0C")
        dedy_l0a0 = mn_context.get("dedy_l0A0")
        stride_h, _ = self.strides
        _, dedy_w = self.shape_4d_dedy[2:]
        c_index = loop_n * cal_n // self.block
        k_dim = (cal_hw + self.block - 1) // self.block
        if filter_w == 1:
            fmap_l1 = mn_context.get("fmap_l1")[0, c_index, 0, 0]
            with self.tik_instance.for_range(0, cal_n // self.block, thread_num=1) as c_i:
                self.tik_instance.load3dv1(fmap_l0b0[0, c_i, 0, 0], fmap_l1, self.padding, cal_hw // fmap_w,
                                           fmap_w, c_i, 0, 0, 0, 0, self.strides[1], self.strides[0], filter_w,
                                           filter_h, 1, 1, cal_n // self.block, 1, k_dim)
        else:
            fmap_l1_h = cal_hw // dedy_w * stride_h + filter_h - 1
            padu, padd, padl, padr = self.padding
            pad_u = self.tik_instance.Scalar("int8", "pad_u")
            pad_d = self.tik_instance.Scalar("int8", "pad_d")
            fmap_h = self.tik_instance.Scalar("int32", "fmap_h")
            left_top_h = self.tik_instance.Scalar("int32", "left_top_h")
            fmap_l1 = mn_context.get("fmap_l1")[0, loop_n, 0, 0]
            with self.tik_instance.if_scope(loop_k == 0):
                pad_u.set_as(padu)
                pad_d.set_as(0)
                fmap_h.set_as(fmap_l1_h - padu)
                left_top_h.set_as(-padu)
            with self.tik_instance.else_scope():
                with self.tik_instance.if_scope(loop_k == split_dict.get("loop_K") - 1):
                    pad_u.set_as(0)
                    pad_d.set_as(padd)
                    fmap_h.set_as(fmap_l1_h - padd)
                    left_top_h.set_as(0)
                with self.tik_instance.else_scope():
                    pad_u.set_as(0)
                    pad_d.set_as(0)
                    fmap_h.set_as(fmap_l1_h)
                    left_top_h.set_as(0)
            with self.tik_instance.for_range(0, cal_n // self.block, thread_num=1) as c_i:
                self.tik_instance.load3dv1(fmap_l0b0[0, c_i, 0, 0], fmap_l1, [padl, padr, pad_u, pad_d], fmap_h,
                                           fmap_w, 0, c_i % filter_w, c_i // filter_w, -padl,
                                           left_top_h, self.strides[1], self.strides[0], filter_w, filter_h, 1,
                                           1, cal_n // self.block, 1, (cal_hw // self.block))

        with self.tik_instance.if_scope(loop_k == 0):
            with self.tik_instance.if_scope(loop_batch == 0):
                self.tik_instance.mmad(dw_l0c[c_index, 0, 0, 0], dedy_l0a0, fmap_l0b0, cal_m, cal_hw, cal_n, 0)
            with self.tik_instance.else_scope():
                self.tik_instance.mmad(dw_l0c[c_index, 0, 0, 0], dedy_l0a0, fmap_l0b0, cal_m, cal_hw, cal_n, 1)
        with self.tik_instance.else_scope():
            self.tik_instance.mmad(dw_l0c[c_index, 0, 0, 0], dedy_l0a0, fmap_l0b0, cal_m, cal_hw, cal_n, 1)

    def _outer_loop_mpp_fun_big_x(self, mn_context, split_dict, block_idx):
        cal_hw = split_dict.get("cal_hw")
        m_loop = split_dict.get("loop_M")
        n_loop = split_dict.get("loop_N")
        k_loop = split_dict.get("loop_K")
        stride_h, stride_w = self.strides
        _, dedy_w = self.shape_4d_dedy[2:]
        fmap_h, fmap_w = self.shape_4d_x[2:]
        kernel_h = self.shape_4d_filters[2]
        cal_n = split_dict.get("cal_N")
        cub_n = split_dict.get("cub_N")
        fmap_l0_shape = [(split_dict.get("cal_hw") + self.block - 1) // self.block,
                         cal_n // self.block, self.block, self.block]
        if k_loop % 2 != 0:
            fmap_l0b0 = self.tik_instance.Tensor("float16", fmap_l0_shape, name="fmap_l0B0", scope=tik.scope_cb)
            mn_context["fmap_l0B0"] = fmap_l0b0
        shape_n = self.shape_4d_filters[2] * self.shape_4d_filters[3] * self.shape_x_5hd[1]
        l0c_shape = [shape_n, split_dict.get("cal_M") // self.block, self.block, self.block]
        cub_shape = [shape_n // cub_n, split_dict.get("cal_M") // self.block, self.block, self.block]
        dw_l0c = self.tik_instance.Tensor("float32", l0c_shape, name="dw_l0C", scope=tik.scope_cc)
        dw_cub = self.tik_instance.Tensor("float32", cub_shape, name="dw_cub", scope=tik.scope_ubuf)
        mn_context["dw_l0C"] = dw_l0c
        mn_context["dw_cub"] = dw_cub
        with self.tik_instance.for_range(0, m_loop, thread_num=min(2, m_loop)) as loop_m:
            pipe_k_loop = k_loop
            k_thread_num = 2
            if k_loop % 2 != 0:
                if k_loop == 1:
                    k_thread_num = 1
                else:
                    pipe_k_loop = k_loop - 1
            with self.tik_instance.for_range(0, pipe_k_loop, thread_num=k_thread_num) as loop_k:
                dedy_l0_shape = [split_dict.get("cal_M") // self.block,
                                 (split_dict.get("cal_hw") + self.block - 1) // self.block, self.block, self.block]
                dedy_l0a0 = self.tik_instance.Tensor("float16", dedy_l0_shape, name="dedy_l0A0", scope=tik.scope_ca)
                mn_context["dedy_l0A0"] = dedy_l0a0
                self._cal_dbn_fm_fun(loop_m, loop_k, mn_context, split_dict, block_idx)
                fmap_l1_shape = fmap_l1_shape = [1, self.shape_x_5hd[1], (cal_hw // fmap_w + kernel_h - 1) * fmap_w,
                                                 self.shape_x_5hd[4]]
                if self.shape_4d_filters[2] == 3:
                    l1_fmap = self.tik_instance.Tensor("float16", fmap_l1_shape, name="fmap_l1", scope=tik.scope_cbuf)
                    padu, padd, _, _ = self.padding
                    with self.tik_instance.if_scope(loop_k == 0):
                        burst = fmap_l1_shape[2] - padu * fmap_w
                        self.tik_instance.data_move(l1_fmap, self.fmap[block_idx, 0, 0, 0, 0], 0, fmap_l1_shape[1],
                                                    burst, fmap_w * fmap_h - burst, padu * fmap_w)
                    with self.tik_instance.else_scope():
                        fmap_gm = self.fmap[block_idx, 0, 2 * loop_k - padu, 0, 0]
                        with self.tik_instance.if_scope(loop_k == k_loop - 1):
                            burst = fmap_l1_shape[2] - padd * fmap_w
                            self.tik_instance.data_move(l1_fmap, fmap_gm, 0, fmap_l1_shape[1], burst,
                                                        fmap_w * fmap_h - burst, padd * fmap_w)
                        with self.tik_instance.else_scope():
                            burst = fmap_l1_shape[2]
                            self.tik_instance.data_move(l1_fmap, fmap_gm, 0, fmap_l1_shape[1], burst,
                                                        fmap_w * fmap_h - burst, 0)
                else:
                    l1_fmap = self.tik_instance.Tensor("float16", fmap_l1_shape, name="fmap_l1", scope=tik.scope_cbuf)
                    fmap_gm = self.fmap[block_idx, 0,
                                        loop_k * stride_h * (cal_hw // dedy_w) + kernel_h - stride_h, 0, 0]
                    self.tik_instance.data_move(l1_fmap, fmap_gm, 0, self.shape_x_5hd[1],
                                                fmap_l1_shape[2], fmap_h * fmap_w - fmap_l1_shape[2], 0)
                mn_context["fmap_l1"] = l1_fmap

                with self.tik_instance.for_range(0, n_loop, thread_num=min(n_loop, 2)) as loop_n:
                    if pipe_k_loop == k_loop:
                        fmap_l0b0 = self.tik_instance.Tensor("float16", fmap_l0_shape, name="fmap_l0B0",
                                                             scope=tik.scope_cb)
                        mn_context["fmap_l0B0"] = fmap_l0b0
                    self._loopk_mpp_big_x(loop_k, loop_n, mn_context, split_dict)

            if pipe_k_loop != k_loop:
                dedy_l0_shape = [split_dict.get("cal_M") // self.block,
                                 (split_dict.get("cal_hw") + self.block - 1) // self.block, self.block, self.block]
                dedy_l0a0 = self.tik_instance.Tensor("float16", dedy_l0_shape, name="dedy_l0A0", scope=tik.scope_ca)
                mn_context["dedy_l0A0"] = dedy_l0a0
                self._cal_dbn_fm_fun(loop_m, pipe_k_loop, mn_context, split_dict, block_idx)
                fmap_l1_shape = [1, self.shape_x_5hd[1], cal_hw * stride_h * stride_w, self.shape_x_5hd[4]]
                l1_fmap = self.tik_instance.Tensor("float16", fmap_l1_shape, name="fmap_l1", scope=tik.scope_cbuf)
                mn_context["fmap_l1"] = l1_fmap
                fmap_gm = self.fmap[block_idx, 0, pipe_k_loop * stride_h * cal_hw // dedy_w, 0, 0]
                self.tik_instance.data_move(l1_fmap, fmap_gm, 0, self.shape_x_5hd[1],
                                            fmap_l1_shape[2], fmap_h * fmap_w - fmap_l1_shape[2], 0)

                with self.tik_instance.for_range(0, n_loop, thread_num=min(n_loop, 2)) as loop_n:
                    self._loopk_mpp_big_x(pipe_k_loop, loop_n, mn_context, split_dict)
            with self.tik_instance.for_range(0, cub_n) as loop_n:
                self._ub_mpp_fun(loop_n, loop_m, split_dict, mn_context, cub_n)

    def _loopk_mpp_conv1(self, loop_k1, loop_k, loop_k2, mn_context, split_dict):
        fmap_w = self.shape_x_5hd[3]
        filter_h, filter_w = self.shape_4d_filters[2:]
        cal_n = split_dict.get("cal_N")
        cal_m = split_dict.get("cal_M")
        k_loop = split_dict.get("loop_K")
        loop_k_bl1 = split_dict.get("loop_K_BL1")

        dw_l0c = mn_context.get("dw_l0C")
        dedy_l0a0 = mn_context.get("dedy_l0A0")
        fmap_l0_shape = [1, cal_n // self.block, self.block, self.block]
        fmap_l0b0 = self.tik_instance.Tensor("float16", fmap_l0_shape, name="fmap_l0B0", scope=tik.scope_cb)

        fmap_l1 = mn_context.get("fmap_l1")
        padu, padd, padl, padr = self.padding
        stride_h, _ = self.strides
        pad_u = self.tik_instance.Scalar("int8", "pad_u")
        pad_d = self.tik_instance.Scalar("int8", "pad_d")
        fmap_h = self.tik_instance.Scalar("int32", "fmap_h")
        left_top_h = self.tik_instance.Scalar("int32", "left_top_h")
        with self.tik_instance.if_scope(loop_k1 == 0):
            pad_u.set_as(padu)
            pad_d.set_as(0)
            fmap_h.set_as(2 * k_loop * stride_h + filter_h - stride_h - padu)
            left_top_h.set_as((2 * loop_k + loop_k2 // filter_w) * stride_h - padu)
        with self.tik_instance.else_scope():
            with self.tik_instance.if_scope(loop_k1 == loop_k_bl1 - 1):
                pad_u.set_as(0)
                pad_d.set_as(padd)
                fmap_h.set_as(2 * k_loop * stride_h + filter_h - stride_h - padd)
                left_top_h.set_as((2 * loop_k + loop_k2 // filter_w) * stride_h)
            with self.tik_instance.else_scope():
                pad_u.set_as(0)
                pad_d.set_as(0)
                fmap_h.set_as(2 * k_loop * stride_h + filter_h - stride_h)
                left_top_h.set_as((2 * loop_k + loop_k2 // filter_w) * stride_h)

        left_top_w = self.tik_instance.Scalar("int32", "left_top_w",
                                              loop_k2 % filter_w * self.strides[1] * self.block - padl)

        self.tik_instance.load3dv1(fmap_l0b0[0, 0, 0, 0], fmap_l1, [padl, padr, pad_u, pad_d], fmap_h, fmap_w, 0,
                                   0, 0, left_top_w, left_top_h, self.strides[1],
                                   self.strides[0], filter_w, filter_h, 1, 1, 1, 0, cal_n // self.block)

        dw_l0c_addr = dw_l0c[0, 0, 0, 0]
        cal_k = 16
        with self.tik_instance.if_scope(loop_k1 + loop_k + loop_k2 == 0):
            self.tik_instance.mmad(dw_l0c_addr, dedy_l0a0, fmap_l0b0, cal_m, cal_k, cal_n, 0)
        with self.tik_instance.else_scope():
            self.tik_instance.mmad(dw_l0c_addr, dedy_l0a0, fmap_l0b0, cal_m, cal_k, cal_n, 1)

    def _outer_loop_mpp_fun_conv1(self, mn_context, split_dict, block_idx):

        k_loop = split_dict.get("loop_K")
        loop_k_bl1 = split_dict.get("loop_K_BL1")
        stride_h, _ = self.strides
        kernel_h = self.shape_4d_filters[2]
        _, fmap_w = self.shape_4d_x[2:]
        padu, padd, _, _ = self.padding
        cub_n = split_dict.get("cub_N")
        cal_hw = split_dict.get("cal_hw")
        cal_m = split_dict.get("cal_M")
        fmap_l1_shape = [1, self.shape_x_5hd[1], (stride_h * 2 * k_loop + kernel_h - stride_h) * fmap_w,
                         self.shape_x_5hd[4]]
        l1_fmap = self.tik_instance.Tensor("float16", fmap_l1_shape, name="fmap_L1", scope=tik.scope_cbuf)
        shape_n = self.shape_4d_filters[2] * self.shape_4d_filters[3] * self.shape_x_5hd[1]
        l0c_shape = [shape_n, split_dict.get("cal_M") // self.block, self.block, self.block]
        cub_shape = [shape_n // cub_n, split_dict.get("cal_M") // self.block, self.block, self.block]
        dw_l0c = self.tik_instance.Tensor("float32", l0c_shape, name="dw_l0C", scope=tik.scope_cc)
        dw_cub = self.tik_instance.Tensor("float32", cub_shape, name="dw_cub", scope=tik.scope_ubuf)
        mn_context["dw_l0C"] = dw_l0c
        mn_context["dw_cub"] = dw_cub
        mn_context["fmap_l1"] = l1_fmap
        with self.tik_instance.for_range(0, loop_k_bl1) as loop_k1:
            with self.tik_instance.if_scope(loop_k1 == 0):
                self.tik_instance.data_move(l1_fmap, self.fmap[block_idx, 0, 0, 0, 0], 0, 1,
                                            fmap_l1_shape[2] - padu * fmap_w, 0, 0)
            with self.tik_instance.else_scope():
                fmap_gm = self.fmap[block_idx, 0, stride_h * loop_k1 * k_loop * 2 - padu, 0, 0]
                with self.tik_instance.if_scope(loop_k1 == loop_k_bl1 - 1):
                    self.tik_instance.data_move(l1_fmap, fmap_gm, 0, 1, fmap_l1_shape[2] - padd * fmap_w, 0, 0)
                with self.tik_instance.else_scope():
                    self.tik_instance.data_move(l1_fmap, fmap_gm, 0, 1, fmap_l1_shape[2], 0, 0)
            with self.tik_instance.for_range(0, k_loop, thread_num=1) as loop_k:
                self._cal_dbn_fm_fun(0, loop_k1 * k_loop + loop_k, mn_context, split_dict, block_idx)
                dedy_l1 = mn_context.get("dedy_l1")
                k2_loop = cal_hw // self.block
                with self.tik_instance.for_range(0, k2_loop, thread_num=2) as loop_k2:
                    dedy_l0_shape = [split_dict.get("cal_M") // self.block, 1, self.block, self.block]
                    dedy_l0a0 = self.tik_instance.Tensor("float16", dedy_l0_shape, name="dedy_l0A0", scope=tik.scope_ca)
                    mn_context["dedy_l0A0"] = dedy_l0a0
                    self.tik_instance.load2dv1(dedy_l0a0, dedy_l1[0, loop_k2 * self.block, 0],
                                               0, cal_m // self.block, k2_loop, 0, True)
                    self._loopk_mpp_conv1(loop_k1, loop_k, loop_k2, mn_context, split_dict)
        with self.tik_instance.for_range(0, cub_n) as loop_n:
            self._ub_mpp_fun(loop_n, 0, split_dict, mn_context, cub_n)

    def _mov_fmap2l1(self, loop_k, mn_context, split_dict, batch_dim):
        cal_hw = split_dict.get("cal_hw")
        k_loop = split_dict.get("loop_K")
        dedy_w = self.shape_4d_dedy[3]
        fmap_h, fmap_w = self.shape_4d_x[2:]
        kernel_h = self.shape_4d_filters[2]
        padu, padd = self.padding[0:2]
        stride_h, _ = self.strides
        fmap_l1_h = cal_hw // dedy_w * stride_h + kernel_h - 1
        fmap_l1_shape = [1, self.shape_x_5hd[1], fmap_l1_h * self.shape_x_5hd[3], self.shape_x_5hd[4]]
        fmap_l1 = self.tik_instance.Tensor("float16", fmap_l1_shape, name="fmap_l1", scope=tik.scope_cbuf)
        repeat_times = self.shape_x_5hd[1]
        with self.tik_instance.if_scope(loop_k == 0):
            fmap_gm = self.fmap[batch_dim, 0, 0, 0, 0]
            burst_hdim = fmap_l1_h - padu
            src_stride_hdim = fmap_h - burst_hdim
            self.tik_instance.data_move(fmap_l1, fmap_gm, 0, repeat_times, burst_hdim * fmap_w,
                                        src_stride_hdim * fmap_w, padu * fmap_w)
        with self.tik_instance.else_scope():
            fmap_gm = self.fmap[batch_dim, 0, cal_hw * loop_k // dedy_w - padu, 0, 0]
            with self.tik_instance.if_scope(loop_k == k_loop - 1):
                burst_hdim = fmap_l1_h - padd
                src_stride_hdim = fmap_h - burst_hdim
                self.tik_instance.data_move(fmap_l1, fmap_gm, 0, repeat_times, burst_hdim * fmap_w,
                                            src_stride_hdim * fmap_w, padd * fmap_w)

            with self.tik_instance.else_scope():
                src_stride_hdim = fmap_h - fmap_l1_h
                self.tik_instance.data_move(fmap_l1, fmap_gm, 0, repeat_times, burst_hdim * fmap_w,
                                            src_stride_hdim * fmap_w, 0)
        mn_context["fmap_l1"] = fmap_l1


    def _outer_loop_mpp_fun_big_xy(self, mn_context, split_dict, block_idx):
        """
        outer_loop_mpp_fun_big_xy.

        Parameters

        --------------
        mn_context: dict.

        split_dict: dict, infor of split.

        block_idx: block informatioin.

        Returns
        -------
        split result.
        """
        m_loop = split_dict.get("loop_M")
        n_loop = split_dict.get("loop_N")
        k_loop = split_dict.get("loop_K")
        fmap_h, fmap_w = self.shape_4d_x[2:]
        cal_n = split_dict.get("cal_N")
        cub_n = split_dict.get("cub_N")
        if k_loop == 1:
            fmap_l1_shape = [Constant.BATCH_PER_CORE, self.shape_x_5hd[1], fmap_h * fmap_w, self.shape_x_5hd[4]]
            l1_fmap = self.tik_instance.Tensor("float16", fmap_l1_shape, name="fmap_l1", scope=tik.scope_cbuf)
            self.tik_instance.data_move(l1_fmap, self.fmap[block_idx * Constant.BATCH_PER_CORE, 0, 0, 0, 0],
                                        0, Constant.BATCH_PER_CORE, fmap_l1_shape[1] * fmap_l1_shape[2], 0, 0)
        fmap_l0_shape = [(split_dict.get("cal_hw") + self.block - 1) // self.block, cal_n // self.block,
                         self.block, self.block]
        shape_n = self.shape_4d_filters[2] * self.shape_4d_filters[3] * self.shape_x_5hd[1]
        l0c_shape = [shape_n, split_dict.get("cal_M") // self.block, self.block, self.block]
        cub_shape = [shape_n // cub_n, split_dict.get("cal_M") // self.block, self.block, self.block]
        dw_l0c = self.tik_instance.Tensor("float32", l0c_shape, name="dw_l0C", scope=tik.scope_cc)
        dw_cub = self.tik_instance.Tensor("float32", cub_shape, name="dw_cub", scope=tik.scope_ubuf)
        mn_context["dw_l0C"] = dw_l0c
        mn_context["dw_cub"] = dw_cub
        with self.tik_instance.for_range(0, m_loop) as loop_m:
            with self.tik_instance.for_range(0, k_loop) as loop_k:
                with self.tik_instance.for_range(0, Constant.BATCH_PER_CORE, thread_num=2) as loop_batch:
                    batch_dim = block_idx * Constant.BATCH_PER_CORE + loop_batch
                    dedy_l0_shape = [split_dict.get("cal_M") // self.block,
                                     (split_dict.get("cal_hw") + self.block - 1) // self.block, self.block, self.block]
                    dedy_l0a0 = self.tik_instance.Tensor("float16", dedy_l0_shape,
                                                         name="dedy_l0A0", scope=tik.scope_ca)
                    mn_context["dedy_l0A0"] = dedy_l0a0
                    self._cal_dbn_fm_fun(loop_m, loop_k, mn_context, split_dict, batch_dim)
                    if k_loop != 1:
                        self._mov_fmap2l1(loop_k, mn_context, split_dict, batch_dim)
                    else:
                        mn_context["fmap_l1"] = l1_fmap[loop_batch, 0, 0, 0]
                    with self.tik_instance.for_range(0, n_loop, thread_num=min(n_loop, 2)) as loop_n:
                        fmap_l0b0 = self.tik_instance.Tensor("float16", fmap_l0_shape, name="fmap_l0B0",
                                                             scope=tik.scope_cb)
                        mn_context["fmap_l0B0"] = fmap_l0b0
                        self._loopk_mpp_big_x(loop_k, loop_n, mn_context, split_dict, loop_batch)
            with self.tik_instance.for_range(0, cub_n) as loop_n:
                self._ub_mpp_fun(loop_n, loop_m, split_dict, mn_context, cub_n)



def _check_shape_and_format(
        x,
        out_backprop,
        y,
        filter_size,
        strides,
        dilations,
        data_format
    ):
    """
    check the shape dims, format and get NCHW format shape
    """
    ori_format_filters = y.get("ori_format")
    ori_shape_filters = y.get("ori_shape")
    ori_format_dedy = out_backprop.get("ori_format")
    ori_shape_dedy = out_backprop.get("ori_shape")
    ori_format_fmap = x.get("ori_format")
    ori_shape_fmap = x.get("ori_shape")

    if list(filter_size) != list(ori_shape_filters):
        error_manager_cube.raise_err_param_should_be_equal(
            "filter_size", "ori_shape of y", filter_size, ori_shape_filters
        )
    shape_4d_filters = util_deconv_comm.get_nchw_shape(
        ori_format_filters, ori_shape_filters, util_deconv_comm.WEIGHT_FORMAT_LIST, "filter")
    shape_4d_x = util_deconv_comm.get_nchw_shape(
        ori_format_fmap, ori_shape_fmap, util_deconv_comm.FMAP_FORMAT_LIST, "feature_map")
    shape_4d_dedy = util_deconv_comm.get_nchw_shape(
        ori_format_dedy, ori_shape_dedy, util_deconv_comm.OUT_BACKPROP_FORMAT_LIST, "out_backprop")

    if len(strides) == 4:
        h_index = data_format.find("H")
        w_index = data_format.find("W")
        strides = [strides[h_index], strides[w_index]]

    dilations = util_deconv_comm.get_nchw_shape(
        data_format, dilations, util_deconv_comm.OUT_BACKPROP_FORMAT_LIST, "out_backprop")
    return [shape_4d_x,
            shape_4d_dedy,
            shape_4d_filters,
            strides,
            dilations]


def fused_dbn_dw(x, grads, dbn_x, diff_scale, diff_offset, scale,
                 batch_mean, batch_variance, dbn_y,
                 y, filter_size, strides, pads, dilations=(1, 1, 1, 1),
                 groups=1, data_format="NHWC", epsilon=0.0001, kernel_name="fuse_dbn2_dw"):
    """
    algorithm: fuse_bn_training_reduce_grad_conv2d_backprop_filter

    Parameters
    ----------
    x: dict
        dict of s, A 5D Tensor for input x of dw.
        source data type, support "float32", "float16".
    grads: dict
        dict of grads, A 5D Tensor for input grads.
        source data type, support "float32", "float16".
    dbn_x: dict
        dict of grads, A 5D Tensor for input x of bn_training_reduce_grad.
        source data type, support "float32", "float16".
    diff_scale: dict
        dict of diff_scale, A 5D Tensor for input diff_scale.
        The output of bn_training_update_grad.
        source data type, support "float32".
    diff_offset: dict
        dict of diff_offset, A 5HD Tensor for input diff_offset.
        The output of bn_training_update_grad.
        source data type, support "float32".
    scale: dict
        dict of scale, A 5HD Tensor for input scale.
        source data type, support "float32".
    batch_mean: dict
        dict of batch_mean, A 5D Tensor for input batch_mean.
        source data type, support "float32".
    batch_variance: dict
        dict of batch_variance, A 5D Tensor for input batch_variance.
        source data type, support "float32".
    dbn_y: dict
        dict of bn_training_reduce_grad's output, A `Tensor`. Has the same type as `grads`.
    y: dict
        dict of conv2d_backprop_filter's output.
    epsilon: float
        A small float number added to the variance of dbn_x.
    filter_size: tuple/list of 4 integers
        The shape of filter. 4-D with shape [filter_height, filter_width, in_channels,
        out_channels] or [out_channels, filter_height, filter_width, in_channels] or
        [out_channels, in_channel, filter_height, filter_width].
    strides: tuple/list of 2 integers
        filter move stride.
    pads: tuple/list of 4 integers
        [pad_top, pad_bottom, pad_left, pad_right].
    dilations: tuple/list of 4 integers
        filter expand size of dilated conv2d_backprop_filter. Default to (1, 1, 1, 1).
    groups: int
        param for group conv2d_backprop_filter. Default to 1.
    data_format: str
        input data format. Specify the data format of the input and output data.
        Default to "NHWC".
    kernel_name: str
        kernel name, default value is "fuse_dbn2_dw"
    Returns
    -------
    None
    """
    shape_util.compare_tensor_dict_key(grads, x, "dtype")
    shape_util.compare_tensor_dict_key(grads, dbn_x, "dtype")
    shape_util.compare_tensor_dict_key(grads, dbn_x, "shape")
    shape_util.compare_tensor_dict_key(grads, dbn_y, "dtype")
    shape_util.compare_tensor_dict_key(grads, dbn_y, "shape")
    shape_util.compare_tensor_dict_key(batch_mean, batch_variance, "shape")
    shape_util.compare_tensor_dict_key(batch_mean, batch_variance, "dtype")
    shape_util.compare_tensor_dict_key(batch_mean, diff_scale, "shape")
    shape_util.compare_tensor_dict_key(batch_mean, diff_scale, "dtype")
    shape_util.compare_tensor_dict_key(batch_mean, diff_offset, "shape")
    shape_util.compare_tensor_dict_key(batch_mean, diff_offset, "dtype")
    shape_util.compare_tensor_dict_key(batch_mean, scale, "shape")
    shape_util.compare_tensor_dict_key(batch_mean, scale, "dtype")
    res = _check_shape_and_format(
        x,
        grads,
        y,
        filter_size,
        strides,
        dilations,
        data_format
    )
    [shape_4d_x, shape_4d_dedy, shape_4d_filters, strides, dilations] = res

    input_fmap_x_dtype = x.get("dtype").lower()
    input_grads_dtype = grads.get("dtype").lower()
    input_dbn_x_dtype = dbn_x.get("dtype").lower()
    batch_mean_dtype = batch_mean.get("dtype").lower()
    dbn_y_dtype = dbn_y.get("dtype").lower()
    y_dtype = y.get("dtype").lower()


    shape_fmap_x = x.get("shape")
    shape_grads = grads.get("shape")
    shape_dbn_x = dbn_x.get("shape")
    shape_batch_mean = batch_mean.get("shape")
    shape_y = y.get("shape")

    para_check.check_dtype(input_fmap_x_dtype, ("float16",), param_name="fmap_x")
    para_check.check_dtype(input_grads_dtype, ("float16"), param_name="grads")
    para_check.check_dtype(input_dbn_x_dtype, ("float16"), param_name="dbn_x")
    para_check.check_dtype(batch_mean_dtype, ("float32",), param_name="batch_mean")

    tik_instance = tik.Tik()

    fmap_x = tik_instance.Tensor(input_fmap_x_dtype, shape_fmap_x, name="fmap_x", scope=tik.scope_gm)
    grads_input = tik_instance.Tensor(input_grads_dtype, shape_grads, name="grads_input", scope=tik.scope_gm)
    diff_scale_input = tik_instance.Tensor(batch_mean_dtype, shape_batch_mean,
                                           name="diff_scale_input", scope=tik.scope_gm)
    diff_offset_input = tik_instance.Tensor(batch_mean_dtype, shape_batch_mean,
                                            name="diff_offset_input", scope=tik.scope_gm)
    scale_input = tik_instance.Tensor(batch_mean_dtype, shape_batch_mean, name="scale_input", scope=tik.scope_gm)
    dbn_x_input = tik_instance.Tensor(input_dbn_x_dtype, shape_dbn_x, name="dbn_x_input", scope=tik.scope_gm)
    batch_mean_input = tik_instance.Tensor(batch_mean_dtype, shape_batch_mean, name="batch_mean_input",
                                           scope=tik.scope_gm)
    batch_variance_input = tik_instance.Tensor(batch_mean_dtype, shape_batch_mean,
                                               name="batch_variance_input", scope=tik.scope_gm)
    epsilon_input = tik_instance.Scalar("float32", "epsilon", epsilon)

    dbn_res = tik_instance.Tensor(dbn_y_dtype, shape_grads, name="dbn_res", scope=tik.scope_gm)
    dw_res = tik_instance.Tensor(y_dtype, shape_y, name="dw_res", scope=tik.scope_gm, is_atomic_add=True)

    para_dict = {
        "shape_4d_filters": shape_4d_filters,
        "shape_4d_x": shape_4d_x,
        "shape_4d_dedy": shape_4d_dedy,
        "strides": strides,
        "padding": pads,
        "dilations": dilations,
        "groups": groups,
        "kernel_name": kernel_name,
    }

    dbn2_dw_fusion = Dbn2Conv2dBackpropFilter(tik_instance, fmap_x, grads_input, dbn_x_input,
                                              diff_scale_input, diff_offset_input, scale_input,
                                              batch_mean_input, batch_variance_input,
                                              epsilon_input, dbn_res, dw_res, para_dict)

    dbn2_dw_fusion.dbn2_dw_compute()
    config = {"double_buffer_non_reuse": True, "InjectSync": {"sync_mode": 4}}
    tik_instance.BuildCCE(kernel_name=kernel_name,
                          inputs=[fmap_x, grads_input, dbn_x_input, diff_scale_input, diff_offset_input, scale_input,
                                  batch_mean_input, batch_variance_input], outputs=[dbn_res, dw_res], config=config)
    return tik_instance
