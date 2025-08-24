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
max_pool_grad_with_argmax_v1
"""
# 'pylint: disable=too-many-lines
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tik
from impl import common_util_v1
from impl import constant_util_v1 as constant


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for Constant
    """
    # size of vector calc one repeat
    ONE_REPEAT = 256
    # max repeat of vector calc
    V_MAX_REPEAT = 255
    # max num of fp16 in one repeat
    FP16_MAX = 128
    # max num of fp32 in one repeat
    FP32_MAX = 64
    # max num of fp16 mask handle one time
    MASK_MAX = 8
    TILING_NUM = 64
    MAX_GM = 2 ** 31 - 1
    L1_SIZE = tbe_platform.get_soc_spec(tbe_platform.L1_SIZE)


# 'pylint: disable=too-many-lines,too-many-instance-attributes,too-many-arguments,unused-argument,too-many-statements
# 'pylint: disable=too-many-locals
class MaxpoolGradBase:
    """
    parameter for max_pool_grad_with_pool
    """

    def __init__(self, grad, argmax, input_x, ksize, strides, padding, dilation, ceil_mode):
        """
        init compare and bit pack base parameters
        Parameters
        ----------
        input_x: input of maxpool, useless for maxpool gard
        grad: input of maxpoolgard or output of maxpool
        argmax:output of maxpool mask or index
        strides: stride , minimum length is 4, just like [1, poolingStrideH, poolingStrideW, 1]
        padding: pad
        Returns
        -------
        None
        """
        self.blocknum = tik.Dprofile("v100", "cloud").get_aicore_num()
        self.ub_size = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)

        self.dtype = grad.get("dtype").lower()
        self.dtype_size = common_util_v1.get_data_size(self.dtype)
        self.nc1 = 1

        self.tik_instance = tik.Tik()
        self.ksize = ksize
        self.strides = strides
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        self.channel = 16
        self.pad_top, self.pad_left = self.padding[1:3]
        self.kernel_h, self.kernel_w = self.ksize[1:3]

        self.data_input = self.tik_instance.Tensor(self.dtype, (Constant.MAX_GM,), name="data_input",
                                                   scope=tik.scope_gm)
        self.data_mask = self.tik_instance.Tensor("uint16", (Constant.MAX_GM,), name="data_mask", scope=tik.scope_gm)
        self.data_input_origin = self.tik_instance.Tensor(self.dtype, (Constant.MAX_GM,), name="data_input_origin",
                                                          scope=tik.scope_gm)
        self.data_output = self.tik_instance.Tensor(self.dtype, (Constant.MAX_GM,), name="data_output",
                                                    scope=tik.scope_gm)
        self.tiling_gm = self.tik_instance.Tensor("int32", (Constant.TILING_NUM,), name="tiling_gm", scope=tik.scope_gm)

        self.tiling_mode = None
        self.real_block = None
        self.block_cycle = None
        self.block_index = None
        self.dxh, self.dxw = None, None
        self.dyh, self.dyw = None, None

        self.stride_h, self.stride_w = None, None
        self.pad_bottom, self.pad_right = None, None
        self.offset_h, self.offset_w = None, None
        self.hoverlap, self.woverlap = None, None
        self.col2img_h, self.col2img_w = None, None
        self.col2img_h_every, self.col2img_h_last = None, None
        self.ho_max, self.wo_max = None, None
        self.ho_max_every, self.ho_max_last = None, None
        self.ho_every, self.ho_last = None, None
        self.ho_count = None

    # 'pylint: disable=too-many-statements
    def get_tiling_params(self):
        """
        get tiling params
        """
        self.tiling_mode = self.tik_instance.Scalar("int32")
        self.real_block = self.tik_instance.Scalar("int32")
        self.block_cycle = self.tik_instance.Scalar("int32")
        self.block_index = self.tik_instance.Scalar("int32")
        self.dxh = self.tik_instance.Scalar("int32")
        self.dxw = self.tik_instance.Scalar("int32")
        self.dyh = self.tik_instance.Scalar("int32")
        self.dyw = self.tik_instance.Scalar("int32")
        self.stride_h = self.tik_instance.Scalar("int32")
        self.stride_w = self.tik_instance.Scalar("int32")
        self.pad_bottom = self.tik_instance.Scalar("int32")
        self.pad_right = self.tik_instance.Scalar("int32")
        self.offset_h = self.tik_instance.Scalar("int32")
        self.offset_w = self.tik_instance.Scalar("int32")
        self.hoverlap = self.tik_instance.Scalar("int32")
        self.woverlap = self.tik_instance.Scalar("int32")
        self.col2img_h = self.tik_instance.Scalar("int32")
        self.col2img_w = self.tik_instance.Scalar("int32")
        self.col2img_h_every = self.tik_instance.Scalar("int32")
        self.col2img_h_last = self.tik_instance.Scalar("int32")
        self.ho_max = self.tik_instance.Scalar("int32")
        self.wo_max = self.tik_instance.Scalar("int32")
        self.ho_max_every = self.tik_instance.Scalar("int32")
        self.ho_max_last = self.tik_instance.Scalar("int32")
        self.ho_every = self.tik_instance.Scalar("int32")
        self.ho_last = self.tik_instance.Scalar("int32")
        self.ho_count = self.tik_instance.Scalar("int32")

        with self.tik_instance.new_stmt_scope():
            tiling_ub = self.tik_instance.Tensor("int32", (Constant.TILING_NUM,), name="tiling_ub",
                                                 scope=tik.scope_ubuf)
            self.tik_instance.data_move(tiling_ub, self.tiling_gm, 0, 1, 8, 0, 0)
            self.tiling_mode.set_as(tiling_ub[0])
            self.real_block.set_as(tiling_ub[1])
            self.block_cycle.set_as(tiling_ub[2])
            self.block_index.set_as(tiling_ub[3])
            self.dxh.set_as(tiling_ub[4])
            self.dxw.set_as(tiling_ub[5])
            self.dyh.set_as(tiling_ub[6])
            self.dyw.set_as(tiling_ub[7])
            self.stride_h.set_as(tiling_ub[8])
            self.stride_w.set_as(tiling_ub[9])
            self.pad_bottom.set_as(tiling_ub[10])
            self.pad_right.set_as(tiling_ub[11])
            self.offset_h.set_as(tiling_ub[12])
            self.offset_w.set_as(tiling_ub[13])
            self.hoverlap.set_as(tiling_ub[14])
            self.woverlap.set_as(tiling_ub[15])
            self.col2img_h.set_as(tiling_ub[16])
            self.col2img_w.set_as(tiling_ub[17])
            self.col2img_h_every.set_as(tiling_ub[18])
            self.col2img_h_last.set_as(tiling_ub[19])
            self.ho_max.set_as(tiling_ub[20])
            self.wo_max.set_as(tiling_ub[21])
            self.ho_max_every.set_as(tiling_ub[22])
            self.ho_max_last.set_as(tiling_ub[23])
            self.ho_every.set_as(tiling_ub[24])
            self.ho_last.set_as(tiling_ub[25])
            self.ho_count.set_as(tiling_ub[26])

    def _move_mask_to_l1(self, mask_l1, mask_gm, mask_idx, repeat_time):
        max_repeat_time = 65535
        with self.tik_instance.if_scope(repeat_time <= max_repeat_time):
            self.tik_instance.data_move(
                mask_l1, mask_gm[mask_idx], constant.SID, 1, repeat_time, 0, 0)
        with self.tik_instance.else_scope():
            iter_time = repeat_time // max_repeat_time
            res_iter_time = repeat_time - iter_time * max_repeat_time
            cur_mask_idx = self.tik_instance.Scalar("int32")
            with self.tik_instance.for_range(0, iter_time) as idx:
                cur_mask_idx.set_as(mask_idx + idx * max_repeat_time)
                self.tik_instance.data_move(
                    mask_l1[idx * max_repeat_time], mask_gm[cur_mask_idx], constant.SID, 1, max_repeat_time, 0, 0)
            with self.tik_instance.if_scope(res_iter_time > 0):
                cur_mask_idx.set_as(mask_idx + iter_time * max_repeat_time)
                self.tik_instance.data_move(
                    mask_l1[iter_time * max_repeat_time], mask_gm[cur_mask_idx], constant.SID, 1, res_iter_time, 0, 0)

    # 'pylint: disable=too-many-locals,too-many-statements
    def tik_instance_cut_nc1_cut_h(self, block_id):
        """
        function for max_pool_grad_with_pool calc for normal shape
        """
        dtype = self.dtype
        dtype_size = self.dtype_size
        nc1 = self.nc1
        wo_max = self.tik_instance.Scalar("int32")
        h_cycle = self.tik_instance.Scalar("int32")
        wo_max.set_as(self.wo_max)

        stride_h, stride_w = self.strides[1:3]
        stridehw = stride_h * stride_w
        if stridehw == 1:
            max_mem_size0 = 16256
            max_mem_size1 = 16448
            max_mem_size2 = 16512
            max_mem_size3 = 1024
            max_mem_size4 = 16256
            max_mem_size5 = 16256
        elif stridehw < 4:
            max_mem_size0 = 10880
            max_mem_size1 = 21920
            max_mem_size2 = 22632
            max_mem_size3 = 1024
            max_mem_size4 = 10880
            max_mem_size5 = 10880
        else:
            max_mem_size0 = 8064
            max_mem_size1 = 32832
            max_mem_size2 = 32896
            max_mem_size3 = 512
            max_mem_size4 = 8064
            max_mem_size5 = 8064

        with self.tik_instance.if_scope(self.hoverlap == 0):
            h_cycle.set_as((self.dyh + self.ho_max - 1) // self.ho_max)
        with self.tik_instance.else_scope():
            h_cycle.set_as((self.dyh - 1 + self.ho_max - 2) // (self.ho_max - 1))

        with self.tik_instance.if_scope(self.dyh == 1):
            self.ho_max.set_as(1)
            h_cycle.set_as(1)

        mask_one_window = ((self.dyh * self.dyw + 15) // 16 + 1) * 16

        # vector_repeat_time
        v_rep_time = self.ho_max * wo_max * self.channel * dtype_size // Constant.ONE_REPEAT
        v_rep_cycle_fp32 = 2 * v_rep_time // Constant.V_MAX_REPEAT
        # v_rep_last
        v_rep_last_fp32 = 2 * v_rep_time % Constant.V_MAX_REPEAT

        v_rep_time_col = (2 * (self.col2img_w * self.channel * self.col2img_h + 64) * dtype_size) // Constant.ONE_REPEAT
        v_rep_cycle_col = v_rep_time_col // Constant.V_MAX_REPEAT
        v_rep_last_col = v_rep_time_col % Constant.V_MAX_REPEAT

        real_cycle = self.tik_instance.Scalar("int32")
        block_base = self.tik_instance.Scalar("int32")
        block_num = self.tik_instance.Scalar("int32")

        with self.tik_instance.if_scope(block_id < self.block_index):
            real_cycle.set_as(self.block_cycle + 1)
            block_base.set_as(block_id * real_cycle)
        with self.tik_instance.else_scope():
            real_cycle.set_as(self.block_cycle)
            block_base.set_as(self.block_index + block_id * self.block_cycle)
        with self.tik_instance.for_range(0, real_cycle) as cycle_id:  # 32
            block_num.set_as(block_base + cycle_id)
            data_vsel_ub_zero = self.tik_instance.Tensor(dtype, (128,), name="data_vsel_ub_zero0",
                                                         scope=tik.scope_ubuf)

            # data move only 16 fp16 every time
            self.tik_instance.data_move(data_vsel_ub_zero[0], self.data_input_origin[0],
                                        constant.SID, constant.DEFAULT_NBURST,
                                        constant.DEFAULT_BURST_LEN, constant.STRIDE_ZERO,
                                        constant.STRIDE_ZERO)

            # clear to zeros
            self.clean_fp16_one_repeat(data_vsel_ub_zero, dtype)

            with self.tik_instance.for_range(0, nc1) as loopc1:  # 1
                # vector_dup ub every time
                dxh_address_offset = self.tik_instance.Scalar("int32")
                dxh_address_offset.set_as(0)
                # 16x6x16
                data_max_ub = self.tik_instance.Tensor(dtype, (max_mem_size0,),
                                                       name="data_max_ub0", scope=tik.scope_ubuf)
                # 33x13x16+64
                data_vmul_ub_col2img_fp32 = \
                    self.tik_instance.Tensor("float32", (max_mem_size1,),
                                             name="data_vmul_ub_col2img_fp320", scope=tik.scope_ubuf)
                # mask define 6x16
                data_mask_ub = self.tik_instance.Tensor("uint16", (max_mem_size3,),
                                                        name="data_mask_ub0", scope=tik.scope_ubuf)

                new_looph = self.tik_instance.Scalar("int32")
                new_looph.set_as(0)
                in_nburst = self.tik_instance.Scalar("int32")
                in_nburst.set_as(self.ho_max)
                true_val = self.tik_instance.Scalar("int32")
                with self.tik_instance.if_scope(
                        self.kernel_h * self.kernel_w * mask_one_window > (Constant.L1_SIZE // 2)):
                    true_val.set_as(0)
                with self.tik_instance.else_scope():
                    with self.tik_instance.if_scope(
                            tik.any(self.dyw % 16 == 0, tik.all(self.ho_max == 1, h_cycle == 1))):
                        true_val.set_as(1)
                    with self.tik_instance.else_scope():
                        true_val.set_as(0)

                l1_len = ((self.kernel_h * self.kernel_w * mask_one_window + 127) // 128) * 128
                # 49 x 2 x 16
                data_mask_l1 = self.tik_instance.Tensor("uint16", (l1_len,),
                                                        name="data_mask_l1_buf0", scope=tik.scope_cbuf,
                                                        max_mem_size=1048544)
                with self.tik_instance.for_range(0, h_cycle) as looph:  # 10
                    with self.tik_instance.if_scope(self.hoverlap == 0):
                        new_looph.set_as(looph * self.ho_max)
                        with self.tik_instance.if_scope(looph == h_cycle - 1):
                            in_nburst.set_as(self.dyh - looph * self.ho_max)
                    with self.tik_instance.else_scope():
                        with self.tik_instance.if_scope(looph != 0):
                            new_looph.set_as(looph * (self.ho_max - 1))
                        with self.tik_instance.if_scope(looph == h_cycle - 1):
                            in_nburst.set_as(self.dyh - looph * (self.ho_max - 1))

                    # clearn to zeors
                    self.clean_fp32_multi_repeat(data_vmul_ub_col2img_fp32, dtype_size * 2)
                    self.clean_fp16_one_repeat(data_max_ub, dtype)
                    self.clean_fp16_one_repeat(data_mask_ub, "uint16")

                    self.tik_instance.data_move(data_max_ub,
                                                self.data_input[(block_num * nc1 * self.dyh + loopc1 * self.dyh +
                                                                 new_looph) * self.dyw * self.channel],
                                                constant.SID, in_nburst, self.dyw,
                                                constant.STRIDE_ZERO, wo_max - self.dyw)
                    with self.tik_instance.if_scope(true_val == 1):
                        mask_idx = self.tik_instance.Scalar("int32")
                        mask_idx.set_as(block_num * nc1 * mask_one_window * self.kernel_w * self.kernel_h +
                                        loopc1 * mask_one_window * self.kernel_w * self.kernel_h)
                        mask_repeat_time = self.kernel_h * self.kernel_w * mask_one_window // 16
                        self._move_mask_to_l1(data_mask_l1, self.data_mask, mask_idx, mask_repeat_time)

                    with self.tik_instance.new_stmt_scope():
                        with self.tik_instance.for_range(0, self.kernel_h * self.kernel_w) as mask_id:  # 7 x 7
                            with self.tik_instance.for_range(0, in_nburst) as cycle:  # 12
                                with self.tik_instance.if_scope(true_val == 1):
                                    self.tik_instance.data_move(
                                        data_mask_ub[cycle * wo_max],
                                        data_mask_l1[(new_looph + cycle) * self.dyw + mask_one_window * mask_id],
                                        constant.SID, 1, wo_max // 16, 0, 0)
                                with self.tik_instance.else_scope():
                                    # mask copy gm to ub 9 x 16
                                    self.tik_instance.data_move(data_mask_ub[cycle * wo_max], self.data_mask[
                                        block_num * nc1 * mask_one_window * self.kernel_w * self.kernel_h +
                                        loopc1 * mask_one_window * self.kernel_w * self.kernel_h + (new_looph + cycle) *
                                        self.dyw + mask_one_window * mask_id], constant.SID, 1, wo_max // 16, 0, 0)

                            # 16 x 6 x 16
                            data_vsel_ub = self.tik_instance.Tensor(dtype, (max_mem_size4,),
                                                                    name="data_vsel_ub0", scope=tik.scope_ubuf)
                            data_vsel_ub_fp32 = self.tik_instance.Tensor("float32", (max_mem_size5,),
                                                                         name="data_vsel_ub_fp320",
                                                                         scope=tik.scope_ubuf)
                            with self.tik_instance.if_scope(v_rep_time > 0):  # 12
                                with self.tik_instance.for_range(0, v_rep_time, thread_num=1) as cycle:
                                    # extract 8 mask, compare 128 number
                                    cmpmask = self.tik_instance.mov_tensor_to_cmpmask(
                                        data_mask_ub[cycle * Constant.MASK_MAX])

                                    # calc 128 number everytime, 16x6x16->12x128
                                    self.tik_instance.vsel(constant.MASK128, 0, data_vsel_ub[cycle * Constant.FP16_MAX],
                                                           cmpmask, data_max_ub[cycle * Constant.FP16_MAX],
                                                           data_vsel_ub_zero[0], constant.REPEAT_TIME_ONCE,
                                                           constant.STRIDE_ONE, constant.STRIDE_ONE,
                                                           constant.STRIDE_ONE, constant.REPEAT_STRIDE_EIGHT,
                                                           constant.REPEAT_STRIDE_EIGHT,
                                                           constant.REPEAT_STRIDE_EIGHT)

                            # fp16 to fp32
                            with self.tik_instance.if_scope(v_rep_cycle_fp32 > 0):  # 0
                                with self.tik_instance.for_range(0, v_rep_cycle_fp32, thread_num=1) as cycle:
                                    self.tik_instance.vconv(constant.MASK64, "",
                                                            data_vsel_ub_fp32[
                                                                cycle * Constant.V_MAX_REPEAT * Constant.FP32_MAX],
                                                            data_vsel_ub[
                                                                cycle * Constant.V_MAX_REPEAT * Constant.FP32_MAX],
                                                            Constant.V_MAX_REPEAT, constant.STRIDE_ONE,
                                                            constant.STRIDE_ONE, constant.REPEAT_STRIDE_EIGHT,
                                                            constant.REPEAT_STRIDE_FOUR)

                            with self.tik_instance.if_scope(v_rep_last_fp32 != 0):  # 24 x 64
                                self.tik_instance.vconv(constant.MASK64, "",
                                                        data_vsel_ub_fp32[
                                                            v_rep_cycle_fp32 * Constant.V_MAX_REPEAT *
                                                            Constant.FP32_MAX],
                                                        data_vsel_ub[v_rep_cycle_fp32 * Constant.V_MAX_REPEAT *
                                                                     Constant.FP32_MAX],
                                                        v_rep_last_fp32, constant.STRIDE_ONE, constant.STRIDE_ONE,
                                                        constant.REPEAT_STRIDE_EIGHT, constant.REPEAT_STRIDE_FOUR)

                            # col2img
                            fetch_filter_w = mask_id % self.kernel_w
                            fetch_filter_h = mask_id // self.kernel_w
                            left_top_w = 0
                            left_top_h = 0

                            self.tik_instance.col2img(data_vmul_ub_col2img_fp32[0],  # 33x13x16 + 64
                                                      data_vsel_ub_fp32[0],  # 16x6x16
                                                      (0, 0, 0, 0), self.col2img_h, self.col2img_w, fetch_filter_w,
                                                      fetch_filter_h, left_top_w, left_top_h,
                                                      self.stride_w, self.stride_h, self.kernel_w, self.kernel_h,
                                                      self.dilation[1], self.dilation[2], self.ho_max * wo_max // 16)
                    # 33x13x16+128
                    data_vmul_ub_col2img_fp16 = \
                        self.tik_instance.Tensor(dtype, (max_mem_size2,),
                                                 name="data_vmul_ub_col2img_fp160", scope=tik.scope_ubuf)
                    self.clean_fp16_multi_repeat(data_vmul_ub_col2img_fp16, dtype_size)

                    # convert fp32 to fp16
                    with self.tik_instance.if_scope(v_rep_cycle_col > 0):  # 0
                        with self.tik_instance.for_range(0, v_rep_cycle_col) as cycle:
                            self.tik_instance.vconv(constant.MASK64, "",
                                                    data_vmul_ub_col2img_fp16[
                                                        cycle * Constant.V_MAX_REPEAT * Constant.FP32_MAX],
                                                    data_vmul_ub_col2img_fp32[
                                                        cycle * Constant.V_MAX_REPEAT * Constant.FP32_MAX],
                                                    Constant.V_MAX_REPEAT, constant.STRIDE_ONE, constant.STRIDE_ONE,
                                                    constant.REPEAT_STRIDE_FOUR, constant.REPEAT_STRIDE_EIGHT)

                    with self.tik_instance.if_scope(v_rep_last_col != 0):  # 108 x 64 = 6912
                        self.tik_instance.vconv(constant.MASK64, "",
                                                data_vmul_ub_col2img_fp16[
                                                    v_rep_cycle_col * Constant.V_MAX_REPEAT * Constant.FP32_MAX],
                                                data_vmul_ub_col2img_fp32[
                                                    v_rep_cycle_col * Constant.V_MAX_REPEAT * Constant.FP32_MAX],
                                                v_rep_last_col, constant.STRIDE_ONE, constant.STRIDE_ONE,
                                                constant.REPEAT_STRIDE_FOUR, constant.REPEAT_STRIDE_EIGHT)

                    # move ub to gm
                    output_cuthline = self.tik_instance.Scalar("int32")
                    output_cuthline.set_as(0)
                    src_address = self.tik_instance.Scalar("int32")
                    src_address.set_as(self.pad_left * self.channel)
                    with self.tik_instance.if_scope(self.hoverlap == 0):
                        output_cuthline.set_as(self.col2img_h)
                        with self.tik_instance.if_scope(looph == 0):
                            src_address.set_as(src_address + self.pad_top * self.col2img_w * self.channel)
                            output_cuthline.set_as(output_cuthline - self.pad_top)
                            with self.tik_instance.if_scope(looph == h_cycle - 1):
                                output_cuthline.set_as(self.dxh)
                        with self.tik_instance.else_scope():
                            with self.tik_instance.if_scope(looph == h_cycle - 1):
                                output_cuthline.set_as(self.dxh - self.col2img_h * (h_cycle - 1) + self.pad_top)
                    with self.tik_instance.else_scope():
                        with self.tik_instance.if_scope(looph == 0):
                            output_cuthline.set_as(self.stride_h * self.ho_max - self.pad_top)
                            src_address.set_as(src_address + self.pad_top * self.col2img_w * self.channel)
                            with self.tik_instance.if_scope(looph == h_cycle - 1):
                                output_cuthline.set_as(self.dxh)
                        with self.tik_instance.else_scope():
                            output_cuthline.set_as(self.stride_h * (self.ho_max - 1))
                            src_address.set_as(src_address + self.stride_h * self.col2img_w * self.channel)
                            with self.tik_instance.if_scope(looph == h_cycle - 1):
                                output_cuthline.set_as(self.dxh - self.stride_h * self.ho_max - (h_cycle - 2) *
                                                       self.stride_h * (self.ho_max - 1) + self.pad_top)

                    idx = block_num * nc1 * self.dxh * self.dxw * self.channel + loopc1 * self.dxh * self.dxw * \
                          self.channel + dxh_address_offset

                    self.tik_instance.data_move(self.data_output[idx],
                                                data_vmul_ub_col2img_fp16[src_address],  # 544
                                                constant.SID, output_cuthline, self.offset_w,  # 12, 12
                                                self.col2img_w - self.offset_w,  # 21
                                                self.dxw - self.offset_w)  # 0

                    dxh_address_offset.set_as(dxh_address_offset + output_cuthline * self.dxw * self.channel)

    # 'pylint: disable=too-many-locals,too-many-statements
    def tik_instance_cut_nc1h_cut_h(self, block_id):
        """
        function for max_pool_grad_with_pool calc for normal shape
        """
        dtype = self.dtype
        dtype_size = self.dtype_size
        dialate_h = self.dilation[1]
        dialate_w = self.dilation[2]
        stride_h, stride_w = self.strides[1:3]
        stridehw = stride_h * stride_w

        if stridehw == 1:
            max_mem_size0 = 16256
            max_mem_size1 = 16448
            max_mem_size2 = 16512
            max_mem_size3 = 1024
            max_mem_size4 = 16256
            max_mem_size5 = 16256
        elif stridehw < 4:
            max_mem_size0 = 10800
            max_mem_size1 = 21600
            max_mem_size2 = 21664
            max_mem_size3 = 675
            max_mem_size4 = 10800
            max_mem_size5 = 10800
        else:
            max_mem_size0 = 7936
            max_mem_size1 = 32368
            max_mem_size2 = 32432
            max_mem_size3 = 512
            max_mem_size4 = 7936
            max_mem_size5 = 7936

        h_cycle_every = self.tik_instance.Scalar("int32")
        h_cycle_last = self.tik_instance.Scalar("int32")
        wo_max = self.tik_instance.Scalar("int32")
        wo_max.set_as(self.wo_max)

        with self.tik_instance.if_scope(self.hoverlap == 0):
            h_cycle_every.set_as((self.ho_every + self.ho_max_every - 1) // self.ho_max_every)
            h_cycle_last.set_as((self.ho_last + self.ho_max_last - 1) // self.ho_max_last)
        with self.tik_instance.else_scope():
            h_cycle_every.set_as((self.ho_every - 1 - self.ho_max_every) // (self.ho_max_every - 1) + 2)
            h_cycle_last.set_as((self.ho_last - 1 - self.ho_max_last) // (self.ho_max_last - 1) + 2)

        mask_one_window = ((self.dyh * self.dyw + 15) // 16 + 1) * 16
        # vector_repeat_time
        v_rep_time_last = self.ho_max_last * wo_max * self.channel * dtype_size // Constant.ONE_REPEAT
        v_rep_cycle_fp32_last = 2 * v_rep_time_last // Constant.V_MAX_REPEAT
        v_rep_last_fp32_last = 2 * v_rep_time_last % Constant.V_MAX_REPEAT
        v_rep_time_col_last = (2 * (self.col2img_w * self.channel * self.col2img_h_last + 64) *
                               dtype_size) // Constant.ONE_REPEAT
        v_rep_cycle_col_last = v_rep_time_col_last // Constant.V_MAX_REPEAT
        v_rep_last_col_last = v_rep_time_col_last % Constant.V_MAX_REPEAT
        v_rep_time_every = self.ho_max_every * wo_max * self.channel * dtype_size // Constant.ONE_REPEAT
        v_rep_cycle_fp32_every = 2 * v_rep_time_every // Constant.V_MAX_REPEAT
        v_rep_last_fp32_every = 2 * v_rep_time_every % Constant.V_MAX_REPEAT
        v_rep_time_col_every = (2 * (self.col2img_w * self.channel * self.col2img_h_every + 64) *
                                dtype_size) // Constant.ONE_REPEAT
        v_rep_cycle_col_every = v_rep_time_col_every // Constant.V_MAX_REPEAT
        v_rep_last_col_every = v_rep_time_col_every % Constant.V_MAX_REPEAT

        real_cycle = self.tik_instance.Scalar("int32")
        block_base = self.tik_instance.Scalar("int32")
        block_num = self.tik_instance.Scalar("int32")
        with self.tik_instance.if_scope(block_id < self.block_index):
            real_cycle.set_as(self.block_cycle + 1)
            block_base.set_as(block_id * real_cycle)
        with self.tik_instance.else_scope():
            real_cycle.set_as(self.block_cycle)
            block_base.set_as(self.block_index + block_id * self.block_cycle)

        with self.tik_instance.for_range(0, real_cycle) as cycle_id:  # 1
            block_num.set_as(block_base + cycle_id)
            data_vsel_ub_zero = self.tik_instance.Tensor(dtype, (128,),
                                                         name="data_vsel_ub_zero1",
                                                         scope=tik.scope_ubuf)
            self.tik_instance.data_move(data_vsel_ub_zero[0],
                                        self.data_input_origin[0],
                                        constant.SID,
                                        constant.DEFAULT_NBURST,
                                        constant.DEFAULT_BURST_LEN,
                                        constant.STRIDE_ZERO,
                                        constant.STRIDE_ZERO)
            self.clean_fp16_one_repeat(data_vsel_ub_zero, dtype)

            block_batch = self.tik_instance.Scalar("int32")
            block_batch.set_as(block_num // self.ho_count)  # 0
            block_h = self.tik_instance.Scalar("int32")
            block_h.set_as(block_num % self.ho_count)  # 0
            with self.tik_instance.if_scope(block_h == self.ho_count - 1):
                data_max_ub = self.tik_instance.Tensor(dtype, (max_mem_size0,),
                                                       name="data_max_ub1",
                                                       scope=tik.scope_ubuf)
                data_vmul_ub_col2img_fp32 = \
                    self.tik_instance.Tensor("float32",
                                             (max_mem_size1,),
                                             name="data_vmul_ub_col2img_fp321",
                                             scope=tik.scope_ubuf)
                data_vmul_ub_col2img_fp16 = \
                    self.tik_instance.Tensor(dtype,
                                             (max_mem_size2,),
                                             name="data_vmul_ub_col2img_fp161",
                                             scope=tik.scope_ubuf)
                # mask define
                data_mask_ub = self.tik_instance.Tensor("uint16", (max_mem_size3,),
                                                        name="data_mask_ub1",
                                                        scope=tik.scope_ubuf)

                new_looph = self.tik_instance.Scalar("int32")
                new_looph.set_as(0)
                in_nburst = self.tik_instance.Scalar("int32")
                in_nburst.set_as(self.ho_max_last)
                in_src_address = self.tik_instance.Scalar("int32")
                mask_address = self.tik_instance.Scalar("int32")
                with self.tik_instance.for_range(0, h_cycle_last) as looph:  # 4
                    in_src_address.set_as(block_batch * self.dyh * self.dyw * self.channel)
                    # 56 x 56 x 3 x 3
                    mask_address.set_as(block_batch * mask_one_window * self.kernel_w * self.kernel_h)
                    with self.tik_instance.if_scope(self.hoverlap == 0):
                        new_looph.set_as(looph * self.ho_max_last)
                        with self.tik_instance.if_scope(looph == h_cycle_last - 1):
                            in_nburst.set_as(self.ho_last - looph * self.ho_max_last)
                        in_src_address.set_as(in_src_address +
                                              (block_h * self.ho_every + new_looph) * self.dyw * self.channel)
                        mask_address.set_as(mask_address +
                                            (block_h * self.ho_every + new_looph) * self.dyw)
                    with self.tik_instance.else_scope():
                        with self.tik_instance.if_scope(looph != 0):
                            new_looph.set_as(looph * (self.ho_max_last - 1))
                        with self.tik_instance.if_scope(looph == h_cycle_last - 1):
                            in_nburst.set_as(self.ho_last - looph * (self.ho_max_last - 1))
                        in_src_address.set_as(in_src_address + (block_h * (self.ho_every - 1) +
                                                                new_looph) * self.dyw * self.channel)
                        mask_address.set_as(mask_address + (block_h * (self.ho_every - 1) +
                                                            new_looph) * self.dyw)

                    self.clean_fp32_multi_repeat(data_vmul_ub_col2img_fp32, dtype_size * 2)
                    self.clean_fp16_multi_repeat(data_vmul_ub_col2img_fp16, dtype_size)
                    self.clean_max_ub(data_max_ub, dtype)
                    self.tik_instance.data_move(data_max_ub,
                                                self.data_input[in_src_address],
                                                constant.SID, in_nburst,
                                                self.dyw, constant.STRIDE_ZERO,
                                                wo_max - self.dyw)

                    with self.tik_instance.for_range(0, self.kernel_h * self.kernel_w) as mask_id:  # 3x3
                        with self.tik_instance.for_range(0, in_nburst) as cycle:  # 7
                            # mask copy gm to ub
                            self.tik_instance.data_move(data_mask_ub[cycle * wo_max],
                                                        self.data_mask[mask_address + cycle * self.dyw +
                                                                       mask_one_window * mask_id],
                                                        constant.SID, 1,
                                                        wo_max // 16, 0, 0)

                        data_vsel_ub = self.tik_instance.Tensor(dtype, (max_mem_size4,),
                                                                name="data_vsel_ub1",
                                                                scope=tik.scope_ubuf)
                        data_vsel_ub_fp32 = self.tik_instance.Tensor("float32",
                                                                     (max_mem_size5,),
                                                                     name="data_vsel_ub_fp321",
                                                                     scope=tik.scope_ubuf)
                        with self.tik_instance.if_scope(v_rep_time_last > 0):
                            with self.tik_instance.for_range(0, v_rep_time_last, thread_num=1) as cycle:  # 56
                                cmpmask = self.tik_instance.mov_tensor_to_cmpmask(
                                    data_mask_ub[cycle * Constant.MASK_MAX])
                                self.tik_instance.vsel(constant.MASK128, 0,
                                                       data_vsel_ub[cycle * Constant.FP16_MAX],
                                                       cmpmask,
                                                       data_max_ub[cycle * Constant.FP16_MAX],
                                                       data_vsel_ub_zero[0],
                                                       constant.REPEAT_TIME_ONCE,
                                                       constant.STRIDE_ONE,
                                                       constant.STRIDE_ONE,
                                                       constant.STRIDE_ONE,
                                                       constant.REPEAT_STRIDE_EIGHT,
                                                       constant.REPEAT_STRIDE_EIGHT,
                                                       constant.REPEAT_STRIDE_EIGHT)

                        # fp16 to fp32
                        with self.tik_instance.if_scope(v_rep_cycle_fp32_last > 0):  # 0
                            with self.tik_instance.for_range(0, v_rep_cycle_fp32_last,
                                                             thread_num=1) as cycle:
                                self.tik_instance.vconv(constant.MASK64, "",
                                                        data_vsel_ub_fp32[
                                                            cycle * Constant.V_MAX_REPEAT * Constant.FP32_MAX],
                                                        data_vsel_ub[cycle * Constant.V_MAX_REPEAT * Constant.FP32_MAX],
                                                        Constant.V_MAX_REPEAT, constant.STRIDE_ONE,
                                                        constant.STRIDE_ONE,
                                                        constant.REPEAT_STRIDE_EIGHT,
                                                        constant.REPEAT_STRIDE_FOUR)
                        with self.tik_instance.if_scope(v_rep_last_fp32_last != 0):  # 112
                            self.tik_instance.vconv(constant.MASK64, "", data_vsel_ub_fp32[
                                v_rep_cycle_fp32_last * Constant.V_MAX_REPEAT * Constant.FP32_MAX],
                                                    data_vsel_ub[v_rep_cycle_fp32_last * Constant.V_MAX_REPEAT *
                                                                 Constant.FP32_MAX],
                                                    v_rep_last_fp32_last, constant.STRIDE_ONE,
                                                    constant.STRIDE_ONE,
                                                    constant.REPEAT_STRIDE_EIGHT,
                                                    constant.REPEAT_STRIDE_FOUR)
                        # col2img
                        fetch_filter_w = mask_id % self.kernel_w
                        fetch_filter_h = mask_id // self.kernel_w
                        left_top_w = 0
                        left_top_h = 0

                        self.tik_instance.col2img(data_vmul_ub_col2img_fp32[0],
                                                  data_vsel_ub_fp32[0],
                                                  (0, 0, 0, 0),
                                                  self.col2img_h_last, self.col2img_w, fetch_filter_w,
                                                  fetch_filter_h, left_top_w, left_top_h,
                                                  self.stride_w, self.stride_h, self.kernel_w, self.kernel_h,
                                                  dialate_w, dialate_h,
                                                  self.ho_max_last * wo_max // 16)

                    with self.tik_instance.if_scope(v_rep_cycle_col_last > 0):  # 0
                        with self.tik_instance.for_range(0, v_rep_cycle_col_last) as cycle:
                            self.tik_instance.vconv(constant.MASK64, "",
                                                    data_vmul_ub_col2img_fp16[
                                                        cycle * Constant.V_MAX_REPEAT * Constant.FP32_MAX],
                                                    data_vmul_ub_col2img_fp32[
                                                        cycle * Constant.V_MAX_REPEAT * Constant.FP32_MAX],
                                                    Constant.V_MAX_REPEAT, constant.STRIDE_ONE,
                                                    constant.STRIDE_ONE,
                                                    constant.REPEAT_STRIDE_FOUR,
                                                    constant.REPEAT_STRIDE_EIGHT)
                    with self.tik_instance.if_scope(v_rep_last_col_last != 0):  # 162
                        self.tik_instance.vconv(constant.MASK64, "",
                                                data_vmul_ub_col2img_fp16[
                                                    v_rep_cycle_col_last * Constant.V_MAX_REPEAT *
                                                    Constant.FP32_MAX],
                                                data_vmul_ub_col2img_fp32[
                                                    v_rep_cycle_col_last * Constant.V_MAX_REPEAT *
                                                    Constant.FP32_MAX],
                                                v_rep_last_col_last, constant.STRIDE_ONE,
                                                constant.STRIDE_ONE,
                                                constant.REPEAT_STRIDE_FOUR,
                                                constant.REPEAT_STRIDE_EIGHT)
                    # move ub to gm
                    output_cuthline = self.tik_instance.Scalar("int32")
                    output_cuthline.set_as(0)
                    src_address = self.tik_instance.Scalar("int32")
                    dst_address = self.tik_instance.Scalar("int32")
                    dst_address.set_as(block_batch * self.dxh * self.dxw * self.channel)

                    with self.tik_instance.if_scope(self.hoverlap == 0):
                        src_address.set_as(self.pad_left * self.channel)
                        output_cuthline.set_as(self.col2img_h_last)
                        dst_address.set_as(dst_address +
                                           (self.ho_count - 1) * self.ho_every * self.stride_h * self.dxw *
                                           self.channel + looph * self.ho_max_last * self.stride_h *
                                           self.dxw * self.channel - self.pad_top * self.dxw * self.channel)
                        with self.tik_instance.if_scope(looph == h_cycle_last - 1):
                            output_cuthline.set_as(self.dxh - self.ho_every *
                                                   (self.ho_count - 1) * self.stride_h -
                                                   looph * self.ho_max_last * self.stride_h +
                                                   self.pad_top)
                    with self.tik_instance.else_scope():  # 1
                        src_address.set_as(self.pad_left * self.channel + self.stride_h * self.col2img_w * self.channel)
                        output_cuthline.set_as((self.ho_max_last - 1) * self.stride_h)
                        with self.tik_instance.if_scope(looph == h_cycle_last - 1):
                            output_cuthline.set_as((self.ho_last - looph * (self.ho_max_last - 1) - 1) * self.stride_h +
                                                   self.kernel_h - self.stride_h - self.pad_bottom + 1)
                        dst_address.set_as(dst_address +
                                           ((block_h * (self.ho_every - 1) +
                                             looph * (self.ho_max_last - 1) + 1) * self.stride_h -
                                            self.pad_top) * self.dxw * self.channel)
                    self.tik_instance.data_move(self.data_output[dst_address],
                                                data_vmul_ub_col2img_fp16[src_address],
                                                constant.SID, output_cuthline, self.offset_w,  # 12 x 112
                                                self.col2img_w - self.offset_w,  # 17
                                                self.dxw - self.offset_w)  # 0
            with self.tik_instance.else_scope():
                data_max_ub = self.tik_instance.Tensor(dtype,
                                                       (max_mem_size0,),
                                                       name="data_max_ub",
                                                       scope=tik.scope_ubuf)  # 2 x 64 x 16
                data_vmul_ub_col2img_fp32 = \
                    self.tik_instance.Tensor("float32",
                                             (max_mem_size1,),  # 129 x 16 x 5
                                             name="data_vmul_ub_col2img_fp32",
                                             scope=tik.scope_ubuf)
                data_vmul_ub_col2img_fp16 = \
                    self.tik_instance.Tensor(dtype,
                                             (max_mem_size2,),  # 129 x 16 x 5
                                             name="data_vmul_ub_col2img_fp16",
                                             scope=tik.scope_ubuf)
                # mask define
                data_mask_ub = self.tik_instance.Tensor("uint16", (max_mem_size3,),  # 2 x 64
                                                        name="data_mask_ub",
                                                        scope=tik.scope_ubuf)
                in_nburst = self.tik_instance.Scalar("int32")
                in_nburst.set_as(self.ho_max_every)
                in_src_address = self.tik_instance.Scalar("int32")
                mask_address = self.tik_instance.Scalar("int32")
                with self.tik_instance.for_range(0, h_cycle_every) as looph:  # 1
                    in_src_address.set_as(block_batch * self.dyh * self.dyw * self.channel)
                    # 1 x 3 x 3 x 56 x 56
                    mask_address.set_as(block_batch * mask_one_window * self.kernel_w * self.kernel_h)
                    with self.tik_instance.if_scope(self.hoverlap == 0):
                        in_src_address.set_as(in_src_address +
                                              (block_h * self.ho_every +
                                               looph * self.ho_max_every) * self.dyw * self.channel)
                        mask_address.set_as(mask_address + (block_h * self.ho_every +
                                                            looph * self.ho_max_every) * self.dyw)
                        with self.tik_instance.if_scope(looph == h_cycle_every - 1):
                            in_nburst.set_as(self.ho_every - looph * self.ho_max_every)
                    with self.tik_instance.else_scope():
                        in_src_address.set_as(in_src_address +
                                              (block_h * (self.ho_every - 1) +
                                               looph * (self.ho_max_every - 1)) * self.dyw * self.channel)
                        mask_address.set_as(mask_address +
                                            (block_h * (self.ho_every - 1) +
                                             looph * (self.ho_max_every - 1)) * self.dyw)
                        with self.tik_instance.if_scope(looph == h_cycle_every - 1):
                            in_nburst.set_as(self.ho_every - looph * (self.ho_max_every - 1))

                    self.clean_fp32_multi_repeat(data_vmul_ub_col2img_fp32, dtype_size * 2)
                    self.clean_fp16_multi_repeat(data_vmul_ub_col2img_fp16, dtype_size)
                    self.clean_max_ub(data_max_ub, dtype)
                    self.tik_instance.data_move(data_max_ub,
                                                self.data_input[in_src_address],
                                                constant.SID, in_nburst,
                                                self.dyw, constant.STRIDE_ZERO,
                                                wo_max - self.dyw)

                    with self.tik_instance.for_range(0, self.kernel_h * self.kernel_w) as mask_id:  # 3 x 3
                        with self.tik_instance.for_range(0, in_nburst) as cycle:  # 2
                            # mask copy gm to ub
                            self.tik_instance.data_move(data_mask_ub[cycle * wo_max],
                                                        self.data_mask[mask_address + cycle * self.dyw +
                                                                       mask_one_window * mask_id],
                                                        constant.SID, 1,
                                                        wo_max // 16, 0, 0)
                        data_vsel_ub = self.tik_instance.Tensor(dtype, (max_mem_size4,),
                                                                name="data_vsel_ub",
                                                                scope=tik.scope_ubuf)
                        data_vsel_ub_fp32 = self.tik_instance.Tensor("float32",
                                                                     (max_mem_size5,),
                                                                     name="data_vsel_ub_fp32",
                                                                     scope=tik.scope_ubuf)
                        with self.tik_instance.if_scope(v_rep_time_every > 0):  # 16
                            with self.tik_instance.for_range(0, v_rep_time_every, thread_num=1) as cycle:
                                cmpmask = self.tik_instance.mov_tensor_to_cmpmask(
                                    data_mask_ub[cycle * Constant.MASK_MAX])
                                self.tik_instance.vsel(constant.MASK128, 0,
                                                       data_vsel_ub[cycle * Constant.FP16_MAX],
                                                       cmpmask,
                                                       data_max_ub[cycle * Constant.FP16_MAX],
                                                       data_vsel_ub_zero[0],
                                                       constant.REPEAT_TIME_ONCE,
                                                       constant.STRIDE_ONE,
                                                       constant.STRIDE_ONE,
                                                       constant.STRIDE_ONE,
                                                       constant.REPEAT_STRIDE_EIGHT,
                                                       constant.REPEAT_STRIDE_EIGHT,
                                                       constant.REPEAT_STRIDE_EIGHT)
                        # fp16 to fp32
                        with self.tik_instance.if_scope(v_rep_cycle_fp32_every > 0):  # 0
                            with self.tik_instance.for_range(0, v_rep_cycle_fp32_every,
                                                             thread_num=1) as cycle:
                                self.tik_instance.vconv(constant.MASK64, "",
                                                        data_vsel_ub_fp32[
                                                            cycle * Constant.V_MAX_REPEAT * Constant.FP32_MAX],
                                                        data_vsel_ub[cycle * Constant.V_MAX_REPEAT * Constant.FP32_MAX],
                                                        Constant.V_MAX_REPEAT, constant.STRIDE_ONE,
                                                        constant.STRIDE_ONE,
                                                        constant.REPEAT_STRIDE_EIGHT,
                                                        constant.REPEAT_STRIDE_FOUR)
                        with self.tik_instance.if_scope(v_rep_last_fp32_every != 0):  # 32
                            self.tik_instance.vconv(constant.MASK64, "",
                                                    data_vsel_ub_fp32[v_rep_cycle_fp32_every *
                                                                      Constant.V_MAX_REPEAT * Constant.FP32_MAX],
                                                    data_vsel_ub[v_rep_cycle_fp32_every * Constant.V_MAX_REPEAT *
                                                                 Constant.FP32_MAX],
                                                    v_rep_last_fp32_every, constant.STRIDE_ONE,
                                                    constant.STRIDE_ONE,
                                                    constant.REPEAT_STRIDE_EIGHT,
                                                    constant.REPEAT_STRIDE_FOUR)
                        # col2img
                        fetch_filter_w = mask_id % self.kernel_w
                        fetch_filter_h = mask_id // self.kernel_w
                        left_top_w = 0
                        left_top_h = 0

                        self.tik_instance.col2img(data_vmul_ub_col2img_fp32[0],
                                                  data_vsel_ub_fp32[0],
                                                  (0, 0, 0, 0),
                                                  self.col2img_h_every, self.col2img_w, fetch_filter_w,
                                                  fetch_filter_h, left_top_w, left_top_h,
                                                  self.stride_w, self.stride_h,
                                                  self.kernel_w, self.kernel_h, dialate_w, dialate_h,
                                                  self.ho_max_every * wo_max // 16)
                    with self.tik_instance.if_scope(v_rep_cycle_col_every > 0):  # 0
                        with self.tik_instance.for_range(0, v_rep_cycle_col_every) as cycle:
                            self.tik_instance.vconv(constant.MASK64, "",
                                                    data_vmul_ub_col2img_fp16[
                                                        cycle * Constant.V_MAX_REPEAT * Constant.FP32_MAX],
                                                    data_vmul_ub_col2img_fp32[
                                                        cycle * Constant.V_MAX_REPEAT * Constant.FP32_MAX],
                                                    Constant.V_MAX_REPEAT, constant.STRIDE_ONE,
                                                    constant.STRIDE_ONE,
                                                    constant.REPEAT_STRIDE_FOUR,
                                                    constant.REPEAT_STRIDE_EIGHT)
                    with self.tik_instance.if_scope(v_rep_last_col_every != 0):  # 162
                        self.tik_instance.vconv(constant.MASK64, "", data_vmul_ub_col2img_fp16[
                            v_rep_cycle_col_every * Constant.V_MAX_REPEAT * Constant.FP32_MAX],
                                                data_vmul_ub_col2img_fp32[
                                                    v_rep_cycle_col_every *
                                                    Constant.V_MAX_REPEAT * Constant.FP32_MAX],
                                                v_rep_last_col_every, constant.STRIDE_ONE,
                                                constant.STRIDE_ONE,
                                                constant.REPEAT_STRIDE_FOUR,
                                                constant.REPEAT_STRIDE_EIGHT)
                    # move ub to gm
                    output_cuthline = self.tik_instance.Scalar("int32")
                    output_cuthline.set_as(0)
                    src_address = self.tik_instance.Scalar("int32")
                    src_address.set_as(self.pad_left * self.channel)
                    dst_address = self.tik_instance.Scalar("int32")
                    dst_address.set_as(block_batch * self.dxh * self.dxw * self.channel)

                    with self.tik_instance.if_scope(self.hoverlap == 0):
                        output_cuthline.set_as(self.col2img_h_every)  # 5
                        dst_address.set_as(dst_address +
                                           ((block_h * self.ho_every +
                                             looph * self.ho_max_every) *
                                            self.stride_h - self.pad_top) * self.dxw * self.channel)
                        with self.tik_instance.if_scope(block_h == 0):
                            with self.tik_instance.if_scope(looph == 0):
                                output_cuthline.set_as(output_cuthline - self.pad_top)
                                src_address.set_as(self.pad_left * self.channel +
                                                   self.pad_top * self.col2img_w * self.channel)
                                dst_address.set_as(block_batch * self.dxh * self.dxw * self.channel)
                        with self.tik_instance.if_scope(looph == h_cycle_every - 1):
                            output_cuthline.set_as(self.ho_every * self.stride_h - self.col2img_h_every *
                                                   (h_cycle_every - 1))
                    with self.tik_instance.else_scope():
                        src_address.set_as(self.pad_left * self.channel + self.stride_h * self.col2img_w * self.channel)
                        output_cuthline.set_as((self.ho_max_every - 1) * self.stride_h)
                        with self.tik_instance.if_scope(looph == h_cycle_every - 1):
                            output_cuthline.set_as((self.ho_every - looph *
                                                    (self.ho_max_every - 1) - 1) * self.stride_h)

                        dst_address.set_as(dst_address +
                                           ((block_h * (self.ho_every - 1) + looph * (self.ho_max_every - 1) -
                                             self.pad_top + 1) * self.stride_h + self.pad_top) *
                                           self.dxw * self.channel)

                        with self.tik_instance.if_scope(block_h == 0):  # 1
                            with self.tik_instance.if_scope(looph == 0):  # 0
                                output_cuthline.set_as(self.ho_max_every * self.stride_h - self.pad_top)
                                src_address.set_as(self.pad_left * self.channel + self.pad_top * self.col2img_w *
                                                   self.channel)
                                dst_address.set_as(block_batch * self.dxh * self.dxw * self.channel)
                    self.tik_instance.data_move(self.data_output[dst_address],
                                                data_vmul_ub_col2img_fp16[src_address],
                                                constant.SID, output_cuthline, self.offset_w,
                                                self.col2img_w - self.offset_w,
                                                self.dxw - self.offset_w)  # 0

    def clean_max_ub(self, data_max_ub, dtype):
        """
        The fun just for clean max ub
        """
        repeat_time = data_max_ub.shape[0] // 128
        data_vsel_scalar = self.tik_instance.Scalar(dtype)
        data_vsel_scalar.set_as(0)
        repeat_time_num = (repeat_time + 254) // 255
        with self.tik_instance.if_scope(repeat_time_num > 255):
            with self.tik_instance.for_range(0, repeat_time_num) as repeat_index:
                with self.tik_instance.if_scope(repeat_index !=
                                                (repeat_time_num - 1)):
                    self.tik_instance.vector_dup(
                        constant.MASK128,
                        data_max_ub[repeat_index * 255 * 128],
                        data_vsel_scalar,
                        255,
                        constant.STRIDE_ONE,
                        constant.REPEAT_STRIDE_EIGHT)
                with self.tik_instance.else_scope():
                    self.tik_instance.vector_dup(
                        constant.MASK128,
                        data_max_ub[repeat_index * 255 * 128],
                        data_vsel_scalar,
                        (repeat_time - repeat_index * 255),
                        constant.STRIDE_ONE,
                        constant.REPEAT_STRIDE_EIGHT)
        with self.tik_instance.else_scope():
            self.tik_instance.vector_dup(constant.MASK128,
                                         data_max_ub,
                                         data_vsel_scalar,
                                         repeat_time,
                                         constant.STRIDE_ONE,
                                         constant.REPEAT_STRIDE_EIGHT)

    def clean_fp16_one_repeat(self, data_vsel_ub_zero, dtype):
        """
        The fun just for clean ub
        """
        data_vsel_scalar = self.tik_instance.Scalar(dtype)
        data_vsel_scalar.set_as(0)
        if data_vsel_ub_zero.shape[0] > 128:
            self.tik_instance.vector_dup(constant.MASK128,
                                         data_vsel_ub_zero[0],
                                         data_vsel_scalar,
                                         (data_vsel_ub_zero.shape[0] + 127) // 128,
                                         constant.STRIDE_ONE,
                                         constant.REPEAT_STRIDE_EIGHT)
        else:
            self.tik_instance.vector_dup(data_vsel_ub_zero.shape[0],
                                         data_vsel_ub_zero[0],
                                         data_vsel_scalar,
                                         constant.REPEAT_TIME_ONCE,
                                         constant.STRIDE_ONE,
                                         constant.REPEAT_STRIDE_EIGHT)

    def clean_fp32_multi_repeat(self, data_vmul_ub_col2img_fp32, dtype_size):
        """
        The fun just for clean ub
        """
        v_rep_clear_time = data_vmul_ub_col2img_fp32.shape[0] * dtype_size // Constant.ONE_REPEAT
        v_rep_clear_cycle = v_rep_clear_time // Constant.V_MAX_REPEAT  # 1
        v_rep_clear_last = v_rep_clear_time % Constant.V_MAX_REPEAT  # 250
        v_res_last = v_rep_clear_cycle * Constant.V_MAX_REPEAT * Constant.FP32_MAX + v_rep_clear_last *\
                     Constant.FP32_MAX
        v_res_time = data_vmul_ub_col2img_fp32.shape[0] - v_res_last
        if v_rep_clear_cycle > 0:
            with self.tik_instance.for_range(0, v_rep_clear_cycle, thread_num=1) as cycle:
                self.tik_instance.vector_dup(constant.MASK64,
                                             data_vmul_ub_col2img_fp32[
                                                 cycle * Constant.V_MAX_REPEAT * Constant.FP32_MAX],
                                             0, Constant.V_MAX_REPEAT, constant.STRIDE_ONE,
                                             constant.REPEAT_STRIDE_EIGHT)
        if v_rep_clear_last != 0:
            self.tik_instance.vector_dup(constant.MASK64,
                                         data_vmul_ub_col2img_fp32[
                                             v_rep_clear_cycle * Constant.V_MAX_REPEAT * Constant.FP32_MAX],
                                         0, v_rep_clear_last, constant.STRIDE_ONE, constant.REPEAT_STRIDE_EIGHT)
        if v_res_time != 0:
            res_repeat_time = v_res_time * dtype_size // constant.BLOCK_SIZE
            self.tik_instance.vector_dup(8, data_vmul_ub_col2img_fp32[v_res_last], 0, res_repeat_time,
                                         constant.STRIDE_ONE, 1)

    def clean_fp16_multi_repeat(self, data_vmul_ub_col2img_fp16, dtype_size):
        """
        The fun just for clean ub
        """
        v_rep_clear_time = data_vmul_ub_col2img_fp16.shape[0] * dtype_size // Constant.ONE_REPEAT
        v_rep_clear_cycle = v_rep_clear_time // Constant.V_MAX_REPEAT  # 1
        v_rep_clear_last = v_rep_clear_time % Constant.V_MAX_REPEAT  # 250
        v_res_last = v_rep_clear_cycle * Constant.V_MAX_REPEAT * Constant.FP16_MAX + v_rep_clear_last *\
                     Constant.FP16_MAX
        v_res_time = data_vmul_ub_col2img_fp16.shape[0] - v_res_last
        mask = 16
        if v_rep_clear_cycle > 0:
            with self.tik_instance.for_range(0, v_rep_clear_cycle, thread_num=1) as cycle:
                self.tik_instance.vector_dup(constant.MASK128,
                                             data_vmul_ub_col2img_fp16[
                                                 cycle * Constant.V_MAX_REPEAT * Constant.FP16_MAX],
                                             0, Constant.V_MAX_REPEAT, constant.STRIDE_ONE,
                                             constant.REPEAT_STRIDE_EIGHT)
        if v_rep_clear_last != 0:
            self.tik_instance.vector_dup(constant.MASK128,
                                         data_vmul_ub_col2img_fp16[
                                             v_rep_clear_cycle * Constant.V_MAX_REPEAT * Constant.FP16_MAX],
                                         0, v_rep_clear_last, constant.STRIDE_ONE, constant.REPEAT_STRIDE_EIGHT)
        if v_res_time != 0:
            res_repeat_time = v_res_time * dtype_size // constant.BLOCK_SIZE
            self.tik_instance.vector_dup(mask, data_vmul_ub_col2img_fp16[v_res_last], 0, res_repeat_time,
                                         constant.STRIDE_ONE, 1)
