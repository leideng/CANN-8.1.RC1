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
max_pool_grad_with_argmax_cut_w_v1
"""
from impl import constant_util_v1 as constant
from impl.dynamic.max_pool_grad_with_argmax_cut_h_v1 import MaxpoolGradBase
from impl.util.platform_adapter import tik


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The Class for Constant
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


# 'pylint: disable=too-many-arguments,useless-super-delegation,super-with-arguments
# 'pylint: disable=too-many-locals,too-many-branches,too-many-statements
class MaxpoolGardObject(MaxpoolGradBase):
    """
    parameter for max_pool_grad_with_pool
    """

    def __init__(self, grad, argmax, input_x, ksize, strides, padding, dilation, ceil_mode):
        """
        init compare and bit pack base parameters
        Parameters
        ----------
        x: input of maxpool, useless for maxpool gard
        grad: input of maxpoolgard or output of maxpool
        argmax:output of maxpool mask or index
        strides: stride , minimum length is 4, just like [1, poolingStrideH, poolingStrideW, 1]
        padding: pad
        Returns
        -------
        None
        """
        super(MaxpoolGardObject, self).__init__(grad, argmax, input_x, ksize, strides, padding, dilation, ceil_mode)

    def tik_instance_cut_nc1_cut_w(self, block_id):
        """
        function for max_pool_grad_with_pool calc for normal shape
        """
        dtype = self.dtype
        dtype_size = self.dtype_size
        nc1 = self.nc1
        stride_h, stride_w = self.strides[1:3]
        stridehw = stride_h * stride_w

        if stridehw == 1:
            max_mem_size0 = 16384
            max_mem_size1 = 16448
            max_mem_size2 = 16512
            max_mem_size3 = 1024
            max_mem_size4 = 16384
            max_mem_size5 = 16384
        elif stridehw < 4:
            max_mem_size0 = 10752
            max_mem_size1 = 21600
            max_mem_size2 = 21664
            max_mem_size3 = 1024
            max_mem_size4 = 10752
            max_mem_size5 = 10752
        else:
            max_mem_size0 = 6144
            max_mem_size1 = 30864
            max_mem_size2 = 30928
            max_mem_size3 = 384
            max_mem_size4 = 6144
            max_mem_size5 = 6144

        wo_max = self.tik_instance.Scalar("int32")
        wo_max.set_as(self.wo_max)

        h_cycle = self.dyh
        w_cycle = self.tik_instance.Scalar("int32")
        with self.tik_instance.if_scope(self.woverlap == 0):
            w_cycle.set_as((self.dyw + wo_max - 1) // wo_max)
        with self.tik_instance.else_scope():
            w_cycle.set_as((self.dyw - 1 + wo_max - 2) // (wo_max - 1))

        mask_one_window = ((self.dyh * self.dyw + 15) // 16 + 1) * 16
        # vector_repeat_time
        v_rep_time = self.ho_max * wo_max * self.channel * dtype_size // Constant.ONE_REPEAT
        v_rep_cycle_fp32 = 2 * v_rep_time // Constant.V_MAX_REPEAT
        # v_rep_last
        v_rep_last_fp32 = 2 * v_rep_time % Constant.V_MAX_REPEAT

        v_rep_time_col = (2 * (self.col2img_w * self.channel * self.col2img_h + 64) * dtype_size) // Constant.ONE_REPEAT
        v_rep_cycle_col = v_rep_time_col // Constant.V_MAX_REPEAT

        v_rep_last_col = v_rep_time_col % Constant.V_MAX_REPEAT

        num_one_c0 = self.dxh * self.dxw * self.channel
        num_one_block = nc1 * num_one_c0

        real_cycle = self.tik_instance.Scalar("int32")
        block_base = self.tik_instance.Scalar("int32")
        block_num = self.tik_instance.Scalar("int32")
        with self.tik_instance.if_scope(block_id < self.block_index):
            real_cycle.set_as(self.block_cycle + 1)
            block_base.set_as(block_id * real_cycle)
        with self.tik_instance.else_scope():
            real_cycle.set_as(self.block_cycle)
            block_base.set_as(self.block_index + block_id * self.block_cycle)
        with self.tik_instance.for_range(0, real_cycle) as cycle_id:
            block_num.set_as(block_base + cycle_id)
            data_vsel_ub_zero = self.tik_instance.Tensor(dtype, (128,), name="data_vsel_ub_zero4",
                                                         scope=tik.scope_ubuf)
            self.tik_instance.data_move(data_vsel_ub_zero[0], self.data_input_origin[0], constant.SID,
                                        constant.DEFAULT_NBURST, constant.DEFAULT_BURST_LEN,
                                        constant.STRIDE_ZERO, constant.STRIDE_ZERO)
            self.clean_fp16_one_repeat(data_vsel_ub_zero, dtype)

            with self.tik_instance.for_range(0, nc1) as loopc1:
                # vector_dup ub every time
                dxh_address_offset = self.tik_instance.Scalar("int32")
                dxh_address_offset.set_as(0)
                data_max_ub = self.tik_instance.Tensor(dtype, (max_mem_size0,),
                                                       name="data_max_ub4", scope=tik.scope_ubuf)
                data_vmul_ub_col2img_fp32 = \
                    self.tik_instance.Tensor("float32", (max_mem_size1,),
                                             name="data_vmul_ub_col2img_fp324", scope=tik.scope_ubuf)
                data_vmul_ub_col2img_fp16 = \
                    self.tik_instance.Tensor(dtype, (max_mem_size2,),
                                             name="data_vmul_ub_col2img_fp164", scope=tik.scope_ubuf)
                # mask define
                data_mask_ub = self.tik_instance.Tensor("uint16", (max_mem_size3,), name="data_mask_ub4",
                                                        scope=tik.scope_ubuf)
                new_looph = self.tik_instance.Scalar("int32")
                new_looph.set_as(0)
                with self.tik_instance.for_range(0, h_cycle) as looph:
                    with self.tik_instance.if_scope(self.hoverlap == 0):
                        new_looph.set_as(looph)
                    with self.tik_instance.else_scope():
                        with self.tik_instance.if_scope(looph != 0):
                            new_looph.set_as(looph - 1)
                    new_loopw = self.tik_instance.Scalar("int32")
                    new_loopw.set_as(0)
                    in_burstlen = self.tik_instance.Scalar("int32")
                    in_burstlen.set_as(wo_max)
                    with self.tik_instance.for_range(0, w_cycle) as loopw:
                        with self.tik_instance.if_scope(self.woverlap == 0):
                            new_loopw.set_as(loopw * wo_max)
                            with self.tik_instance.if_scope(loopw == w_cycle - 1):
                                in_burstlen.set_as(self.dyw - loopw * wo_max)
                        with self.tik_instance.else_scope():
                            with self.tik_instance.if_scope(loopw != 0):
                                new_loopw.set_as(loopw * (wo_max - 1))
                            with self.tik_instance.if_scope(loopw == w_cycle - 1):
                                in_burstlen.set_as(self.dyw - loopw * (wo_max - 1))

                        self.clean_fp32_multi_repeat(data_vmul_ub_col2img_fp32, dtype_size * 2)
                        self.clean_fp16_multi_repeat(data_vmul_ub_col2img_fp16, dtype_size)
                        self.clean_max_ub(data_max_ub, dtype)

                        self.tik_instance.data_move(
                            data_max_ub,
                            self.data_input[(block_num * nc1 * self.dyh + loopc1 * self.dyh + new_looph) * self.dyw *
                                            self.channel + new_loopw * self.channel],
                            constant.SID, self.ho_max, in_burstlen, self.dyw - in_burstlen, wo_max - in_burstlen)

                        with self.tik_instance.for_range(0, self.kernel_h * self.kernel_w) as mask_id:
                            with self.tik_instance.for_range(0, self.ho_max) as cycle:
                                # mask copy gm to ub
                                self.tik_instance.data_move(
                                    data_mask_ub[cycle * wo_max],
                                    self.data_mask[block_num * nc1 * mask_one_window * self.kernel_w * self.kernel_h +
                                                   loopc1 * mask_one_window * self.kernel_w * self.kernel_h +
                                                   (
                                                           new_looph + cycle) * self.dyw + mask_one_window * mask_id
                                                   + new_loopw],
                                    constant.SID, 1, (in_burstlen + 15) // 16, 0, 0)
                            data_vsel_ub = self.tik_instance.Tensor(
                                dtype, (max_mem_size4,), name="data_vsel_ub4", scope=tik.scope_ubuf)
                            data_vsel_ub_fp32 = \
                                self.tik_instance.Tensor("float32", (max_mem_size5,),
                                                         name="data_vsel_ub_fp324", scope=tik.scope_ubuf)
                            with self.tik_instance.if_scope(v_rep_time > 0):
                                with self.tik_instance.for_range(0, v_rep_time, thread_num=1) as cycle:
                                    cmpmask = self.tik_instance.mov_tensor_to_cmpmask(
                                        data_mask_ub[cycle * Constant.MASK_MAX])
                                    self.tik_instance.vsel(
                                        constant.MASK128, 0, data_vsel_ub[cycle * Constant.FP16_MAX],
                                        cmpmask, data_max_ub[cycle * Constant.FP16_MAX], data_vsel_ub_zero[0],
                                        constant.REPEAT_TIME_ONCE, constant.STRIDE_ONE, constant.STRIDE_ONE,
                                        constant.STRIDE_ONE, constant.REPEAT_STRIDE_EIGHT,
                                        constant.REPEAT_STRIDE_EIGHT, constant.REPEAT_STRIDE_EIGHT)
                            # fp16 to fp32
                            with self.tik_instance.if_scope(v_rep_cycle_fp32 > 0):
                                with self.tik_instance.for_range(0, v_rep_cycle_fp32) as cycle:
                                    self.tik_instance.vconv(
                                        constant.MASK64, "",
                                        data_vsel_ub_fp32[cycle * Constant.V_MAX_REPEAT * Constant.FP32_MAX],
                                        data_vsel_ub[cycle * Constant.V_MAX_REPEAT * Constant.FP32_MAX],
                                        Constant.V_MAX_REPEAT, constant.STRIDE_ONE, constant.STRIDE_ONE,
                                        constant.REPEAT_STRIDE_EIGHT, constant.REPEAT_STRIDE_FOUR)
                            with self.tik_instance.if_scope(v_rep_last_fp32 != 0):
                                self.tik_instance.vconv(
                                    constant.MASK64, "",
                                    data_vsel_ub_fp32[v_rep_cycle_fp32 * Constant.V_MAX_REPEAT * Constant.FP32_MAX],
                                    data_vsel_ub[v_rep_cycle_fp32 * Constant.V_MAX_REPEAT * Constant.FP32_MAX],
                                    v_rep_last_fp32, constant.STRIDE_ONE, constant.STRIDE_ONE,
                                    constant.REPEAT_STRIDE_EIGHT, constant.REPEAT_STRIDE_FOUR)
                            # col2img
                            fetch_filter_w = mask_id % self.kernel_w
                            fetch_filter_h = mask_id // self.kernel_w
                            left_top_w = 0
                            left_top_h = 0
                            self.tik_instance.col2img(
                                data_vmul_ub_col2img_fp32[0], data_vsel_ub_fp32[0], (0, 0, 0, 0),
                                self.col2img_h, self.col2img_w, fetch_filter_w, fetch_filter_h, left_top_w, left_top_h,
                                self.stride_w, self.stride_h, self.kernel_w, self.kernel_h, 1, 1,
                                self.ho_max * wo_max // 16)
                        with self.tik_instance.if_scope(v_rep_cycle_col > 0):
                            with self.tik_instance.for_range(0, v_rep_cycle_col) as cycle:
                                self.tik_instance.vconv(
                                    constant.MASK64, "",
                                    data_vmul_ub_col2img_fp16[cycle * Constant.V_MAX_REPEAT * Constant.FP32_MAX],
                                    data_vmul_ub_col2img_fp32[cycle * Constant.V_MAX_REPEAT * Constant.FP32_MAX],
                                    Constant.V_MAX_REPEAT, constant.STRIDE_ONE, constant.STRIDE_ONE,
                                    constant.REPEAT_STRIDE_FOUR, constant.REPEAT_STRIDE_EIGHT)
                        with self.tik_instance.if_scope(v_rep_last_col != 0):
                            self.tik_instance.vconv(
                                constant.MASK64, "",
                                data_vmul_ub_col2img_fp16[v_rep_cycle_col * Constant.V_MAX_REPEAT * Constant.FP32_MAX],
                                data_vmul_ub_col2img_fp32[v_rep_cycle_col * Constant.V_MAX_REPEAT * Constant.FP32_MAX],
                                v_rep_last_col, constant.STRIDE_ONE, constant.STRIDE_ONE,
                                constant.REPEAT_STRIDE_FOUR, constant.REPEAT_STRIDE_EIGHT)

                        nburst = self.tik_instance.Scalar("int32")
                        nburst.set_as(self.stride_h)
                        burst_len = self.tik_instance.Scalar("int32")
                        burst_len.set_as(1)
                        src_stride = self.tik_instance.Scalar("int32")
                        src_stride.set_as(0)
                        dst_stride = self.tik_instance.Scalar("int32")
                        dst_stride.set_as(0)
                        src_address = self.tik_instance.Scalar("int32")
                        src_address.set_as(0)
                        dst_address = self.tik_instance.Scalar("int32")
                        dst_address.set_as(block_num * num_one_block + loopc1 * num_one_c0)
                        with self.tik_instance.if_scope(self.hoverlap != 0):
                            src_address.set_as(self.stride_h * self.col2img_w * self.channel)
                        with self.tik_instance.if_scope(looph == 0):
                            nburst.set_as(self.stride_h - self.pad_top)
                            src_address.set_as(self.pad_top * self.col2img_w * self.channel)
                        with self.tik_instance.else_scope():
                            dst_address.set_as(
                                dst_address + (looph * self.stride_h - self.pad_top) * self.dxw * self.channel)
                            with self.tik_instance.if_scope(looph == h_cycle - 1):
                                nburst.set_as(self.dxh - looph * self.stride_h + self.pad_top)

                        with self.tik_instance.if_scope(self.woverlap == 0):
                            with self.tik_instance.if_scope(loopw == 0):
                                burst_len.set_as(self.col2img_w - self.pad_left)
                                src_address.set_as(src_address + self.pad_left * self.channel)
                                with self.tik_instance.if_scope(loopw == w_cycle - 1):
                                    burst_len.set_as(self.dxw)
                            with self.tik_instance.else_scope():
                                burst_len.set_as(self.col2img_w)
                                dst_address.set_as(
                                    dst_address + (loopw * self.col2img_w - self.pad_left) * self.channel)
                                with self.tik_instance.if_scope(loopw == w_cycle - 1):
                                    burst_len.set_as(self.dxw - self.col2img_w * loopw + self.pad_left)
                        with self.tik_instance.else_scope():
                            with self.tik_instance.if_scope(loopw == 0):
                                burst_len.set_as(self.stride_w * wo_max - self.pad_left)
                                src_address.set_as(src_address + self.pad_left * self.channel)
                                with self.tik_instance.if_scope(loopw == w_cycle - 1):
                                    burst_len.set_as(self.dxw)
                            with self.tik_instance.else_scope():
                                burst_len.set_as(self.stride_w * (wo_max - 1))
                                src_address.set_as(src_address + self.stride_w * self.channel)
                                dst_address.set_as(
                                    dst_address + ((loopw - 1) * self.stride_w * (wo_max - 1) +
                                                   self.stride_w * wo_max - self.pad_left) * self.channel)
                                with self.tik_instance.if_scope(loopw == w_cycle - 1):
                                    burst_len.set_as(self.dxw - self.stride_w * wo_max - (w_cycle - 2) *
                                                     self.stride_w * (wo_max - 1) + self.pad_left)
                        src_stride.set_as(self.col2img_w - burst_len)
                        dst_stride.set_as(self.dxw - burst_len)
                        # move ub to gm
                        self.tik_instance.data_move(self.data_output[dst_address],
                                                    data_vmul_ub_col2img_fp16[src_address],
                                                    constant.SID, nburst, burst_len, src_stride, dst_stride)

    def tik_instance_cut_nc1h_cut_w(self, block_id):
        """
        function for max_pool_grad_with_pool calc for normal shape
        """
        dtype = self.dtype
        dtype_size = self.dtype_size
        dilate_h, dilate_w = self.dilation[1:3]
        stride_h, stride_w = self.strides[1:3]
        stridehw = stride_h * stride_w

        if stridehw == 1:
            max_mem_size0 = 16384
            max_mem_size1 = 16448
            max_mem_size2 = 16512
            max_mem_size3 = 1024
            max_mem_size4 = 16384
            max_mem_size5 = 16384
        elif stridehw < 4:
            max_mem_size0 = 10752
            max_mem_size1 = 21600
            max_mem_size2 = 21664
            max_mem_size3 = 1024
            max_mem_size4 = 10752
            max_mem_size5 = 10752
        else:
            max_mem_size0 = 7168
            max_mem_size1 = 32832
            max_mem_size2 = 32896
            max_mem_size3 = 512
            max_mem_size4 = 7168
            max_mem_size5 = 7168

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

        w_cycle = self.tik_instance.Scalar("int32")
        with self.tik_instance.if_scope(self.woverlap == 0):
            w_cycle.set_as((self.dyw + wo_max - 1) // wo_max)
        with self.tik_instance.else_scope():
            w_cycle.set_as((self.dyw - 1 + wo_max - 2) // (wo_max - 1))

        mask_one_window = ((self.dyh * self.dyw + 15) // 16 + 1) * 16
        # vector_repeat_time
        v_rep_time_last = self.ho_max_last * wo_max * self.channel * dtype_size // Constant.ONE_REPEAT
        v_rep_cycle_fp32_last = 2 * v_rep_time_last // Constant.V_MAX_REPEAT
        # v_rep_last
        v_rep_last_fp32_last = 2 * v_rep_time_last % Constant.V_MAX_REPEAT

        v_rep_time_col_last = (2 * (
                self.col2img_w * self.channel * self.col2img_h_last + 64) * dtype_size) // Constant.ONE_REPEAT
        v_rep_cycle_col_last = v_rep_time_col_last // Constant.V_MAX_REPEAT
        v_rep_last_col_last = v_rep_time_col_last % Constant.V_MAX_REPEAT

        # vector_repeat_time
        v_rep_time_every = self.ho_max_every * wo_max * self.channel * dtype_size // Constant.ONE_REPEAT
        v_rep_cycle_fp32_every = 2 * v_rep_time_every // Constant.V_MAX_REPEAT
        # v_rep_last
        v_rep_last_fp32_every = 2 * v_rep_time_every % Constant.V_MAX_REPEAT

        v_rep_time_col_every = (2 * (
                self.col2img_w * self.channel * self.col2img_h_every + 64) * dtype_size) // Constant.ONE_REPEAT
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
        with self.tik_instance.for_range(0, real_cycle) as cycle_id:
            block_num.set_as(block_base + cycle_id)
            data_vsel_ub_zero = self.tik_instance.Tensor(
                dtype, (128,), name="data_vsel_ub_zero5", scope=tik.scope_ubuf)
            self.tik_instance.data_move(data_vsel_ub_zero[0], self.data_input_origin[0], constant.SID,
                                        constant.DEFAULT_NBURST, constant.DEFAULT_BURST_LEN,
                                        constant.STRIDE_ZERO, constant.STRIDE_ZERO)
            self.clean_fp16_one_repeat(data_vsel_ub_zero, dtype)
            block_batch = self.tik_instance.Scalar("int32")
            block_batch.set_as(block_num // self.ho_count)
            block_h = self.tik_instance.Scalar("int32")
            block_h.set_as(block_num % self.ho_count)
            with self.tik_instance.if_scope(block_h == self.ho_count - 1):
                data_max_ub = self.tik_instance.Tensor(
                    dtype, (max_mem_size0,), name="data_max_ub5", scope=tik.scope_ubuf)
                data_vmul_ub_col2img_fp32 = \
                    self.tik_instance.Tensor("float32", (max_mem_size1,),
                                             name="data_vmul_ub_col2img_fp325", scope=tik.scope_ubuf)
                data_vmul_ub_col2img_fp16 = \
                    self.tik_instance.Tensor(dtype, (max_mem_size2,),
                                             name="data_vmul_ub_col2img_fp165", scope=tik.scope_ubuf)
                # mask define
                data_mask_ub = self.tik_instance.Tensor("uint16", (max_mem_size3,),
                                                        name="data_mask_ub5", scope=tik.scope_ubuf)
                new_looph = self.tik_instance.Scalar("int32")
                new_looph.set_as(0)
                in_nburst = self.tik_instance.Scalar("int32")
                in_nburst.set_as(self.ho_max_last)
                in_src_address = self.tik_instance.Scalar("int32")
                mask_address = self.tik_instance.Scalar("int32")
                with self.tik_instance.for_range(0, h_cycle_last) as looph:
                    in_src_address.set_as(block_batch * self.dyh * self.dyw * self.channel)
                    mask_address.set_as(block_batch * mask_one_window * self.kernel_w * self.kernel_h)
                    with self.tik_instance.if_scope(self.hoverlap == 0):
                        new_looph.set_as(looph * self.ho_max_last)
                        with self.tik_instance.if_scope(looph == h_cycle_last - 1):
                            in_nburst.set_as(self.ho_last - looph * self.ho_max_last)
                        in_src_address.set_as(
                            in_src_address + (block_h * self.ho_every + new_looph) * self.dyw * self.channel)
                        mask_address.set_as(mask_address + (block_h * self.ho_every + new_looph) * self.dyw)
                    with self.tik_instance.else_scope():
                        with self.tik_instance.if_scope(looph != 0):
                            new_looph.set_as(looph * (self.ho_max_last - 1))
                        with self.tik_instance.if_scope(looph == h_cycle_last - 1):
                            in_nburst.set_as(self.ho_last - looph * (self.ho_max_last - 1))
                        in_src_address.set_as(
                            in_src_address + (block_h * (self.ho_every - 1) + new_looph) * self.dyw * self.channel)
                        mask_address.set_as(mask_address + (block_h * (self.ho_every - 1) + new_looph) * self.dyw)
                    new_loopw = self.tik_instance.Scalar("int32")
                    new_loopw.set_as(0)
                    in_burstlen = self.tik_instance.Scalar("int32")
                    in_burstlen.set_as(wo_max)
                    with self.tik_instance.for_range(0, w_cycle) as loopw:
                        with self.tik_instance.if_scope(self.woverlap == 0):
                            new_loopw.set_as(loopw * wo_max)
                            with self.tik_instance.if_scope(loopw == w_cycle - 1):
                                in_burstlen.set_as(self.dyw - loopw * wo_max)
                        with self.tik_instance.else_scope():
                            with self.tik_instance.if_scope(loopw != 0):
                                new_loopw.set_as(loopw * (wo_max - 1))
                            with self.tik_instance.if_scope(loopw == w_cycle - 1):
                                in_burstlen.set_as(self.dyw - loopw * (wo_max - 1))

                        self.clean_fp32_multi_repeat(data_vmul_ub_col2img_fp32, dtype_size * 2)
                        self.clean_fp16_multi_repeat(data_vmul_ub_col2img_fp16, dtype_size)
                        self.clean_max_ub(data_max_ub, dtype)

                        self.tik_instance.data_move(data_max_ub,
                                                    self.data_input[in_src_address + new_loopw * self.channel],
                                                    constant.SID, in_nburst, in_burstlen, self.dyw - in_burstlen,
                                                    wo_max - in_burstlen)

                        with self.tik_instance.for_range(0, self.kernel_h * self.kernel_w) as mask_id:
                            with self.tik_instance.for_range(0, in_nburst) as cycle:
                                # mask copy gm to ub
                                self.tik_instance.data_move(
                                    data_mask_ub[cycle * wo_max],
                                    self.data_mask[
                                        mask_address + cycle * self.dyw + mask_one_window * mask_id + new_loopw],
                                    constant.SID, 1, (in_burstlen + 15) // 16, 0, 0)
                            data_vsel_ub = self.tik_instance.Tensor(
                                dtype, (max_mem_size4,),
                                name="data_vsel_ub5", scope=tik.scope_ubuf)
                            data_vsel_ub_fp32 = \
                                self.tik_instance.Tensor(
                                    "float32", (max_mem_size5,),
                                    name="data_vsel_ub_fp325", scope=tik.scope_ubuf)
                            with self.tik_instance.if_scope(v_rep_time_last > 0):
                                with self.tik_instance.for_range(0, v_rep_time_last, thread_num=1) as cycle:
                                    cmpmask = self.tik_instance.mov_tensor_to_cmpmask(
                                        data_mask_ub[cycle * Constant.MASK_MAX])
                                    self.tik_instance.vsel(
                                        constant.MASK128, 0, data_vsel_ub[cycle * Constant.FP16_MAX],
                                        cmpmask, data_max_ub[cycle * Constant.FP16_MAX],
                                        data_vsel_ub_zero[0], constant.REPEAT_TIME_ONCE, constant.STRIDE_ONE,
                                        constant.STRIDE_ONE, constant.STRIDE_ONE, constant.REPEAT_STRIDE_EIGHT,
                                        constant.REPEAT_STRIDE_EIGHT, constant.REPEAT_STRIDE_EIGHT)
                            # fp16 to fp32
                            with self.tik_instance.if_scope(v_rep_cycle_fp32_last > 0):
                                with self.tik_instance.for_range(0, v_rep_cycle_fp32_last, thread_num=1) as cycle:
                                    self.tik_instance.vconv(
                                        constant.MASK64, "",
                                        data_vsel_ub_fp32[cycle * Constant.V_MAX_REPEAT * Constant.FP32_MAX],
                                        data_vsel_ub[cycle * Constant.V_MAX_REPEAT * Constant.FP32_MAX],
                                        Constant.V_MAX_REPEAT, constant.STRIDE_ONE, constant.STRIDE_ONE,
                                        constant.REPEAT_STRIDE_EIGHT, constant.REPEAT_STRIDE_FOUR)
                            with self.tik_instance.if_scope(v_rep_last_fp32_last != 0):
                                self.tik_instance.vconv(
                                    constant.MASK64, "",
                                    data_vsel_ub_fp32[
                                        v_rep_cycle_fp32_last * Constant.V_MAX_REPEAT * Constant.FP32_MAX],
                                    data_vsel_ub[v_rep_cycle_fp32_last * Constant.V_MAX_REPEAT * Constant.FP32_MAX],
                                    v_rep_last_fp32_last, constant.STRIDE_ONE, constant.STRIDE_ONE,
                                    constant.REPEAT_STRIDE_EIGHT, constant.REPEAT_STRIDE_FOUR)
                            # col2img
                            fetch_filter_w = mask_id % self.kernel_w
                            fetch_filter_h = mask_id // self.kernel_w
                            left_top_w = 0
                            left_top_h = 0
                            self.tik_instance.col2img(data_vmul_ub_col2img_fp32[0], data_vsel_ub_fp32[0],
                                                      (0, 0, 0, 0), self.col2img_h_last, self.col2img_w, fetch_filter_w,
                                                      fetch_filter_h, left_top_w, left_top_h,
                                                      self.stride_w, self.stride_h, self.kernel_w, self.kernel_h,
                                                      dilate_w, dilate_h,
                                                      self.ho_max_last * wo_max // 16)
                        with self.tik_instance.if_scope(v_rep_cycle_col_last > 0):
                            with self.tik_instance.for_range(0, v_rep_cycle_col_last) as cycle:
                                self.tik_instance.vconv(
                                    constant.MASK64, "",
                                    data_vmul_ub_col2img_fp16[cycle * Constant.V_MAX_REPEAT * Constant.FP32_MAX],
                                    data_vmul_ub_col2img_fp32[cycle * Constant.V_MAX_REPEAT * Constant.FP32_MAX],
                                    Constant.V_MAX_REPEAT, constant.STRIDE_ONE, constant.STRIDE_ONE,
                                    constant.REPEAT_STRIDE_FOUR, constant.REPEAT_STRIDE_EIGHT)
                        with self.tik_instance.if_scope(v_rep_last_col_last != 0):
                            self.tik_instance.vconv(
                                constant.MASK64, "",
                                data_vmul_ub_col2img_fp16[
                                    v_rep_cycle_col_last * Constant.V_MAX_REPEAT * Constant.FP32_MAX],
                                data_vmul_ub_col2img_fp32[
                                    v_rep_cycle_col_last * Constant.V_MAX_REPEAT * Constant.FP32_MAX],
                                v_rep_last_col_last, constant.STRIDE_ONE, constant.STRIDE_ONE,
                                constant.REPEAT_STRIDE_FOUR, constant.REPEAT_STRIDE_EIGHT)
                        # move ub to gm
                        output_cuthline = self.tik_instance.Scalar("int32")
                        output_cuthline.set_as(0)
                        src_address = self.tik_instance.Scalar("int32")
                        src_address.set_as(0)
                        dst_address = self.tik_instance.Scalar("int32")
                        dst_address.set_as(block_batch * self.dxh * self.dxw * self.channel)
                        burst_len = self.tik_instance.Scalar("int32")
                        burst_len.set_as(1)
                        with self.tik_instance.if_scope(self.hoverlap == 0):
                            output_cuthline.set_as(self.col2img_h_last)
                            dst_address.set_as(dst_address + (self.ho_count - 1) * self.ho_every * self.stride_h *
                                               self.dxw * self.channel + looph * self.ho_max_last *
                                               self.stride_h * self.dxw * self.channel -
                                               self.pad_top * self.dxw * self.channel)
                            with self.tik_instance.if_scope(looph == h_cycle_last - 1):
                                output_cuthline.set_as(self.dxh - self.ho_every * (self.ho_count - 1) * self.stride_h -
                                                       looph * self.ho_max_every * self.stride_h +
                                                       self.pad_top)
                        with self.tik_instance.else_scope():
                            src_address.set_as(self.stride_h * self.col2img_w * self.channel)
                            output_cuthline.set_as((self.ho_max_last - 1) * self.stride_h)
                            with self.tik_instance.if_scope(looph == h_cycle_last - 1):
                                output_cuthline.set_as(
                                    (self.ho_last - looph * (self.ho_max_last - 1) - 1) * self.stride_h +
                                    self.kernel_h - self.stride_h - self.pad_bottom)
                            dst_address.set_as(dst_address +
                                               ((block_h * (self.ho_every - 1) + looph * (self.ho_max_last - 1) + 1) *
                                                self.stride_h - self.pad_top) * self.dxw * self.channel)
                        with self.tik_instance.if_scope(self.woverlap == 0):
                            with self.tik_instance.if_scope(loopw == 0):
                                src_address.set_as(src_address + self.pad_left * self.channel)
                                burst_len.set_as(self.col2img_w - self.pad_left)
                                with self.tik_instance.if_scope(loopw == w_cycle - 1):
                                    burst_len.set_as(self.dxw)
                            with self.tik_instance.else_scope():
                                burst_len.set_as(self.col2img_w)
                                dst_address.set_as(
                                    dst_address + (loopw * self.col2img_w - self.pad_left) * self.channel)
                                with self.tik_instance.if_scope(loopw == w_cycle - 1):
                                    burst_len.set_as(self.dxw - self.col2img_w * loopw + self.pad_left)
                        with self.tik_instance.else_scope():
                            with self.tik_instance.if_scope(loopw == 0):
                                burst_len.set_as(self.stride_w * wo_max - self.pad_left)
                                src_address.set_as(src_address + self.pad_left * self.channel)
                                with self.tik_instance.if_scope(loopw == w_cycle - 1):
                                    burst_len.set_as(self.dxw)
                            with self.tik_instance.else_scope():
                                burst_len.set_as(self.stride_w * (wo_max - 1))
                                src_address.set_as(src_address + self.stride_w * self.channel)
                                dst_address.set_as(dst_address + ((loopw - 1) * self.stride_w *
                                                                  (wo_max - 1) + self.stride_w *
                                                                  wo_max - self.pad_left) * self.channel)
                                with self.tik_instance.if_scope(loopw == w_cycle - 1):
                                    burst_len.set_as(self.dxw - self.stride_w * wo_max - (w_cycle - 2) *
                                                     self.stride_w * (wo_max - 1) + self.pad_left)
                        self.tik_instance.data_move(self.data_output[dst_address],
                                                    data_vmul_ub_col2img_fp16[src_address],
                                                    constant.SID, output_cuthline, burst_len,
                                                    self.col2img_w - burst_len,
                                                    self.dxw - burst_len)
            with self.tik_instance.else_scope():
                data_max_ub = self.tik_instance.Tensor(dtype, (max_mem_size0,),
                                                       name="data_max_ub5", scope=tik.scope_ubuf)
                data_vmul_ub_col2img_fp32 = \
                    self.tik_instance.Tensor("float32",
                                             (max_mem_size1,),
                                             name="data_vmul_ub_col2img_fp325",
                                             scope=tik.scope_ubuf)
                data_vmul_ub_col2img_fp16 = \
                    self.tik_instance.Tensor(dtype,
                                             (max_mem_size2,),
                                             name="data_vmul_ub_col2img_fp165",
                                             scope=tik.scope_ubuf)
                # mask define
                data_mask_ub = self.tik_instance.Tensor("uint16", (max_mem_size3,),
                                                        name="data_mask_ub5",
                                                        scope=tik.scope_ubuf)
                in_nburst = self.tik_instance.Scalar("int32")
                in_nburst.set_as(self.ho_max_every)
                in_src_address = self.tik_instance.Scalar("int32")
                mask_address = self.tik_instance.Scalar("int32")
                with self.tik_instance.for_range(0, h_cycle_every) as looph:
                    in_src_address.set_as(block_batch * self.dyh * self.dyw * self.channel)
                    mask_address.set_as(block_batch * mask_one_window * self.kernel_w * self.kernel_h)
                    with self.tik_instance.if_scope(self.hoverlap == 0):
                        in_src_address.set_as(in_src_address + (block_h * self.ho_every + looph * self.ho_max_every) *
                                              self.dyw * self.channel)
                        mask_address.set_as(
                            mask_address + (block_h * self.ho_every + looph * self.ho_max_every) * self.dyw)
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
                    new_loopw = self.tik_instance.Scalar("int32")
                    new_loopw.set_as(0)
                    in_burstlen = self.tik_instance.Scalar("int32")
                    in_burstlen.set_as(wo_max)
                    with self.tik_instance.for_range(0, w_cycle) as loopw:
                        with self.tik_instance.if_scope(self.woverlap == 0):
                            new_loopw.set_as(loopw * wo_max)
                            with self.tik_instance.if_scope(loopw == w_cycle - 1):
                                in_burstlen.set_as(self.dyw - loopw * wo_max)
                        with self.tik_instance.else_scope():
                            with self.tik_instance.if_scope(loopw != 0):
                                new_loopw.set_as(loopw * (wo_max - 1))
                            with self.tik_instance.if_scope(loopw == w_cycle - 1):
                                in_burstlen.set_as(self.dyw - loopw * (wo_max - 1))
                        self.clean_fp32_multi_repeat(data_vmul_ub_col2img_fp32, dtype_size * 2)
                        self.clean_fp16_multi_repeat(data_vmul_ub_col2img_fp16, dtype_size)
                        self.clean_max_ub(data_max_ub, dtype)

                        self.tik_instance.data_move(
                            data_max_ub,
                            self.data_input[in_src_address + new_loopw * self.channel],
                            constant.SID, in_nburst, in_burstlen, self.dyw - in_burstlen, wo_max - in_burstlen)

                        with self.tik_instance.for_range(0, self.kernel_h * self.kernel_w) as mask_id:
                            with self.tik_instance.for_range(0, in_nburst) as cycle:
                                # mask copy gm to ub
                                self.tik_instance.data_move(
                                    data_mask_ub[cycle * wo_max],
                                    self.data_mask[
                                        mask_address + cycle * self.dyw + mask_one_window * mask_id + new_loopw],
                                    constant.SID, 1, (in_burstlen + 15) // 16, 0, 0)
                            data_vsel_ub = self.tik_instance.Tensor(
                                dtype, (max_mem_size4,), name="data_vsel_ub5",
                                scope=tik.scope_ubuf)
                            data_vsel_ub_fp32 = \
                                self.tik_instance.Tensor(
                                    "float32", (max_mem_size5,),
                                    name="data_vsel_ub_fp325", scope=tik.scope_ubuf)
                            with self.tik_instance.if_scope(v_rep_time_every > 0):
                                with self.tik_instance.for_range(0, v_rep_time_every, thread_num=1) as cycle:
                                    cmpmask = self.tik_instance.mov_tensor_to_cmpmask(
                                        data_mask_ub[cycle * Constant.MASK_MAX])
                                    self.tik_instance.vsel(constant.MASK128, 0, data_vsel_ub[cycle * Constant.FP16_MAX],
                                                           cmpmask, data_max_ub[cycle * Constant.FP16_MAX],
                                                           data_vsel_ub_zero[0], constant.REPEAT_TIME_ONCE,
                                                           constant.STRIDE_ONE, constant.STRIDE_ONE,
                                                           constant.STRIDE_ONE, constant.REPEAT_STRIDE_EIGHT,
                                                           constant.REPEAT_STRIDE_EIGHT,
                                                           constant.REPEAT_STRIDE_EIGHT)
                            # fp16 to fp32
                            with self.tik_instance.if_scope(v_rep_cycle_fp32_every > 0):
                                with self.tik_instance.for_range(0, v_rep_cycle_fp32_every,
                                                                 thread_num=1) as cycle:
                                    self.tik_instance.vconv(
                                        constant.MASK64, "",
                                        data_vsel_ub_fp32[cycle * Constant.V_MAX_REPEAT * Constant.FP32_MAX],
                                        data_vsel_ub[cycle * Constant.V_MAX_REPEAT * Constant.FP32_MAX],
                                        Constant.V_MAX_REPEAT, constant.STRIDE_ONE, constant.STRIDE_ONE,
                                        constant.REPEAT_STRIDE_EIGHT, constant.REPEAT_STRIDE_FOUR)
                            with self.tik_instance.if_scope(v_rep_last_fp32_every != 0):
                                self.tik_instance.vconv(
                                    constant.MASK64, "",
                                    data_vsel_ub_fp32[
                                        v_rep_cycle_fp32_every * Constant.V_MAX_REPEAT * Constant.FP32_MAX],
                                    data_vsel_ub[v_rep_cycle_fp32_every * Constant.V_MAX_REPEAT * Constant.FP32_MAX],
                                    v_rep_last_fp32_every, constant.STRIDE_ONE, constant.STRIDE_ONE,
                                    constant.REPEAT_STRIDE_EIGHT, constant.REPEAT_STRIDE_FOUR)
                            # col2img
                            fetch_filter_w = mask_id % self.kernel_w
                            fetch_filter_h = mask_id // self.kernel_w
                            left_top_w = 0
                            left_top_h = 0
                            self.tik_instance.col2img(data_vmul_ub_col2img_fp32[0],
                                                      data_vsel_ub_fp32[0],
                                                      (0, 0, 0, 0),
                                                      self.col2img_h_every, self.col2img_w,
                                                      fetch_filter_w, fetch_filter_h,
                                                      left_top_w, left_top_h,
                                                      self.stride_w, self.stride_h,
                                                      self.kernel_w, self.kernel_h, dilate_w, dilate_h,
                                                      self.ho_max_every * wo_max // 16)
                        with self.tik_instance.if_scope(v_rep_cycle_col_every > 0):
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
                        with self.tik_instance.if_scope(v_rep_last_col_every != 0):
                            self.tik_instance.vconv(constant.MASK64, "",
                                                    data_vmul_ub_col2img_fp16[
                                                        v_rep_cycle_col_every *
                                                        Constant.V_MAX_REPEAT * Constant.FP32_MAX],
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
                        src_address.set_as(0)
                        dst_address = self.tik_instance.Scalar("int32")
                        dst_address.set_as(block_batch * self.dxh * self.dxw * self.channel)
                        burst_len = self.tik_instance.Scalar("int32")
                        burst_len.set_as(1)
                        with self.tik_instance.if_scope(self.hoverlap == 0):
                            output_cuthline.set_as(self.col2img_h_every)
                            dst_address.set_as(dst_address + ((block_h * self.ho_every + looph * self.ho_max_every) *
                                                              self.stride_h - self.pad_top) * self.dxw * self.channel)
                            with self.tik_instance.if_scope(block_h == 0):
                                with self.tik_instance.if_scope(looph == 0):
                                    output_cuthline.set_as(output_cuthline - self.pad_top)
                                    src_address.set_as(self.pad_top * self.col2img_w * self.channel)
                                    dst_address.set_as(block_batch * self.dxh * self.dxw * self.channel)
                        with self.tik_instance.else_scope():
                            src_address.set_as(self.stride_h * self.col2img_w * self.channel)
                            output_cuthline.set_as((self.ho_max_every - 1) * self.stride_h)
                            with self.tik_instance.if_scope(looph == h_cycle_every - 1):
                                output_cuthline.set_as(
                                    (self.ho_every - looph * (self.ho_max_every - 1) - 1) * self.stride_h)
                            dst_address.set_as(
                                dst_address + ((block_h * (self.ho_every - 1) + (looph - 1) * (self.ho_max_every - 1) +
                                                self.ho_max_every) * self.stride_h - self.pad_top) *
                                self.dxw * self.channel)
                            with self.tik_instance.if_scope(block_h == 0):
                                with self.tik_instance.if_scope(looph == 0):
                                    output_cuthline.set_as(self.ho_max_every * self.stride_h - self.pad_top)
                                    src_address.set_as(self.pad_top * self.col2img_w * self.channel)
                                    dst_address.set_as(block_batch * self.dxh * self.dxw * self.channel)
                        with self.tik_instance.if_scope(self.woverlap == 0):
                            with self.tik_instance.if_scope(loopw == 0):
                                burst_len.set_as(self.col2img_w - self.pad_left)
                                src_address.set_as(src_address + self.pad_left * self.channel)
                                with self.tik_instance.if_scope(loopw == w_cycle - 1):
                                    burst_len.set_as(self.dxw)
                            with self.tik_instance.else_scope():
                                burst_len.set_as(self.col2img_w)
                                dst_address.set_as(
                                    dst_address + (loopw * self.col2img_w - self.pad_left) * self.channel)
                                with self.tik_instance.if_scope(loopw == w_cycle - 1):
                                    burst_len.set_as(self.dxw - self.col2img_w * loopw + self.pad_left)
                        with self.tik_instance.else_scope():
                            with self.tik_instance.if_scope(loopw == 0):
                                burst_len.set_as(self.stride_w * wo_max - self.pad_left)
                                src_address.set_as(src_address + self.pad_left * self.channel)
                                with self.tik_instance.if_scope(loopw == w_cycle - 1):
                                    burst_len.set_as(self.dxw)
                            with self.tik_instance.else_scope():
                                burst_len.set_as(self.stride_w * (wo_max - 1))
                                src_address.set_as(src_address + self.stride_w * self.channel)
                                dst_address.set_as(
                                    dst_address + ((loopw - 1) * self.stride_w * (wo_max - 1) +
                                                   self.stride_w * wo_max - self.pad_left) * self.channel)
                                with self.tik_instance.if_scope(loopw == w_cycle - 1):
                                    burst_len.set_as(self.dxw - self.stride_w * wo_max - (w_cycle - 2) *
                                                     self.stride_w * (wo_max - 1) + self.pad_left)
                        self.tik_instance.data_move(self.data_output[dst_address],
                                                    data_vmul_ub_col2img_fp16[src_address],
                                                    constant.SID, output_cuthline, burst_len,
                                                    self.col2img_w - burst_len, self.dxw - burst_len)
