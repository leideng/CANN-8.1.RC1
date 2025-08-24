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
sign_bits_pack
"""
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context


class Constant:
    """
    Constant Num
    """
    # reserved ub size
    RESERVED_UB_SIZE = 8 * 1024
    # double buffer
    THREAD_NUM = 1
    # bytes of one block
    BLOCK_SIZE = 32
    # tiling param num
    TILING_ARG_NUM = 32
    # max int32
    MAX_INT32 = 2 ** 31 - 1

    def _init__(self):
        pass


class SignBitsPack():
    """
    Function: use to finish SignBitPack main functions
    """

    def __init__(self, input_x, y, size, kernel_name):
        self.init_tik_inst()
        self.kernel_name = kernel_name
        self.dtype = input_x.get("dtype").lower()
        self.loop_num = None
        self.loop_left = None
        self.repeat_time = None
        self.repeat_left = None
        self.act_core_num = None
        self.core_ele = None
        self.one_core_ele = None
        self.last_core_ele = None
        self.core_overlap = None
        self.ub_a = None
        self.ub_b = None
        self.dst_ub = None
        self.pack_rate = 8
        self.u8_size = tbe_platform.get_bit_len("uint8") // 8
        self.u8_each_block = Constant.BLOCK_SIZE // self.u8_size
        if self.dtype == "float16":
            self.mask = 128
            self.block = 16
        elif self.dtype == "float32":
            self.mask = 64
            self.block = 8
        self.shape_padding = 256
        self.repeat_max = 255

        self.dtype_size = tbe_platform.get_bit_len(self.dtype) // 8
        self.num_each_block = Constant.BLOCK_SIZE // self.dtype_size
        # 256 numbers need to be packed at the same time
        self.align_unit = 256
        self.core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.core_num_var = self.tik_inst.Scalar(dtype="int64", name="core_num_var", init_value=self.core_num)
        self.ub_ele = (tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) -
                       Constant.RESERVED_UB_SIZE) // self.dtype_size // self.align_unit
        self.ub_split = 6
        self.max_ele = self.ub_ele // self.ub_split
        if self.max_ele * self.align_unit % self.pack_rate == 0:
            self.dst_ub_shape = self.max_ele * self.align_unit // self.pack_rate
        else:
            self.dst_ub_shape = self.max_ele * self.align_unit // self.pack_rate + 1
        if self.max_ele * self.align_unit % self.pack_rate == 0:
            self.out_div8 = self.max_ele * self.align_unit // self.pack_rate
        else:
            self.out_div8 = self.max_ele * self.align_unit // self.pack_rate + 1

        self.init_gm_tensor()

        self.shape_ceil8 = None
        self.neg1_to_fill = None
        self.shape_div8 = None
        self.shape_div_size = None
        self.overlap = None
        self.one_core_ele = None
        self.act_core_num = None
        self.last_core_ele = None
        self.one_core_loop_num = None
        self.one_core_loop_left = None
        self.last_core_loop_num = None
        self.last_core_loop_left = None
        self.last_core_number = None
        self.last_core_number_fill8 = None
        self.last_core_input_move_para = None
        self.size = None
        self.tiling_ub = None

    def tiling_args(self):
        """Get runtime params from tiling
        """
        self.shape_ceil8 = self.tik_inst.Scalar("int64", name="shape_ceil8")
        self.shape_ceil8.set_as(self.tiling_ub[0])
        self.neg1_to_fill = self.tik_inst.Scalar("int64", name="neg1_to_fill")
        self.neg1_to_fill.set_as(self.tiling_ub[1])
        self.shape_div8 = self.tik_inst.Scalar("int64", name="shape_div8")
        self.shape_div8.set_as(self.tiling_ub[2])
        self.shape_div_size = self.tik_inst.Scalar("int64", name="shape_div_size")
        self.shape_div_size.set_as(self.tiling_ub[3])
        self.overlap = self.tik_inst.Scalar("int64", name="overlap")
        self.overlap.set_as(self.tiling_ub[4])
        self.one_core_ele = self.tik_inst.Scalar("int64", name="one_core_ele")
        self.one_core_ele.set_as(self.tiling_ub[5])
        self.act_core_num = self.tik_inst.Scalar("int64", name="act_core_num")
        self.act_core_num.set_as(self.tiling_ub[6])
        self.last_core_ele = self.tik_inst.Scalar("int64", name="last_core_ele")
        self.last_core_ele.set_as(self.tiling_ub[7])
        self.one_core_loop_num = self.tik_inst.Scalar("int64", name="one_core_loop_num")
        self.one_core_loop_num.set_as(self.tiling_ub[8])
        self.one_core_loop_left = self.tik_inst.Scalar("int64", name="one_core_loop_left")
        self.one_core_loop_left.set_as(self.tiling_ub[9])
        self.last_core_loop_num = self.tik_inst.Scalar("int64", name="last_core_loop_num")
        self.last_core_loop_num.set_as(self.tiling_ub[10])
        self.last_core_loop_left = self.tik_inst.Scalar("int64", name="last_core_loop_left")
        self.last_core_loop_left.set_as(self.tiling_ub[11])
        self.last_core_number = self.tik_inst.Scalar("int64", name="last_core_number")
        self.last_core_number.set_as(self.tiling_ub[12])
        self.last_core_number_fill8 = self.tik_inst.Scalar("int64", name="last_core_number_fill8")
        self.last_core_number_fill8.set_as(self.tiling_ub[13])
        self.last_core_input_move_para = self.tik_inst.Scalar("int64", name="last_core_input_move_para")
        self.last_core_input_move_para.set_as(self.tiling_ub[14])
        self.size = self.tik_inst.Scalar("int64", name="size")
        self.size.set_as(self.tiling_ub[15])
        self.core_num_var.set_as(self.tiling_ub[16])

    def init_tik_inst(self):
        """init_tik_inst
        """
        self.tik_inst = tik.Tik()

    def init_gm_tensor(self):
        """init_gm_tensor
        """
        self.tiling_gm = self.tik_inst.Tensor("int64", (Constant.TILING_ARG_NUM,), name="tiling_gm",
                                              scope=tik.scope_gm)
        self.src_gm = self.tik_inst.Tensor(self.dtype, (Constant.MAX_INT32,), name="src_gm", scope=tik.scope_gm)
        self.dst_gm = self.tik_inst.Tensor("uint8", (Constant.MAX_INT32,), name="dst_gm", scope=tik.scope_gm)

    def init_ub_tensor(self):
        """init_ub_tensor
        """
        self.ub_a = self.tik_inst.Tensor(self.dtype, (self.max_ele * self.align_unit,),
                                         name="ub_a", scope=tik.scope_ubuf)
        self.ub_b = self.tik_inst.Tensor(self.dtype, (self.max_ele * self.align_unit,),
                                         name="ub_b", scope=tik.scope_ubuf)
        self.dst_ub = self.tik_inst.Tensor("uint8", (self.dst_ub_shape,), name="dst_ub", scope=tik.scope_ubuf)

    def init_ub_scalar(self):
        """init_ub_scalar
        """
        self.core_ele = self.tik_inst.Scalar("int64", name="core_ele")
        self.loop_num = self.tik_inst.Scalar("int64", name="loop_num")
        self.loop_left = self.tik_inst.Scalar("int64", name="loop_left")
        self.repeat_time = self.tik_inst.Scalar("int64", name="repeat_time")
        self.repeat_left = self.tik_inst.Scalar("int64", name="repeat_left")
        self.core_overlap = self.tik_inst.Scalar("int64", name="core_overlap")

    def move_in_gm_data(self, offset, ub_ele, overlap, core_idx, last_round):
        """
        :param offset: the offset for gm
        :param ub_ele: the data length
        :param overlap: the overlap for 32B
        :return: None
        """
        with self.tik_inst.if_scope(overlap > 0):
            with self.tik_inst.if_scope(ub_ele - 1 > 0):
                self.tik_inst.data_move(self.ub_a,
                                        self.src_gm[offset:],
                                        0, 1, (ub_ele - 1) * self.align_unit // self.block, 0, 0)

            align_offset = offset + (ub_ele - 1) * self.align_unit - overlap
            ub_offset = (ub_ele - 1) * self.align_unit
            self.tik_inst.data_move(self.ub_a[ub_offset:],
                                    self.src_gm[align_offset:],
                                    0, 1, 1 * self.align_unit // self.block, 0, 0)

            with self.tik_inst.if_scope(core_idx == self.act_core_num - 1):
                begin = self.tik_inst.Scalar("int64", "begin", init_value=0)
                end = self.tik_inst.Scalar("int64", "end", init_value=0)
                begin.set_as(ub_ele * self.align_unit - self.neg1_to_fill)
                end.set_as(ub_ele * self.align_unit)

                with self.tik_inst.for_range(begin, end) as fill_idx:
                    self.ub_a[fill_idx].set_as(-1.0)

        with self.tik_inst.else_scope():
            with self.tik_inst.if_scope(ub_ele != 0):
                self.tik_inst.data_move(self.ub_a,
                                        self.src_gm[offset:],
                                        0, 1, ub_ele * self.align_unit // self.block, 0, 0)
                with self.tik_inst.if_scope(tik.all(core_idx == self.act_core_num - 1, last_round)):
                    begin = self.tik_inst.Scalar("int64", "begin", init_value=0)
                    end = self.tik_inst.Scalar("int64", "end", init_value=0)
                    begin.set_as(ub_ele * self.align_unit - self.neg1_to_fill)
                    end.set_as(ub_ele * self.align_unit)
                    with self.tik_inst.for_range(begin, end) as fill_idx_1:
                        self.ub_a[fill_idx_1].set_as(-1.0)
            with self.tik_inst.else_scope():
                self.tik_inst.data_move(self.ub_a,
                                        self.src_gm[offset:],
                                        0, 1, self.last_core_input_move_para, 0, 0)
                with self.tik_inst.for_range(self.last_core_number, self.last_core_number_fill8) as fill_idx:
                    self.ub_a[fill_idx].set_as(-1.0)

    def move_out_ub_data(self, offset, ub_ele, overlap):
        """
        :param offset: the offset for gm
        :param ub_ele: the data length
        :param overlap: the overlap for 32B
        :return: None
        """
        dst_ub = self.dst_ub
        with self.tik_inst.if_scope(overlap > 0):
            with self.tik_inst.if_scope(ub_ele - 1 > 0):
                self.tik_inst.data_move(self.dst_gm[offset:], self.dst_ub,
                                        0, 1, (ub_ele - 1) * self.align_unit //
                                        self.u8_each_block // self.pack_rate, 0, 0)
            align_offset = offset + (ub_ele - 1) * self.align_unit // self.pack_rate - overlap // self.pack_rate
            ub_offset = (ub_ele - 1) * self.align_unit // self.pack_rate

            self.tik_inst.data_move(self.dst_gm[align_offset:],
                                    self.dst_ub[ub_offset:],
                                    0, 1, 1 * self.align_unit // self.u8_each_block // self.pack_rate, 0, 0)

        with self.tik_inst.else_scope():
            with self.tik_inst.if_scope(ub_ele != 0):
                self.tik_inst.data_move(self.dst_gm[offset:],
                                        self.dst_ub,
                                        0, 1, ub_ele * self.align_unit // self.u8_each_block // self.pack_rate, 0, 0)
            with self.tik_inst.else_scope():
                self.tik_inst.data_move(self.dst_gm[offset:],
                                        self.dst_ub,
                                        0, 1, 1, 0, 0)

    def pack_bit_compute(self, repeat_time, repeat_left):
        offset = self.repeat_time * self.mask
        out_offset = self.repeat_time * self.mask // self.pack_rate
        with self.tik_inst.if_scope(repeat_time > 0):
            input_x = self.ub_a
            is_le = self.dst_ub
            input_y = self.ub_b
            loop_num = self.repeat_time // self.repeat_max
            loop_left = self.repeat_time % self.repeat_max
            with self.tik_inst.for_range(0, loop_num) as loop_i:
                rep_offset = loop_i * self.repeat_max * self.mask
                rep_out_offset = rep_offset // self.pack_rate
                self.tik_inst.vector_dup(self.mask, input_y[rep_offset:], 0, self.repeat_max, 1, 8)
                self.tik_inst.vec_cmpv_ge(is_le[rep_out_offset:], input_x[rep_offset:],
                                          input_y[rep_offset:], self.repeat_max, 8, 8)
            with self.tik_inst.if_scope(loop_left > 0):
                rep_offset = loop_num * self.repeat_max * self.mask
                rep_out_offset = rep_offset // self.pack_rate
                self.tik_inst.vector_dup(self.mask, input_y[rep_offset:], 0, loop_left, 1, 8)
                self.tik_inst.vec_cmpv_ge(is_le[rep_out_offset:], input_x[rep_offset:],
                                          input_y[rep_offset:], loop_left, 8, 8)

        with self.tik_inst.if_scope(repeat_left > 0):
            input_x = self.ub_a[offset:]
            is_le = self.dst_ub[out_offset:]
            input_y = self.ub_b
            self.tik_inst.vector_dup(self.mask, input_y, 0, 1, 1, 8)
            self.tik_inst.vec_cmpv_ge(is_le, input_x, input_y, 1, 8, 8)

    def calculation_process(self, core_idx, loop_num, loop_left, overlap):
        """
        :param core_idx: the core index
        :param loop_num: the loop num
        :param loop_left: the loop left
        :param overlap: the overlap
        :return: None
        """
        # base input output offset
        base_offset = core_idx * self.one_core_ele * self.align_unit
        base_offset_out = core_idx * self.one_core_ele * self.align_unit // self.pack_rate
        with self.tik_inst.for_range(0, loop_num, thread_num=Constant.THREAD_NUM) as cyc_idx:
            self.repeat_time.set_as(self.max_ele * self.align_unit // self.mask)
            self.repeat_left.set_as(self.max_ele * self.align_unit % self.mask)
            offset = base_offset + cyc_idx * self.max_ele * self.align_unit
            offset_out = base_offset_out + cyc_idx * self.max_ele * self.align_unit // self.pack_rate
            with self.tik_inst.if_scope(tik.all(loop_left == 0, cyc_idx == loop_num - 1)):
                self.move_in_gm_data(offset, self.max_ele, overlap, core_idx, True)
            with self.tik_inst.else_scope():
                self.move_in_gm_data(offset, self.max_ele, 0, core_idx, False)
            self.pack_bit_compute(self.repeat_time, self.repeat_left)
            with self.tik_inst.if_scope(tik.all(loop_left == 0, cyc_idx == loop_num - 1)):
                self.move_out_ub_data(offset_out, self.max_ele, overlap)
            with self.tik_inst.else_scope():
                self.move_out_ub_data(offset_out, self.max_ele, 0)

        with self.tik_inst.if_scope(loop_left > 0):
            self.repeat_time.set_as(loop_left * self.align_unit // self.mask)
            self.repeat_left.set_as(loop_left * self.align_unit % self.mask)
            offset = base_offset + loop_num * self.max_ele * self.align_unit
            offset_out = base_offset_out + loop_num * self.max_ele * self.align_unit // self.pack_rate
            with self.tik_inst.if_scope(tik.all(loop_left == 1, self.act_core_num == 1)):
                self.move_in_gm_data(offset, 0, 0, core_idx, True)
            with self.tik_inst.else_scope():
                self.move_in_gm_data(offset, loop_left, overlap, core_idx, True)
            self.pack_bit_compute(self.repeat_time, self.repeat_left)
            with self.tik_inst.if_scope(tik.all(loop_left == 1, 1 == self.act_core_num)):
                self.move_out_ub_data(offset_out, 0, 0)
            with self.tik_inst.else_scope():
                self.move_out_ub_data(offset_out, loop_left, overlap)

    def pack_compute_tiling(self):
        """pack_compute_tiling
        """
        with self.tik_inst.for_range(0, self.core_num_var, block_num=self.core_num_var) as core_idx:
            self.tiling_ub = self.tik_inst.Tensor("int64", (Constant.TILING_ARG_NUM,),
                                                  name="tiling_ub",
                                                  scope=tik.scope_ubuf)
            self.tik_inst.data_move(self.tiling_ub, self.tiling_gm, 0, 1, 4, 0, 0)
            self.tiling_args()
            with self.tik_inst.if_scope(core_idx < self.act_core_num):
                self.init_ub_scalar()
                self.init_ub_tensor()
                with self.tik_inst.if_scope(core_idx == self.act_core_num - 1):
                    self.core_overlap.set_as(self.overlap)
                with self.tik_inst.else_scope():
                    self.core_overlap.set_as(0)
                with self.tik_inst.if_scope(core_idx < self.act_core_num - 1):
                    self.core_ele.set_as(self.one_core_ele)
                    self.loop_num.set_as(self.one_core_loop_num)
                    self.loop_left.set_as(self.one_core_loop_left)
                with self.tik_inst.if_scope(core_idx == self.act_core_num - 1):
                    self.core_ele.set_as(self.last_core_ele)
                    self.loop_num.set_as(self.last_core_loop_num)
                    self.loop_left.set_as(self.last_core_loop_left)
                self.calculation_process(core_idx, self.loop_num, self.loop_left, self.core_overlap)

    def pack_bits_operator(self):
        """tik_inst_function
        """
        self.pack_compute_tiling()
        opt_config = {"out_of_bound_sync_check": True, "enable_const_fold": True}
        tbe_context.get_context().add_compile_info(
            "vars", {
                "pack_rate": self.pack_rate,
                "core_num": self.core_num,
                "align_unit": self.align_unit,
                "max_ele": self.max_ele,
                "block": self.block
            })
        self.tik_inst.BuildCCE(kernel_name=self.kernel_name,
                               inputs=[self.src_gm],
                               outputs=[self.dst_gm],
                               flowtable=[self.tiling_gm],
                               config=opt_config)


def _check_param(input_x):
    """
    check paramters, if one is invalid, then raise error
    -----------------------
    :param input_x: tensor of numbers to be compressed
    "return: None
    """
    dtype = input_x.get("dtype").lower()
    
    if dtype != "float32" and dtype != "float16":
        raise RuntimeError("dtype is neither float32 nor float16.")


@register_operator("SignBitsPack")
def sign_bits_pack(x, y, size, kernel_name="sign_bits_pack"):
    """
    implementation of sign_bits_pack and return the tik instance
    ----------------------------------------------------------------
    :param size: output first dimension
    :param x: dict of shape and dtype of input numbers to be compressed
    :param y: dict of shape and dtype of packed output
    :param kernel_name: the kernel's name
    :return: tik instance
    """
    _check_param(x)
    obj = SignBitsPack(x, y, size, kernel_name)
    obj.pack_bits_operator()

    return obj.tik_inst
