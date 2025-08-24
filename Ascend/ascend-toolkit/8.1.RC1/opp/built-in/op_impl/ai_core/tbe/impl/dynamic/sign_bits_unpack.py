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
sign_bits_unpack
"""
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from tbe.common.platform import get_bit_len


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
    # float32 dtype
    DT_FLOAT = 0
    # float32 dtype
    DT_FLOAT16 = 1

    def __init__(self):
        pass


# 'pylint: disable=too-many-lines,invalid-name,too-many-arguments,consider-using-in
# 'pylint: disable=too-many-branches,too-many-instance-attributes,too-many-locals
# 'pylint: disable=too-many-statements,no-self-use,too-few-public-methods
# 'pylint: disable=unused-argument
class SignBitsUnpack():
    """
    Function: use to finish SignBitsUnpack main functions
    """

    def __init__(self, x, y, size, dtype, kernel_name):
        """
        init SignBitsUnpack parameters
        ---------------------------------------------------------------
        :param x: dict of shape and dtype for input
        :param y: dict of shape and dtype for output
        :param size: dim of output
        :param kernel_name: the kernel's name
        :return: None
        """
        self.init_tik_inst()
        self.dtype = y.get("dtype").lower()
        self.kernel_name = kernel_name
        self.num_pack_rate = 8
        self.per_mask_u8 = 16
        self.mask_f16 = 128
        if self.dtype == "float32":
            self.mask = 64
            self.pack_rate = 32
            self.tensor_dtype = "float32"
        elif self.dtype == "float16":
            self.mask = 128
            self.pack_rate = 16
            self.tensor_dtype = "float16"

        self.dtype_size = get_bit_len(self.tensor_dtype) // 8
        self.num_each_block = Constant.BLOCK_SIZE // self.dtype_size

        self.u8_size = get_bit_len("uint8") // 8
        self.u8_each_block = Constant.BLOCK_SIZE // self.u8_size

        self.core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)

        self.align_unit = self.pack_rate * self.num_each_block
        self.ub_ele = (tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) -
                       Constant.RESERVED_UB_SIZE) // self.dtype_size // self.align_unit
        self.ub_spilt = 4

        self.act_core_num = None
        self.one_core_ele = None
        self.last_core_ele = None
        self.one_core_loop_num = None
        self.one_core_loop_left = None
        self.last_core_loop_num = None
        self.last_core_loop_left = None
        self.overlap = None
        self.max_ele = None
        self.align_block = None
        self.shape = None
        self.x_ub = None
        self.f16_ub = None
        self.y_ub = None
        self.tiling_ub = None
        self.nega_tensor = None
        self.one_tensor = None
        self.core_ele = None
        self.loop_num = None
        self.loop_left = None
        self.repeat_time = None
        self.repeat_left = None
        self.core_overlap = None
        self.dim = None
        self.core_num_var = None

        self.init_gm_tensor()

    def init_tik_inst(self):
        """init_tik_inst
        """
        self.tik_inst = tik.Tik()

    def tiling_args(self):
        """tiling_param_caculate
        """
        self.act_core_num = self.tik_inst.Scalar("int64", name="act_core_num")
        self.one_core_ele = self.tik_inst.Scalar("int64", name="one_core_ele")
        self.last_core_ele = self.tik_inst.Scalar("int64", name="last_core_ele")
        self.one_core_loop_num = self.tik_inst.Scalar("int64", name="one_core_loop_num")
        self.one_core_loop_left = self.tik_inst.Scalar("int64", name="one_core_loop_left")
        self.last_core_loop_num = self.tik_inst.Scalar("int64", name="last_core_loop_num")
        self.last_core_loop_left = self.tik_inst.Scalar("int64", name="last_core_loop_left")
        self.overlap = self.tik_inst.Scalar("int64", name="overlap")
        self.max_ele = self.tik_inst.Scalar("int64", name="max_ele")
        self.align_block = self.tik_inst.Scalar("int64", name="align_block")
        self.shape = self.tik_inst.Scalar("int64", name="shape")
        self.dim = self.tik_inst.Scalar("int64", name="dim")
        self.core_num_var = self.tik_inst.Scalar("int64", name="core_num_var")
        self.tiling_ub = self.tik_inst.Tensor("int64", (Constant.TILING_ARG_NUM,),
                                              name="tiling_ub",
                                              scope=tik.scope_ubuf)
        self.tik_inst.data_move(self.tiling_ub, self.tiling_gm, 0, 1, 4, 0, 0)
        self.act_core_num.set_as(self.tiling_ub[0])
        self.one_core_ele.set_as(self.tiling_ub[1])
        self.last_core_ele.set_as(self.tiling_ub[2])
        self.one_core_loop_num.set_as(self.tiling_ub[3])
        self.one_core_loop_left.set_as(self.tiling_ub[4])
        self.last_core_loop_num.set_as(self.tiling_ub[5])
        self.last_core_loop_left.set_as(self.tiling_ub[6])
        self.overlap.set_as(self.tiling_ub[7])
        self.max_ele.set_as(self.tiling_ub[8])
        self.align_block.set_as(self.tiling_ub[9])
        self.shape.set_as(self.tiling_ub[10])
        self.dim.set_as(self.tiling_ub[11])
        self.core_num_var.set_as(self.tiling_ub[12])

    def init_gm_tensor(self):
        """init_gm_tensor
        """
        self.tiling_gm = self.tik_inst.Tensor("int64", (Constant.TILING_ARG_NUM,), name="tiling_gm",
                                              scope=tik.scope_gm)
        self.x_gm = self.tik_inst.Tensor("uint8", (Constant.MAX_INT32,), name="x_gm", scope=tik.scope_gm)
        self.y_gm = self.tik_inst.Tensor(self.tensor_dtype, (Constant.MAX_INT32,), name="y_gm", scope=tik.scope_gm)

    def init_ub_tensor(self):
        """init_ub_tensor
        """
        self.x_ub = self.tik_inst.Tensor("uint8", (self.max_ele * self.align_unit // self.num_pack_rate,),
                                         name="x_ub", scope=tik.scope_ubuf)

        self.f16_ub = self.tik_inst.Tensor("float16", (self.max_ele * self.align_unit,),
                                           name="f16_ub", scope=tik.scope_ubuf)

        self.y_ub = self.tik_inst.Tensor(self.tensor_dtype, (self.max_ele * self.align_unit,),
                                         name="y_ub", scope=tik.scope_ubuf)

        self.nega_tensor = self.tik_inst.Tensor("float16", (self.mask_f16,),
                                                name="nega_tensor", scope=tik.scope_ubuf)

        self.one_tensor = self.tik_inst.Tensor("float16", (self.mask_f16,),
                                               name="one_tensor", scope=tik.scope_ubuf)

        self.tik_inst.vector_dup(self.mask_f16, self.nega_tensor, -1, 1, 1, 8)
        self.tik_inst.vector_dup(self.mask_f16, self.one_tensor, 1, 1, 1, 8)

    def init_ub_scalar(self):
        """init_ub_tensor
        """
        self.core_ele = self.tik_inst.Scalar("int64", name="core_ele")
        self.loop_num = self.tik_inst.Scalar("int64", name="loop_num")
        self.loop_left = self.tik_inst.Scalar("int64", name="loop_left")
        self.repeat_time = self.tik_inst.Scalar("int64", name="repeat_time")
        self.repeat_left = self.tik_inst.Scalar("int64", name="repeat_left")
        self.core_overlap = self.tik_inst.Scalar("int64", name="core_overlap")

    def move_in_gm_data(self, offset, data_len, overlap):
        """
        :param offset: the offset for gm
        :param data_len: the data lenght
        :param overlap: the overlap for 32B
        :return: None
        """

        with self.tik_inst.if_scope(overlap > 0):
            with self.tik_inst.if_scope(data_len - 1 > 0):
                block_len = (data_len - 1) * self.align_unit // self.num_pack_rate // self.u8_each_block
                self.tik_inst.data_move(self.x_ub, self.x_gm[offset:], 0, 1, block_len, 0, 0)

            align_offset = offset + (
                        data_len - 1) * self.align_unit // self.num_pack_rate - overlap // self.num_pack_rate
            ub_offset = (data_len - 1) * self.align_unit // self.num_pack_rate
            self.tik_inst.data_move(self.x_ub[ub_offset:], self.x_gm[align_offset:], 0, 1,
                                    self.align_unit // self.num_pack_rate // self.u8_each_block, 0, 0)

        with self.tik_inst.else_scope():
            with self.tik_inst.if_scope(self.shape >= self.align_unit):
                block_len = data_len * self.align_unit // self.num_pack_rate // self.u8_each_block
                self.tik_inst.data_move(self.x_ub, self.x_gm[offset:], 0, 1, block_len, 0, 0)

            with self.tik_inst.else_scope():
                self.tik_inst.data_move(self.x_ub, self.x_gm[offset:], 0, 1, 1, 0, 0)

    def move_out_ub_data(self, offset, data_len, overlap):
        """
        :param offset: the offset for gm
        :param data_len: the data lenght
        :param overlap: the overlap for 32B
        :return: None
        """
        with self.tik_inst.if_scope(overlap > 0):
            with self.tik_inst.if_scope(data_len - 1 > 0):
                block_len = (data_len - 1) * self.align_unit // self.num_each_block
                self.tik_inst.data_move(self.y_gm[offset:], self.y_ub, 0, 1, block_len, 0, 0)

            align_offset = offset + (data_len - 1) * self.align_unit - overlap
            ub_offset = (data_len - 1) * self.align_unit
            self.tik_inst.data_move(self.y_gm[align_offset:], self.y_ub[ub_offset:], 0, 1,
                                    self.align_unit // self.num_each_block, 0, 0)

        with self.tik_inst.else_scope():
            with self.tik_inst.if_scope(self.shape >= self.align_unit):
                block_len = data_len * self.align_unit // self.num_each_block
                self.tik_inst.data_move(self.y_gm[offset:], self.y_ub, 0, 1, block_len, 0, 0)

            with self.tik_inst.else_scope():
                self.tik_inst.data_move(self.y_gm[offset:], self.y_ub, 0, 1, self.align_block, 0, 0)

    def unpack_bit_compute(self, repeat_time, repeat_left, offset):
        """
        :param repeat_time: int, the repeat time for tik instruction
        :param repeat_left: int, the repeat left for tik instrucion
        :param offset: int, the repeat offset for tik instruction
        :return: None
        """
        if self.dtype == "float16":
            dst_ub = self.y_ub
        else:
            dst_ub = self.f16_ub

        with self.tik_inst.for_range(0, repeat_time) as cmp_idx:
            cmpmask = self.tik_inst.mov_tensor_to_cmpmask(self.x_ub[cmp_idx * self.per_mask_u8])
            self.tik_inst.vsel(self.mask_f16, 0,
                               dst_ub[cmp_idx * self.mask_f16],
                               cmpmask, self.one_tensor, self.nega_tensor,
                               1, 1, 1, 1, 8, 8, 8)

        if self.dtype == "float32":
            self.tik_inst.vconv(self.mask, 'none', self.y_ub, dst_ub,
                                repeat_time * 2, 1, 1, 8, 4)

    def calculation_process(self, core_idx, loop_num, loop_left, overlap):
        """
        :param core_idx: the core index
        :param loop_num: the loop num
        :param loop_left: the loop left
        :param overlap: the overlap
        :return: None
        """
        base_offset = core_idx * self.one_core_ele * self.align_unit
        with self.tik_inst.for_range(0, loop_num, thread_num=Constant.THREAD_NUM) as cyc_idx:
            self.repeat_time.set_as(self.max_ele * self.align_unit // self.mask_f16)
            self.repeat_left.set_as(self.max_ele * self.align_unit % self.mask_f16)
            offset = base_offset + cyc_idx * self.max_ele * self.align_unit

            self.move_in_gm_data(offset // self.num_pack_rate, self.max_ele, 0)
            self.unpack_bit_compute(self.repeat_time, self.max_ele, self.repeat_time * self.mask_f16)
            self.move_out_ub_data(offset, self.max_ele, 0)

        with self.tik_inst.if_scope(loop_left > 0):
            self.repeat_time.set_as(loop_left * self.align_unit // self.mask_f16)
            self.repeat_left.set_as(loop_left * self.align_unit % self.mask_f16)
            offset = base_offset + loop_num * self.max_ele * self.align_unit
            self.move_in_gm_data(offset // self.num_pack_rate, loop_left, overlap)
            self.unpack_bit_compute(self.repeat_time, self.repeat_left, self.repeat_time * self.mask_f16)
            self.move_out_ub_data(offset, loop_left, overlap)

    def unpack_compute_tiling(self):
        """unpack_compute_tiling
        """
        self.tiling_args()
        with self.tik_inst.for_range(0, self.core_num_var, block_num=self.core_num_var) as core_idx:
            with self.tik_inst.if_scope(core_idx < self.act_core_num):
                self.init_ub_scalar()
                self.init_ub_tensor()
                with self.tik_inst.if_scope(core_idx < self.act_core_num - 1):
                    self.core_overlap.set_as(0)
                    self.core_ele.set_as(self.one_core_ele)
                    self.loop_num.set_as(self.one_core_loop_num)
                    self.loop_left.set_as(self.one_core_loop_left)
                with self.tik_inst.if_scope(core_idx == self.act_core_num - 1):
                    self.core_overlap.set_as(self.overlap)
                    self.core_ele.set_as(self.last_core_ele)
                    self.loop_num.set_as(self.last_core_loop_num)
                    self.loop_left.set_as(self.last_core_loop_left)
                self.calculation_process(core_idx, self.loop_num, self.loop_left, self.core_overlap)

    def tik_inst_function(self):
        """tik_inst_function
        """
        self.unpack_compute_tiling()
        opt_config = {"out_of_bound_sync_check": True, "enable_const_fold": True}
        tbe_context.get_context().add_compile_info(
            "vars", {
                "ub_ele": self.ub_ele,
                "align_unit": self.align_unit,
                "ub_spilt": self.ub_spilt,
                "num_each_block": self.num_each_block,
                "core_num": self.core_num,
            })
        self.tik_inst.BuildCCE(kernel_name=self.kernel_name,
                               inputs=[self.x_gm],
                               outputs=[self.y_gm],
                               flowtable=[self.tiling_gm],
                               config=opt_config)


def _check_param(x):
    """
    check parameters, if one is invalid, then raise error
    -----------------------
    :param x: tensor of numbers to be uncompressed
    :return: None
    """
    x_dtype = x.get("dtype").lower()

    if x_dtype != "uint8":
        raise RuntimeError("input x dtype must be uint8.")


# 'pylint: disable=dangerous-default-value,too-many-locals,
# 'pylint: disable=too-many-arguments,,unused-argument,invalid-name
@register_operator("SignBitsUnpack")
def sign_bits_unpack(x, y, size=1, dtype=Constant.DT_FLOAT, kernel_name="sign_bits_unpack"):
    """
    implementation of sign_bits_unpack and return the tik instance
    ----------------------------------------------------------------
    :param anchor_box: dict of shape and dtype of input anchor bbox
    :param gt_box: dict of shape and dtype of input gt bbox
    :param y: dict of shape and dtype of output encode bbox
    :param weight: the weight for encode bounding box
    :param kernel_name: the kernel's name
    :return: tik instance
    """
    _check_param(x)
    obj = SignBitsUnpack(x, y, size, dtype, kernel_name)
    obj.tik_inst_function()

    return obj.tik_inst
