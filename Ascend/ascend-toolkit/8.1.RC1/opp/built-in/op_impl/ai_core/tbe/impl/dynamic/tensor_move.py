#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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
tensor_move
"""
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from tbe.common.platform import get_bit_len


# 'pylint:disable=too-few-public-methods,too-many-instance-attributes
class Constant:
    """
    The class for constant
    """
    # max int64
    MAX_INT64 = 2 ** 63 - 1
    # bytes of one block
    BLOCK_BYTES = 32
    # proposal num
    PROPOSAL_NUM = 8
    # ting param num
    TILING_ARG_NUM = 4
    # reserved ub size
    RESERVED_UB_SIZE = 8 * 1024


def _apply_mem(tik_instance, dtype, shape, name, scope):
    """apply mem fuc

    Parameters
    ----------
    tik_instance: tik_instance
        tik_instance
    dtype: str
        ub dtype
    shape: list
        ub shape
    name: str
        ub name
    scope: scope
        scope_ubuf or scope_gm
    Returns
    -------
    Tensor: Tensor
    """
    return tik_instance.Tensor(dtype, shape, name=name, scope=scope)


# 'pylint: disable=locally-disabled,too-many-instance-attributes
class TensorMove:
    """
    Function: use to finish TensorMove main functions
    """

    def __init__(self, src, dst):
        """
        init TensorMove parameters

        Parameters
        ----------
        src : dict
            shape and dtype of input
        dst: dict
            shape and dtype of output, should be same shape and type as input

        Returns
        -------
        None
        """
        self.src_dtype = src.get("dtype").lower()
        self.dst_dtype = dst.get("dtype").lower()
        if self.dst_dtype == "bool":
            self.src_dtype = "int8"
            self.dst_dtype = "int8"
        if self.dst_dtype == "double":
            self.src_dtype = "int64"
            self.dst_dtype = "int64"
        # get dtype size, float16 size = 2 byte or float32 size = 4 byte
        self.dtype_size = \
            get_bit_len(self.src_dtype) // Constant.PROPOSAL_NUM
        # get one block data size, block align len
        # the len in one block = 16 fp16 and = 8 fp32
        self.data_len_one_block = Constant.BLOCK_BYTES // self.dtype_size
        self.data_len_one_vector = self.data_len_one_block * Constant.PROPOSAL_NUM

        self.ub_availble = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) - Constant.RESERVED_UB_SIZE
        self.ub_max_data = self.ub_availble // self.dtype_size

        self.tik_instance = tik.Tik()
        self.core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)

        self.src_gm = self.tik_instance.Tensor(
            self.src_dtype,
            [Constant.MAX_INT64],
            name="src_gm",
            scope=tik.scope_gm)
        self.dst_gm = self.tik_instance.Tensor(
            self.dst_dtype,
            [Constant.MAX_INT64],
            name="dst_gm",
            scope=tik.scope_gm)
        self.tiling_gm = self.tik_instance.Tensor(
            "int64",
            (Constant.TILING_ARG_NUM,),
            name="tiling_gm",
            scope=tik.scope_gm)
        self.data_ub = None
        self.tiling_mode = None
        self.tiling_ub = None
        self.data_size = None
        self.core_used_temp = None
        self._result = self.tik_instance.Scalar("int64", name="_result")
        self.tiling_mode = self.tik_instance.Scalar("int64", name="tiling_mode")
        self.data_size = self.tik_instance.Scalar("int64", name="data_size")
        self.core_used_temp = self.tik_instance.Scalar("int64", name="core_used_temp")
        self.core_num_var = self.tik_instance.Scalar("int64", name="core_num_var")

    def tensor_move(self):
        """
        copy all data from src to des
        """
        # core scedule
        core_data = self.tik_instance.Scalar("int64", name="core_data")
        with self.tik_instance.if_scope(self.core_used_temp <= 1):
            core_data.set_as(self.data_size)
        with self.tik_instance.elif_scope(self.data_size % self.core_used_temp == 0):
            core_data.set_as(self.data_size // self.core_used_temp)
        with self.tik_instance.else_scope():
            core_data.set_as(self.data_size // (self.core_used_temp - 1))
        with self.tik_instance.if_scope(core_data == 0):
            core_data.set_as(1)
        self._get_ceil_int(core_data, self.data_len_one_block)
        core_data.set_as(self._result * self.data_len_one_block)

        core_used = self.tik_instance.Scalar("int64", name="core_used")
        self._get_ceil_int(self.data_size, core_data)
        core_used.set_as(self._result)
        core_last = self.tik_instance.Scalar("int64", name="core_last")
        core_last.set_as(self.data_size - (core_data * (core_used - 1)))
        # calcu max copy segment
        copy_segment = self.tik_instance.Scalar("int64", name="copy_segment")
        copy_segment.set_as(self.ub_max_data // 2)
        self._get_ceil_int(copy_segment, self.data_len_one_block)
        copy_segment.set_as((self._result - 1) * self.data_len_one_block)
        # core process
        copy_loop = self.tik_instance.Scalar("int64", name="copy_loop")
        copy_tail = self.tik_instance.Scalar("int64", name="copy_tail")
        gm_in_offset = self.tik_instance.Scalar("int64", name="gm_in_offset")
        gm_out_offset = self.tik_instance.Scalar("int64", name="gm_out_offset")
        with self.tik_instance.for_range(
                0, self.core_used_temp, block_num=self.core_used_temp) as core_index:
            with self.tik_instance.if_scope(core_index < (core_used - 1)):
                copy_loop.set_as(core_data // copy_segment)
                copy_tail.set_as(core_data % copy_segment)
                thread_num = 2
                with self.tik_instance.if_scope(copy_loop < 2):
                    thread_num = 1
                with self.tik_instance.for_range(
                        0, copy_loop, thread_num=thread_num) as loop_index:
                    gm_in_offset.set_as(core_index * core_data +
                                        loop_index * copy_segment)
                    gm_out_offset.set_as(core_index * core_data +
                                         loop_index * copy_segment)
                    self._copy_in_to_out(copy_segment,
                                         gm_in_offset,
                                         gm_out_offset)
                with self.tik_instance.if_scope(copy_tail != 0):
                    gm_in_offset.set_as(core_index * core_data +
                                        copy_loop * copy_segment)
                    gm_out_offset.set_as(core_index * core_data +
                                         copy_loop * copy_segment)
                    self._copy_in_to_out(copy_tail,
                                         gm_in_offset,
                                         gm_out_offset)
            with self.tik_instance.if_scope(core_index == (core_used - 1)):
                copy_loop.set_as(core_last // copy_segment)
                copy_tail.set_as(core_last % copy_segment)
                thread_num = 2
                with self.tik_instance.if_scope(copy_loop < 2):
                    thread_num = 1
                with self.tik_instance.for_range(
                        0, copy_loop, thread_num=thread_num) as loop_index:
                    gm_in_offset.set_as(core_index * core_data +
                                        loop_index * copy_segment)
                    gm_out_offset.set_as(core_index * core_data +
                                         loop_index * copy_segment)
                    self._copy_in_to_out(copy_segment,
                                         gm_in_offset, gm_out_offset)
                with self.tik_instance.if_scope(copy_tail != 0):
                    gm_in_offset.set_as(core_index * core_data +
                                        copy_loop * copy_segment)
                    gm_out_offset.set_as(core_index * core_data +
                                         copy_loop * copy_segment)
                    self._copy_in_to_out(copy_tail,
                                         gm_in_offset, gm_out_offset)

    def run_tik(self, kernel_name):
        """
        cal tik_instance according to mode
        """
        self._tiling_args()
        tbe_context.get_context().add_compile_info(
            "vars", {
                "core_num": self.core_num,
                "data_len_one_block": self.data_len_one_block
            })
        self.tensor_move()
        opt_config = {"enable_const_fold": True}
        self.tik_instance.BuildCCE(
            kernel_name=kernel_name,
            inputs=[self.src_gm],
            outputs=[self.dst_gm],
            flowtable=[self.tiling_gm],
            config=opt_config)
        return self.tik_instance

    def _get_ceil_int(self, int1, int2):
        """
        Function: Round up
        """
        _remainder = self.tik_instance.Scalar("int64", name="_remainder")
        _remainder.set_as(int1 % int2)
        with self.tik_instance.if_scope(_remainder == 0):
            self._result.set_as(int1 // int2)
        with self.tik_instance.else_scope():
            self._result.set_as(int1 // int2 + 1)

    def _tiling_args(self):
        """
        get runtime params from tiling

        Parameters
        ----------
        tiling_ub: tensor, runtime params from tensor_move tiling

        Returns
        -------
        None
        """
        self.tiling_ub = self.tik_instance.Tensor("int64", (Constant.TILING_ARG_NUM,),
                                                  name="tiling_ub", scope=tik.scope_ubuf)
        self.tik_instance.data_move(self.tiling_ub, self.tiling_gm, 0, 1, 1, 0, 0)
        self.tiling_mode.set_as(self.tiling_ub[0])
        self.data_size.set_as(self.tiling_ub[1])
        self.core_used_temp.set_as(self.tiling_ub[2])
        self.core_num_var.set_as(self.tiling_ub[3])

    def _copy_in_to_out(self, copy_len, copy_in_offset, copy_out_offset):
        """
        copy ub to gm
        """
        nbust = self.tik_instance.Scalar("int64", name="nbust")
        self._get_ceil_int(copy_len, self.data_len_one_block)
        nbust.set_as(self._result)
        data_shape = self.tik_instance.Scalar("int64", name="data_shape")
        data_shape.set_as(nbust * self.data_len_one_block)
        self.data_ub = self.tik_instance.Tensor(self.dst_dtype, [data_shape],
                                                name="data_ub", scope=tik.scope_ubuf)
        if tbe_platform.api_check_support("tik.data_move_pad", self.src_dtype):
            self.tik_instance.data_move_pad(self.data_ub, self.src_gm[copy_in_offset], 1, copy_len * self.dtype_size,
                                            0, 0)
            self.tik_instance.data_move_pad(self.dst_gm[copy_out_offset], self.data_ub, 1, copy_len * self.dtype_size,
                                            0, 0)
        elif tbe_platform.api_check_support("tik.data_move_pad") and self.dtype_size == Constant.PROPOSAL_NUM:
            s8_src_gm = self.src_gm.reinterpret_cast_to("int8")
            s8_data_ub = self.data_ub.reinterpret_cast_to("int8")
            s8_dst_gm = self.dst_gm.reinterpret_cast_to("int8")
            self.tik_instance.data_move_pad(s8_data_ub, s8_src_gm[copy_in_offset * 8], 1,
                                            copy_len * self.dtype_size, 0, 0)
            self.tik_instance.data_move_pad(s8_dst_gm[copy_out_offset * 8], s8_data_ub, 1,
                                            copy_len * self.dtype_size, 0, 0)
        else:
            self.tik_instance.data_move(self.data_ub[0],
                                        self.src_gm[copy_in_offset],
                                        0, 1, nbust, 0, 0)
            self.tik_instance.data_move(self.dst_gm[copy_out_offset],
                                        self.data_ub[0],
                                        0, 1, nbust, 0, 0)


@register_operator("TensorMove")
@para_check.check_op_params(para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def tensor_move(src, dst, kernel_name="tensor_move"):
    """
    algorithm: tensor_move

    Parameters
    ----------
    src : dict
        shape and dtype of input
    dst: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name: str
        kernel name

    Returns
    -------
    None
    """
    res = TensorMove(src, dst)

    return res.run_tik(kernel_name)
