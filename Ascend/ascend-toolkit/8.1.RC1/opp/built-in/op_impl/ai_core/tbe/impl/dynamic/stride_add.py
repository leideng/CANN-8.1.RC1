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
stride_add
"""
import functools
import math
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import register_operator

RESERVE_SIZE = 16 * 1024
BLOCK_SIZE = 32
DTYPE_BYTES = {"float16": 2, "float32": 4, "int32": 4, "bfloat16": 2}
TILING_ARG_NUM = 64
TILING_MODE_DEFAULT = 0
SCALAR_TENSOR_SIZE = 32
MAX_INT32 = 2 ** 31 - 1
TILING_DTYPE = "int32"
ATTR_DTYPE = "int32"
MAX_REPEAT_TIME = 255


# 'pylint: disable=invalid-name,too-many-arguments,useless-return
# 'pylint: disable=too-many-instance-attributes,attribute-defined-outside-init
@register_operator("StrideAdd")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_INT,
                            para_check.KERNEL_NAME)
def stride_add(x1, x2, y, x1_c1_offset, x2_c1_offset, c1_len, kernel_name="stride_add"):
    """
    the external interfaces of op stride_add

    Parameters
    ----------
    x1: dict including shape, format and dtype
        dtype supports float16, float32; format only support NC1HWC0
    x2: dict including shape, format and dtype
        dtype supports float16, float32; format only support NC1HWC0
    y: dict including shape, format and dtype
        dtype supports float16, float32; format only support NC1HWC0
    x1_c1_offset: offset of c1 dim from x1 tensor input
    x2_c1_offset: offset of c1 dim from x2 tensor input
    c1_len: len of c1 dim to conduct stride add
    kernel_name: cce kernel name

    Returns
    -------
    tik_instance: tik_instance
    """
    input_dtype = x1.get("dtype").lower()
    check_list = ("bfloat16", "float32", "float16")
    para_check.check_dtype(input_dtype, check_list, param_name="x1")
    stride_add_process = StrideAdd(
        x1, x2, y, x1_c1_offset, x2_c1_offset, c1_len, kernel_name)
    tik_instance = stride_add_process.compute_stride_add()
    return tik_instance


class StrideAdd():
    """
    the main class of op stride_add
    """
    def __init__(self, x1, x2, y, x1_c1_offset, x2_c1_offset, c1_len, kernel_name="stride_add"):
        """
        the constructor function of class StrideAdd

        Parameters
        ----------
        input_dict: the dict including the basic input info

        Returns
        -------
        None
        """
        self.dtype_x1 = x1.get('dtype')
        self.dtype_x2 = x2.get('dtype')
        self.dtype_y = y.get('dtype')
        self.data_type = self.dtype_x1
        self.dtype_x1_c1_offset = ATTR_DTYPE
        self.dtype_x2_c1_offset = ATTR_DTYPE
        self.dtype_c1_len = ATTR_DTYPE
        self.kernel_name = kernel_name
        self.tiling_dtype = TILING_DTYPE
        self.aicore_num = tik.Dprofile().get_aicore_num()

        # maximum elements of each tensor on the UB
        if self.dtype_x1 == "bfloat16":
            self.data_type = "float32"
        self.dsize_x1 = DTYPE_BYTES.get(self.data_type)
        self.data_each_block = BLOCK_SIZE // self.dsize_x1
        self.vector_mask_max = 8 * self.data_each_block
        available_ub_size = (tik.Dprofile().get_unified_buffer_size() - RESERVE_SIZE)
        if self.dtype_x1 == "bfloat16":
            self.ub_max_num = (available_ub_size // self.dsize_x1 // 4 // self.data_each_block * self.data_each_block)
        else:
            self.ub_max_num = (available_ub_size // self.dsize_x1 // 3 // self.data_each_block * self.data_each_block)

        # tiling gm
        self.tik_instance = tik.Tik(tik.Dprofile())
        self.dtype_bytes_size_tiling = DTYPE_BYTES.get(self.tiling_dtype)
        self.tiling_each_block = BLOCK_SIZE // self.dtype_bytes_size_tiling
        self.tiling_gm = self.tik_instance.Tensor(self.tiling_dtype, (TILING_ARG_NUM,),
                                                  name='tiling_gm', scope=tik.scope_gm)

        # ub & gm
        self.x1_ub = None
        self.x1_gm = None
        self.x2_ub = None
        self.x2_gm = None
        self.tmp_bf16_ub = None
        self.y_ub = None
        self.y_gm = None
        self.c1_len_ub = None
        self.c1_len_gm = None

        # tiling parameters
        self.x1_n = None
        self.x1_c1 = None
        self.x1_h = None
        self.x1_w = None
        self.x1_c0 = None
        self.x2_n = None
        self.x2_c1 = None
        self.x2_h = None
        self.x2_w = None
        self.x2_c0 = None
        self.x1_c1_offset = None
        self.x2_c1_offset = None
        self.c1_len = None
        self.tiling_mode = None
        self.core_used = None
        self.core_num_var = None

    def prepare_gm_for_data(self):
        """
        Prepare GM for the input data of StrideAdd OP

        Parameters
        ----------

        Returns
        -------
        None
        """
        self.x1_gm = self.tik_instance.Tensor(self.dtype_x1,
                                              (MAX_INT32,),
                                              name='x1_gm',
                                              scope=tik.scope_gm)

        self.x2_gm = self.tik_instance.Tensor(self.dtype_x1,
                                              (MAX_INT32,),
                                              name='x2_gm',
                                              scope=tik.scope_gm)

        self.y_gm = self.tik_instance.Tensor(self.dtype_x1,
                                             (MAX_INT32,),
                                             name='y_gm',
                                             scope=tik.scope_gm)
        return

    def prepare_ub_for_data(self):
        """
        Prepare UB for the input data of StrideAdd OP

        TODO: Allocate actual ub size according to tilling info, other than maximum

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.x1_ub = self.tik_instance.Tensor(self.data_type,
                                              (self.ub_max_num,),
                                              name="x1_ub",
                                              scope=tik.scope_ubuf)

        self.x2_ub = self.tik_instance.Tensor(self.data_type,
                                              (self.ub_max_num,),
                                              name="x2_ub",
                                              scope=tik.scope_ubuf)

        self.y_ub = self.tik_instance.Tensor(self.data_type,
                                             (self.ub_max_num,),
                                             name="y_ub",
                                             scope=tik.scope_ubuf)
        if self.dtype_x1 == "bfloat16":
            self.tmp_bf16_ub = self.tik_instance.Tensor(self.dtype_x1,
                                                (self.ub_max_num,),
                                                name="tmp_bf16_ub",
                                                scope=tik.scope_ubuf)

    def get_tiling_args(self):
        """
        get tiling args from tiling_ub

        Parameters
        ----------
        tiling_ub: tensor with tiling_args in ub

        Returns
        -------
        None
        """
        tiling_ub = self.tik_instance.Tensor(self.tiling_dtype,
                                             (TILING_ARG_NUM,),
                                             name='tiling_ub',
                                             scope=tik.scope_ubuf)
        self.tik_instance.data_move(tiling_ub,
                                    self.tiling_gm,
                                    0,
                                    1,
                                    SCALAR_TENSOR_SIZE //
                                    self.tiling_each_block,
                                    0,
                                    0)
        self.x1_n = self.tik_instance.Scalar(
            self.tiling_dtype, name='x1_n')
        self.x1_c1 = self.tik_instance.Scalar(
            self.tiling_dtype, name='x1_c1')
        self.x1_h = self.tik_instance.Scalar(
            self.tiling_dtype, name='x1_h')
        self.x1_w = self.tik_instance.Scalar(
            self.tiling_dtype, name='x1_w')
        self.x1_c0 = self.tik_instance.Scalar(
            self.tiling_dtype, name='x1_c0')
        self.x2_n = self.tik_instance.Scalar(
            self.tiling_dtype, name='x2_n')
        self.x2_c1 = self.tik_instance.Scalar(
            self.tiling_dtype, name='x2_c1')
        self.x2_h = self.tik_instance.Scalar(
            self.tiling_dtype, name='x2_h')
        self.x2_w = self.tik_instance.Scalar(
            self.tiling_dtype, name='x2_w')
        self.x2_c0 = self.tik_instance.Scalar(
            self.tiling_dtype, name='x2_c0')
        self.x1_c1_offset = self.tik_instance.Scalar(
            self.tiling_dtype, name='x1_c1_offset')
        self.x2_c1_offset = self.tik_instance.Scalar(
            self.tiling_dtype, name='x2_c1_offset')
        self.c1_len = self.tik_instance.Scalar(
            self.tiling_dtype, name='c1_len')
        self.tiling_mode = self.tik_instance.Scalar(
            self.tiling_dtype, name='tiling_mode')
        self.core_used = self.tik_instance.Scalar(
            self.tiling_dtype, name='core_used')
        self.core_num_var = self.tik_instance.Scalar(
            self.tiling_dtype, name="core_num_var", init_value=self.aicore_num)
        self.x1_n.set_as(tiling_ub[0])
        self.x1_c1.set_as(tiling_ub[1])
        self.x1_h.set_as(tiling_ub[2])
        self.x1_w.set_as(tiling_ub[3])
        self.x1_c0.set_as(tiling_ub[4])
        self.x2_n.set_as(tiling_ub[5])
        self.x2_c1.set_as(tiling_ub[6])
        self.x2_h.set_as(tiling_ub[7])
        self.x2_w.set_as(tiling_ub[8])
        self.x2_c0.set_as(tiling_ub[9])
        self.x1_c1_offset.set_as(tiling_ub[10])
        self.x2_c1_offset.set_as(tiling_ub[11])
        self.c1_len.set_as(tiling_ub[12])
        self.tiling_mode.set_as(tiling_ub[13])
        self.core_used.set_as(tiling_ub[14])
        self.core_num_var.set_as(tiling_ub[15])

    def compute_stride_add(self):
        """
        the main function of computing stride add

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        self.get_tiling_args()

        self.prepare_gm_for_data()

        (used_aicore_num,
         batch_num_per_aicore_process,
         batch_tail) = self.split_aicore()

        with self.tik_instance.for_range(0, self.core_num_var, block_num=self.core_num_var) as block_id:
            with self.tik_instance.if_scope(block_id < self.core_used):
                self.prepare_ub_for_data()
                self.compute_stride_add_dynamic(
                    block_id, batch_num_per_aicore_process, batch_tail)

        opt_config = {"enable_const_fold": True}
        tbe_context.get_context().add_compile_info('vars', {'core_num': self.aicore_num})
        self.tik_instance.BuildCCE(
            kernel_name=self.kernel_name,
            inputs=[self.x1_gm, self.x2_gm],
            outputs=[self.y_gm],
            flowtable=[self.tiling_gm],
            config=opt_config)

        return self.tik_instance

    def compute_stride_add_dynamic(self, aicore_id, batch_num_per_aicore_process, batch_tail):

        batch_id = self.tik_instance.Scalar("int32")
        self.compute_uniformly_divided_batches(
            aicore_id, batch_id,
            batch_tail, batch_num_per_aicore_process)
        self.compute_tail_batches(aicore_id, batch_id, batch_tail)

    def compute_uniformly_divided_batches(self, aicore_id,
                                          batch_id, batch_tail,
                                          batch_num_per_aicore_process):
        """
        compute the uniformly divided batches on each aicore

        Parameters
        ----------
        aicore_id: the aicore index
        batch_id: the batch index
        batch_tail: the tail batch num after uniformly divided
        batch_num_per_aicore_process:
            batch num per aicore process

        Returns
        -------
        None
        """
        with self.tik_instance.for_range(
                0, batch_num_per_aicore_process) as inner_cycle:
            batch_id.set_as(
                aicore_id * batch_num_per_aicore_process + inner_cycle)

            with self.tik_instance.if_scope(batch_tail > 0):
                with self.tik_instance.if_scope(aicore_id < batch_tail):
                    batch_id.set_as(batch_id + aicore_id)

                with self.tik_instance.else_scope():
                    batch_id.set_as(batch_id + batch_tail)
            self.compute_each_batch(batch_id)

        return

    def compute_tail_batches(self, aicore_id, batch_id, batch_tail):
        """
        compute the tail batches on each aicore

        Parameters
        ----------
        aicore_id: the aicore index
        batch_id: the batch index
        batch_tail: the tail batch num after uniformly divided

        Returns
        -------
        None
        """
        with self.tik_instance.if_scope(batch_tail > 0):
            with self.tik_instance.if_scope(aicore_id < batch_tail):
                batch_id.set_as(batch_id + 1)
                self.compute_each_batch(batch_id)

        return

    def compute_each_batch(self, batch_id):
        """
        compute stride add on each batch

        Parameters
        ----------
        batch_id: batch index

        Returns
        -------
        None
        """
        compute_num = self.tik_instance.Scalar("int32")
        compute_num.set_as(self.c1_len * self.x1_h * self.x1_w * self.x1_c0)

        # the base offset of each batch
        x1_base_offset = self.compute_offset(
            batch_id, self.x1_c1, self.x1_h, self.x1_w, self.x1_c0, self.x1_c1_offset)
        x2_base_offset = self.compute_offset(
            batch_id, self.x2_c1, self.x2_h, self.x2_w, self.x2_c0, self.x2_c1_offset)

        y_base_offset = self.tik_instance.Scalar("int32")
        y_base_offset.set_as(batch_id * compute_num)

        x1_base_offset_init = self.tik_instance.Scalar("int32")
        x1_base_offset_init.set_as(x1_base_offset)
        x2_base_offset_init = self.tik_instance.Scalar("int32")
        x2_base_offset_init.set_as(x2_base_offset)
        y_base_offset_init = self.tik_instance.Scalar("int32")
        y_base_offset_init.set_as(y_base_offset)

        loop_time = self.tik_instance.Scalar("int32")
        loop_time.set_as(compute_num // self.ub_max_num)

        with self.tik_instance.if_scope(loop_time > 0):
            with self.tik_instance.for_range(0, loop_time) as loop_index:
                x1_base_offset = x1_base_offset + loop_index * self.ub_max_num
                x2_base_offset = x2_base_offset + loop_index * self.ub_max_num
                y_base_offset = y_base_offset + loop_index * self.ub_max_num
                self.compute_each_loop(x1_base_offset,
                                       x2_base_offset,
                                       y_base_offset,
                                       self.ub_max_num)

            x1_base_offset = x1_base_offset_init + loop_time * self.ub_max_num
            x2_base_offset = x2_base_offset_init + loop_time * self.ub_max_num
            y_base_offset = y_base_offset_init + loop_time * self.ub_max_num

        last_num_data = self.tik_instance.Scalar("int32")
        last_num_data.set_as(compute_num % self.ub_max_num)
        with self.tik_instance.if_scope(last_num_data > 0):
            self.compute_each_loop(x1_base_offset,
                                   x2_base_offset,
                                   y_base_offset,
                                   last_num_data)

        return

    def vconv_bf16_to_fp32(self, dst, src, compute_num):
        vconv_loop = compute_num // self.vector_mask_max // MAX_REPEAT_TIME
        repeat_tail = compute_num % (self.vector_mask_max * MAX_REPEAT_TIME) // self.vector_mask_max
        last_tail = compute_num % (self.vector_mask_max * MAX_REPEAT_TIME) % self.vector_mask_max
        with self.tik_instance.for_range(0, vconv_loop) as vconv_loop_idx:
            loop_offset = vconv_loop_idx * self.vector_mask_max * MAX_REPEAT_TIME
            self.tik_instance.vconv(self.vector_mask_max, "",
                                dst[loop_offset], src[loop_offset], MAX_REPEAT_TIME, 1, 1, 8, 4)

        with self.tik_instance.if_scope(repeat_tail > 0):
            loop_offset = vconv_loop * self.vector_mask_max * MAX_REPEAT_TIME
            self.tik_instance.vconv(self.vector_mask_max, "",
                                dst[loop_offset], src[loop_offset], repeat_tail, 1, 1, 8, 4)

        with self.tik_instance.if_scope(last_tail > 0):
            loop_offset = vconv_loop * self.vector_mask_max * MAX_REPEAT_TIME + repeat_tail * self.vector_mask_max
            self.tik_instance.vconv(last_tail, "",
                                dst[loop_offset], src[loop_offset], 1, 1, 1, 8, 4)

    def vconv_fp32_to_bf16(self, dst, src, compute_num):
        vconv_loop = compute_num // self.vector_mask_max // MAX_REPEAT_TIME
        repeat_tail = compute_num % (self.vector_mask_max * MAX_REPEAT_TIME) // self.vector_mask_max
        last_tail = compute_num % (self.vector_mask_max * MAX_REPEAT_TIME) % self.vector_mask_max
        with self.tik_instance.for_range(0, vconv_loop) as vconv_loop_idx:
            loop_offset = vconv_loop_idx * self.vector_mask_max * MAX_REPEAT_TIME
            self.tik_instance.vconv(self.vector_mask_max, "round",
                                dst[loop_offset], src[loop_offset], MAX_REPEAT_TIME, 1, 1, 4, 8)

        with self.tik_instance.if_scope(repeat_tail > 0):
            loop_offset = vconv_loop * self.vector_mask_max * MAX_REPEAT_TIME
            self.tik_instance.vconv(self.vector_mask_max, "round",
                                dst[loop_offset], src[loop_offset], repeat_tail, 1, 1, 4, 8)

        with self.tik_instance.if_scope(last_tail > 0):
            loop_offset = vconv_loop * self.vector_mask_max * MAX_REPEAT_TIME + repeat_tail * self.vector_mask_max
            self.tik_instance.vconv(last_tail, "round", dst[loop_offset], src[loop_offset], 1, 1, 1, 4, 8)


    def compute_each_loop(self, x1_offset, x2_offset,
                          y_offset, compute_num):
        """
        compute stride add on each loop

        Parameters
        ----------
        x1_offset: offset of x1
        x2_offset: offset of x2
        y_offset: offset of y
        compute_num: number of computing in this loop

        Returns
        -------
        None
        """
        burst_len = self.tik_instance.Scalar("int32")
        burst_len.set_as((compute_num - 1) // self.data_each_block + 1)
        if self.dtype_x1 == "bfloat16":
            data_each_block_bf16 = BLOCK_SIZE // DTYPE_BYTES.get(self.dtype_x1)
            burst_len_bf16 = (compute_num - 1) // data_each_block_bf16 + 1
            self.tik_instance.data_move(self.tmp_bf16_ub, self.x1_gm[x1_offset], 0, 1, burst_len_bf16, 0, 0)
            self.vconv_bf16_to_fp32(self.x1_ub, self.tmp_bf16_ub, compute_num)

            self.tik_instance.data_move(self.tmp_bf16_ub, self.x2_gm[x2_offset], 0, 1, burst_len_bf16, 0, 0)
            self.vconv_bf16_to_fp32(self.x2_ub, self.tmp_bf16_ub, compute_num)
        else:
            self.tik_instance.data_move(self.x1_ub, self.x1_gm[x1_offset], 0, 1, burst_len, 0, 0)

            self.tik_instance.data_move(self.x2_ub, self.x2_gm[x2_offset], 0, 1, burst_len, 0, 0)

        add_loop = self.tik_instance.Scalar("int32")
        add_loop.set_as(compute_num // (self.vector_mask_max * 255))

        add_offset = self.tik_instance.Scalar("int32")
        add_offset.set_as(0)
        with self.tik_instance.if_scope(add_loop > 0):
            with self.tik_instance.for_range(0, add_loop) as index:
                add_offset.set_as(index * self.vector_mask_max * 255)
                self.tik_instance.vec_add(self.vector_mask_max,
                                          self.y_ub[add_offset],
                                          self.x1_ub[add_offset],
                                          self.x2_ub[add_offset],
                                          255, 8, 8, 8)

        repeat_time = self.tik_instance.Scalar("int32")
        repeat_time.set_as((compute_num % (self.vector_mask_max * 255) // self.vector_mask_max))

        with self.tik_instance.if_scope(repeat_time > 0):
            add_offset.set_as(self.vector_mask_max * 255 * add_loop)
            self.tik_instance.vec_add(self.vector_mask_max,
                                      self.y_ub[add_offset],
                                      self.x1_ub[add_offset],
                                      self.x2_ub[add_offset],
                                      repeat_time, 8, 8, 8)

        left_num = self.tik_instance.Scalar("int32")
        left_num.set_as(compute_num % self.vector_mask_max)
        with self.tik_instance.if_scope(left_num > 0):
            add_offset.set_as(compute_num // self.vector_mask_max * self.vector_mask_max)
            self.tik_instance.vec_add(left_num,
                                      self.y_ub[add_offset],
                                      self.x1_ub[add_offset],
                                      self.x2_ub[add_offset],
                                      1, 8, 8, 8)

        if self.dtype_x1 == "bfloat16":
            self.vconv_fp32_to_bf16(self.tmp_bf16_ub, self.y_ub, compute_num)
            data_each_block_bf16 = BLOCK_SIZE // DTYPE_BYTES.get(self.dtype_x1)
            burst_len_bf16 = (compute_num - 1) // data_each_block_bf16 + 1
            self.tik_instance.data_move(self.y_gm[y_offset], self.tmp_bf16_ub, 0, 1, burst_len_bf16, 0, 0)
        else:
            self.tik_instance.data_move(self.y_gm[y_offset], self.y_ub, 0, 1, burst_len, 0, 0)

        return

    def compute_offset(self, batch_id, x_c1, x_h, x_w, x_c0, c1_offset):
        """
        compute the offset of input and output tensors

        Parameters
        ----------
        batch_id: batch index
        x_id: indicates which input tensor; can be x1 or x2

        Returns
        -------
        None
        """
        c1_dim_size = self.tik_instance.Scalar("int32")
        c1_dim_size.set_as(x_c1)

        hwc0 = self.tik_instance.Scalar("int32")
        hwc0.set_as(x_h * x_w * x_c0)

        offset = self.tik_instance.Scalar("int32")
        offset.set_as(c1_offset * hwc0 + batch_id * c1_dim_size * hwc0)

        return offset

    def split_aicore(self):
        """
        the aicore split scheme

        Parameters
        ----------
        None

        Returns
        -------
        used_aicore_num: used aicore num
        batch_num_per_aicore_process:
            batch num per aicore process
        batch_tail:
            the tail batch num after uniformly divided
        """
        used_aicore_num = self.tik_instance.Scalar("int32")
        batch_num_per_aicore_process = self.tik_instance.Scalar("int32")
        batch_tail = self.tik_instance.Scalar("int32")
        with self.tik_instance.if_scope(self.x1_n < self.core_num_var):
            used_aicore_num.set_as(self.x1_n)
            batch_num_per_aicore_process.set_as(1)
            batch_tail.set_as(0)
        with self.tik_instance.else_scope():
            used_aicore_num.set_as(self.core_num_var)
            batch_num_per_aicore_process.set_as(self.x1_n // used_aicore_num)
            batch_tail.set_as(self.x1_n % used_aicore_num)

        return (used_aicore_num,
                batch_num_per_aicore_process,
                batch_tail)
