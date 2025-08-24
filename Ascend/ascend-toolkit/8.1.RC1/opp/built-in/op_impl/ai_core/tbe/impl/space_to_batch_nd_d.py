#!/usr/bin/python
# -*- coding: utf-8 -*-
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
space_to_batch_nd_d
"""
import functools

from impl.copy_only import copy_only
from impl.transpose_d import transpose_d
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import para_check
from tbe.dsl.instrinsic.cce_intrin_md import reset_mask_insn
import te.lang.cce as tbe
from impl.util.platform_adapter import tbe_build
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tvm
from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json
from impl.util.util_select_op_base import get_op_cal_info
from tbe.common.platform import get_bit_len

# repeat up limit
REPEAT_LIMIT = 255


# 'pylint: disable=too-many-arguments,too-many-locals
def _vector_dup_continuous(tik_inst, src, size):
    """vector_dup continuous function, set ubuf to 0
    """
    mask_len = 128
    if src.dtype.lower() == "float32":
        mask_len = 64

    if size > 0:
        size_loop = size // mask_len
        size_left = size % mask_len
        repeat_loop = size_loop // REPEAT_LIMIT
        repeat_left = size_loop % REPEAT_LIMIT
        dup_value = float(0)

        if repeat_loop > 0:
            with tik_inst.for_range(0, repeat_loop) as repeat_loop_idx:
                repeat_offset = repeat_loop_idx * REPEAT_LIMIT * mask_len
                tik_inst.vector_dup(mask_len, src[repeat_offset], dup_value, REPEAT_LIMIT, 1, 8)

        if repeat_left > 0:
            repeat_offset = repeat_loop * REPEAT_LIMIT * mask_len
            tik_inst.vector_dup(mask_len, src[repeat_offset], dup_value, repeat_left, 1, 8)

        if size_left > 0:
            size_offset = size_loop * mask_len
            tik_inst.vector_dup(size_left, src[size_offset], dup_value, 1, 1, 8)


def _vector_dup_discrete(tik_inst, src, repeat, size, block_ele, dst_blk=1, dst_rep=8):
    """vector_dup discrete function, set ubuf to 0
    """
    mask_len = 128
    if src.dtype.lower() == "float32":
        mask_len = 64

    if size > 0:
        size_loop = size // mask_len
        size_left = size % mask_len
        repeat_loop = repeat // REPEAT_LIMIT
        repeat_left = repeat % REPEAT_LIMIT
        dup_value = float(0)

        def _inner(src, mask_len):
            """exec repeat
            """
            if repeat_loop > 0:
                with tik_inst.for_range(0, repeat_loop) as repeat_loop_idx:
                    repeat_offset = repeat_loop_idx * REPEAT_LIMIT * dst_rep * block_ele
                    tik_inst.vector_dup(mask_len, src[repeat_offset], dup_value, REPEAT_LIMIT, dst_blk, dst_rep)
            if repeat_left > 0:
                repeat_offset = repeat_loop * REPEAT_LIMIT * dst_rep * block_ele
                tik_inst.vector_dup(mask_len, src[repeat_offset], dup_value, repeat_left, dst_blk, dst_rep)

        # exec size
        if size_loop > 0:
            with tik_inst.for_range(0, size_loop) as size_loop_idx:
                size_offset = size_loop_idx * mask_len
                _inner(src[size_offset], mask_len)
        if size_left > 0:
            size_offset = size_loop * mask_len
            _inner(src[size_offset], size_left)


# 'pylint: disable=unused-argument,useless-object-inheritance,too-many-branches,too-many-lines,invalid-name
# 'pylint: disable=too-many-instance-attributes,too-many-statements,too-many-boolean-expressions
class SpaceToBatchNdFive(object):
    """Function: use to finish SpaceToBatchNd main functions to reset data.
    """

    def __init__(self, shape, dtype, block_shape, paddings):
        """init SpaceToBatchNd parameters.
        """
        self.shape = shape
        self.dtype = dtype
        self.batch = self.shape[0]
        self.channel1 = self.shape[1]
        self.input_height = self.shape[2]
        self.input_width = self.shape[3]
        self.channel0 = self.shape[4]

        self.pad_top = paddings[0][0]
        self.pad_bottom = paddings[0][1]
        self.pad_left = paddings[1][0]
        self.pad_right = paddings[1][1]
        self.pad_height = self.pad_top + self.pad_bottom
        self.pad_width = self.pad_left + self.pad_right
        self.padded_height = self.input_height + self.pad_height
        self.padded_width = self.input_width + self.pad_width

        self.block_height = block_shape[0]
        self.block_width = block_shape[1]
        self.block_size = self.block_height * self.block_width

        self.output_height = self.padded_height // self.block_height
        self.output_width = self.padded_width // self.block_width

        self.padded_shape = [self.batch, self.channel1, self.padded_height, self.padded_width, self.channel0]

        self.output_shape = [
            self.batch * self.block_size, self.channel1, self.output_height, self.output_width, self.channel0
        ]

        self.tile_shape = [self.block_height, self.output_height, self.block_width, self.output_width, self.channel0]

    def tile_axis(self):
        """tile axis.
        """
        ub_size = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
        dtype_size = get_bit_len(self.dtype) // 8
        total_cnt = ub_size // dtype_size // 2

        tile_axis = 1
        for i, _ in enumerate(self.tile_shape):
            if i > 0:
                ele_cnt = functools.reduce(lambda x, y: x * y, self.tile_shape[i:])
                if total_cnt // ele_cnt > 0:
                    tile_axis = i
                    break

        return tile_axis

    def new_alloc(self, i_b, shape, name, scope):
        """new alloc.
        """
        buf_var = i_b.allocate(self.dtype, shape, name=name, scope=scope)
        new_buffer = tvm.decl_buffer(shape, buf_var.dtype, name=name, scope=scope, data=buf_var)

        return new_buffer

    def vector_dup(self, i_b, buf, size):
        """vector dup.
        """
        one_cnt = 128
        if self.dtype == "float32":
            one_cnt = 64
        repeat = size // one_cnt
        remainder = size % one_cnt
        loop_repeat = repeat // 255
        loop_remainder = repeat % 255

        with i_b.if_scope(loop_repeat > 0):
            with i_b.for_range(0, loop_repeat) as i:
                mask = one_cnt
                offset_repeat = i * one_cnt * 255
                reset_mask_insn(i_b, self.dtype, bits=mask)
                i_b.emit(
                    tvm.call_extern(self.dtype, "vector_dup", buf.access_ptr("w", offset=offset_repeat),
                                    tvm.const(0, dtype=self.dtype), 255, 1, 1, 8, 8))

        offset_remainder = loop_repeat * one_cnt * 255
        with i_b.if_scope(loop_remainder > 0):
            mask = one_cnt
            reset_mask_insn(i_b, self.dtype, bits=mask)
            i_b.emit(
                tvm.call_extern(self.dtype, "vector_dup", buf.access_ptr("w", offset=offset_remainder),
                                tvm.const(0, dtype=self.dtype), loop_remainder, 1, 1, 8, 8))

        offset = one_cnt * loop_remainder + loop_repeat * one_cnt * 255
        with i_b.if_scope(remainder > 0):
            mask = remainder
            reset_mask_insn(i_b, self.dtype, bits=mask)
            i_b.emit(
                tvm.call_extern(self.dtype, "vector_dup", buf.access_ptr("w", offset=offset),
                                tvm.const(0, dtype=self.dtype), 1, 1, 1, 8, 8))

    def kernel_ir(self, dst, src):
        """kernel ir.
        """
        i_b = tvm.tir.ir_builder.create()
        input_data = src[0]
        output_data = dst[0]
        dtype = self.dtype
        pad_t = self.pad_top
        pad_l = self.pad_left
        block_h = self.block_height
        block_w = self.block_width
        pad_w = self.pad_width
        input_h = self.input_height
        input_w = self.input_width
        padded_w = self.padded_width
        output_h = self.output_height
        output_w = self.output_width
        c_0 = self.channel0
        num = self.batch * self.channel1
        num_outer = num
        num_inner = 1
        if num > 65535:
            for i in reversed(list(range(1, 65535))):
                if num % i == 0:
                    num_outer = i
                    num_inner = num // i
                    break
        block_idx = tvm.thread_axis("blockIdx.x")
        i_b.scope_attr(block_idx, "thread_extent", num_outer)
        var = block_idx.var
        burst = 1
        if dtype == "float32":
            burst = 2

        tile_axis = self.tile_axis()
        if tile_axis == 1 and input_w * (block_h - 1) * burst <= 65535 and pad_w * burst <= 65535 and (
                block_w - 1) * burst <= 65535:
            size = output_h * padded_w * c_0
            pad_data = self.new_alloc(i_b, [size], name="pad_data", scope=tbe_platform.scope_ubuf)
            pad_data2 = self.new_alloc(i_b, [size], name="pad_data2", scope=tbe_platform.scope_ubuf)

            divisor = block_h // 2
            remainder = block_h % 2
            with i_b.for_range(0, num_inner) as n_i:
                with i_b.for_range(0, divisor) as b_h:
                    # set buffer to all zero
                    self.vector_dup(i_b, pad_data, size)

                    # move data from GM to UB
                    start = (pad_t - b_h + block_h - 1) // block_h
                    end = (pad_t + input_h - b_h + block_h - 1) // block_h
                    offset_base = (var * num_inner + n_i) * input_h * input_w * c_0
                    offset_src = (b_h + start * block_h - pad_t) * input_w * c_0 + offset_base
                    offset_dst = start * padded_w * c_0 + pad_l * c_0
                    with i_b.if_scope(end - start > 0):
                        i_b.emit(
                            tvm.call_extern(dtype, "copy_gm_to_ubuf", pad_data.access_ptr("w", offset=offset_dst),
                                            input_data.access_ptr("r", offset=offset_src), 0, end - start,
                                            input_w * burst,
                                            input_w * (block_h - 1) * burst, pad_w * burst))

                    # move data from UB to GM
                    with i_b.for_range(0, block_w) as b_w:
                        offset_out = ((b_h * block_w + b_w) * num + (var * num_inner + n_i)) * output_h * output_w * c_0
                        i_b.emit(
                            tvm.call_extern(dtype, "copy_ubuf_to_gm", output_data.access_ptr("w", offset=offset_out),
                                            pad_data.access_ptr("r", offset=b_w * c_0), 0, output_h * output_w, burst,
                                            (block_w - 1) * burst, 0))

                    # set buffer to all zero
                    self.vector_dup(i_b, pad_data2, size)

                    # move data from GM to UB
                    start = (pad_t - (b_h + divisor) + block_h - 1) // block_h
                    end = (pad_t + input_h - (b_h + divisor) + block_h - 1) // block_h
                    offset_base = (var * num_inner + n_i) * input_h * input_w * c_0
                    offset_src = ((b_h + divisor) + start * block_h - pad_t) * input_w * c_0 + offset_base
                    offset_dst = start * padded_w * c_0 + pad_l * c_0
                    with i_b.if_scope(end - start > 0):
                        i_b.emit(
                            tvm.call_extern(dtype, "copy_gm_to_ubuf", pad_data2.access_ptr("w", offset=offset_dst),
                                            input_data.access_ptr("r", offset=offset_src), 0, end - start,
                                            input_w * burst,
                                            input_w * (block_h - 1) * burst, pad_w * burst))

                    # move data from UB to GM
                    with i_b.for_range(0, block_w) as b_w:
                        offset_out = (((b_h + divisor) * block_w + b_w) * num +
                                      (var * num_inner + n_i)) * output_h * output_w * c_0
                        i_b.emit(
                            tvm.call_extern(dtype, "copy_ubuf_to_gm", output_data.access_ptr("w", offset=offset_out),
                                            pad_data2.access_ptr("r", offset=b_w * c_0), 0, output_h * output_w, burst,
                                            (block_w - 1) * burst, 0))

                if remainder != 0:
                    # set buffer to all zero
                    self.vector_dup(i_b, pad_data, size)

                    # move data from GM to UB
                    start = pad_t // block_h
                    end = (pad_t + input_h) // block_h
                    offset_base = (var * num_inner + n_i) * input_h * input_w * c_0
                    offset_src = ((block_h - 1) + start * block_h - pad_t) * input_w * c_0 + offset_base
                    offset_dst = start * padded_w * c_0 + pad_l * c_0
                    with i_b.if_scope(end - start > 0):
                        i_b.emit(
                            tvm.call_extern(dtype, "copy_gm_to_ubuf", pad_data.access_ptr("w", offset=offset_dst),
                                            input_data.access_ptr("r", offset=offset_src), 0, end - start,
                                            input_w * burst,
                                            input_w * (block_h - 1) * burst, pad_w * burst))

                    # move data from UB to GM
                    with i_b.for_range(0, block_w) as b_w:
                        offset_out = (((block_h - 1) * block_w + b_w) * num +
                                      (var * num_inner + n_i)) * output_h * output_w * c_0
                        i_b.emit(
                            tvm.call_extern(dtype, "copy_ubuf_to_gm", output_data.access_ptr("w", offset=offset_out),
                                            pad_data.access_ptr("r", offset=b_w * c_0), 0, output_h * output_w, burst,
                                            (block_w - 1) * burst, 0))
        elif tile_axis in (1, 2) and (block_w - 1) * burst <= 65535:
            size = padded_w * c_0
            pad_data = self.new_alloc(i_b, [size], name="pad_data", scope=tbe_platform.scope_ubuf)

            with i_b.for_range(0, num_inner) as n_i:
                with i_b.for_range(0, block_h) as b_h:
                    with i_b.for_range(0, output_h) as o_h:
                        # set buffer to all zero
                        self.vector_dup(i_b, pad_data, size)

                        # move data from GM to UB
                        with i_b.if_scope(tvm.all(o_h * block_h + b_h >= pad_t, o_h * block_h + b_h < input_h + pad_t)):
                            offset_base = (var * num_inner + n_i) * input_h * input_w * c_0
                            offset_src = (o_h * block_h + b_h - pad_t) * input_w * c_0 + offset_base
                            i_b.emit(
                                tvm.call_extern(dtype, "copy_gm_to_ubuf", pad_data.access_ptr("w", offset=pad_l * c_0),
                                                input_data.access_ptr("r", offset=offset_src), 0, 1, input_w * burst, 0,
                                                0))

                        # move data from UB to GM
                        with i_b.for_range(0, block_w) as b_w:
                            offset_out = (((b_h * block_w + b_w) * num +
                                           (var * num_inner + n_i)) * output_h + o_h) * output_w * c_0
                            i_b.emit(
                                tvm.call_extern(dtype, "copy_ubuf_to_gm", output_data.access_ptr("w",
                                                                                                 offset=offset_out),
                                                pad_data.access_ptr("r", offset=b_w * c_0), 0, output_w, burst,
                                                (block_w - 1) * burst, 0))
        elif tile_axis == 3 and (block_w - 1) * burst <= 65535:
            size = output_w * c_0
            pad_data = self.new_alloc(i_b, [size], name="pad_data", scope=tbe_platform.scope_ubuf)

            with i_b.for_range(0, num_inner) as n_i:
                with i_b.for_range(0, block_h) as b_h:
                    with i_b.for_range(0, output_h) as o_h:
                        with i_b.for_range(0, block_w) as b_w:
                            # set buffer to all zero
                            self.vector_dup(i_b, pad_data, size)

                            # move data from GM to UB
                            with i_b.if_scope(
                                    tvm.all(o_h * block_h + b_h >= pad_t, o_h * block_h + b_h < input_h + pad_t)):
                                start = (pad_l - b_w + block_w - 1) // block_w
                                end = (pad_l + input_w - b_w + block_w - 1) // block_w
                                offset_base = (var * num_inner + n_i) * input_h * input_w * c_0
                                offset_src = (o_h * block_h + b_h - pad_t) * input_w * c_0 + (b_w + start * block_w -
                                                                                              pad_l) * c_0 + offset_base
                                with i_b.if_scope(end - start > 0):
                                    i_b.emit(
                                        tvm.call_extern(dtype, "copy_gm_to_ubuf",
                                                        pad_data.access_ptr("w", offset=start * c_0),
                                                        input_data.access_ptr("r", offset=offset_src), 0, end - start,
                                                        burst, (block_w - 1) * burst, 0))

                            # move data from UB to GM
                            offset_out = (((b_h * block_w + b_w) * num +
                                           (var * num_inner + n_i)) * output_h + o_h) * output_w * c_0
                            i_b.emit(
                                tvm.call_extern(dtype, "copy_ubuf_to_gm", output_data.access_ptr("w",
                                                                                                 offset=offset_out),
                                                pad_data.access_ptr("r", offset=0), 0, output_w, burst, 0, 0))
        else:
            size = c_0
            pad_data = self.new_alloc(i_b, [size], name="pad_data", scope=tbe_platform.scope_ubuf)

            with i_b.for_range(0, num_inner) as n_i:
                with i_b.for_range(0, block_h) as b_h:
                    with i_b.for_range(0, output_h) as o_h:
                        with i_b.for_range(0, block_w) as b_w:
                            with i_b.for_range(0, output_w) as o_w:
                                # set buffer to all zero
                                self.vector_dup(i_b, pad_data, size)

                                # move data from GM to UB
                                with i_b.if_scope(
                                        tvm.all(o_h * block_h + b_h >= pad_t, o_h * block_h + b_h < input_h + pad_t,
                                                o_w * block_w + b_w >= pad_l, o_w * block_w + b_w < input_w + pad_l)):
                                    offset_base = (var * num_inner + n_i) * input_h * input_w * c_0
                                    offset_src = (o_h * block_h + b_h - pad_t) * input_w * c_0 + (
                                        o_w * block_w + b_w - pad_l) * c_0 + offset_base
                                    i_b.emit(
                                        tvm.call_extern(dtype, "copy_gm_to_ubuf", pad_data.access_ptr("w", offset=0),
                                                        input_data.access_ptr("r", offset=offset_src), 0, 1, burst, 0,
                                                        0))

                                # move data from UB to GM
                                offset_out = ((((b_h * block_w + b_w) * num +
                                                (var * num_inner + n_i)) * output_h + o_h) * output_w + o_w) * c_0
                                i_b.emit(
                                    tvm.call_extern(dtype, "copy_ubuf_to_gm",
                                                    output_data.access_ptr("w", offset=offset_out),
                                                    pad_data.access_ptr("r", offset=0), 0, 1, burst, 0, 0))

        return i_b.get()


class SpaceToBatchNdSix(object):
    """Function: use to finish SpaceToBatchNd main functions to reset data.
    """

    def __init__(self, shape, dtype, block_shape, paddings, kernel_name):
        """init SpaceToBatchNd parameters.
        """
        self.shape = shape
        self.dtype = dtype
        self.block_shape = block_shape
        self.pads = paddings
        self.kernel_name = kernel_name
        self.ub_size = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
        self.dtype_size = get_bit_len(self.dtype) // 8
        self.block_ele = 32 // self.dtype_size
        self.ub_ele = self.ub_size // self.dtype_size - self.block_ele
        self.half_ub_ele = self.ub_ele // 2
        self.core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.tik_instance = tik.Tik()
        # input batch and output batch
        self.input_b = self.shape[0]
        self.output_b = self.input_b * block_shape[0] * block_shape[1] * block_shape[2]
        # input depth and output depth
        self.input_d = self.shape[1]
        self.pad_d = self.input_d + self.pads[0][0] + self.pads[0][1]
        self.output_d = self.pad_d // self.block_shape[0]
        # input heigth and output height
        self.input_h = self.shape[3]
        self.pad_h = self.input_h + self.pads[1][0] + self.pads[1][1]
        self.output_h = self.pad_h // self.block_shape[1]
        # input wieth and output height
        self.input_w = self.shape[4]
        self.pad_w = self.input_w + self.pads[2][0] + self.pads[2][1]
        self.output_w = self.pad_w // self.block_shape[2]
        # channel one and zero and burst_len
        self.c_1 = self.shape[2]
        self.c_0 = self.shape[5]
        # output shape
        self.out_shape = [self.output_b, self.output_d, self.c_1, self.output_h, self.output_w, self.c_0]
        # input tensor and output tensor
        self.input_tensor = self.tik_instance.Tensor(self.dtype, self.shape, name="gm_in", scope=tik.scope_gm)
        self.output_tensor = self.tik_instance.Tensor(self.dtype, self.out_shape, name="gm_out", scope=tik.scope_gm)

    def run_height(self):
        """run height function.
        """
        thread = 2 if (self.pad_h + 2) * self.pad_w * self.c_0 <= self.half_ub_ele and self.c_1 > 1 else 1
        with self.tik_instance.for_range(0, self.output_d, block_num=self.output_d) as idx_d:
            with self.tik_instance.for_range(0, self.input_b) as idx_b:
                with self.tik_instance.for_range(0, self.block_shape[0]) as idx_blk_0:
                    # open double buffer at c1
                    with self.tik_instance.for_range(0, self.c_1, thread_num=thread) as idx_c1:
                        # define ub
                        ub_size = self.pad_h * self.pad_w * self.c_0
                        ub_tensor = self.tik_instance.Tensor(self.dtype, [ub_size],
                                                             name="ub_tensor",
                                                             scope=tik.scope_ubuf)
                        flag_d = idx_d * self.block_shape[0] + idx_blk_0
                        # only vector dup
                        with self.tik_instance.if_scope(
                                tik.any(flag_d < self.pads[0][0], flag_d >= self.pads[0][0] + self.input_d)):
                            _vector_dup_continuous(self.tik_instance, ub_tensor, ub_size)
                        # move in and vector dup
                        with self.tik_instance.else_scope():
                            # move in
                            offset_b = idx_b * self.input_d * self.c_1 * self.input_h * self.input_w * self.c_0
                            offset_d = (idx_d * self.block_shape[0] + idx_blk_0 -
                                        self.pads[0][0]) * self.c_1 * self.input_h * self.input_w * self.c_0
                            offset_c1 = idx_c1 * self.input_h * self.input_w * self.c_0
                            offset_gm_in = offset_b + offset_d + offset_c1
                            offset_ub = self.pads[1][0] * self.pad_w * self.c_0 + self.pads[2][0] * self.c_0
                            n_burst = self.input_h
                            burst_len = self.input_w * self.c_0 // self.block_ele
                            dst_stride = (self.pads[2][1] + self.pads[2][0]) * self.c_0 // self.block_ele
                            self.tik_instance.data_move(ub_tensor[offset_ub], self.input_tensor[offset_gm_in], 0,
                                                        n_burst, burst_len, 0, dst_stride)
                            # vector dup
                            size_1 = self.pads[1][0] * self.pad_w * self.c_0 + self.pads[2][0] * self.c_0
                            _vector_dup_continuous(self.tik_instance, ub_tensor, size_1)
                            offset_2 = size_1 + self.input_w * self.c_0
                            repeat = self.input_h - 1
                            size_2 = (self.pads[2][1] + self.pads[2][0]) * self.c_0
                            dst_rep = self.pad_w * self.c_0 // self.block_ele
                            _vector_dup_discrete(self.tik_instance, ub_tensor[offset_2:], repeat, size_2,
                                                 self.block_ele, 1, dst_rep)
                            offset_3 = (self.pad_h -
                                        self.pads[1][1]) * self.pad_w * self.c_0 - self.pads[2][1] * self.c_0
                            size_3 = self.pads[1][1] * self.pad_w * self.c_0 + self.pads[2][1] * self.c_0
                            _vector_dup_continuous(self.tik_instance, ub_tensor[offset_3:], size_3)
                        # move out and open double buffer
                        thread = 2 if self.output_h > 1 else 1
                        with self.tik_instance.for_range(0, self.output_h, thread_num=thread) as idx_h:
                            with self.tik_instance.for_range(0, self.block_shape[1]) as idx_blk_1:
                                # define ub
                                tmp_size = self.pad_w * self.c_0
                                tmp_tensor = self.tik_instance.Tensor(self.dtype, [tmp_size],
                                                                      name="tmp_tensor",
                                                                      scope=tik.scope_ubuf)
                                with self.tik_instance.for_range(0, self.block_shape[2]) as idx_blk_2:
                                    # permute at ub
                                    offset_ub_in = (
                                        idx_h * self.block_shape[1] + idx_blk_1
                                    ) * self.output_w * self.block_shape[2] * self.c_0 + idx_blk_2 * self.c_0
                                    offset_ub_out = idx_blk_2 * self.output_w * self.c_0
                                    n_burst = self.output_w
                                    burst_len = self.c_0 // self.block_ele
                                    src_stride = (self.block_shape[2] * self.c_0 - self.c_0) // self.block_ele
                                    self.tik_instance.data_move(tmp_tensor[offset_ub_out], ub_tensor[offset_ub_in], 0,
                                                                n_burst, burst_len, src_stride, 0)
                                    # move out
                                    offset_gm_out = (((((
                                        (idx_blk_0 * self.block_shape[1] + idx_blk_1) * self.block_shape[2] + idx_blk_2)
                                        * self.input_b + idx_b) * self.output_d + idx_d) * self.c_1 +
                                        idx_c1) * self.output_h + idx_h) * self.output_w * self.c_0
                                    offset_ub = idx_blk_2 * self.output_w * self.c_0
                                    burst_len = self.output_w * self.c_0 // self.block_ele
                                    self.tik_instance.data_move(self.output_tensor[offset_gm_out],
                                                                tmp_tensor[offset_ub], 0, 1, burst_len, 0, 0)

    def run_width(self):
        """run width function.
        """
        thread = 2 if 2 * self.pad_w * self.c_0 <= self.half_ub_ele and self.c_1 > 1 else 1
        with self.tik_instance.for_range(0, self.output_d, block_num=self.output_d) as idx_d:
            with self.tik_instance.for_range(0, self.input_b) as idx_b:
                with self.tik_instance.for_range(0, self.block_shape[0]) as idx_blk_0:
                    # open double buffer at c1
                    with self.tik_instance.for_range(0, self.c_1, thread_num=thread) as idx_c1:
                        with self.tik_instance.for_range(0, self.output_h) as idx_h:
                            with self.tik_instance.for_range(0, self.block_shape[1]) as idx_blk_1:
                                # define ub
                                ub_size = self.pad_w * self.c_0
                                ub_tensor = self.tik_instance.Tensor(self.dtype, [ub_size],
                                                                     name="ub_tensor",
                                                                     scope=tik.scope_ubuf)
                                flag_d = idx_d * self.block_shape[0] + idx_blk_0
                                flag_h = idx_h * self.block_shape[1] + idx_blk_1
                                # only vector dup
                                with self.tik_instance.if_scope(
                                        tik.any(flag_d < self.pads[0][0], flag_d >= self.pads[0][0] + self.input_d,
                                                flag_h < self.pads[1][0], flag_h >= self.pads[1][0] + self.input_h)):
                                    _vector_dup_continuous(self.tik_instance, ub_tensor, ub_size)
                                # move in and vector dup
                                with self.tik_instance.else_scope():
                                    # move in
                                    offset_b = idx_b * self.input_d * self.c_1 * self.input_h * self.input_w * self.c_0
                                    offset_d = (idx_d * self.block_shape[0] + idx_blk_0 -
                                                self.pads[0][0]) * self.c_1 * self.input_h * self.input_w * self.c_0
                                    offset_c1 = idx_c1 * self.input_h * self.input_w * self.c_0
                                    offset_h = (idx_h * self.block_shape[1] + idx_blk_1 -
                                                self.pads[1][0]) * self.input_w * self.c_0
                                    offset_gm_in = offset_b + offset_d + offset_c1 + offset_h
                                    offset_ub = self.pads[2][0] * self.c_0
                                    burst_len = self.input_w * self.c_0 // self.block_ele
                                    self.tik_instance.data_move(ub_tensor[offset_ub], self.input_tensor[offset_gm_in],
                                                                0, 1, burst_len, 0, 0)
                                    # vector dup
                                    size_1 = self.pads[2][0] * self.c_0
                                    _vector_dup_continuous(self.tik_instance, ub_tensor, size_1)
                                    offset_2 = (self.pad_w - self.pads[2][1]) * self.c_0
                                    size_2 = self.pads[2][1] * self.c_0
                                    _vector_dup_continuous(self.tik_instance, ub_tensor[offset_2:], size_2)

                                # move out
                                with self.tik_instance.for_range(0, self.block_shape[2]) as idx_blk_2:
                                    # define ub
                                    tmp_size = self.pad_w * self.c_0
                                    tmp_tensor = self.tik_instance.Tensor(self.dtype, [tmp_size],
                                                                          name="tmp_tensor",
                                                                          scope=tik.scope_ubuf)
                                    # permute at ub
                                    offset_ub_in = idx_blk_2 * self.c_0
                                    offset_ub_out = idx_blk_2 * self.output_w * self.c_0
                                    n_burst = self.output_w
                                    burst_len = self.c_0 // self.block_ele
                                    src_stride = (self.block_shape[2] * self.c_0 - self.c_0) // self.block_ele
                                    self.tik_instance.data_move(tmp_tensor[offset_ub_out], ub_tensor[offset_ub_in], 0,
                                                                n_burst, burst_len, src_stride, 0)
                                    # move out
                                    offset_gm_out = (((((
                                        (idx_blk_0 * self.block_shape[1] + idx_blk_1) * self.block_shape[2] + idx_blk_2)
                                        * self.input_b + idx_b) * self.output_d + idx_d) * self.c_1 +
                                        idx_c1) * self.output_h + idx_h) * self.output_w * self.c_0
                                    offset_ub = idx_blk_2 * self.output_w * self.c_0
                                    burst_len = self.output_w * self.c_0 // self.block_ele
                                    self.tik_instance.data_move(self.output_tensor[offset_gm_out],
                                                                tmp_tensor[offset_ub], 0, 1, burst_len, 0, 0)

    def run_last(self):
        """run channel_zero function.
        """
        thread = 2 if self.c_0 <= self.half_ub_ele and self.c_1 > 1 else 1
        with self.tik_instance.for_range(0, self.output_d, block_num=self.output_d) as idx_d:
            with self.tik_instance.for_range(0, self.input_b) as idx_b:
                with self.tik_instance.for_range(0, self.block_shape[0]) as idx_blk_0:
                    with self.tik_instance.for_range(0, self.c_1, thread_num=thread) as idx_c1:
                        with self.tik_instance.for_range(0, self.output_h) as idx_h:
                            with self.tik_instance.for_range(0, self.block_shape[1]) as idx_blk_1:
                                with self.tik_instance.for_range(0, self.output_w) as idx_w:
                                    with self.tik_instance.for_range(0, self.block_shape[2]) as idx_blk_2:
                                        # define ub
                                        ub_size = self.c_0
                                        ub_tensor = self.tik_instance.Tensor(self.dtype, [ub_size],
                                                                             name="ub_tensor",
                                                                             scope=tik.scope_ubuf)
                                        burst_len = self.c_0 // self.block_ele
                                        flag_d = idx_d * self.block_shape[0] + idx_blk_0
                                        flag_h = idx_h * self.block_shape[1] + idx_blk_1
                                        flag_w = idx_w * self.block_shape[2] + idx_blk_2
                                        # vector dup
                                        with self.tik_instance.if_scope(
                                                tik.any(flag_d < self.pads[0][0],
                                                        flag_d >= self.pads[0][0] + self.input_d,
                                                        flag_h < self.pads[1][0],
                                                        flag_h >= self.pads[1][0] + self.input_h,
                                                        flag_w < self.pads[2][0],
                                                        flag_w >= self.pads[2][0] + self.input_w)):
                                            _vector_dup_continuous(self.tik_instance, ub_tensor, ub_size)
                                        # move in
                                        with self.tik_instance.else_scope():
                                            offset_b = idx_b * self.input_d * self.c_1 * self.input_h * \
                                                self.input_w * self.c_0
                                            offset_d = (idx_d * self.block_shape[0] + idx_blk_0 - self.pads[0][0]
                                                        ) * self.c_1 * self.input_h * self.input_w * self.c_0
                                            offset_c1 = idx_c1 * self.input_h * self.input_w * self.c_0
                                            offset_h = (idx_h * self.block_shape[1] + idx_blk_1 -
                                                        self.pads[1][0]) * self.input_w * self.c_0
                                            offset_w = (idx_w * self.block_shape[2] + idx_blk_2 -
                                                        self.pads[2][0]) * self.c_0
                                            offset_gm_in = offset_b + offset_d + offset_c1 + offset_h + offset_w
                                            self.tik_instance.data_move(ub_tensor, self.input_tensor[offset_gm_in], 0,
                                                                        1, burst_len, 0, 0)
                                        # move out
                                        offset_gm_out = ((
                                            (((((idx_blk_0 * self.block_shape[1] + idx_blk_1) * self.block_shape[2] +
                                                idx_blk_2) * self.input_b + idx_b) * self.output_d + idx_d) * self.c_1 +
                                             idx_c1) * self.output_h + idx_h) * self.output_w + idx_w) * self.c_0
                                        self.tik_instance.data_move(self.output_tensor[offset_gm_out], ub_tensor, 0, 1,
                                                                    burst_len, 0, 0)

    def run(self):
        """run function.
        """
        # select branch
        if (self.pad_h + 2) * self.pad_w * self.c_0 <= self.ub_ele:
            self.run_height()
        elif 2 * self.pad_w * self.c_0 <= self.ub_ele:
            self.run_width()
        else:
            self.run_last()
        # build cce
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[self.input_tensor],
                                   outputs=[self.output_tensor])
        return self.tik_instance


def check_supported(x, y, block_shape, paddings, kernel_name="space_to_batch_nd_d"):
    """
    dynamic is not selected when any condition is true: \n
        ori shape must be [?,1168,?], block_shape length must be 1, paddings length must be 2 \n
    """
    ori_format = x.get("ori_format")
    ori_shape = x.get("ori_shape")
    block_shape = list(block_shape)
    paddings = list(paddings)
    if tbe_platform.api_check_support("tik.vcopy"):
        return False, ""
    if ori_format in ("NHWC",) and len(ori_shape) == 3 and len(block_shape) == 1 and len(paddings) == 2:
        if ori_shape[1] == 1168:
            return True, ""
    if ori_format in ("NHWC",) and len(ori_shape) == 4 and len(block_shape) == 2 and len(paddings) == 4:
        if (ori_shape[1], ori_shape[2]) in ((66, 66),):
            return True, ""

    return False, ""


def get_op_support_info(x, y, block_shape, paddings, kernel_name="space_to_batch_nd_d"):
    """get op support info.
    """
    axis_split_list = None
    axis_reduce_list = None
    op_cal_info_in_json = get_op_cal_info(axis_split_list, axis_reduce_list)
    return op_cal_info_in_json


def op_select_format(x, y, block_shape, paddings, kernel_name="space_to_batch_nd_d"):
    """
    1. when input x's ori_shape in ["NDHWC", "NCDHW"], the Op
    SpaceToBatchNDD can support NDC1HWC0.
    > for example:
    > x : Tensor of (shape=(16, 1, 1, 16, 16, 16), "NDC1HWC0")
    > the Op SpaceToBatchNDD can process with NC1HWC0:
    > x : Tensor of (shape=(16, 1, 16, 16, 16), "NC1HWC0")
    """
    dtype = "float16, float"
    input_format = "NC1HWC0, NC1HWC0"
    ori_format = x.get("ori_format")
    if ori_format in ("NDHWC", "NCDHW"):
        dtype = "float16, float"
        input_format = "NDC1HWC0, NDC1HWC0"

    input0 = gen_param(classify="input0", name="x", datatype=dtype, format=input_format)
    output0 = gen_param(classify="output0", name="y", datatype=dtype, format=input_format)

    param_list = [input0, output0]
    param_dynamic_in_json = get_dynamic_param_in_json(param_list)
    return param_dynamic_in_json


def check_parms_6hd(shape, dtype, block_shape, paddings, kernel_name):
    """check the parameters including shape, dtype, block_shape, paddings and kernel_name.
    """
    dtype_list = ("float16", "float32")
    para_check.check_shape(shape, param_name="x")
    para_check.check_dtype(dtype, dtype_list, param_name="x")

    if len(shape) != 6:
        error_detail = "the shape'rank of x should be 6 bug got: %d" % len(shape)
        error_manager_vector.raise_err_input_shape_invalid(kernel_name, "x", error_detail)

    if len(block_shape) != 3:
        error_detail = "the shape'rank of block_shape should be 3 bug got: %d" % len(block_shape)
        error_manager_vector.raise_err_input_shape_invalid(kernel_name, "block_shape", error_detail)

    if len(paddings) != 3 or len(paddings[0]) != 2 or len(paddings[1]) != 2 or len(paddings[2]) != 2:
        error_detail = "the shape of paddings should be 3x2"
        error_manager_vector.raise_err_input_shape_invalid(kernel_name, "paddings", error_detail)

    if not (isinstance(block_shape[0], int) and isinstance(block_shape[1], int) and isinstance(block_shape[2], int) and
            block_shape[0] > 0 and block_shape[1] > 0 and block_shape[2] > 0):
        error_detail = "the value of block_shape should be integer and be greater to 0"
        error_manager_vector.raise_err_input_shape_invalid(kernel_name, "block_shape", error_detail)

    if not (isinstance(paddings[0][0], int) and paddings[0][0] >= 0 and isinstance(paddings[0][1], int) and
            paddings[0][1] >= 0 and isinstance(paddings[1][0], int) and paddings[1][0] >= 0 and
            isinstance(paddings[1][1], int) and paddings[1][1] >= 0 and isinstance(paddings[2][0], int) and
            paddings[2][0] >= 0 and isinstance(paddings[2][1], int) and paddings[2][1] >= 0):
        error_detail = "the value of paddings should be integer and be greater to 0"
        error_manager_vector.raise_err_input_shape_invalid(kernel_name, "paddings", error_detail)

    if (shape[1] + paddings[0][0] + paddings[0][1]) % block_shape[0] != 0:
        error_detail = "paddings depth should be exactly divisible by block depth"
        error_manager_vector.raise_err_input_shape_invalid(kernel_name, "paddings", error_detail)
    if (shape[3] + paddings[1][0] + paddings[1][1]) % block_shape[1] != 0:
        error_detail = "paddings height should be exactly divisible by block height"
        error_manager_vector.raise_err_input_shape_invalid(kernel_name, "paddings", error_detail)
    if (shape[4] + paddings[2][0] + paddings[2][1]) % block_shape[2] != 0:
        error_detail = "paddings width should be exactly divisible by block width"
        error_manager_vector.raise_err_input_shape_invalid(kernel_name, "paddings", error_detail)


def space_to_batch_nd_d_6hd(x, y, block_shape, paddings, kernel_name="space_to_batch_nd_d_6hd"):
    """SpaceToBatch for N-D tensors.

    Parameters
    ----------
    x: dict
        the dict of input tensor.
    y: dict
        the dict of output tensor.
    block_shape: list or tuple
        1-D with shape [3].
    paddings: list or tuple
        2-D with shape [3, 2], paddings[i] = [pad_start, pad_end].
    kernel_name: str
        cce kernel name, default value is "space_to_batch_nd_d_6hd".

    Returns
    -------
    None.
    """
    shape = x.get("shape")
    dtype = x.get("dtype").lower()
    ori_format = x.get("ori_format")

    # check format and set block_shape to 1-D with shape [3] ans set crops to 2-D with shape [3, 2]
    if ori_format in ("NDHWC",):
        if len(block_shape) == 3 and len(paddings) == 6:
            paddings = [[paddings[0], paddings[1]], [paddings[2], paddings[3]], [paddings[4], paddings[5]]]
    elif ori_format in ("NCDHW",):
        if len(block_shape) == 4 and block_shape[0] == 1:
            block_shape = [block_shape[1], block_shape[2], block_shape[3]]
            if len(paddings) == 8 and paddings[0] == 0 and paddings[1] == 0:
                paddings = [[paddings[2], paddings[3]], [paddings[4], paddings[5]], [paddings[6], paddings[7]]]
            if len(paddings) == 4 and len(paddings[0]) == 2 and len(paddings[1]) == 2 and len(paddings[2]) == 2 and len(
                    paddings[3]) == 2 and paddings[0][0] == 0 and paddings[0][1] == 0:
                paddings = [[paddings[1][0], paddings[1][1]], [paddings[2][0], paddings[2][1]],
                            [paddings[3][0], paddings[3][1]]]
    else:
        error_manager_vector.raise_err_input_format_invalid(kernel_name, "x", "NDHWC,NCDHW", ori_format)

    # check input params
    check_parms_6hd(shape, dtype, block_shape, paddings, kernel_name)

    # if output is the same as input, call the function 'copy_only'
    if block_shape[0] == 1 and block_shape[1] == 1 and block_shape[2] == 1 and paddings[0][0] == 0 and paddings[0][
            1] == 0 and paddings[1][0] == 0 and paddings[1][1] == 0 and paddings[2][0] == 0 and paddings[2][1] == 0:
        copy_only(x, x, kernel_name)
        return

    # run tik
    batch_to_space_nd = SpaceToBatchNdSix(shape, dtype, block_shape, paddings, kernel_name)
    batch_to_space_nd.run()


def check_parms_5hd(shape, dtype, block_shape, paddings, kernel_name):
    """check the parameters including shape, dtype, block_shape, paddings and kernel_name.
    """
    dtype_list = ("float16", "float32")
    para_check.check_shape(shape, param_name="x")
    para_check.check_dtype(dtype, dtype_list, param_name="x")

    if len(shape) != 5:
        error_detail = "the shape'rank of x should be 5 bug got: %d" % len(shape)
        error_manager_vector.raise_err_input_shape_invalid(kernel_name, "x", error_detail)

    if len(block_shape) != 2:
        error_detail = "the shape'rank of block_shape should be 2 bug got: %d" % len(block_shape)
        error_manager_vector.raise_err_input_shape_invalid(kernel_name, "block_shape", error_detail)

    if len(paddings) != 2 or len(paddings[0]) != 2 or len(paddings[1]) != 2:
        error_detail = "the shape of paddings should be 2x2"
        error_manager_vector.raise_err_input_shape_invalid(kernel_name, "paddings", error_detail)

    if not (isinstance(block_shape[0], int) and isinstance(block_shape[1], int) and block_shape[0] > 0 and
            block_shape[1] > 0):
        error_detail = "the value of block_shape should be integer and be greater to 0"
        error_manager_vector.raise_err_input_shape_invalid(kernel_name, "block_shape", error_detail)

    if not (isinstance(paddings[0][0], int) and paddings[0][0] >= 0 and isinstance(paddings[0][1], int) and
            paddings[0][1] >= 0 and isinstance(paddings[1][0], int) and paddings[1][0] >= 0 and
            isinstance(paddings[1][1], int) and paddings[1][1] >= 0):
        error_detail = "the value of paddings should be integer and be greater or equal to 0"
        error_manager_vector.raise_err_input_shape_invalid(kernel_name, "paddings", error_detail)

    if (shape[2] + paddings[0][0] + paddings[0][1]) % block_shape[0] != 0:
        error_detail = "paddings height should be exactly divisible by block height"
        error_manager_vector.raise_err_input_shape_invalid(kernel_name, "paddings", error_detail)
    if (shape[3] + paddings[1][0] + paddings[1][1]) % block_shape[1] != 0:
        error_detail = "paddings width should be exactly divisible by block width"
        error_manager_vector.raise_err_input_shape_invalid(kernel_name, "paddings", error_detail)


def space_to_batch_nd_d_5hd(x, y, block_shape, paddings, kernel_name="space_to_batch_nd_d_5hd"):
    """SpaceToBatch for N-D tensors.

    Parameters
    ----------
    x: dict
        the dict of input tensor.
    y: dict
        the dict of output tensor.
    block_shape: list or tuple
        1-D with shape [1] or [2].
    paddings: list or tuple
        2-D with shape [1, 2] or [2, 2], paddings[i] = [pad_start, pad_end].
    kernel_name: str
        cce kernel name, default value is "space_to_batch_nd_d_5hd".

    Returns
    -------
    None.
    """
    shape = x.get("shape")
    dtype = x.get("dtype").lower()
    ori_format = x.get("ori_format")

    if ori_format in ("NHWC",):
        if len(block_shape) == 1:
            block_shape = [1, block_shape[0]]
            if len(paddings) == 2:
                paddings = [[0, 0], [paddings[0], paddings[1]]]
            if len(paddings) == 1 and len(paddings[0]) == 2:
                paddings = [[0, 0], [paddings[0][0], paddings[0][1]]]
        if len(block_shape) == 2 and len(paddings) == 4:
            paddings = [[paddings[0], paddings[1]], [paddings[2], paddings[3]]]
    elif ori_format in ("NCHW",):
        if len(block_shape) == 3 and block_shape[0] == 1:
            block_shape = [block_shape[1], block_shape[2]]
            if len(paddings) == 6 and paddings[0] == 0 and paddings[1] == 0:
                paddings = [[paddings[2], paddings[3]], [paddings[4], paddings[5]]]
            if len(paddings) == 3 and len(paddings[0]) == 2 and len(paddings[1]) == 2 and len(
                    paddings[2]) == 2 and paddings[0][0] == 0 and paddings[0][1] == 0:
                paddings = [[paddings[1][0], paddings[1][1]], [paddings[2][0], paddings[2][1]]]
    else:
        error_manager_vector.raise_err_input_format_invalid(kernel_name, "x", "NHWC,NCHW", ori_format)

    check_parms_5hd(shape, dtype, block_shape, paddings, kernel_name)

    if block_shape[0] == 1 and block_shape[1] == 1 and paddings[0][0] == 0 and paddings[0][1] == 0 and paddings[1][
            0] == 0 and paddings[1][1] == 0:
        copy_only(x, x, kernel_name)
        return

    if paddings[0][0] == 0 and paddings[0][1] == 0 and paddings[1][0] == 0 and paddings[1][1] == 0:
        new_shape_input = (shape[0], shape[1], shape[2] // block_shape[0], block_shape[0], shape[3] // block_shape[1],
                           block_shape[1], shape[4])
        new_shape_output = (block_shape[0], block_shape[1], shape[0], shape[1], shape[2] // block_shape[0],
                            shape[3] // block_shape[1], shape[4])
        x.update({"shape": new_shape_input})
        y.update({"shape": new_shape_output})
        transpose_d(x, y, [3, 5, 0, 1, 2, 4, 6], kernel_name)
        return

    data = tvm.placeholder(shape, name="data", dtype=dtype)
    space = SpaceToBatchNdFive(shape, dtype, block_shape, paddings)
    res = tvm.extern([space.output_shape], [data],
                     lambda ins, outs: space.kernel_ir(outs, ins),
                     dtype=dtype,
                     name="res")
    sch = tvm.create_schedule(res.op)
    with tbe_build.build_config():
        tvm.build(sch, [data, res], "cce", name=kernel_name)


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_LIST_INT,
                            (para_check.REQUIRED_ATTR_LIST_INT, para_check.REQUIRED_ATTR_LIST_LIST_INT),
                            para_check.KERNEL_NAME)
def space_to_batch_nd_d(x, y, block_shape, paddings, kernel_name="space_to_batch_nd_d"):
    """SpaceToBatch for N-D tensors.

    Parameters
    ----------
    x: dict
        the dict of input tensor.
    y: dict
        the dict of output tensor.
    block_shape: list or tuple
        1-D with shape [1], [2] or [3].
    paddings: list or tuple
        2-D with shape [1, 2], [2, 2] or [3, 2], paddings[i] = [pad_start, pad_end].
    kernel_name: str
        cce kernel name, default value is "space_to_batch_nd_d".

    Returns
    -------
    None.
    """
    input_format = x.get("format")

    if input_format not in ("NC1HWC0", "NDC1HWC0"):
        error_manager_vector.raise_err_input_format_invalid(kernel_name, "x", "NC1HWC0", input_format)

    if input_format in ("NC1HWC0",):
        space_to_batch_nd_d_5hd(x, y, block_shape, paddings, kernel_name)
        return

    if input_format in ("NDC1HWC0",):
        space_to_batch_nd_d_6hd(x, y, block_shape, paddings, kernel_name)
        return
