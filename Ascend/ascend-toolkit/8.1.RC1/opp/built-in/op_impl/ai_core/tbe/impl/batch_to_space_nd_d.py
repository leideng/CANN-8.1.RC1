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
batch_to_space_nd_d
"""
import functools

from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tbe_build
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import error_manager_vector
from impl.copy_only import copy_only
from impl.transpose_d import transpose_d
from impl.util.util_select_op_base import SplitInput
from impl.util.util_select_op_base import SplitOutput
from impl.util.util_select_op_base import get_op_cal_info
from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json
from impl.util.util_binary import get_bit_len


# 'pylint: disable=unused-argument,useless-object-inheritance
# 'pylint: disable=too-many-locals,too-many-branches,too-many-lines,invalid-name
# 'pylint: disable=too-many-instance-attributes,too-many-statements,too-many-boolean-expressions,too-many-arguments
class BatchToSpaceNdFive(object):
    """Function: use to finish BatchToSpaceNd main functions to reset data.
    """

    def __init__(self, shape, dtype, block_shape, crops):
        """init BatchToSpaceNd parameters.
        """
        self.shape = shape
        self.dtype = dtype
        self.batch = self.shape[0]
        self.channel1 = self.shape[1]
        self.input_height = self.shape[2]
        self.input_width = self.shape[3]
        self.channel0 = self.shape[4]

        self.crop_top = crops[0][0]
        self.crop_bottom = crops[0][1]
        self.crop_left = crops[1][0]
        self.crop_right = crops[1][1]
        self.crop_height = self.crop_top + self.crop_bottom
        self.crop_width = self.crop_left + self.crop_right

        self.block_height = block_shape[0]
        self.block_width = block_shape[1]
        self.block_size = self.block_height * self.block_width
        self.padded_height = self.input_height * self.block_height
        self.padded_width = self.input_width * self.block_width
        self.ibatch = self.batch // self.block_height // self.block_width

        self.output_height = self.padded_height - self.crop_height
        self.output_width = self.padded_width - self.crop_width

        self.permute_shape = [self.ibatch, self.channel1, self.padded_height, self.padded_width, self.channel0]
        self.output_shape = [self.ibatch, self.channel1, self.output_height, self.output_width, self.channel0]
        self.tile_shape = [self.block_height, self.input_height, self.block_width, self.input_width, self.channel0]

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

    def kernel_ir(self, dst, src):
        """ kernel ir.
        """
        i_b = tvm.tir.ir_builder.create()
        input_data = src[0]
        output_data = dst[0]
        dtype = self.dtype
        crop_t = self.crop_top
        crop_l = self.crop_left
        block_h = self.block_height
        block_w = self.block_width
        crop_w = self.crop_width
        input_h = self.input_height
        input_w = self.input_width
        padded_w = self.padded_width
        output_h = self.output_height
        output_w = self.output_width
        c_0 = self.channel0
        num = self.ibatch * self.channel1
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
        if tile_axis == 1 and output_w * (block_h - 1) * burst <= 65535 and (
                block_w - 1) * burst <= 65535 and crop_w * burst <= 65535 and (input_h * input_w <= 4095):
            size = input_h * padded_w * c_0
            padded_data = self.new_alloc(i_b, [size], name="padded_data", scope=tbe_platform.scope_ubuf)

            padded_data2 = self.new_alloc(i_b, [size], name="padded_data2", scope=tbe_platform.scope_ubuf)

            divisor = block_h // 2
            remainder = block_h % 2
            with i_b.for_range(0, num_inner) as n_i:
                with i_b.for_range(0, divisor) as b_h:
                    # move data from GM to UB
                    with i_b.for_range(0, block_w) as b_w:
                        offset_src = ((b_h * block_w + b_w) * num + (var * num_inner + n_i)) * input_h * input_w * c_0
                        offset_dst = b_w * c_0
                        i_b.emit(
                            tvm.call_extern(dtype, "copy_gm_to_ubuf", padded_data.access_ptr('w', offset=offset_dst),
                                            input_data.access_ptr('r', offset=offset_src), 0, input_h * input_w, burst,
                                            0, (block_w - 1) * burst))

                    # move data from UB to GM
                    start = (crop_t - b_h + block_h - 1) // block_h
                    end = (crop_t + output_h - b_h + block_h - 1) // block_h
                    offset_base = (var * num_inner + n_i) * output_h * output_w * c_0
                    offset_out = (b_h + start * block_h - crop_t) * output_w * c_0 + offset_base
                    offset_pad = start * padded_w * c_0 + crop_l * c_0
                    with i_b.if_scope(end - start > 0):
                        i_b.emit(
                            tvm.call_extern(dtype, "copy_ubuf_to_gm", output_data.access_ptr('w', offset=offset_out),
                                            padded_data.access_ptr('r', offset=offset_pad), 0, end - start,
                                            output_w * burst, crop_w * burst,
                                            output_w * (block_h - 1) * burst))

                    # move data from GM to UB
                    with i_b.for_range(0, block_w) as b_w:
                        offset_src = (((b_h + divisor) * block_w + b_w) * num +
                                      (var * num_inner + n_i)) * input_h * input_w * c_0
                        offset_dst = b_w * c_0
                        i_b.emit(
                            tvm.call_extern(dtype, "copy_gm_to_ubuf", padded_data2.access_ptr('w', offset=offset_dst),
                                            input_data.access_ptr('r', offset=offset_src), 0, input_h * input_w, burst,
                                            0, (block_w - 1) * burst))

                    # move data from UB to GM
                    start = (crop_t - (b_h + divisor) + block_h - 1) // block_h
                    end = (crop_t + output_h - (b_h + divisor) + block_h - 1) // block_h
                    offset_base = (var * num_inner + n_i) * output_h * output_w * c_0
                    offset_out = ((b_h + divisor) + start * block_h - crop_t) * output_w * c_0 + offset_base
                    offset_pad = start * padded_w * c_0 + crop_l * c_0
                    with i_b.if_scope(end - start > 0):
                        i_b.emit(
                            tvm.call_extern(dtype, "copy_ubuf_to_gm", output_data.access_ptr('w', offset=offset_out),
                                            padded_data2.access_ptr('r', offset=offset_pad), 0, end - start,
                                            output_w * burst, crop_w * burst,
                                            output_w * (block_h - 1) * burst))
                if remainder != 0:
                    # move data from GM to UB
                    with i_b.for_range(0, block_w) as b_w:
                        offset_src = (((block_h - 1) * block_w + b_w) * num +
                                      (var * num_inner + n_i)) * input_h * input_w * c_0
                        offset_dst = b_w * c_0
                        i_b.emit(
                            tvm.call_extern(dtype, "copy_gm_to_ubuf", padded_data.access_ptr('w', offset=offset_dst),
                                            input_data.access_ptr('r', offset=offset_src), 0, input_h * input_w, burst,
                                            0, (block_w - 1) * burst))

                    # move data from UB to GM
                    start = (crop_t - (block_h - 1) + block_h - 1) // block_h
                    end = (crop_t + output_h - (block_h - 1) + block_h - 1) // block_h
                    offset_base = (var * num_inner + n_i) * output_h * output_w * c_0
                    offset_out = ((block_h - 1) + start * block_h - crop_t) * output_w * c_0 + offset_base
                    offset_pad = start * padded_w * c_0 + crop_l * c_0
                    with i_b.if_scope(end - start > 0):
                        i_b.emit(
                            tvm.call_extern(dtype, "copy_ubuf_to_gm", output_data.access_ptr('w', offset=offset_out),
                                            padded_data.access_ptr('r', offset=offset_pad), 0, end - start,
                                            output_w * burst, crop_w * burst,
                                            output_w * (block_h - 1) * burst))

        elif tile_axis in (1, 2) and (block_w - 1) * burst <= 65535 and (input_w <= 4095):
            size = padded_w * c_0
            padded_data = self.new_alloc(i_b, [size], name="padded_data", scope=tbe_platform.scope_ubuf)

            with i_b.for_range(0, num_inner) as n_i:
                with i_b.for_range(0, block_h) as b_h:
                    with i_b.for_range(0, input_h) as i_h:
                        # move data from GM to UB
                        with i_b.for_range(0, block_w) as b_w:
                            offset_src = (((b_h * block_w + b_w) * num +
                                           (var * num_inner + n_i)) * input_h + i_h) * input_w * c_0
                            i_b.emit(
                                tvm.call_extern(dtype, "copy_gm_to_ubuf", padded_data.access_ptr("w", offset=b_w * c_0),
                                                input_data.access_ptr("r", offset=offset_src), 0, input_w, burst, 0,
                                                (block_w - 1) * burst))

                        # move data from UB to GM
                        with i_b.if_scope(
                                tvm.all(i_h * block_h + b_h >= crop_t, i_h * block_h + b_h < output_h + crop_t)):
                            offset_base = (var * num_inner + n_i) * output_h * output_w * c_0
                            offset_out = (i_h * block_h + b_h - crop_t) * output_w * c_0 + offset_base
                            i_b.emit(
                                tvm.call_extern(dtype, "copy_ubuf_to_gm", output_data.access_ptr("w",
                                                                                                 offset=offset_out),
                                                padded_data.access_ptr("r", offset=crop_l * c_0), 0, 1,
                                                output_w * burst, 0, 0))

        elif tile_axis == 3 and (block_w - 1) * burst <= 65535 and (input_w <= 4095):
            size = input_w * c_0
            padded_data = self.new_alloc(i_b, [size], name="padded_data", scope=tbe_platform.scope_ubuf)

            with i_b.for_range(0, num_inner) as n_i:
                with i_b.for_range(0, block_h) as b_h:
                    with i_b.for_range(0, input_h) as i_h:
                        with i_b.for_range(0, block_w) as b_w:
                            # move data from GM to UB
                            offset_src = (((b_h * block_w + b_w) * num +
                                           (var * num_inner + n_i)) * input_h + i_h) * input_w * c_0
                            i_b.emit(
                                tvm.call_extern(dtype, "copy_gm_to_ubuf", padded_data.access_ptr("w", offset=0),
                                                input_data.access_ptr("r", offset=offset_src), 0, input_w, burst, 0, 0))

                            # move data from UB to GM
                            with i_b.if_scope(
                                    tvm.all(i_h * block_h + b_h >= crop_t, i_h * block_h + b_h < output_h + crop_t)):
                                start = (crop_l - b_w + block_w - 1) // block_w
                                end = (crop_l + output_w - b_w + block_w - 1) // block_w
                                offset_base = (var * num_inner + n_i) * output_h * output_w * c_0
                                offset_dst = (i_h * block_h + b_h - crop_t) * output_w * c_0 + (
                                    b_w + start * block_w - crop_l) * c_0 + offset_base
                                with i_b.if_scope(end - start > 0):
                                    i_b.emit(
                                        tvm.call_extern(dtype, "copy_ubuf_to_gm",
                                                        output_data.access_ptr("w", offset=offset_dst),
                                                        padded_data.access_ptr("r", offset=start * c_0), 0, end - start,
                                                        burst, 0, (block_w - 1) * burst))

        else:
            size = c_0
            padded_data = self.new_alloc(i_b, [size], name="padded_data", scope=tbe_platform.scope_ubuf)

            with i_b.for_range(0, num_inner) as n_i:
                with i_b.for_range(0, block_h) as b_h:
                    with i_b.for_range(0, input_h) as i_h:
                        with i_b.for_range(0, block_w) as b_w:
                            with i_b.for_range(0, input_w) as i_w:
                                # move data from GM to UB
                                offset_src = ((((b_h * block_w + b_w) * num +
                                                (var * num_inner + n_i)) * input_h + i_h) * input_w + i_w) * c_0
                                i_b.emit(
                                    tvm.call_extern(dtype, "copy_gm_to_ubuf", padded_data.access_ptr("w", offset=0),
                                                    input_data.access_ptr("r", offset=offset_src), 0, 1, burst, 0, 0))

                                # move data from UB to GM
                                with i_b.if_scope(
                                        tvm.all(i_h * block_h + b_h >= crop_t, i_h * block_h + b_h < output_h + crop_t,
                                                i_w * block_w + b_w >= crop_l,
                                                i_w * block_w + b_w < output_w + crop_l)):
                                    offset_base = (var * num_inner + n_i) * output_h * output_w * c_0
                                    offset_out = (i_h * block_h + b_h - crop_t) * output_w * c_0 + (
                                        i_w * block_w + b_w - crop_l) * c_0 + offset_base
                                    i_b.emit(
                                        tvm.call_extern(dtype, "copy_ubuf_to_gm",
                                                        output_data.access_ptr("w", offset=offset_out),
                                                        padded_data.access_ptr("r", offset=0), 0, 1, burst, 0, 0))

        return i_b.get()


class BatchToSpaceNdSix(object):
    """Function: use to finish BatchToSpaceNd main functions to reset data.
    """

    def __init__(self, shape, dtype, block_shape, crops, kernel_name):
        """init BatchToSpaceNd parameters.
        """
        self.shape = shape
        self.dtype = dtype
        self.block_shape = block_shape
        self.crops = crops
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
        self.output_b = self.input_b // block_shape[0] // block_shape[1] // block_shape[2]
        # input depth and output depth
        self.input_d = self.shape[1]
        self.permute_d = self.input_d * self.block_shape[0]
        self.output_d = self.permute_d - self.crops[0][0] - self.crops[0][1]
        # input heigth and output height
        self.input_h = self.shape[3]
        self.permute_h = self.input_h * self.block_shape[1]
        self.output_h = self.permute_h - self.crops[1][0] - self.crops[1][1]
        # input wieth and output height
        self.input_w = self.shape[4]
        self.permute_w = self.input_w * self.block_shape[2]
        self.output_w = self.permute_w - self.crops[2][0] - self.crops[2][1]
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
        thread = 2 if self.permute_h * self.permute_w * self.c_0 <= self.half_ub_ele and self.c_1 > 1 else 1
        with self.tik_instance.for_range(0, self.input_d, block_num=self.input_d) as idx_d:
            with self.tik_instance.for_range(0, self.output_b) as idx_b:
                with self.tik_instance.for_range(0, self.block_shape[0]) as idx_blk_0:
                    with self.tik_instance.for_range(0, self.c_1, thread_num=thread) as idx_c1:
                        # define ub
                        ub_size = self.permute_h * self.permute_w * self.c_0
                        ub_tensor = self.tik_instance.Tensor(self.dtype, [ub_size],
                                                             name="ub_tensor",
                                                             scope=tik.scope_ubuf)
                        flag_d = idx_d * self.block_shape[0] + idx_blk_0
                        with self.tik_instance.if_scope(
                                tik.all(flag_d >= self.crops[0][0], flag_d < self.crops[0][0] + self.output_d)):
                            # move in
                            with self.tik_instance.for_range(0, self.input_h) as idx_h:
                                with self.tik_instance.for_range(0, self.block_shape[1]) as idx_blk_1:
                                    with self.tik_instance.for_range(0, self.block_shape[2]) as idx_blk_2:
                                        offset_gm_in = ((((
                                            ((idx_blk_0 * self.block_shape[1] + idx_blk_1) * self.block_shape[2] +
                                             idx_blk_2) * self.output_b + idx_b) * self.input_d + idx_d) * self.c_1 +
                                                         idx_c1) * self.input_h + idx_h) * self.input_w * self.c_0
                                        offset_ub = (
                                            idx_h * self.block_shape[1] + idx_blk_1
                                        ) * self.input_w * self.block_shape[2] * self.c_0 + idx_blk_2 * self.c_0
                                        n_burst = self.input_w
                                        burst_len = self.c_0 // self.block_ele
                                        dst_stride = (self.block_shape[2] * self.c_0 - self.c_0) // self.block_ele
                                        self.tik_instance.data_move(ub_tensor[offset_ub],
                                                                    self.input_tensor[offset_gm_in], 0, n_burst,
                                                                    burst_len, 0, dst_stride)
                            # move out
                            offset_b = idx_b * self.output_d * self.c_1 * self.output_h * self.output_w * self.c_0
                            offset_d = (idx_d * self.block_shape[0] + idx_blk_0 -
                                        self.crops[0][0]) * self.c_1 * self.output_h * self.output_w * self.c_0
                            offset_c1 = idx_c1 * self.output_h * self.output_w * self.c_0
                            offset_gm_out = offset_b + offset_d + offset_c1
                            offset_ub = self.crops[1][0] * self.permute_w * self.c_0 + self.crops[2][0] * self.c_0
                            n_burst = self.output_h
                            burst_len = self.output_w * self.c_0 // self.block_ele
                            src_stride = (self.crops[2][1] + self.crops[2][0]) * self.c_0 // self.block_ele
                            self.tik_instance.data_move(self.output_tensor[offset_gm_out], ub_tensor[offset_ub], 0,
                                                        n_burst, burst_len, src_stride, 0)

    def run_width(self):
        """run width function.
        """
        thread = 2 if self.permute_w * self.c_0 <= self.half_ub_ele and self.c_1 > 1 else 1
        with self.tik_instance.for_range(0, self.input_d, block_num=self.input_d) as idx_d:
            with self.tik_instance.for_range(0, self.output_b) as idx_b:
                with self.tik_instance.for_range(0, self.block_shape[0]) as idx_blk_0:
                    with self.tik_instance.for_range(0, self.c_1, thread_num=thread) as idx_c1:
                        with self.tik_instance.for_range(0, self.input_h) as idx_h:
                            with self.tik_instance.for_range(0, self.block_shape[1]) as idx_blk_1:
                                # define ub
                                ub_size = self.permute_w * self.c_0
                                ub_tensor = self.tik_instance.Tensor(self.dtype, [ub_size],
                                                                     name="ub_tensor",
                                                                     scope=tik.scope_ubuf)
                                flag_d = idx_d * self.block_shape[0] + idx_blk_0
                                flag_h = idx_h * self.block_shape[1] + idx_blk_1
                                with self.tik_instance.if_scope(
                                        tik.all(flag_d >= self.crops[0][0], flag_d < self.crops[0][0] + self.output_d,
                                                flag_h >= self.crops[1][0], flag_h < self.crops[1][0] + self.output_h)):
                                    # move in
                                    with self.tik_instance.for_range(0, self.block_shape[2]) as idx_blk_2:
                                        offset_gm_in = ((((
                                            ((idx_blk_0 * self.block_shape[1] + idx_blk_1) * self.block_shape[2] +
                                             idx_blk_2) * self.output_b + idx_b) * self.input_d + idx_d) * self.c_1 +
                                                         idx_c1) * self.input_h + idx_h) * self.input_w * self.c_0
                                        offset_ub = idx_blk_2 * self.c_0
                                        n_burst = self.input_w
                                        burst_len = self.c_0 // self.block_ele
                                        dst_stride = (self.block_shape[2] * self.c_0 - self.c_0) // self.block_ele
                                        self.tik_instance.data_move(ub_tensor[offset_ub],
                                                                    self.input_tensor[offset_gm_in], 0, n_burst,
                                                                    burst_len, 0, dst_stride)
                                    # move out
                                    offset_b = idx_b * self.output_d * self.c_1 * self.output_h * \
                                               self.output_w * self.c_0
                                    offset_d = (idx_d * self.block_shape[0] + idx_blk_0 -
                                                self.crops[0][0]) * self.c_1 * self.output_h * self.output_w * self.c_0
                                    offset_c1 = idx_c1 * self.output_h * self.output_w * self.c_0
                                    offset_h = (idx_h * self.block_shape[1] + idx_blk_1 -
                                                self.crops[1][0]) * self.output_w * self.c_0
                                    offset_gm_out = offset_b + offset_d + offset_c1 + offset_h
                                    offset_ub = self.crops[2][0] * self.c_0
                                    burst_len = self.output_w * self.c_0 // self.block_ele
                                    self.tik_instance.data_move(self.output_tensor[offset_gm_out], ub_tensor[offset_ub],
                                                                0, 1, burst_len, 0, 0)

    def run_last(self):
        """run channel_zero function.
        """
        thread = 2 if self.c_0 <= self.half_ub_ele and self.c_1 > 1 else 1
        with self.tik_instance.for_range(0, self.input_d, block_num=self.input_d) as idx_d:
            with self.tik_instance.for_range(0, self.output_b) as idx_b:
                with self.tik_instance.for_range(0, self.block_shape[0]) as idx_blk_0:
                    with self.tik_instance.for_range(0, self.c_1, thread_num=thread) as idx_c1:
                        with self.tik_instance.for_range(0, self.input_h) as idx_h:
                            with self.tik_instance.for_range(0, self.block_shape[1]) as idx_blk_1:
                                with self.tik_instance.for_range(0, self.input_w) as idx_w:
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
                                        with self.tik_instance.if_scope(
                                                tik.all(flag_d >= self.crops[0][0],
                                                        flag_d < self.crops[0][0] + self.output_d,
                                                        flag_h >= self.crops[1][0],
                                                        flag_h < self.crops[1][0] + self.output_h,
                                                        flag_w >= self.crops[2][0],
                                                        flag_w < self.crops[2][0] + self.output_w)):
                                            # move in
                                            offset_gm_in = ((((((
                                                (idx_blk_0 * self.block_shape[1] + idx_blk_1) * self.block_shape[2] +
                                                idx_blk_2) * self.output_b + idx_b) * self.input_d + idx_d) * self.c_1 +
                                                              idx_c1) * self.input_h + idx_h) * self.input_w +
                                                            idx_w) * self.c_0
                                            self.tik_instance.data_move(ub_tensor, self.input_tensor[offset_gm_in], 0,
                                                                        1, burst_len, 0, 0)
                                            # move out
                                            offset_b = idx_b * self.output_d * self.c_1 * self.output_h * \
                                                       self.output_w * self.c_0
                                            offset_d = (idx_d * self.block_shape[0] + idx_blk_0 - self.crops[0][0]
                                                       ) * self.c_1 * self.output_h * self.output_w * self.c_0
                                            offset_c1 = idx_c1 * self.output_h * self.output_w * self.c_0
                                            offset_h = (idx_h * self.block_shape[1] + idx_blk_1 -
                                                        self.crops[1][0]) * self.output_w * self.c_0
                                            offset_w = (idx_w * self.block_shape[2] + idx_blk_2 -
                                                        self.crops[2][0]) * self.c_0
                                            offset_gm_out = offset_b + offset_d + offset_c1 + offset_h + offset_w
                                            self.tik_instance.data_move(self.output_tensor[offset_gm_out], ub_tensor, 0,
                                                                        1, burst_len, 0, 0)

    def run(self):
        """run function.
        """
        # select branch
        if self.permute_h * self.permute_w * self.c_0 <= self.ub_ele:
            self.run_height()
        elif self.permute_w * self.c_0 <= self.ub_ele:
            self.run_width()
        else:
            self.run_last()
        # build cce
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[self.input_tensor],
                                   outputs=[self.output_tensor])
        return self.tik_instance


def check_supported(x, y, block_shape, crops, kernel_name="batch_to_space_nd_d"):
    """
    dynamic is not selected when any condition is true: \n
        ori shape must be [?,584,?], block_shape length must be 1, crops length must be 2 \n
    """
    ori_format = x.get("ori_format")
    ori_shape = x.get("ori_shape")
    block_shape = list(block_shape)
    crops = list(crops)
    if tbe_platform.api_check_support("tik.vcopy"):
        return False, ""
    if ori_format in ("NHWC",) and len(ori_shape) == 3 and len(block_shape) == 1 and len(crops) == 2:
        if ori_shape[1] == 584:
            return True, ""
    if ori_format in ("NHWC",) and len(ori_shape) == 4 and len(block_shape) == 2 and len(crops) == 4:
        if (ori_shape[1], ori_shape[2]) in ((33, 33), (22, 22), (11, 11), (8, 8), (6, 6), (5, 5), (4, 4)):
            return True, ""

    return False, ""


def get_op_support_info(x, y, block_shape, crops, kernel_name="batch_to_space_nd_d"):
    """get op support info."""
    format_x = x.get("format").upper()
    if format_x == "NC1HWC0":
        axis_split_matrix = [[SplitInput([0, [1], [-1], [-1]]), SplitOutput([0, [1]])]]
        axis_reduce_list = None

    else:
        axis_split_matrix = None
        axis_reduce_list = None
    op_cal_info_in_json = get_op_cal_info(axis_split_matrix, axis_reduce_list, 0, 0)
    return op_cal_info_in_json


def op_select_format(x, y, block_shape, crops, kernel_name="batch_to_space_nd_d"):
    """
    1. when x's ori_format is in ["NDHWC", "NCDHW"], the Op BatchToSpaceNDD
    can support NDC1HWC0.
    > for example:
    > x : Tensor of (shape=(16, 1, 1, 16, 16, 16), "NDC1HWC0")
    the Op Select can process with NC1HWC0:
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


def check_parms_6hd(shape, dtype, block_shape, crops, kernel_name):
    """check the parameters including shape, dtype, block_shape, crops and kernel_name.
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

    if len(crops) != 3 or len(crops[0]) != 2 or len(crops[1]) != 2 or len(crops[2]) != 2:
        error_detail = "the shape of crops should be 3x2"
        error_manager_vector.raise_err_input_shape_invalid(kernel_name, "crops", error_detail)

    if not (isinstance(block_shape[0], int) and isinstance(block_shape[1], int) and isinstance(block_shape[2], int) and
            block_shape[0] > 0 and block_shape[1] > 0 and block_shape[2] > 0):
        error_detail = "the value of block_shape should be integer and be greater to 0"
        error_manager_vector.raise_err_input_shape_invalid(kernel_name, "block_shape", error_detail)

    if not (isinstance(crops[0][0], int) and crops[0][0] >= 0 and isinstance(crops[0][1], int) and crops[0][1] >= 0 and
            isinstance(crops[1][0], int) and crops[1][0] >= 0 and isinstance(crops[1][1], int) and crops[1][1] >= 0 and
            isinstance(crops[2][0], int) and crops[2][0] >= 0 and isinstance(crops[2][1], int) and crops[2][1] >= 0):
        error_detail = "the value of crops should be integer and be greater to 0"
        error_manager_vector.raise_err_input_shape_invalid(kernel_name, "crops", error_detail)

    if crops[0][0] + crops[0][1] >= shape[1] * block_shape[0]:
        error_detail = "crops in depth dimension should less than (x depth)*(block_shape)"
        error_manager_vector.raise_err_input_shape_invalid(kernel_name, "crops", error_detail)

    if crops[1][0] + crops[1][1] >= shape[3] * block_shape[1]:
        error_detail = "crops in height dimension should less than (x height)*(block_shape)"
        error_manager_vector.raise_err_input_shape_invalid(kernel_name, "crops", error_detail)

    if crops[2][0] + crops[2][1] >= shape[4] * block_shape[2]:
        error_detail = "crops in width dimension should less than (x width)*(block_shape)"
        error_manager_vector.raise_err_input_shape_invalid(kernel_name, "crops", error_detail)

    if shape[0] % (block_shape[0] * block_shape[1] * block_shape[2]) != 0:
        error_detail = "x'batch size/(block depth*block height*block width) should be integer"
        error_manager_vector.raise_err_input_shape_invalid(kernel_name, "x", error_detail)


def batch_to_space_nd_d_6hd(x, y, block_shape, crops, kernel_name="batch_to_space_nd_d_6hd"):
    """BatchToSpace for N-D tensors.

    Parameters
    ----------
    x: dict
        the dict of input tensor.
    y: dict
        the dict of output tensor.
    block_shape: list or tuple
        1-D with shape [3].
    crops: list or tuple
        2-D with shape [3, 2], crops[i] = [crop_start, crop_end].
    kernel_name: str
        cce kernel name, default value is "batch_to_space_nd_d_6hd".

    Returns
    -------
    None.
    """
    shape = x.get("shape")
    dtype = x.get("dtype").lower()
    ori_format = x.get("ori_format")

    # check format and set block_shape to 1-D with shape [3] ans set crops to 2-D with shape [3, 2]
    if ori_format in ("NDHWC",):
        if len(block_shape) == 3 and len(crops) == 6:
            crops = [[crops[0], crops[1]], [crops[2], crops[3]], [crops[4], crops[5]]]
    elif ori_format in ("NCDHW",):
        if len(block_shape) == 4 and block_shape[0] == 1:
            block_shape = [block_shape[1], block_shape[2], block_shape[3]]
            if len(crops) == 8 and crops[0] == 0 and crops[1] == 0:
                crops = [[crops[2], crops[3]], [crops[4], crops[5]], [crops[6], crops[7]]]
            if len(crops) == 4 and len(crops[0]) == 2 and len(crops[1]) == 2 and len(crops[2]) == 2 and len(
                    crops[3]) == 2 and crops[0][0] == 0 and crops[0][1] == 0:
                crops = [[crops[1][0], crops[1][1]], [crops[2][0], crops[2][1]], [crops[3][0], crops[3][1]]]
    else:
        error_manager_vector.raise_err_input_format_invalid(kernel_name, "x", "NDHWC,NCDHW", ori_format)

    # check input params
    check_parms_6hd(shape, dtype, block_shape, crops, kernel_name)

    # if output is the same as input, call the function 'copy_only'
    if block_shape[0] == 1 and block_shape[1] == 1 and block_shape[2] == 1 and crops[0][0] == 0 and crops[0][
            1] == 0 and crops[1][0] == 0 and crops[1][1] == 0 and crops[2][0] == 0 and crops[2][1] == 0:
        copy_only(x, x, kernel_name)
        return

    # run tik
    batch_to_space_nd = BatchToSpaceNdSix(shape, dtype, block_shape, crops, kernel_name)
    batch_to_space_nd.run()


def check_parms_5hd(shape, dtype, block_shape, crops, kernel_name):
    """check the parameters including shape, dtype, block_shape, crops and kernel_name.
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

    if len(crops) != 2 or len(crops[0]) != 2 or len(crops[1]) != 2:
        error_detail = "the shape of crops should be 2x2"
        error_manager_vector.raise_err_input_shape_invalid(kernel_name, "crops", error_detail)

    if not (isinstance(block_shape[0], int) and isinstance(block_shape[1], int) and block_shape[0] > 0 and
            block_shape[1] > 0):
        error_detail = "the value of block_shape should be integer and be greater to 0"
        error_manager_vector.raise_err_input_shape_invalid(kernel_name, "block_shape", error_detail)

    if not (isinstance(crops[0][0], int) and crops[0][0] >= 0 and isinstance(crops[0][1], int) and crops[0][1] >= 0 and
            isinstance(crops[1][0], int) and crops[1][0] >= 0 and isinstance(crops[1][1], int) and crops[1][1] >= 0):
        error_detail = "the value of crops should be integer and be greater to 0"
        error_manager_vector.raise_err_input_shape_invalid(kernel_name, "crops", error_detail)

    if crops[0][0] + crops[0][1] >= shape[2] * block_shape[0]:
        error_detail = "crops in height dimension should less than (x height)*(x height)"
        error_manager_vector.raise_err_input_shape_invalid(kernel_name, "crops", error_detail)

    if crops[1][0] + crops[1][1] >= shape[3] * block_shape[1]:
        error_detail = "crops in width dimension should less than (x width)*(x width)"
        error_manager_vector.raise_err_input_shape_invalid(kernel_name, "crops", error_detail)

    if shape[0] % (block_shape[0] * block_shape[1]) != 0:
        error_detail = "x'batch size/(block height*block width) should be integer"
        error_manager_vector.raise_err_input_shape_invalid(kernel_name, "x", error_detail)


def batch_to_space_nd_d_5hd(x, y, block_shape, crops, kernel_name="batch_to_space_nd_d_5hd"):
    """BatchToSpace for N-D tensors.

    Parameters
    ----------
    x: dict
        the dict of input tensor.
    y: dict
        the dict of output tensor.
    block_shape: list or tuple
        1-D with shape [1] or [2].
    crops: list or tuple
        2-D with shape [1, 2] or [2, 2], crops[i] = [crop_start, crop_end].
    kernel_name: str
        cce kernel name, default value is "batch_to_space_nd_d_5hd".

    Returns
    -------
    None.
    """
    shape = x.get("shape")
    dtype = x.get("dtype").lower()
    ori_format = x.get("ori_format")

    # check format and set block_shape to 1-D with shape [2] ans set crops to 2-D with shape [2, 2]
    if ori_format in ("NHWC",):
        if len(block_shape) == 1:
            block_shape = [1, block_shape[0]]
            if len(crops) == 2:
                crops = [[0, 0], [crops[0], crops[1]]]
            if len(crops) == 1 and len(crops[0]) == 2:
                crops = [[0, 0], [crops[0][0], crops[0][1]]]
        if len(block_shape) == 2 and len(crops) == 4:
            crops = [[crops[0], crops[1]], [crops[2], crops[3]]]
    elif ori_format in ("NCHW",):
        if len(block_shape) == 3 and block_shape[0] == 1:
            block_shape = [block_shape[1], block_shape[2]]
            if len(crops) == 6 and crops[0] == 0 and crops[1] == 0:
                crops = [[crops[2], crops[3]], [crops[4], crops[5]]]
            if len(crops) == 3 and len(crops[0]) == 2 and len(crops[1]) == 2 and len(
                    crops[2]) == 2 and crops[0][0] == 0 and crops[0][1] == 0:
                crops = [[crops[1][0], crops[1][1]], [crops[2][0], crops[2][1]]]
    else:
        error_manager_vector.raise_err_input_format_invalid(kernel_name, "x", "NHWC,NCHW", ori_format)

    # check input params
    check_parms_5hd(shape, dtype, block_shape, crops, kernel_name)

    # if output is the same as input, call the function 'copy_only'
    if block_shape[0] == 1 and block_shape[1] == 1 and crops[0][0] == 0 and crops[0][1] == 0 and crops[1][
            0] == 0 and crops[1][1] == 0:
        copy_only(x, x, kernel_name)
        return

    # if crops is zero, call the function 'transpose_d'
    if crops[0][0] == 0 and crops[0][1] == 0 and crops[1][0] == 0 and crops[1][1] == 0:
        new_shape_input = (block_shape[0], block_shape[1], shape[0] // block_shape[0] // block_shape[1], shape[1],
                           shape[2], shape[3], shape[4])
        new_shape_output = (shape[0] // block_shape[0] // block_shape[1], shape[1], shape[2], block_shape[0], shape[3],
                            block_shape[1], shape[4])
        x.update({"shape": new_shape_input})
        y.update({"shape": new_shape_output})
        transpose_d(x, y, [2, 3, 4, 0, 5, 1, 6], kernel_name)
        return

    # call ir build
    data = tvm.placeholder(shape, name="data", dtype=dtype)
    batch = BatchToSpaceNdFive(shape, dtype, block_shape, crops)
    res = tvm.extern([batch.output_shape], [data],
                     lambda ins, outs: batch.kernel_ir(outs, ins),
                     dtype=dtype,
                     name="res")
    sch = tvm.create_schedule(res.op)
    with tbe_build.build_config():
        tvm.build(sch, [data, res], "cce", name=kernel_name)


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_LIST_INT,
                            (para_check.REQUIRED_ATTR_LIST_INT, para_check.REQUIRED_ATTR_LIST_LIST_INT),
                            para_check.KERNEL_NAME)
def batch_to_space_nd_d(x, y, block_shape, crops, kernel_name="batch_to_space_nd_d"):
    """BatchToSpace for N-D tensors.

    Parameters
    ----------
    x: dict
        the dict of input tensor.
    y: dict
        the dict of output tensor.
    block_shape: list or tuple
        1-D with shape [1], [2] or [3].
    crops: list or tuple
        2-D with shape [1, 2], [2, 2] or [3, 2], crops[i] = [crop_start, crop_end].
    kernel_name: str
        cce kernel name, default value is "batch_to_space_nd_d".

    Returns
    -------
    None.
    """
    input_format = x.get("format")

    if input_format not in ("NC1HWC0", "NDC1HWC0"):
        error_manager_vector.raise_err_input_format_invalid(kernel_name, "x", "NC1HWC0,NDC1HWC0", input_format)

    if input_format in ("NC1HWC0",):
        batch_to_space_nd_d_5hd(x, y, block_shape, crops, kernel_name)
        return

    if input_format in ("NDC1HWC0",):
        batch_to_space_nd_d_6hd(x, y, block_shape, crops, kernel_name)
        return
