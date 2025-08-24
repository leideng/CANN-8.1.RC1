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
anchor_response_flags.py
"""

from impl.util.platform_adapter import tik
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe_platform


class AnchorResponseFlags():
    """
    class of AnchorResponseFlags op
    """
    # 'pylint: disable=too-many-arguments
    def __init__(self, gt_bboxes, flags, featmap_size, strides, num_base_anchors,
                 kernel_name="anchor_response_flags"):
        self.input_shape = gt_bboxes.get("shape")
        self.input_dtype = gt_bboxes.get("dtype")
        self.output_shape = flags.get("shape")
        self.output_dtype = flags.get("dtype")
        self.featmap_h = featmap_size[0]
        self.featmap_w = featmap_size[1]
        self.hw = self.featmap_h * self.featmap_w
        self.stride_h = strides[0]
        self.stride_w = strides[1]
        self.num_base_anchors = num_base_anchors
        self.kernel_name = kernel_name
        self.n = self.input_shape[0]
        self.aligned_n = (self.n + 7) // 8 * 8
        self.input_size = self.input_shape[0] * self.input_shape[1]
        self.output_size = self.output_shape[0]

        self.tik_instance = tik.Tik()
        self.block_byte_size = 32

        self.input_gm = self.tik_instance.Tensor(self.input_dtype, self.input_shape, scope=tik.scope_gm,
                                                 name="input_gm")
        self.output_gm = self.tik_instance.Tensor(self.output_dtype, self.output_shape, scope=tik.scope_gm,
                                                  name="output_gm")

    @staticmethod
    def get_dtype_size(dtype):
        """
        get byte size of dtype
        """
        dtype_dict = {"float32": 4, "uint8": 1, "int32": 4, "float16": 2}
        return dtype_dict.get(dtype)

    # 'pylint: disable=too-many-arguments
    def data_move(self, dst, src, offsets, num, src_stride=0, dst_stride=0):
        """
        move data
        """
        dst_offset, src_offset = offsets
        sid = 0
        nburst = 1
        dtype_byte_size = self.get_dtype_size(dst.dtype)
        data_each_block = self.block_byte_size // dtype_byte_size
        burst_len = (num + data_each_block - 1) // data_each_block
        self.tik_instance.data_move(dst[dst_offset], src[src_offset], sid, nburst, burst_len, src_stride=src_stride,
                                    dst_stride=dst_stride)

    def dup_value(self, dst, num, dup_value=0, offset=0):
        """
        dup value to ub
        """
        dtype_byte_size = self.get_dtype_size(dst.dtype)
        mask = 256 // dtype_byte_size
        stride = 8

        loop = num // (mask * 255)
        if loop > 0:
            with self.tik_instance.for_range(0, loop) as index:
                tmp_offset = offset + index * mask * 255
                self.tik_instance.vec_dup(mask, dst[tmp_offset], dup_value, 255, stride)
            offset += loop * mask * 255

        repeat_time = (num % (mask * 255)) // mask
        if repeat_time > 0:
            self.tik_instance.vec_dup(mask, dst[offset], dup_value, repeat_time, stride)
            offset += repeat_time * mask

        last_num = num % mask
        if last_num > 0:
            self.tik_instance.vec_dup(last_num, dst[offset], dup_value, 1, stride)

    # 'pylint: disable=too-many-locals, too-many-arguments
    def bboxes_add(self, dst, src0, src1, offsets, num, dst_stride=8, src0_stride=8, src1_stride=8):
        """
        add ub tensor
        """
        dst_offset, src0_offset, src1_offset = offsets
        dtype_byte_size = self.get_dtype_size(dst.dtype)
        mask = 256 // dtype_byte_size
        loop = num // (mask * 255)
        if loop > 0:
            with self.tik_instance.for_range(0, loop) as index:
                tmp_dst = dst_offset + index * mask * 255
                tmp_src0 = src0_offset + index * mask * 255
                tmp_src1 = src1_offset + index * mask * 255
                self.tik_instance.vec_add(mask, dst[tmp_dst], src0[tmp_src0], src1[tmp_src1], 255, dst_stride,
                                          src0_stride, src1_stride)
            dst_offset += loop * mask * 255
            src0_offset += loop * mask * 255
            src1_offset += loop * mask * 255

        repeat_time = (num % (mask * 255)) // mask
        if repeat_time > 0:
            self.tik_instance.vec_add(mask, dst[dst_offset], src0[src0_offset], src1[src1_offset], repeat_time,
                                      dst_stride, src0_stride, src1_stride)
            dst_offset += mask * repeat_time
            src0_offset += mask * repeat_time
            src1_offset += mask * repeat_time

        tail_num = num % mask
        if tail_num > 0:
            self.tik_instance.vec_add(tail_num, dst[dst_offset], src0[src0_offset], src1[src1_offset], 1,
                                      dst_stride, src0_stride, src1_stride)

    # 'pylint: disable=too-many-locals, too-many-arguments
    def bboxes_muls(self, dst, src, scalar, offsets, num, dst_stride=8, src_stride=8):
        """
        mul ub tensor and scalar
        """
        dst_offset, src_offset = offsets
        dtype_byte_size = self.get_dtype_size(dst.dtype)
        mask = 256 // dtype_byte_size
        loop = num // (mask * 255)

        if loop > 0:
            with self.tik_instance.for_range(0, loop) as index:
                tmp_dst = dst_offset + index * mask * 255
                tmp_src = src_offset + index * mask * 255
                self.tik_instance.vec_muls(mask, dst[tmp_dst], src[tmp_src], scalar, 255, dst_stride, src_stride)
            dst_offset += loop * mask * 255
            src_offset += loop * mask * 255

        repeat_time = (num % (mask * 255)) // mask
        if repeat_time > 0:
            self.tik_instance.vec_muls(mask, dst[dst_offset], src[src_offset], scalar, repeat_time, dst_stride,
                                       src_stride)
            dst_offset += mask * repeat_time
            src_offset += mask * repeat_time

        tail_num = num % mask
        if tail_num > 0:
            self.tik_instance.vec_muls(tail_num, dst[dst_offset], src[src_offset], scalar, 1, dst_stride,
                                       src_stride)

    def bboxes_mul(self, dst, src0, src1, offsets, num, dst_stride=8, src0_stride=8, src1_stride=8):
        """
        mul ub tensor
        """
        dst_offset, src0_offset, src1_offset = offsets
        dtype_byte_size = self.get_dtype_size(dst.dtype)
        mask = 256 // dtype_byte_size
        loop = num // (mask * 255)

        if loop > 0:
            with self.tik_instance.for_range(0, loop) as index:
                tmp_dst = dst_offset + index * mask * 255
                tmp_src0 = src0_offset + index * mask * 255
                tmp_src1 = src1_offset + index * mask * 255
                self.tik_instance.vec_mul(mask, dst[tmp_dst], src0[tmp_src0], src1[tmp_src1], 255, dst_stride,
                                          src0_stride, src1_stride)
            dst_offset += loop * mask * 255
            src0_offset += loop * mask * 255
            if src1_stride > 0:
                src1_offset += loop * mask * 255

        repeat_time = (num % (mask * 255)) // mask
        if repeat_time > 0:
            self.tik_instance.vec_mul(mask, dst[dst_offset], src0[src0_offset], src1[src1_offset], repeat_time,
                                      dst_stride, src0_stride, src1_stride)
            dst_offset += mask * repeat_time
            src0_offset += mask * repeat_time
            if src1_stride > 0:
                src1_offset += mask * repeat_time

        tail_num = num % mask
        if tail_num > 0:
            self.tik_instance.vec_mul(tail_num, dst[dst_offset], src0[src0_offset], src1[src1_offset], 1,
                                      dst_stride, src0_stride, src1_stride)

    # 'pylint: disable=too-many-locals, too-many-arguments
    def bboxes_conv(self, dst, src, round_mode, offsets, num, dst_stride=8, src_stride=8):
        """
        conv data type
        """
        dst_offset, src_offset = offsets
        byte_size0 = self.get_dtype_size(dst.dtype)
        byte_size1 = self.get_dtype_size(src.dtype)
        dtype_byte_size = byte_size0 if byte_size0 >= byte_size1 else byte_size1
        mask = 256 // dtype_byte_size
        loop = num // (mask * 255)

        if loop > 0:
            with self.tik_instance.for_range(0, loop) as index:
                tmp_dst = dst_offset + index * mask * 255
                tmp_src = src_offset + index * mask * 255
                self.tik_instance.vec_conv(mask, round_mode, dst[tmp_dst], src[tmp_src], 255, dst_stride, src_stride)
            dst_offset += loop * mask * 255
            src_offset += loop * mask * 255

        repeat_time = (num % (mask * 255)) // mask
        if repeat_time > 0:
            self.tik_instance.vec_conv(mask, round_mode, dst[dst_offset], src[src_offset], repeat_time, dst_stride,
                                       src_stride)
            dst_offset += mask * repeat_time
            src_offset += mask * repeat_time

        tail_num = num % mask
        if tail_num > 0:
            self.tik_instance.vec_conv(tail_num, round_mode, dst[dst_offset], src[src_offset], 1, dst_stride,
                                       src_stride)

    def jump_add(self, bboxes_tensor, add_tensor, start_idx=0):
        """
        jump add ub tensor
        """
        scalar_left = self.tik_instance.Scalar("float32")
        scalar_right = self.tik_instance.Scalar("float32")
        with self.tik_instance.for_range(0, self.n) as idx:
            scalar_left.set_as(bboxes_tensor[start_idx + idx * 4])
            scalar_right.set_as(bboxes_tensor[start_idx + idx * 4 + 2])
            add_tensor[start_idx * self.aligned_n + idx].set_as(scalar_left + scalar_right)

    def get_bboxes_idx(self, idx_tensor):
        """
        get index of gt_bboxes
        """
        with self.tik_instance.new_stmt_scope():
            bboxes_tensor = self.tik_instance.Tensor(self.input_dtype, self.input_shape, scope=tik.scope_ubuf,
                                                     name="bboxes_tensor")
            add_tensor = self.tik_instance.Tensor(self.input_dtype, [self.aligned_n * 2], scope=tik.scope_ubuf,
                                                  name="add_tensor")
            featmap_tensor = self.tik_instance.Tensor("int32", (64,), scope=tik.scope_ubuf, name="featmap_tensor")

            self.data_move(bboxes_tensor, self.input_gm, [0, 0], num=self.input_size)
            self.jump_add(bboxes_tensor, add_tensor, 0)
            self.jump_add(bboxes_tensor, add_tensor, 1)

            self.bboxes_muls(add_tensor, add_tensor, 0.5 / self.stride_h, [0, 0], num=self.n)
            self.bboxes_muls(add_tensor, add_tensor, 0.5 / self.stride_w, [self.aligned_n, self.aligned_n], num=self.n)

            self.bboxes_conv(idx_tensor, add_tensor, "floor", [0, 0], num=self.aligned_n * 2)

            self.dup_value(featmap_tensor, num=64, dup_value=self.featmap_w)
            self.bboxes_mul(idx_tensor, idx_tensor, featmap_tensor, [self.aligned_n, self.aligned_n, 0], num=self.n,
                            src1_stride=0)
            self.bboxes_add(idx_tensor, idx_tensor, idx_tensor, [0, 0, self.aligned_n], num=self.n)

    def get_grid(self):
        """
        get grid
        """
        idx_tensor = self.tik_instance.Tensor("int32", [self.aligned_n * 2], scope=tik.scope_ubuf, name="idx_tensor")
        self.get_bboxes_idx(idx_tensor)

        idx_scalar = self.tik_instance.Scalar("int32")
        scalar_one = self.tik_instance.Scalar("float16", init_value=1.0)
        grid_tensor = self.tik_instance.Tensor(self.output_dtype, self.output_shape, scope=tik.scope_ubuf,
                                               name="grid_tensor")
        grid_tensor_fp16 = self.tik_instance.Tensor("float16", self.output_shape, scope=tik.scope_ubuf,
                                                    name="grid_tensor_fp16")
        self.dup_value(grid_tensor_fp16, num=self.output_size)

        with self.tik_instance.for_range(0, self.n) as idx:
            idx_scalar.set_as(idx_tensor[idx])
            with self.tik_instance.if_scope(idx_scalar < 0):
                idx_scalar.set_as(self.hw + idx_scalar)

            with self.tik_instance.for_range(0, self.num_base_anchors) as num_idx:
                grid_tensor_fp16[idx_scalar * self.num_base_anchors + num_idx].set_as(scalar_one)

        self.bboxes_conv(grid_tensor, grid_tensor_fp16, "", [0, 0], num=self.output_size, dst_stride=4, src_stride=8)
        self.data_move(self.output_gm, grid_tensor, [0, 0], num=self.output_size)

    def compute(self):
        """
        op compute
        """
        self.get_grid()
        self.tik_instance.BuildCCE(self.kernel_name, inputs=[self.input_gm], outputs=[self.output_gm])
        return self.tik_instance


def check_params(gt_bboxes, flags, featmap_size, num_base_anchors):
    """
    check parameters of anchor_response_flags
    """
    input_shape = gt_bboxes.get("shape")
    input_dtype = gt_bboxes.get("dtype")
    output_shape = flags.get("shape")
    output_dtype = flags.get("dtype")
    featmap_h = featmap_size[0]
    featmap_w = featmap_size[1]
    output_size = output_shape[0]
    calc_size = featmap_w * featmap_h * num_base_anchors
    ub_size = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)

    max_n = (ub_size // 4 - 256) // 8

    use_ub = 8 * max_n
    left_ub = ub_size - use_ub
    max_out = left_ub // 3

    n = input_shape[0]
    c = input_shape[1]

    if n > max_n:
        raise RuntimeError("the first shape of gt_bboxes must be smaller than %s" % max_n)
    if c != 4:
        raise RuntimeError("the second shape of gt_bboxes must be 4")
    if input_dtype != "float32":
        raise RuntimeError("input dtype must be float32")
    if output_dtype != "uint8":
        raise RuntimeError("output dtype must be uint8")
    if output_size != calc_size:
        raise RuntimeError("output_size should be equal to featmap_w * featmap_h * num_base_anchors")
    if calc_size > max_out:
        raise RuntimeError("output_size must be smaller than %s" % max_out)


# 'pylint: disable=too-many-arguments
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_LIST_INT, para_check.REQUIRED_ATTR_LIST_INT,
                            para_check.REQUIRED_ATTR_INT, para_check.KERNEL_NAME)
def anchor_response_flags(gt_bboxes, flags, featmap_size, strides, num_base_anchors,
                          kernel_name="anchor_response_flags"):
    """
    Generate the responsible flags of anchor in a single feature map.
    :param gt_bboxes: Ground truth box, 2-D Tensor with shape `[batch, 4]`.
    :param flags: The valid flags of each anchor in a single level.
    :param featmap_size: The size of feature maps, listint.
    :param strides: Stride of current level, listint
    :param num_base_anchors: The number of base anchors.
    :param kernel_name: kernel_name
    """
    check_params(gt_bboxes, flags, featmap_size, num_base_anchors)
    obj = AnchorResponseFlags(gt_bboxes, flags, featmap_size, strides, num_base_anchors, kernel_name)
    inst = obj.compute()
    return inst
