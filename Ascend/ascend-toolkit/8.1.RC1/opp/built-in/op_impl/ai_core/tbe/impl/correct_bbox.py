# Copyright 2023 Huawei Technologies Co., Ltd
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
correct_bbox.py
"""

import math
from tbe import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import PlatformApi


# 'pylint: disable=too-many-arguments,too-many-locals
def check_supported(x, grid, anchor_grid, y, stride, yolo_version):
    """
    check supported, if one is invalid, then return false
    """
    soc_version = PlatformApi.get_soc_spec(PlatformApi.SHORT_SOC_VERSION)
    support_version = ("Ascend310P",)
    if soc_version not in support_version:
        return False, f"soc_version not support {soc_version}, only support Ascend310P"

    support_dtype = ("float16",)
    input_params = (x, grid, anchor_grid, y)
    for input_param in input_params:
        data_dtype = input_param.get("dtype").lower()
        if data_dtype not in support_dtype:
            return False, f"dtype not support {data_dtype}, only support float16"

    x_shape = tuple(x.get("ori_shape"))
    if len(x_shape) != 5:
        return False, "input shape not support"
    bs, na, no, ny, nx = x_shape
    support_shape = ((80, 80), (40, 40), (20, 20), (10, 10))
    if (ny, nx) not in support_shape:
        return False, "input shape not support"

    if stride <= 0:
        return False, "stride must be larger than zero."

    support_version = ("V3", "V5", "V7")
    if yolo_version not in support_version:
        return False, f"yolo_version not support {yolo_version}"

    return True, "check support for correct_bbox op success"


class CorrectBBox():
    """
    class of CorrectBBox op
    """

    def __init__(self, x, grid, anchor_grid, y, stride, yolo_version, kernel_name="correct_bbox"):
        # input parameters
        self.x_shape = x.get("shape")
        self.x_dtype = x.get("dtype")
        self.bs, self.na, self.no, self.ny, self.nx = self.x_shape

        self.stride = stride
        self.yolo_version = yolo_version
        self.kernel_name = kernel_name

        # general parameters
        self.tik_inst = tik.Tik()
        self.block_byte_size = 32
        self.ub_size = PlatformApi.get_soc_spec(PlatformApi.UB_SIZE)
        self.core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.dtype_byte_size = self.get_dtype_size(self.x_dtype)
        self.data_each_block = self.block_byte_size // self.dtype_byte_size

        # init gm data
        self.init_gm_data(x, grid, anchor_grid, y)

        # calculate data num each core
        self.total_data_num = self.bs * self.na
        self.num_each_core = math.ceil(self.total_data_num / self.core_num)
        self.block_num = math.ceil(self.total_data_num / self.num_each_core)
        self.num_last_core = self.total_data_num - self.num_each_core * (self.block_num - 1)

        # calculate data size each core
        self.data_num = self.no * self.ny * self.nx
        self.xy_num = 2 * self.ny * self.nx
        self.hw_num = 2 * self.ny * self.nx
        self.cls_num = (self.no - 4) * self.ny * self.nx
        self.cls_loop_num = (self.no - 4) * 16 * 16 if self.ny >= 16 else (self.no - 4) * self.ny * self.nx

        self.cls_loop = self.cls_num // self.cls_loop_num
        self.cls_tail_num = self.cls_num % self.cls_loop_num

    @staticmethod
    def get_dtype_size(dtype):
        """
        get byte size of dtype
        """
        dtype_dict = {"float32": 4, "float16": 2}
        return dtype_dict.get(dtype)

    def init_gm_data(self, x, grid, anchor_grid, y):
        """
        init gm data
        """
        # init input data
        self.x_gm = self.tik_inst.Tensor(x.get("dtype"), x.get("shape"), name="x_gm", scope=tik.scope_gm)
        self.grid_gm = self.tik_inst.Tensor(grid.get("dtype"), grid.get("shape"), name="grid_gm", scope=tik.scope_gm)
        self.anchor_grid_gm = self.tik_inst.Tensor(anchor_grid.get("dtype"), anchor_grid.get("shape"),
                                                   name="anchor_grid_gm", scope=tik.scope_gm)

        # init output data
        self.y_gm = self.tik_inst.Tensor(y.get("dtype"), y.get("shape"), name="y_gm", scope=tik.scope_gm)

    def compute(self):
        """
        compute entrance
        """
        with self.tik_inst.for_range(0, self.block_num, block_num=self.block_num) as block_id:
            n_num = self.tik_inst.Scalar("int32")
            with self.tik_inst.if_scope(block_id != self.block_num - 1):
                n_num.set_as(self.num_each_core)
            with self.tik_inst.else_scope():
                n_num.set_as(self.num_last_core)

            self.compute_each_block(block_id, n_num)

        self.tik_inst.BuildCCE(kernel_name=self.kernel_name, inputs=[self.x_gm, self.grid_gm, self.anchor_grid_gm],
                               outputs=[self.y_gm])
        return self.tik_inst

    def compute_each_block(self, block_id, n_num):
        """
        compute each block
        """
        with self.tik_inst.for_range(0, n_num) as n_id:
            n_data = self.num_each_core * block_id + n_id
            block_offset = n_data * self.data_num

            # data fp16-->fp32
            self.dtype_byte_size = 4
            self.x_dtype = "float32"

            # compute_xy_each_block
            xy_offset = block_offset
            grid_offset = (n_data % self.na) * self.xy_num
            with self.tik_inst.new_stmt_scope():
                self.compute_xy_each_block(xy_offset, grid_offset, self.xy_num)

            # compute_hw_each_block
            hw_offset = block_offset + self.xy_num
            if self.yolo_version in ("V3", "V5"):
                anchor_grid_offset = (n_data % self.na) * self.hw_num
            else:
                anchor_grid_offset = (n_data % self.na) * 2
            with self.tik_inst.new_stmt_scope():
                self.compute_hw_each_block(hw_offset, anchor_grid_offset, self.hw_num)

            # data fp32-->fp16
            self.dtype_byte_size = 2
            self.x_dtype = "float16"

            # compute_cls_each_loop
            if self.cls_loop > 0:
                with self.tik_inst.for_range(0, self.cls_loop, thread_num=2) as loop_id:
                    self.compute_cls_each_loop(block_offset + self.xy_num + self.hw_num + loop_id * self.cls_loop_num, 
                                               self.cls_loop_num)

            if self.cls_tail_num > 0:
                with self.tik_inst.new_stmt_scope():
                    self.compute_cls_each_loop(block_offset + self.xy_num + self.hw_num + 
                                               self.cls_loop * self.cls_loop_num, self.cls_tail_num)

    def compute_xy_each_block(self, xy_offset, grid_offset, xy_num):
        """
        compute each block for xy
        """
        ub_num = math.ceil(xy_num / self.data_each_block) * self.data_each_block
        xy_ub = self.tik_inst.Tensor("float16", [ub_num], scope=tik.scope_ubuf, name="xy_ub")
        grid_ub = self.tik_inst.Tensor("float16", [ub_num], scope=tik.scope_ubuf, name="grid_ub")
        xy_ub_fp32 = self.tik_inst.Tensor("float32", [ub_num], scope=tik.scope_ubuf, name="xy_ub_fp32")
        grid_ub_fp32 = self.tik_inst.Tensor("float32", [ub_num], scope=tik.scope_ubuf, name="grid_ub_fp32")
        grid_offset = grid_offset if self.yolo_version in ("V3", "V5") else 0

        # data move gm2ub
        self.data_move_align(xy_ub, self.x_gm, xy_offset, xy_num, "gm2ub")
        self.data_move_align(grid_ub, self.grid_gm, grid_offset, xy_num, "gm2ub")

        # data fp16-->fp32
        self.data_conv(xy_ub_fp32, xy_ub, [0, 0], ub_num, 8, 4)
        self.data_conv(grid_ub_fp32, grid_ub, [0, 0], ub_num, 8, 4)

        #compute
        self.data_sigmoid(xy_ub_fp32, xy_ub_fp32, [0, 0, 0], ub_num)
        if self.yolo_version in ("V3", "V5"):
            self.data_muls(xy_ub_fp32, xy_ub_fp32, 2.0, [0, 0], ub_num)
            self.data_adds(xy_ub_fp32, xy_ub_fp32, -0.5, [0, 0], ub_num)
            self.data_add(xy_ub_fp32, xy_ub_fp32, grid_ub_fp32, [0, 0, 0], ub_num)
            self.data_muls(xy_ub_fp32, xy_ub_fp32, self.stride, [0, 0], ub_num)
        elif self.yolo_version in ("V7",):
            self.data_muls(xy_ub_fp32, xy_ub_fp32, self.stride, [0, 0], ub_num)
            self.data_add(xy_ub_fp32, xy_ub_fp32, grid_ub_fp32, [0, 0, 0], ub_num)

        # data fp32-->fp16
        self.data_conv(xy_ub, xy_ub_fp32, [0, 0], ub_num, 4, 8)

        # data move ub2gm
        self.data_move_align(self.y_gm, xy_ub, xy_offset, xy_num, "ub2gm")

    def compute_hw_each_block(self, hw_offset, anchor_grid_offset, hw_num):
        """
        compute each block for hw
        """
        if self.yolo_version in ("V3", "V5"):
            ub_num = math.ceil(hw_num / self.data_each_block) * self.data_each_block
            hw_ub = self.tik_inst.Tensor("float16", [ub_num], scope=tik.scope_ubuf, name="hw_ub")
            anchor_grid_ub = self.tik_inst.Tensor("float16", [ub_num], scope=tik.scope_ubuf, name="anchor_grid_ub")
            hw_ub_fp32 = self.tik_inst.Tensor("float32", [ub_num], scope=tik.scope_ubuf, name="hw_ub_fp32")
            anchor_grid_ub_fp32 = self.tik_inst.Tensor("float32", [ub_num], scope=tik.scope_ubuf, 
                                                       name="anchor_grid_ub_fp32")

            # data move gm2ub
            self.data_move_align(hw_ub, self.x_gm, hw_offset, hw_num, "gm2ub")
            self.data_move_align(anchor_grid_ub, self.anchor_grid_gm, anchor_grid_offset, hw_num, "gm2ub")

            # data fp16-->fp32
            self.data_conv(hw_ub_fp32, hw_ub, [0, 0], ub_num, 8, 4)
            self.data_conv(anchor_grid_ub_fp32, anchor_grid_ub, [0, 0], ub_num, 8, 4)

            #compute
            self.data_sigmoid(hw_ub_fp32, hw_ub_fp32, [0, 0, 0], ub_num)
            self.data_muls(hw_ub_fp32, hw_ub_fp32, 2.0, [0, 0], ub_num)
            self.data_mul(hw_ub_fp32, hw_ub_fp32, hw_ub_fp32, [0, 0, 0], ub_num)
            self.data_mul(hw_ub_fp32, hw_ub_fp32, anchor_grid_ub_fp32, [0, 0, 0], ub_num)

            # data fp32-->fp16
            self.data_conv(hw_ub, hw_ub_fp32, [0, 0], ub_num, 4, 8)

            # data move ub2gm
            self.data_move_align(self.y_gm, hw_ub, hw_offset, hw_num, "ub2gm")

        elif self.yolo_version in ("V7",):
            hw_num = self.ny * self.nx
            ub_num = math.ceil(hw_num / self.data_each_block) * self.data_each_block
            hw_ub = self.tik_inst.Tensor("float16", [ub_num], scope=tik.scope_ubuf, name="hw_ub")
            anchor_grid_ub = self.tik_inst.Tensor("float16", [2], scope=tik.scope_ubuf, name="anchor_grid_ub")
            anchor_grid_ub_fp32 = self.tik_inst.Tensor("float32", [2], scope=tik.scope_ubuf, name="anchor_grid_ub_fp32")
            hw_ub_fp32 = self.tik_inst.Tensor("float32", [ub_num], scope=tik.scope_ubuf, name="hw_ub_fp32")

            anchor_grid_scalar = self.tik_inst.Scalar("float32")

            self.data_move(anchor_grid_ub, self.anchor_grid_gm[anchor_grid_offset], num=2)

            with self.tik_inst.for_range(0, 2) as hw_id:
                # data move gm2ub
                self.data_move_align(hw_ub, self.x_gm, hw_offset + hw_id * hw_num, hw_num, "gm2ub")

                # data fp16-->fp32
                self.data_conv(hw_ub_fp32, hw_ub, [0, 0], ub_num, 8, 4)
                self.data_conv(anchor_grid_ub_fp32, anchor_grid_ub, [0, 0], 2, 8, 4)

                # compute
                with self.tik_inst.new_stmt_scope():
                    self.data_sigmoid(hw_ub_fp32, hw_ub_fp32, [0, 0, 0], ub_num)
                self.data_mul(hw_ub_fp32, hw_ub_fp32, hw_ub_fp32, [0, 0, 0], ub_num)
                anchor_grid_scalar.set_as(anchor_grid_ub_fp32[hw_id])
                self.data_muls(hw_ub_fp32, hw_ub_fp32, anchor_grid_scalar, [0, 0], ub_num)

                # data fp32-->fp16
                self.data_conv(hw_ub, hw_ub_fp32, [0, 0], ub_num, 4, 8)

                # data move ub2gm
                self.data_move_align(self.y_gm, hw_ub, hw_offset + hw_id * hw_num, hw_num, "ub2gm")

    def compute_cls_each_loop(self, cls_offset, cls_num):
        """
        compute each block for class
        """
        ub_num = math.ceil(cls_num / self.data_each_block) * self.data_each_block
        cls_ub = self.tik_inst.Tensor(self.x_dtype, [ub_num], scope=tik.scope_ubuf, name="cls_ub")

        # data move gm2ub
        self.data_move_align(cls_ub, self.x_gm, cls_offset, cls_num, "gm2ub")
        # compute
        self.data_sigmoid(cls_ub, cls_ub, [0, 0, 0], ub_num)
        # data move ub2gm
        self.data_move_align(self.y_gm, cls_ub, cls_offset, cls_num, "ub2gm")

    def data_move_align(self, dst, src, offset, num, trans):
        """
        move data to align
        """
        num_align = num // self.data_each_block * self.data_each_block
        num_extra = num % self.data_each_block

        if trans == "gm2ub":
            if num_align > 0:
                self.data_move(dst, src[offset], num=num_align)
            if num_extra > 0:
                self.data_move(dst[num_align], src[offset + num - self.data_each_block], num=self.data_each_block)

        if trans == "ub2gm":
            if num_align > 0:
                self.data_move(dst[offset], src, num=num_align)
            if num_extra > 0:
                self.data_move(dst[offset + num - self.data_each_block], src[num_align], num=self.data_each_block)

    def data_move(self, dst, src, num, nburst=1):
        """
        move data
        """
        sid, src_stride, dst_stride = 0, 0, 0

        burst_len = (num + self.data_each_block - 1) // self.data_each_block
        self.tik_inst.data_move(dst, src, sid, nburst, burst_len, src_stride=src_stride, dst_stride=dst_stride)

    def data_sigmoid(self, dst, src, offsets, num=0, dst_blk_stride=1, src0_blk_stride=1,
                     src1_blk_stride=1, dst_rep_stride=8, src0_rep_stride=8, src1_rep_stride=8):
        """
        tik sigmoid
        """
        with self.tik_inst.for_range(0, 1, thread_num=1):
            tmp1 = self.tik_inst.Tensor(self.x_dtype, [num], scope=tik.scope_ubuf, name="tmp1")
            tmp2 = self.tik_inst.Tensor(self.x_dtype, [num], scope=tik.scope_ubuf, name="tmp2")

            self.data_muls(dst, src, -1.0, offsets, num)
            self.data_exp(dst, src, offsets, num)
            self.data_adds(dst, src, 1.0, offsets, num)
            self.data_rec(tmp1, dst, offsets, num)

            self.data_mul(tmp2, tmp1, dst, offsets, num)
            self.data_muls(tmp2, tmp2, -1.0, offsets, num)
            self.data_adds(tmp2, tmp2, 2.0, offsets, num)
            self.data_mul(tmp2, tmp1, tmp2, offsets, num)

            self.data_mul(dst, dst, tmp2, offsets, num)
            self.data_muls(dst, dst, -1.0, offsets, num)
            self.data_adds(dst, dst, 2.0, offsets, num)
            self.data_mul(dst, dst, tmp2, offsets, num)

    def data_adds(self, dst, src, scalar, offsets, num=0, dst_stride=8, src_stride=8):
        """
        tik adds
        """
        self.single_operator_template(self.tik_inst.vec_adds, dst, src, scalar, offsets, num, dst_stride,
                                      src_stride)

    def data_add(self, dst, src0, src1, offsets, num=0, dst_stride=8, src0_stride=8, src1_stride=8):
        """
        tik add
        """
        self.double_operator_template(self.tik_inst.vec_add, dst, src0, src1, offsets, num, dst_stride, src0_stride,
                                      src1_stride)

    def data_muls(self, dst, src, scalar, offsets, num=0, dst_stride=8, src_stride=8):
        """
        tik muls
        """
        self.single_operator_template(self.tik_inst.vec_muls, dst, src, scalar, offsets, num, dst_stride,
                                      src_stride)

    def data_mul(self, dst, src0, src1, offsets, num=0, dst_stride=8, src0_stride=8, src1_stride=8):
        """
        tik mul
        """
        self.double_operator_template(self.tik_inst.vec_mul, dst, src0, src1, offsets, num, dst_stride, src0_stride,
                                      src1_stride)

    def data_rec(self, dst, src, offsets, num=0, dst_stride=8, src_stride=8):
        """
        tik rec
        """
        dst_offset, src_offset = offsets[0], offsets[1]
        vector_mask_max = 256 // self.dtype_byte_size

        tensor_size = num
        loop = tensor_size // (vector_mask_max * 255)
        
        with self.tik_inst.if_scope(loop > 0):
            with self.tik_inst.for_range(0, loop) as index:
                self.tik_inst.vec_rec(vector_mask_max, dst[dst_offset + index * vector_mask_max * 255], 
                                      src[src_offset + index * vector_mask_max * 255], 255, dst_stride, src_stride)
    
        repeat_time = (tensor_size % (vector_mask_max * 255)) // vector_mask_max
    
        with self.tik_inst.if_scope(repeat_time > 0):
            self.tik_inst.vec_rec(vector_mask_max, dst[dst_offset + loop * vector_mask_max * 255], 
                                  src[src_offset + loop * vector_mask_max * 255], 
                                  repeat_time, dst_stride, src_stride)
    
        last_num = tensor_size % vector_mask_max
        with self.tik_inst.if_scope(last_num > 0):
            self.tik_inst.vec_rec(last_num, 
                                  dst[dst_offset + loop * vector_mask_max * 255 + repeat_time * vector_mask_max],
                                  src[src_offset + loop * vector_mask_max * 255 + repeat_time * vector_mask_max], 
                                  1, dst_stride, src_stride)

    def data_exp(self, dst, src, offsets, num=0, dst_stride=8, src_stride=8):
        """
        tik exp
        """
        dst_offset, src_offset = offsets[0], offsets[1]
        vector_mask_max = 256 // self.dtype_byte_size

        tensor_size = num
        loop = tensor_size // (vector_mask_max * 255)

        with self.tik_inst.if_scope(loop > 0):
            with self.tik_inst.for_range(0, loop) as index:
                self.tik_inst.vec_exp(vector_mask_max, dst[dst_offset + index * vector_mask_max * 255], 
                                      src[src_offset + index * vector_mask_max * 255], 
                                      255, dst_stride, src_stride)

        repeat_time = (tensor_size % (vector_mask_max * 255)) // vector_mask_max

        with self.tik_inst.if_scope(repeat_time > 0):
            self.tik_inst.vec_exp(vector_mask_max, dst[dst_offset + loop * vector_mask_max * 255], 
                                  src[src_offset + loop * vector_mask_max * 255], 
                                  repeat_time, dst_stride, src_stride)

        last_num = tensor_size % vector_mask_max
        with self.tik_inst.if_scope(last_num > 0):
            self.tik_inst.vec_exp(last_num, 
                                  dst[dst_offset + loop * vector_mask_max * 255 + repeat_time * vector_mask_max], 
                                  src[src_offset + loop * vector_mask_max * 255 + repeat_time * vector_mask_max], 
                                  1, dst_stride, src_stride)

    def single_operator_template(self, op_obj, dst, src, scalar, offsets, num=0, dst_stride=8, src_stride=8):
        """
        tik api template
        """
        dst_offset, src_offset = offsets[0], offsets[1]
        vector_mask_max = 256 // self.dtype_byte_size

        tensor_size = num
        loop = tensor_size // (vector_mask_max * 255)

        with self.tik_inst.if_scope(loop > 0):
            with self.tik_inst.for_range(0, loop) as index:
                op_obj(vector_mask_max, dst[ dst_offset + index * vector_mask_max * 255], 
                       src[src_offset + index * vector_mask_max * 255], scalar, 255, dst_stride, src_stride)

        repeat_time = (tensor_size % (vector_mask_max * 255)) // vector_mask_max

        with self.tik_inst.if_scope(repeat_time > 0):
            op_obj(vector_mask_max, dst[dst_offset + loop * vector_mask_max * 255], 
                   src[src_offset + loop * vector_mask_max * 255], scalar, repeat_time, dst_stride, src_stride)

        last_num = tensor_size % vector_mask_max
        with self.tik_inst.if_scope(last_num > 0):
            op_obj(last_num, dst[dst_offset + loop * vector_mask_max * 255 + repeat_time * vector_mask_max], 
                   src[src_offset + loop * vector_mask_max * 255 + repeat_time * vector_mask_max], scalar, 
                   1, dst_stride, src_stride)

    def double_operator_template(self, op_obj, dst, src0, src1, offsets, num=0, dst_stride=8,
                                 src0_stride=8, src1_stride=8):
        """
        tik api template
        """
        dst_offset, src0_offset, src1_offset = offsets[0], offsets[1], offsets[2]
        vector_mask_max = 256 // self.dtype_byte_size

        tensor_size = num
        loop = tensor_size // (vector_mask_max * 255)

        with self.tik_inst.if_scope(loop > 0):
            with self.tik_inst.for_range(0, loop) as index:
                op_obj(vector_mask_max, dst[dst_offset + index * vector_mask_max * 255], 
                       src0[src0_offset + index * vector_mask_max * 255], 
                       src1[src1_offset + index * vector_mask_max * 255],
                       255, dst_stride, src0_stride, src1_stride)

        repeat_time = (tensor_size % (vector_mask_max * 255)) // vector_mask_max

        with self.tik_inst.if_scope(repeat_time > 0):
            op_obj(vector_mask_max, dst[dst_offset + loop * vector_mask_max * 255], 
                   src0[src0_offset + loop * vector_mask_max * 255], 
                   src1[src1_offset + loop * vector_mask_max * 255], 
                   repeat_time, dst_stride, src0_stride, src1_stride)

        last_num = tensor_size % vector_mask_max
        with self.tik_inst.if_scope(last_num > 0):
            op_obj(last_num, dst[dst_offset + loop * vector_mask_max * 255 + repeat_time * vector_mask_max], 
                   src0[src0_offset + loop * vector_mask_max * 255 + repeat_time * vector_mask_max], 
                   src1[src1_offset + loop * vector_mask_max * 255 + repeat_time * vector_mask_max], 
                   1, dst_stride, src0_stride, src1_stride)

    def data_conv(self, dst, src, offsets, num=0, dst_rep_stride=8, src_rep_stride=8):
        """
        conv fp16 <--> fp32
        """
        round_mode = 'none'
        dst_offset, src_offset, vector_mask_max = offsets[0], offsets[1], 64

        tensor_size = num
        loop = tensor_size // (vector_mask_max * 255)
        with self.tik_inst.if_scope(loop > 0):
            with self.tik_inst.for_range(0, loop) as index:
                self.tik_inst.vec_conv(vector_mask_max, round_mode, dst[dst_offset + index * vector_mask_max * 255],
                                       src[src_offset + index * vector_mask_max * 255], 
                                       255, dst_rep_stride, src_rep_stride)

        repeat_time = (tensor_size % (vector_mask_max * 255)) // vector_mask_max

        with self.tik_inst.if_scope(repeat_time > 0):
            self.tik_inst.vec_conv(vector_mask_max, round_mode, dst[dst_offset + loop * vector_mask_max * 255],
                                   src[src_offset + loop * vector_mask_max * 255], 
                                   repeat_time, dst_rep_stride, src_rep_stride)

        last_num = tensor_size % vector_mask_max

        with self.tik_inst.if_scope(last_num > 0):
            self.tik_inst.vec_conv(last_num, round_mode, 
                                   dst[dst_offset + loop * vector_mask_max * 255 + repeat_time * vector_mask_max],
                                   src[src_offset + loop * vector_mask_max * 255 + repeat_time * vector_mask_max], 
                                   1, dst_rep_stride, src_rep_stride)


# 'pylint: disable=too-many-arguments,too-many-locals
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_INT, para_check.REQUIRED_ATTR_STR,
                            para_check.KERNEL_NAME)
def correct_bbox(x, grid, anchor_grid, y, stride, yolo_version, kernel_name="correct_bbox"):
    """
    implementation of correct_bbox and return the tik instance
    ------------------------------------------------------------------
    [input]
    x: A 5D Tensor with shape (N, na, no, H, W),
       na indicates the number of anchors,
       no indicates the number of outputs per anchor, including [xywh, class_num, conf_score].
    grid: A 5D Tensor with shape (1, na, 2, H, W) for V3/V5 and (1, 1, 2, H, W) for V7,
          the value "2" indicates offsets of coordinates.
    anchor_grid: A 5D Tensor with shape (1, na, 2, H, W) for V3/V5 and (1, 1, 2, 1, 1) for V7,
                 the value "2" indicates anchors relative to the original image.

    [output]
    y: A 5D Tensor of type float16 with shape (N, na, no, H, W), same as the input x.

    [attribute]
    stride: A required int32, scale for each box.
    yolo_version: A required string, specifying the YOLO version, optional [V3, V5, V7].
    """
    obj = CorrectBBox(x, grid, anchor_grid, y, stride, yolo_version, kernel_name)
    inst = obj.compute()

    return inst
