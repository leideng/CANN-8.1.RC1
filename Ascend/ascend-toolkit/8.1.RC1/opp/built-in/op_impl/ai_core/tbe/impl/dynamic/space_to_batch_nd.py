# Copyright 2021 Huawei Technologies Co., Ltd
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
dynamic space_to_batch_nd
"""
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import error_manager_vector
from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_op_cal_info
from impl.util.util_select_op_base import get_dynamic_param_in_json
from tbe.common.platform import get_bit_len


# 'pylint:disable=too-few-public-methods,too-many-instance-attributes
class Constant:
    """
    The class for constant
    """
    # max int32
    MAX_INT32 = 2**31 - 1
    # tiling param num
    TILING_ARG_NUM = 24
    # reserved ub size
    RESERVED_UB_SIZE = 8 * 1024
    # 8 bit
    EIGHT_BIT = 8
    # bytes of one block
    BLOCK_BYTES = 32
    # repeat limit
    REPEAT_LIMIT = 255


# 'pylint: disable=too-many-lines,unused-argument,invalid-name
def get_op_support_info(x, block_shape, paddings, y, kernel_name="space_to_batch_nd"):
    """
    get op support info.
    """
    axis_split_list = None
    axis_reduce_list = None
    op_cal_info_in_json = get_op_cal_info(axis_split_list, axis_reduce_list)
    return op_cal_info_in_json


# 'pylint: disable=invalid-name,too-many-locals,unnecessary-pass,too-many-return-statements
def check_supported(x, block_shape, paddings, y, kernel_name="space_to_batch_nd"):
    """
    check supported dynamiclly. \n
    only spported ori_format NHWC,NCHW,NDHWC,NCDHW \n
    ori_format:NHWC \n
        ori shape must be 4([-1,-1,-1,-1]), block_shape must be 1([2]), paddings must be 2([2,2])
        ori shape must be 3([-1,-1,-1]), block_shape must be 1([1]), paddings must be 2([1,2])
    ori format:NCHW \n
        ori shape must be 4([-1,-1,-1,-1]), block_shape must be 1([3]), paddings must be 2([3,2])
    ori format:NDHWC \n
        ori shape must be 5([-1,-1,-1,-1,-1]), block_shape must be 1([3]), paddings must be 2([3,2])
    ori format:NCDHW \n
        ori shape must be 5([-1,-1,-1,-1,-1]), block_shape must be 1([4]), paddings must be 2([4,2])
    """
    ori_format = x.get("ori_format")
    ori_shape = x.get("ori_shape")
    block_s = block_shape.get("shape")
    pad_s = paddings.get("shape")
    if ori_format not in ("NHWC", "NCHW", "NDHWC", "NCDHW"):
        reason = 'ori_format[%s] is not in ("NHWC", "NCHW", "NDHWC", "NCDHW")' % ori_format
        return False, reason
    if len(block_s) != 1 or len(pad_s) != 2 or pad_s[1] != 2:
        reason = "shape of input is not supported, block_s is [%s], pad_s is [%s]" % (str(block_s), str(pad_s))
        return False, reason
    reason = "when ori_format is [%s], shape of input is not supported, ori_shape is [%s], " \
             "block_s is [%s], pad_s is [%s]" % (ori_format, str(ori_shape), str(block_s), str(pad_s))
    if ori_format in ("NHWC",):
        if len(ori_shape) != 4 or block_s[0] != 2 or pad_s[0] != 2:
            if len(ori_shape) != 4 or block_s[0] != 1 or pad_s[0] != 1:
                if len(ori_shape) != 3 or block_s[0] != 1 or pad_s[0] != 1:
                    return False, reason
    elif ori_format in ("NCHW",):
        if len(ori_shape) != 4 or block_s[0] != 3 or pad_s[0] != 3:
            return False, reason
    elif ori_format in ("NDHWC",):
        if len(ori_shape) != 5 or block_s[0] != 3 or pad_s[0] != 3:
            return False, reason
    elif ori_format in ("NCDHW",):
        if len(ori_shape) != 5 or block_s[0] != 4 or pad_s[0] != 4:
            return False, reason

    return True, ""


def op_select_format(x, block_shape, paddings, y, kernel_name="space_to_batch_nd"):
    """
    select format dynamiclly. \n
    op_select_format support desc: \n
    1.when ori_format is 'NHWC' or 'NCHW', input_format is 'NC1HWC0'

        for example:
            ori:
                x              shape = [16,16,16,16]           format = 'NHWC'
                block_shape    shape = [2,]                    format = 'ND'
                pads           shape = [2,2]                   format = 'ND'
                y              shape = [None,None,None,16]     format = 'NHWC'
            format transformer:
                x              shape = [16,1,16,16,16]         format = 'NC1HWC0'
                block_shape    shape = [2,]                    format = 'ND'
                pads           shape = [2,2]                   format = 'ND'
                y              shape = [None,1,None,None,16]   format = 'NC1HWC0'
    2.when ori_format is 'NDHWC' or 'NCDHW', input_format is 'NDC1HWC0'

        for example:
            ori:
                x              shape = [16,16,16,16,16]              format = 'NDHWC'
                block_shape    shape = [3,]                          format = 'ND'
                pads           shape = [3,2]                         format = 'ND'
                y              shape = [None,None,None,None,16]      format = 'NDHWC'
            format transformer:
                x              shape = [16,16,1,16,16,16]            format = 'NDC1HWC0'
                block_shape    shape = [3,]                          format = 'ND'
                pads           shape = [3,2]                         format = 'ND'
                y              shape = [None,None,1,None,None,16]    format = 'NDC1HWC0'
    """
    input_dtype = "float16, float, float16, float, float16, float16, float, float,\
                   bfloat16, bfloat16, bfloat16, bfloat16"
    input_format = "NC1HWC0, NC1HWC0, NC1HWC0, NC1HWC0, NC1HWC0, NC1HWC0, NC1HWC0, NC1HWC0,\
                    NC1HWC0, NC1HWC0, NC1HWC0, NC1HWC0"
    ori_format = x.get("ori_format")
    if ori_format in ("NDHWC", "NCDHW"):
        input_dtype = "float16, float, float16, float, float16, float16, float, float,\
                       bfloat16, bfloat16, bfloat16, bfloat16"
        input_format = "NDC1HWC0, NDC1HWC0, NDC1HWC0, NDC1HWC0, NDC1HWC0, NDC1HWC0, NDC1HWC0, NDC1HWC0,\
                        NDC1HWC0, NDC1HWC0, NDC1HWC0, NDC1HWC0"

    attr_dtype_1 = "int32, int32, int64, int64, int32, int64, int32, int64, int32, int32, int64, int64"
    attr_format = "ND, ND, ND, ND, ND, ND, ND, ND, ND, ND, ND, ND"
    attr_dtype_2 = "int32, int32, int64, int64, int64, int32, int64, int32, int32, int64, int32, int64"

    input0 = gen_param(classify="input0",
                       name="x",
                       datatype=input_dtype,
                       format=input_format,
                       unknownshape_format=input_format)
    input1 = gen_param(classify="input1",
                       name="block_shape",
                       datatype=attr_dtype_1,
                       format=attr_format,
                       unknownshape_format=attr_format)
    input2 = gen_param(classify="input2",
                       name="paddings",
                       datatype=attr_dtype_2,
                       format=attr_format,
                       unknownshape_format=attr_format)
    output0 = gen_param(classify="output0",
                        name="y",
                        datatype=input_dtype,
                        format=input_format,
                        unknownshape_format=input_format)

    param_list = [input0, input1, input2, output0]
    param_dynamic_in_json = get_dynamic_param_in_json(param_list)
    return param_dynamic_in_json


# 'pylint: disable=too-many-function-args,too-many-public-methods
# 'pylint: disable=too-many-instance-attributes,unexpected-keyword-arg
# 'pylint: disable=too-many-arguments,attribute-defined-outside-init,too-many-statements
class SpaceToBatchND:
    """Performs space_to_batch_nd on input tensor
    5HD:
        input:             input_b  c1  input_h  input_w  c0
        pad+reshape(deal): input_b  c1  output_h  block_h  output_w  block_w  c0
        permute:           block_h  block_w  input_b  c1  output_h  output_w  c0
        output:            output_b  c1  output_h  output_w  c0
    6HD:
        input:             input_b  input_d  c1  input_h  input_w  c0
        pad+reshape(deal): input_b  output_d  block_d  c1  output_h  block_h  output_w  block_w  c0
        permute:           block_d  block_h  block_w  input_b  output_d  c1  output_h  output_w  c0
        output:            output_b  output_d  c1  output_h  output_w  c0
    """

    def __init__(self, dtype, block_size, kernel_name):
        """Init batch_to_space_nd parameters
        """
        self.dtype = dtype
        # zero means space_to_batch_nd; not zeros means space_to_batch
        self.block_size = block_size
        self.kernel_name = kernel_name
        self.tik_instance = tik.Tik()
        self.core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.dtype_size = get_bit_len(self.dtype) // Constant.EIGHT_BIT
        self.ub_size = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) - Constant.RESERVED_UB_SIZE
        self.ub_ele = self.ub_size // self.dtype_size
        self.blk_ele = Constant.BLOCK_BYTES // self.dtype_size
        self.init_gm_tensor()
        self.mask_len = 64 if dtype.lower() == "float32" else 128
        self.core_num_var = self.tik_instance.Scalar(name="core_num_var", init_value=self.core_num)
        self.tiling_mode = self.tik_instance.Scalar("int64", name="tiling_mode")
        self.act_core_num = self.tik_instance.Scalar("int64", name="act_core_num")
        self.one_core_ele = self.tik_instance.Scalar("int64", name="one_core_ele")
        self.last_core_ele = self.tik_instance.Scalar("int64", name="last_core_ele")
        self.input_b = self.tik_instance.Scalar("int64", name="input_b")
        self.block_d = self.tik_instance.Scalar("int64", name="block_d")
        self.block_h = self.tik_instance.Scalar("int64", name="block_h")
        self.block_w = self.tik_instance.Scalar("int64", name="block_w")
        self.pads_f = self.tik_instance.Scalar("int64", name="pads_f")
        self.pads_a = self.tik_instance.Scalar("int64", name="pads_a")
        self.pads_t = self.tik_instance.Scalar("int64", name="pads_t")
        self.pads_b = self.tik_instance.Scalar("int64", name="pads_b")
        self.pads_l = self.tik_instance.Scalar("int64", name="pads_l")
        self.pads_r = self.tik_instance.Scalar("int64", name="pads_r")
        self.input_d = self.tik_instance.Scalar("int64", name="input_d")
        self.channel_one = self.tik_instance.Scalar("int64", name="channel_one")
        self.input_h = self.tik_instance.Scalar("int64", name="input_h")
        self.input_w = self.tik_instance.Scalar("int64", name="input_w")
        self.channel_zero = self.tik_instance.Scalar("int64", name="channel_zero")
        self.output_b = self.tik_instance.Scalar("int64", name="output_b")
        self.output_d = self.tik_instance.Scalar("int64", name="output_d")
        self.output_h = self.tik_instance.Scalar("int64", name="output_h")
        self.output_w = self.tik_instance.Scalar("int64", name="output_w")
        self.pad_w = None
        self.tiling_ub = None

    def set_running_core_num(self, tiling_core_num):
        self.core_num_var.set_as(tiling_core_num)

    def tiling_args(self):
        """Get runtime params from tiling
        """
        self.tiling_mode.set_as(self.tiling_ub[0])
        self.act_core_num.set_as(self.tiling_ub[1])
        self.one_core_ele.set_as(self.tiling_ub[2])
        self.last_core_ele.set_as(self.tiling_ub[3])
        self.input_b.set_as(self.tiling_ub[4])
        self.block_d.set_as(self.tiling_ub[5])
        self.block_h.set_as(self.tiling_ub[6])
        self.block_w.set_as(self.tiling_ub[7])
        self.pads_f.set_as(self.tiling_ub[8])
        self.pads_a.set_as(self.tiling_ub[9])
        self.pads_t.set_as(self.tiling_ub[10])
        self.pads_b.set_as(self.tiling_ub[11])
        self.pads_l.set_as(self.tiling_ub[12])
        self.pads_r.set_as(self.tiling_ub[13])
        self.input_d.set_as(self.tiling_ub[14])
        self.channel_one.set_as(self.tiling_ub[15])
        self.input_h.set_as(self.tiling_ub[16])
        self.input_w.set_as(self.tiling_ub[17])
        self.channel_zero.set_as(self.tiling_ub[18])
        self.output_b.set_as(self.tiling_ub[19])
        self.output_d.set_as(self.tiling_ub[20])
        self.output_h.set_as(self.tiling_ub[21])
        self.output_w.set_as(self.tiling_ub[22])
        self.set_running_core_num(self.tiling_ub[23])
        self.pad_w = self.input_w + self.pads_l + self.pads_r

    def init_gm_tensor(self):
        """Init gm tensor
        """
        self.tiling_gm = self.tik_instance.Tensor("int64", (Constant.TILING_ARG_NUM,),
                                                  name="tiling_gm",
                                                  scope=tik.scope_gm)
        self.input_gm = self.tik_instance.Tensor(self.dtype, (Constant.MAX_INT32,), name="input_gm", scope=tik.scope_gm)
        self.block_gm = self.tik_instance.Tensor("int32", (Constant.MAX_INT32,), name="block_shape", scope=tik.scope_gm)
        self.paddings_gm = self.tik_instance.Tensor("int32", (Constant.MAX_INT32,), name="paddings", scope=tik.scope_gm)
        self.output_gm = self.tik_instance.Tensor(self.dtype, (Constant.MAX_INT32,),
                                                  name="output_gm",
                                                  scope=tik.scope_gm)

    def vector_dup_continuous(self, src, size):
        """vector_dup continuous function, set ubuf to 0
        """
        with self.tik_instance.if_scope(size > 0):
            dup_value = float(0)
            size_loop = size // self.mask_len
            size_left = size % self.mask_len
            repeat_loop = size_loop // Constant.REPEAT_LIMIT
            repeat_left = size_loop % Constant.REPEAT_LIMIT

            with self.tik_instance.for_range(0, repeat_loop) as repeat_loop_idx:
                repeat_offset = repeat_loop_idx * Constant.REPEAT_LIMIT * self.mask_len
                self.tik_instance.vector_dup(self.mask_len, src[repeat_offset], dup_value, Constant.REPEAT_LIMIT, 1, 8)
            with self.tik_instance.if_scope(repeat_left > 0):
                repeat_offset = repeat_loop * Constant.REPEAT_LIMIT * self.mask_len
                self.tik_instance.vector_dup(self.mask_len, src[repeat_offset], dup_value, repeat_left, 1, 8)
            with self.tik_instance.if_scope(size_left > 0):
                size_offset = size_loop * self.mask_len
                self.tik_instance.vector_dup(size_left, src[size_offset], dup_value, 1, 1, 8)

    def vector_dup_discrete(self, src, repeat, size, dst_blk=1, dst_rep=8):
        """vector_dup discrete function, set ubuf to 0, dst_blk <= 65535, dst_rep <=255
        """
        with self.tik_instance.if_scope(size > 0):
            with self.tik_instance.if_scope(dst_rep <= 255):
                dup_value = float(0)
                size_loop = size // self.mask_len
                size_left = size % self.mask_len
                repeat_loop = repeat // Constant.REPEAT_LIMIT
                repeat_left = repeat % Constant.REPEAT_LIMIT

                def _inner(src, mask_len):
                    """exec repeat
                    """
                    with self.tik_instance.for_range(0, repeat_loop) as repeat_loop_idx:
                        repeat_offset = repeat_loop_idx * Constant.REPEAT_LIMIT * dst_rep * self.blk_ele
                        self.tik_instance.vector_dup(mask_len, src[repeat_offset], dup_value, Constant.REPEAT_LIMIT,
                                                     dst_blk, dst_rep)
                    with self.tik_instance.if_scope(repeat_left > 0):
                        repeat_offset = repeat_loop * Constant.REPEAT_LIMIT * dst_rep * self.blk_ele
                        self.tik_instance.vector_dup(mask_len, src[repeat_offset], dup_value, repeat_left, dst_blk,
                                                     dst_rep)

                with self.tik_instance.for_range(0, size_loop) as size_loop_idx:
                    size_offset = size_loop_idx * self.mask_len
                    _inner(src[size_offset:], self.mask_len)
                with self.tik_instance.if_scope(size_left > 0):
                    size_offset = size_loop * self.mask_len
                    _inner(src[size_offset:], size_left)
            with self.tik_instance.else_scope():
                with self.tik_instance.for_range(0, repeat) as repeat_loop_idx:
                    repeat_offset = repeat_loop_idx * dst_rep * self.blk_ele
                    self.vector_dup_continuous(src[repeat_offset:], size)

    # function for 5hd
    def run_block_h_5hd(self, ub_a, ub_b, core_idx, ele_idx, idx_bh):
        """run block height for 5hd function.
        """
        # vector dup and move in
        start = (self.pads_t - idx_bh + self.block_h - 1) // self.block_h
        end = (self.pads_t + self.input_h - idx_bh + self.block_h - 1) // self.block_h
        offset_gm_in = (core_idx * self.one_core_ele + ele_idx) * self.input_h * self.input_w * \
                       self.channel_zero + (idx_bh + start * self.block_h - self.pads_t) * self.input_w * \
                       self.channel_zero
        offset_ub_in = start * self.output_w * self.block_w * self.channel_zero + self.pads_l * self.channel_zero
        src_stride_out = (self.block_h - 1) * self.input_w * self.channel_zero // self.blk_ele
        dst_stride_out = (self.pads_l + self.pads_r) * self.channel_zero // self.blk_ele
        with self.tik_instance.new_stmt_scope(disable_sync=True):
            with self.tik_instance.if_scope(end > start):
                # move in
                self.tik_instance.data_move(ub_a[offset_ub_in], self.input_gm[offset_gm_in], 0, end - start,
                                            self.input_w * self.channel_zero // self.blk_ele, src_stride_out,
                                            dst_stride_out)
                # vector dup
                self.vector_dup_continuous(ub_a, offset_ub_in)
                offset_2 = offset_ub_in + self.input_w * self.channel_zero
                repeat_2 = end - start - 1
                size_2 = (self.pads_l + self.pads_r) * self.channel_zero
                dst_rep = self.pad_w * self.channel_zero // self.blk_ele
                self.vector_dup_discrete(ub_a[offset_2:], repeat_2, size_2, 1, dst_rep)
                offset_3 = end * self.pad_w * self.channel_zero - self.pads_r * self.channel_zero
                size_3 = (self.output_h - end) * self.pad_w * self.channel_zero + self.pads_r * self.channel_zero
                self.vector_dup_continuous(ub_a[offset_3:], size_3)
            with self.tik_instance.else_scope():
                self.vector_dup_continuous(ub_a, self.output_h * self.output_w * self.block_w * self.channel_zero)

        # permute and move out
        src_stride_pt = (self.block_w - 1) * self.channel_zero // self.blk_ele
        with self.tik_instance.new_stmt_scope(disable_sync=True):
            with self.tik_instance.for_range(0, self.block_w) as idx_bw:
                # permute
                offset_ub_pt = idx_bw * self.channel_zero
                offset_ub_out = idx_bw * self.output_h * self.output_w * self.channel_zero
                self.tik_instance.data_move(ub_b[offset_ub_out], ub_a[offset_ub_pt], 0, self.output_h * self.output_w,
                                            self.channel_zero // self.blk_ele, src_stride_pt, 0)
        with self.tik_instance.new_stmt_scope(disable_sync=True):
            with self.tik_instance.for_range(0, self.block_w) as idx_bw:
                # move out
                offset_ub_out = idx_bw * self.output_h * self.output_w * self.channel_zero
                offset_gm_out = (idx_bh * self.block_w + idx_bw) * self.input_b * self.channel_one * self.output_h * \
                                self.output_w * self.channel_zero + (core_idx * self.one_core_ele + ele_idx) * \
                                self.output_h * self.output_w * self.channel_zero
                self.tik_instance.data_move(self.output_gm[offset_gm_out], ub_b[offset_ub_out], 0, 1,
                                            self.output_h * self.output_w * self.channel_zero // self.blk_ele, 0, 0)

    def run_block_h_open_db_5hd(self, core_idx, core_ele):
        """run block height for 5hd function, open double buffer.
        """
        with self.tik_instance.new_stmt_scope():
            with self.tik_instance.for_range(0, self.block_h) as idx_bh:
                ub_a = self.tik_instance.Tensor(self.dtype, (self.ub_ele // 4,), name="ub_a", scope=tik.scope_ubuf)
                ub_b = self.tik_instance.Tensor(self.dtype, (self.ub_ele // 4,), name="ub_b", scope=tik.scope_ubuf)
                ub_c = self.tik_instance.Tensor(self.dtype, (self.ub_ele // 4,), name="ub_c", scope=tik.scope_ubuf)
                ub_d = self.tik_instance.Tensor(self.dtype, (self.ub_ele // 4,), name="ub_d", scope=tik.scope_ubuf)
                with self.tik_instance.for_range(0, core_ele // 2) as ele_idx:
                    self.run_block_h_5hd(ub_a, ub_b, core_idx, ele_idx * 2, idx_bh)
                    self.run_block_h_5hd(ub_c, ub_d, core_idx, ele_idx * 2 + 1, idx_bh)
                with self.tik_instance.if_scope(core_ele % 2 == 1):
                    self.run_block_h_5hd(ub_a, ub_b, core_idx, core_ele - 1, idx_bh)

    def run_block_h_close_db_5hd(self, core_idx, core_ele):
        """run block height for 5hd function, close double buffer.
        """
        with self.tik_instance.new_stmt_scope():
            with self.tik_instance.for_range(0, self.block_h) as idx_bh:
                ub_a = self.tik_instance.Tensor(self.dtype, (self.ub_ele // 2,), name="ub_a", scope=tik.scope_ubuf)
                ub_b = self.tik_instance.Tensor(self.dtype, (self.ub_ele // 2,), name="ub_b", scope=tik.scope_ubuf)
                with self.tik_instance.for_range(0, core_ele) as ele_idx:
                    self.run_block_h_5hd(ub_a, ub_b, core_idx, ele_idx, idx_bh)

    def run_output_h_5hd(self, ub_a, ub_b, core_idx, ele_idx, idx_oh, idx_bh):
        """run output height for 5hd function.
        """
        # vector dup and move in
        flag_h = idx_oh * self.block_h + idx_bh
        with self.tik_instance.new_stmt_scope(disable_sync=True):
            with self.tik_instance.if_scope(tik.all(flag_h >= self.pads_t, flag_h < self.pads_t + self.input_h)):
                offset_gm_in = (core_idx * self.one_core_ele + ele_idx) * self.input_h * self.input_w * \
                               self.channel_zero + (idx_oh * self.block_h + idx_bh - self.pads_t) * self.input_w * \
                               self.channel_zero
                offset_ub_in = self.pads_l * self.channel_zero
                # move in
                self.tik_instance.data_move(ub_a[offset_ub_in], self.input_gm[offset_gm_in], 0, 1,
                                            self.input_w * self.channel_zero // self.blk_ele, 0, 0)
                # vector dup
                self.vector_dup_continuous(ub_a, offset_ub_in)
                offset_3 = (self.pad_w - self.pads_r) * self.channel_zero
                size_3 = self.pads_r * self.channel_zero
                self.vector_dup_continuous(ub_a[offset_3:], size_3)
            with self.tik_instance.else_scope():
                self.vector_dup_continuous(ub_a, self.output_w * self.block_w * self.channel_zero)

        # permute ane move out
        src_stride_pt = (self.block_w - 1) * self.channel_zero // self.blk_ele
        with self.tik_instance.new_stmt_scope(disable_sync=True):
            with self.tik_instance.for_range(0, self.block_w) as idx_bw:
                # permute
                offset_ub_pt = idx_bw * self.channel_zero
                offset_ub_out = idx_bw * self.output_w * self.channel_zero
                self.tik_instance.data_move(ub_b[offset_ub_out], ub_a[offset_ub_pt], 0, self.output_w,
                                            self.channel_zero // self.blk_ele, src_stride_pt, 0)
        with self.tik_instance.new_stmt_scope(disable_sync=True):
            with self.tik_instance.for_range(0, self.block_w) as idx_bw:
                # move out
                offset_ub_out = idx_bw * self.output_w * self.channel_zero
                offset_gm_out = (idx_bh * self.block_w + idx_bw) * self.input_b * self.channel_one * self.output_h * \
                                self.output_w * self.channel_zero + (core_idx * self.one_core_ele + ele_idx) * \
                                self.output_h * self.output_w * self.channel_zero + idx_oh * self.output_w * \
                                self.channel_zero
                self.tik_instance.data_move(self.output_gm[offset_gm_out], ub_b[offset_ub_out], 0, 1,
                                            self.output_w * self.channel_zero // self.blk_ele, 0, 0)

    def run_output_h_open_db_5hd(self, core_idx, core_ele):
        """run output height for 5hd function, open double buffer.
        """
        with self.tik_instance.new_stmt_scope():
            with self.tik_instance.for_range(0, self.output_h) as idx_oh:
                with self.tik_instance.for_range(0, self.block_h) as idx_bh:
                    ub_a = self.tik_instance.Tensor(self.dtype, (self.ub_ele // 4,), name="ub_a", scope=tik.scope_ubuf)
                    ub_b = self.tik_instance.Tensor(self.dtype, (self.ub_ele // 4,), name="ub_b", scope=tik.scope_ubuf)
                    ub_c = self.tik_instance.Tensor(self.dtype, (self.ub_ele // 4,), name="ub_c", scope=tik.scope_ubuf)
                    ub_d = self.tik_instance.Tensor(self.dtype, (self.ub_ele // 4,), name="ub_d", scope=tik.scope_ubuf)
                    with self.tik_instance.for_range(0, core_ele // 2) as ele_idx:
                        self.run_output_h_5hd(ub_a, ub_b, core_idx, ele_idx * 2, idx_oh, idx_bh)
                        self.run_output_h_5hd(ub_c, ub_d, core_idx, ele_idx * 2 + 1, idx_oh, idx_bh)
                    with self.tik_instance.if_scope(core_ele % 2 == 1):
                        self.run_output_h_5hd(ub_a, ub_b, core_idx, core_ele - 1, idx_oh, idx_bh)

    def run_output_h_close_db_5hd(self, core_idx, core_ele):
        """run output height for 5hd function, close double buffer.
        """
        with self.tik_instance.new_stmt_scope():
            with self.tik_instance.for_range(0, self.output_h) as idx_oh:
                with self.tik_instance.for_range(0, self.block_h) as idx_bh:
                    ub_a = self.tik_instance.Tensor(self.dtype, (self.ub_ele // 2,), name="ub_a", scope=tik.scope_ubuf)
                    ub_b = self.tik_instance.Tensor(self.dtype, (self.ub_ele // 2,), name="ub_b", scope=tik.scope_ubuf)
                    with self.tik_instance.for_range(0, core_ele) as ele_idx:
                        self.run_output_h_5hd(ub_a, ub_b, core_idx, ele_idx, idx_oh, idx_bh)

    def split_output_h_5hd(self, ub_a, ub_b, idx_bh, idx_ib, idx_c1, core_idx, ele_idx):
        """run output height for 5hd function.
        """
        # vector dup and move in
        flag_h = (core_idx * self.one_core_ele + ele_idx) * self.block_h + idx_bh
        with self.tik_instance.new_stmt_scope(disable_sync=True):
            with self.tik_instance.if_scope(tik.all(flag_h >= self.pads_t, flag_h < self.pads_t + self.input_h)):
                offset_gm_in = (idx_ib * self.channel_one + idx_c1) * self.input_h * self.input_w * \
                               self.channel_zero + ((core_idx * self.one_core_ele + ele_idx) * self.block_h + \
                               idx_bh - self.pads_t) * self.input_w * self.channel_zero
                offset_ub_in = self.pads_l * self.channel_zero
                # move in
                self.tik_instance.data_move(ub_a[offset_ub_in], self.input_gm[offset_gm_in], 0, 1,
                                            self.input_w * self.channel_zero // self.blk_ele, 0, 0)
                # vector dup
                self.vector_dup_continuous(ub_a, offset_ub_in)
                offset_3 = (self.pad_w - self.pads_r) * self.channel_zero
                size_3 = self.pads_r * self.channel_zero
                self.vector_dup_continuous(ub_a[offset_3:], size_3)
            with self.tik_instance.else_scope():
                self.vector_dup_continuous(ub_a, self.output_w * self.block_w * self.channel_zero)

        # permute ane move out
        src_stride_pt = (self.block_w - 1) * self.channel_zero // self.blk_ele
        with self.tik_instance.new_stmt_scope(disable_sync=True):
            with self.tik_instance.for_range(0, self.block_w) as idx_bw:
                # permute
                offset_ub_pt = idx_bw * self.channel_zero
                offset_ub_out = idx_bw * self.output_w * self.channel_zero
                self.tik_instance.data_move(ub_b[offset_ub_out], ub_a[offset_ub_pt], 0, self.output_w,
                                            self.channel_zero // self.blk_ele, src_stride_pt, 0)
        with self.tik_instance.new_stmt_scope(disable_sync=True):
            with self.tik_instance.for_range(0, self.block_w) as idx_bw:
                # move out
                offset_ub_out = idx_bw * self.output_w * self.channel_zero
                offset_gm_out = (idx_bh * self.block_w + idx_bw) * self.input_b * self.channel_one * self.output_h * \
                                self.output_w * self.channel_zero + (idx_ib * self.channel_one + idx_c1) * \
                                self.output_h * self.output_w * self.channel_zero + \
                                (core_idx * self.one_core_ele + ele_idx) * self.output_w * self.channel_zero
                self.tik_instance.data_move(self.output_gm[offset_gm_out], ub_b[offset_ub_out], 0, 1,
                                            self.output_w * self.channel_zero // self.blk_ele, 0, 0)

    def split_output_h_open_db_5hd(self, core_idx, core_ele):
        """run output height for 5hd function, open double buffer.
        """
        with self.tik_instance.new_stmt_scope():
            with self.tik_instance.for_range(0, self.block_h) as idx_bh:
                with self.tik_instance.for_range(0, self.input_b) as idx_ib:
                    with self.tik_instance.for_range(0, self.channel_one) as idx_c1:
                        ub_a = self.tik_instance.Tensor(self.dtype, (self.ub_ele // 4,),
                                                        name="ub_a",
                                                        scope=tik.scope_ubuf)
                        ub_b = self.tik_instance.Tensor(self.dtype, (self.ub_ele // 4,),
                                                        name="ub_b",
                                                        scope=tik.scope_ubuf)
                        ub_c = self.tik_instance.Tensor(self.dtype, (self.ub_ele // 4,),
                                                        name="ub_c",
                                                        scope=tik.scope_ubuf)
                        ub_d = self.tik_instance.Tensor(self.dtype, (self.ub_ele // 4,),
                                                        name="ub_d",
                                                        scope=tik.scope_ubuf)
                        with self.tik_instance.for_range(0, core_ele // 2) as ele_idx:
                            self.split_output_h_5hd(ub_a, ub_b, idx_bh, idx_ib, idx_c1, core_idx, ele_idx * 2)
                            self.split_output_h_5hd(ub_c, ub_d, idx_bh, idx_ib, idx_c1, core_idx, ele_idx * 2 + 1)
                        with self.tik_instance.if_scope(core_ele % 2 == 1):
                            self.split_output_h_5hd(ub_a, ub_b, idx_bh, idx_ib, idx_c1, core_idx, core_ele - 1)

    def split_output_h_close_db_5hd(self, core_idx, core_ele):
        """run output height for 5hd function, close double buffer.
        """
        with self.tik_instance.new_stmt_scope():
            with self.tik_instance.for_range(0, self.block_h) as idx_bh:
                with self.tik_instance.for_range(0, self.input_b) as idx_ib:
                    with self.tik_instance.for_range(0, self.channel_one) as idx_c1:
                        ub_a = self.tik_instance.Tensor(self.dtype, (self.ub_ele // 2,),
                                                        name="ub_a",
                                                        scope=tik.scope_ubuf)
                        ub_b = self.tik_instance.Tensor(self.dtype, (self.ub_ele // 2,),
                                                        name="ub_b",
                                                        scope=tik.scope_ubuf)
                        with self.tik_instance.for_range(0, core_ele) as ele_idx:
                            self.split_output_h_5hd(ub_a, ub_b, idx_bh, idx_ib, idx_c1, core_idx, ele_idx)

    def run_block_w_5hd(self, core_idx, core_ele):
        """run block width for 5hd function, close double buffer.
        """
        with self.tik_instance.new_stmt_scope():
            ub_a = self.tik_instance.Tensor(self.dtype, (self.ub_ele,), name="ub_a", scope=tik.scope_ubuf)
            with self.tik_instance.for_range(0, self.output_h) as idx_oh:
                with self.tik_instance.for_range(0, self.block_h) as idx_bh:
                    with self.tik_instance.for_range(0, self.block_w) as idx_bw:
                        with self.tik_instance.for_range(0, core_ele) as ele_idx:
                            # vector dup and move in
                            flag_h = idx_oh * self.block_h + idx_bh
                            self.vector_dup_continuous(ub_a, self.output_w * self.channel_zero)
                            with self.tik_instance.if_scope(
                                    tik.all(flag_h >= self.pads_t, flag_h < self.pads_t + self.input_h)):
                                start = (self.pads_l - idx_bw + self.block_w - 1) // self.block_w
                                end = (self.pads_l + self.input_w - idx_bw + self.block_w - 1) // self.block_w
                                offset_gm_in = (core_idx * self.one_core_ele + ele_idx) * self.input_h * \
                                               self.input_w * self.channel_zero + \
                                               (idx_oh * self.block_h + idx_bh - self.pads_t) * self.input_w * \
                                               self.channel_zero + (idx_bw + start * self.block_w - self.pads_l) * \
                                               self.channel_zero
                                offset_ub_in = start * self.channel_zero
                                src_stride_out = (self.block_w - 1) * self.channel_zero // self.blk_ele
                                with self.tik_instance.if_scope(end > start):
                                    self.tik_instance.data_move(ub_a[offset_ub_in], self.input_gm[offset_gm_in], 0,
                                                                end - start, self.channel_zero // self.blk_ele,
                                                                src_stride_out, 0)
                            # move out
                            offset_gm_out = (idx_bh * self.block_w + idx_bw) * self.input_b * self.channel_one * \
                                            self.output_h * self.output_w * self.channel_zero + \
                                            (core_idx * self.one_core_ele + ele_idx) * self.output_h * self.output_w * \
                                            self.channel_zero + idx_oh * self.output_w * self.channel_zero
                            self.tik_instance.data_move(self.output_gm[offset_gm_out], ub_a, 0, 1,
                                                        self.output_w * self.channel_zero // self.blk_ele, 0, 0)

    def run_output_w_5hd(self, core_idx, core_ele):
        """run output width for 5hd function, close double buffer.
        """
        with self.tik_instance.new_stmt_scope():
            ub_a = self.tik_instance.Tensor(self.dtype, (self.ub_ele,), name="ub_a", scope=tik.scope_ubuf)
            with self.tik_instance.for_range(0, self.output_h) as idx_oh:
                with self.tik_instance.for_range(0, self.block_h) as idx_bh:
                    with self.tik_instance.for_range(0, self.output_w) as idx_ow:
                        with self.tik_instance.for_range(0, self.block_w) as idx_bw:
                            with self.tik_instance.for_range(0, core_ele) as ele_idx:
                                # vector dup and move in
                                flag_h = idx_oh * self.block_h + idx_bh
                                flag_w = idx_ow * self.block_w + idx_bw
                                self.vector_dup_continuous(ub_a, self.channel_zero)
                                with self.tik_instance.if_scope(
                                        tik.all(flag_h >= self.pads_t, flag_h < self.pads_t + self.input_h,
                                                flag_w >= self.pads_l, flag_w < self.pads_l + self.input_w)):
                                    offset_gm_in = (core_idx * self.one_core_ele + ele_idx) * self.input_h * \
                                                   self.input_w * self.channel_zero + \
                                                   (idx_oh * self.block_h + idx_bh - self.pads_t) * self.input_w * \
                                                   self.channel_zero + \
                                                   (idx_ow * self.block_w + idx_bw - self.pads_l) * self.channel_zero
                                    self.tik_instance.data_move(ub_a, self.input_gm[offset_gm_in], 0, 1,
                                                                self.channel_zero // self.blk_ele, 0, 0)
                                # move out
                                offset_gm_out = (idx_bh * self.block_w + idx_bw) * self.input_b * self.channel_one * \
                                                self.output_h * self.output_w * self.channel_zero + \
                                                (core_idx * self.one_core_ele + ele_idx) * self.output_h * \
                                                self.output_w * self.channel_zero + \
                                                (idx_oh * self.output_w + idx_ow) * self.channel_zero
                                self.tik_instance.data_move(self.output_gm[offset_gm_out], ub_a, 0, 1,
                                                            self.channel_zero // self.blk_ele, 0, 0)

    # function for 6hd
    def run_block_h_6hd(self, ub_a, ub_b, core_idx, ele_idx, idx_bd, idx_b, c_idx, idx_bh):
        """run block height for 6hd function.
        """
        flag_d = (core_idx * self.one_core_ele + ele_idx) * self.block_d + idx_bd
        with self.tik_instance.if_scope(tik.all(flag_d >= self.pads_f, flag_d < self.pads_f + self.input_d)):
            # vector dup and move in
            start = (self.pads_t - idx_bh + self.block_h - 1) // self.block_h
            end = (self.pads_t + self.input_h - idx_bh + self.block_h - 1) // self.block_h
            offset_gm_in = idx_b * self.input_d * self.channel_one * self.input_h * self.input_w * \
                           self.channel_zero + ((core_idx * self.one_core_ele + ele_idx) * self.block_d + idx_bd - \
                           self.pads_f) * self.channel_one * self.input_h * self.input_w * self.channel_zero + \
                           c_idx * self.input_h * self.input_w * self.channel_zero + \
                           (idx_bh + start * self.block_h - self.pads_t) * self.input_w * self.channel_zero
            offset_ub_in = start * self.output_w * self.block_w * self.channel_zero + self.pads_l * self.channel_zero
            src_stride_out = (self.block_h - 1) * self.input_w * self.channel_zero // self.blk_ele
            dst_stride_out = (self.pads_l + self.pads_r) * self.channel_zero // self.blk_ele
            with self.tik_instance.new_stmt_scope(disable_sync=True):
                with self.tik_instance.if_scope(end > start):
                    # move in
                    self.tik_instance.data_move(ub_a[offset_ub_in], self.input_gm[offset_gm_in], 0, end - start,
                                                self.input_w * self.channel_zero // self.blk_ele, src_stride_out,
                                                dst_stride_out)
                    # vector dup
                    self.vector_dup_continuous(ub_a, offset_ub_in)
                    offset_2 = offset_ub_in + self.input_w * self.channel_zero
                    repeat_2 = end - start - 1
                    size_2 = (self.pads_l + self.pads_r) * self.channel_zero
                    dst_rep = self.pad_w * self.channel_zero // self.blk_ele
                    self.vector_dup_discrete(ub_a[offset_2:], repeat_2, size_2, 1, dst_rep)
                    offset_3 = end * self.pad_w * self.channel_zero - self.pads_r * self.channel_zero
                    size_3 = (self.output_h - end) * self.pad_w * self.channel_zero + self.pads_r * self.channel_zero
                    self.vector_dup_continuous(ub_a[offset_3:], size_3)
                with self.tik_instance.else_scope():
                    self.vector_dup_continuous(ub_a, self.output_h * self.output_w * self.block_w * self.channel_zero)
        with self.tik_instance.else_scope():
            self.vector_dup_continuous(ub_a, self.output_h * self.output_w * self.block_w * self.channel_zero)

        # permute and move out
        src_stride_pt = (self.block_w - 1) * self.channel_zero // self.blk_ele
        with self.tik_instance.new_stmt_scope(disable_sync=True):
            with self.tik_instance.for_range(0, self.block_w) as idx_bw:
                # permute
                offset_ub_pt = idx_bw * self.channel_zero
                offset_ub_out = idx_bw * self.output_h * self.output_w * self.channel_zero
                self.tik_instance.data_move(ub_b[offset_ub_out], ub_a[offset_ub_pt], 0, self.output_h * self.output_w,
                                            self.channel_zero // self.blk_ele, src_stride_pt, 0)
        with self.tik_instance.new_stmt_scope(disable_sync=True):
            with self.tik_instance.for_range(0, self.block_w) as idx_bw:
                # move out
                offset_ub_out = idx_bw * self.output_h * self.output_w * self.channel_zero
                offset_gm_out = (((idx_bd * self.block_h + idx_bh) * self.block_w + idx_bw) * self.input_b + idx_b) * \
                                self.output_d * self.channel_one * self.output_h * self.output_w * self.channel_zero + \
                                (core_idx * self.one_core_ele + ele_idx) * self.channel_one * self.output_h * \
                                self.output_w * self.channel_zero + c_idx * self.output_h * self.output_w * \
                                self.channel_zero
                self.tik_instance.data_move(self.output_gm[offset_gm_out], ub_b[offset_ub_out], 0, 1,
                                            self.output_h * self.output_w * self.channel_zero // self.blk_ele, 0, 0)

    def run_block_h_open_db_6hd(self, core_idx, core_ele):
        """run block height for 6hd function, open double buffer.
        """
        with self.tik_instance.new_stmt_scope():
            with self.tik_instance.for_range(0, core_ele) as ele_idx:
                with self.tik_instance.for_range(0, self.block_d) as idx_bd:
                    with self.tik_instance.for_range(0, self.block_h) as idx_bh:
                        with self.tik_instance.for_range(0, self.input_b) as idx_b:
                            ub_a = self.tik_instance.Tensor(self.dtype, (self.ub_ele // 4,),
                                                            name="ub_a",
                                                            scope=tik.scope_ubuf)
                            ub_b = self.tik_instance.Tensor(self.dtype, (self.ub_ele // 4,),
                                                            name="ub_b",
                                                            scope=tik.scope_ubuf)
                            ub_c = self.tik_instance.Tensor(self.dtype, (self.ub_ele // 4,),
                                                            name="ub_c",
                                                            scope=tik.scope_ubuf)
                            ub_d = self.tik_instance.Tensor(self.dtype, (self.ub_ele // 4,),
                                                            name="ub_d",
                                                            scope=tik.scope_ubuf)
                            with self.tik_instance.for_range(0, self.channel_one // 2) as c_idx:
                                self.run_block_h_6hd(ub_a, ub_b, core_idx, ele_idx, idx_bd, idx_b, c_idx * 2, idx_bh)
                                self.run_block_h_6hd(ub_c, ub_d, core_idx, ele_idx, idx_bd, idx_b, c_idx * 2 + 1,
                                                     idx_bh)
                            with self.tik_instance.if_scope(self.channel_one % 2 == 1):
                                self.run_block_h_6hd(ub_a, ub_b, core_idx, ele_idx, idx_bd, idx_b, self.channel_one - 1,
                                                     idx_bh)

    def run_block_h_close_db_6hd(self, core_idx, core_ele):
        """run block height for 6hd function, close double buffer.
        """
        with self.tik_instance.new_stmt_scope():
            with self.tik_instance.for_range(0, core_ele) as ele_idx:
                with self.tik_instance.for_range(0, self.block_d) as idx_bd:
                    with self.tik_instance.for_range(0, self.block_h) as idx_bh:
                        with self.tik_instance.for_range(0, self.input_b) as idx_b:
                            ub_a = self.tik_instance.Tensor(self.dtype, (self.ub_ele // 2,),
                                                            name="ub_a",
                                                            scope=tik.scope_ubuf)
                            ub_b = self.tik_instance.Tensor(self.dtype, (self.ub_ele // 2,),
                                                            name="ub_b",
                                                            scope=tik.scope_ubuf)
                            with self.tik_instance.for_range(0, self.channel_one) as c_idx:
                                self.run_block_h_6hd(ub_a, ub_b, core_idx, ele_idx, idx_bd, idx_b, c_idx, idx_bh)

    def run_output_h_6hd(self, ub_a, ub_b, core_idx, ele_idx, idx_bd, idx_b, c_idx, idx_oh, idx_bh):
        """run output height for 6hd function.
        """
        # vector dup and move in
        flag_d = (core_idx * self.one_core_ele + ele_idx) * self.block_d + idx_bd
        flag_h = idx_oh * self.block_h + idx_bh
        with self.tik_instance.new_stmt_scope(disable_sync=True):
            with self.tik_instance.if_scope(
                    tik.all(flag_d >= self.pads_f, flag_d < self.pads_f + self.input_d, flag_h >= self.pads_t,
                            flag_h < self.pads_t + self.input_h)):
                offset_gm_in = idx_b * self.input_d * self.channel_one * self.input_h * self.input_w * \
                               self.channel_zero + ((core_idx * self.one_core_ele + ele_idx) * self.block_d + idx_bd - \
                               self.pads_f) * self.channel_one * self.input_h * self.input_w * self.channel_zero + \
                               c_idx * self.input_h * self.input_w * self.channel_zero + \
                               (idx_oh * self.block_h + idx_bh - self.pads_t) * self.input_w * self.channel_zero
                offset_ub_in = self.pads_l * self.channel_zero
                # move in
                self.tik_instance.data_move(ub_a[offset_ub_in], self.input_gm[offset_gm_in], 0, 1,
                                            self.input_w * self.channel_zero // self.blk_ele, 0, 0)
                # vector dup
                self.vector_dup_continuous(ub_a, offset_ub_in)
                offset_3 = (self.pad_w - self.pads_r) * self.channel_zero
                size_3 = self.pads_r * self.channel_zero
                self.vector_dup_continuous(ub_a[offset_3:], size_3)
            with self.tik_instance.else_scope():
                self.vector_dup_continuous(ub_a, self.output_w * self.block_w * self.channel_zero)

        # permute ane move out
        src_stride_pt = (self.block_w - 1) * self.channel_zero // self.blk_ele
        with self.tik_instance.new_stmt_scope(disable_sync=True):
            with self.tik_instance.for_range(0, self.block_w) as idx_bw:
                # permute
                offset_ub_pt = idx_bw * self.channel_zero
                offset_ub_out = idx_bw * self.output_w * self.channel_zero
                self.tik_instance.data_move(ub_b[offset_ub_out], ub_a[offset_ub_pt], 0, self.output_w,
                                            self.channel_zero // self.blk_ele, src_stride_pt, 0)
        with self.tik_instance.new_stmt_scope(disable_sync=True):
            with self.tik_instance.for_range(0, self.block_w) as idx_bw:
                # move out
                offset_ub_out = idx_bw * self.output_w * self.channel_zero
                offset_gm_out = (((idx_bd * self.block_h + idx_bh) * self.block_w + idx_bw) * self.input_b + idx_b) * \
                                self.output_d * self.channel_one * self.output_h * self.output_w * self.channel_zero + \
                                (core_idx * self.one_core_ele + ele_idx) * self.channel_one * self.output_h * \
                                self.output_w * self.channel_zero + c_idx * self.output_h * self.output_w * \
                                self.channel_zero + idx_oh * self.output_w * self.channel_zero
                self.tik_instance.data_move(self.output_gm[offset_gm_out], ub_b[offset_ub_out], 0, 1,
                                            self.output_w * self.channel_zero // self.blk_ele, 0, 0)

    def run_output_h_open_db_6hd(self, core_idx, core_ele):
        """run output height for 6hd function, open double buffer.
        """
        with self.tik_instance.new_stmt_scope():
            with self.tik_instance.for_range(0, core_ele) as ele_idx:
                with self.tik_instance.for_range(0, self.block_d) as idx_bd:
                    with self.tik_instance.for_range(0, self.output_h) as idx_oh:
                        with self.tik_instance.for_range(0, self.block_h) as idx_bh:
                            with self.tik_instance.for_range(0, self.input_b) as idx_b:
                                ub_a = self.tik_instance.Tensor(self.dtype, (self.ub_ele // 4,),
                                                                name="ub_a",
                                                                scope=tik.scope_ubuf)
                                ub_b = self.tik_instance.Tensor(self.dtype, (self.ub_ele // 4,),
                                                                name="ub_b",
                                                                scope=tik.scope_ubuf)
                                ub_c = self.tik_instance.Tensor(self.dtype, (self.ub_ele // 4,),
                                                                name="ub_c",
                                                                scope=tik.scope_ubuf)
                                ub_d = self.tik_instance.Tensor(self.dtype, (self.ub_ele // 4,),
                                                                name="ub_d",
                                                                scope=tik.scope_ubuf)
                                with self.tik_instance.for_range(0, self.channel_one // 2) as c_idx:
                                    self.run_output_h_6hd(ub_a, ub_b, core_idx, ele_idx, idx_bd, idx_b, c_idx * 2,
                                                          idx_oh, idx_bh)
                                    self.run_output_h_6hd(ub_c, ub_d, core_idx, ele_idx, idx_bd, idx_b, c_idx * 2 + 1,
                                                          idx_oh, idx_bh)
                                with self.tik_instance.if_scope(self.channel_one % 2 == 1):
                                    self.run_output_h_6hd(ub_a, ub_b, core_idx, ele_idx, idx_bd, idx_b,
                                                          self.channel_one - 1, idx_oh, idx_bh)

    def run_output_h_close_db_6hd(self, core_idx, core_ele):
        """run output height for 6hd function, close double buffer.
        """
        with self.tik_instance.new_stmt_scope():
            with self.tik_instance.for_range(0, core_ele) as ele_idx:
                with self.tik_instance.for_range(0, self.block_d) as idx_bd:
                    with self.tik_instance.for_range(0, self.output_h) as idx_oh:
                        with self.tik_instance.for_range(0, self.block_h) as idx_bh:
                            with self.tik_instance.for_range(0, self.input_b) as idx_b:
                                ub_a = self.tik_instance.Tensor(self.dtype, (self.ub_ele // 2,),
                                                                name="ub_a",
                                                                scope=tik.scope_ubuf)
                                ub_b = self.tik_instance.Tensor(self.dtype, (self.ub_ele // 2,),
                                                                name="ub_b",
                                                                scope=tik.scope_ubuf)
                                with self.tik_instance.for_range(0, self.channel_one) as c_idx:
                                    self.run_output_h_6hd(ub_a, ub_b, core_idx, ele_idx, idx_bd, idx_b, c_idx, idx_oh,
                                                          idx_bh)

    def run_block_w_6hd(self, core_idx, core_ele):
        """run block width for 6hd function, close double buffer.
        """
        with self.tik_instance.new_stmt_scope():
            ub_a = self.tik_instance.Tensor(self.dtype, (self.ub_ele,), name="ub_a", scope=tik.scope_ubuf)
            with self.tik_instance.for_range(0, core_ele) as ele_idx:
                with self.tik_instance.for_range(0, self.block_d) as idx_bd:
                    with self.tik_instance.for_range(0, self.output_h) as idx_oh:
                        with self.tik_instance.for_range(0, self.block_h) as idx_bh:
                            with self.tik_instance.for_range(0, self.block_w) as idx_bw:
                                with self.tik_instance.for_range(0, self.input_b) as idx_b:
                                    with self.tik_instance.for_range(0, self.channel_one) as c_idx:
                                        flag_d = (core_idx * self.one_core_ele + ele_idx) * self.block_d + idx_bd
                                        flag_h = idx_oh * self.block_h + idx_bh
                                        self.vector_dup_continuous(ub_a, self.output_w * self.channel_zero)
                                        with self.tik_instance.if_scope(
                                                tik.all(flag_d >= self.pads_f, flag_d < self.pads_f + self.input_d,
                                                        flag_h >= self.pads_t, flag_h < self.pads_t + self.input_h)):
                                            start = (self.pads_l - idx_bw + self.block_w - 1) // self.block_w
                                            end = (self.pads_l + self.input_w - idx_bw + self.block_w -
                                                   1) // self.block_w
                                            offset_gm_in = idx_b * self.input_d * self.channel_one * \
                                                           self.input_h * self.input_w * self.channel_zero + \
                                                           ((core_idx * self.one_core_ele + ele_idx) * self.block_d + \
                                                           idx_bd - self.pads_f) * self.channel_one * self.input_h * \
                                                           self.input_w * self.channel_zero + c_idx * \
                                                           self.input_h * self.input_w * self.channel_zero + \
                                                           (idx_oh * self.block_h + idx_bh - self.pads_t) * \
                                                           self.input_w * self.channel_zero + (idx_bw + start * \
                                                           self.block_w - self.pads_l) * self.channel_zero
                                            offset_ub_in = start * self.channel_zero
                                            src_stride_out = (self.block_w - 1) * self.channel_zero // self.blk_ele
                                            with self.tik_instance.if_scope(end > start):
                                                self.tik_instance.data_move(ub_a[offset_ub_in],
                                                                            self.input_gm[offset_gm_in], 0, end - start,
                                                                            self.channel_zero // self.blk_ele,
                                                                            src_stride_out, 0)
                                        # move out
                                        offset_gm_out = (((idx_bd * self.block_h + idx_bh) * self.block_w + idx_bw) * \
                                                        self.input_b + idx_b) * self.output_d * self.channel_one * \
                                                        self.output_h * self.output_w * self.channel_zero + \
                                                        (core_idx * self.one_core_ele + ele_idx) * self.channel_one * \
                                                        self.output_h * self.output_w * self.channel_zero + c_idx * \
                                                        self.output_h * self.output_w * self.channel_zero + idx_oh * \
                                                        self.output_w * self.channel_zero
                                        self.tik_instance.data_move(self.output_gm[offset_gm_out], ub_a, 0, 1,
                                                                    self.output_w * self.channel_zero // self.blk_ele,
                                                                    0, 0)

    def run_output_w_6hd(self, core_idx, core_ele):
        """run output width for 6hd function, close double buffer.
        """
        with self.tik_instance.new_stmt_scope():
            ub_a = self.tik_instance.Tensor(self.dtype, (self.ub_ele,), name="ub_a", scope=tik.scope_ubuf)
            with self.tik_instance.for_range(0, core_ele) as ele_idx:
                with self.tik_instance.for_range(0, self.block_d) as idx_bd:
                    with self.tik_instance.for_range(0, self.output_h) as idx_oh:
                        with self.tik_instance.for_range(0, self.block_h) as idx_bh:
                            with self.tik_instance.for_range(0, self.output_w) as idx_ow:
                                with self.tik_instance.for_range(0, self.block_w) as idx_bw:
                                    with self.tik_instance.for_range(0, self.input_b) as idx_b:
                                        with self.tik_instance.for_range(0, self.channel_one) as c_idx:
                                            flag_d = (core_idx * self.one_core_ele + ele_idx) * self.block_d + idx_bd
                                            flag_h = idx_oh * self.block_h + idx_bh
                                            flag_w = idx_ow * self.block_w + idx_bw
                                            self.vector_dup_continuous(ub_a, self.output_w * self.channel_zero)
                                            with self.tik_instance.if_scope(
                                                    tik.all(flag_d >= self.pads_f, flag_d < self.pads_f + self.input_d,
                                                            flag_h >= self.pads_t, flag_h < self.pads_t + self.input_h,
                                                            flag_w >= self.pads_l,
                                                            flag_w < self.pads_l + self.input_w)):
                                                offset_gm_in = idx_b * self.input_d * self.channel_one * \
                                                               self.input_h * self.input_w * self.channel_zero + \
                                                               ((core_idx * self.one_core_ele + ele_idx) * \
                                                               self.block_d + idx_bd - self.pads_f) * \
                                                               self.channel_one * self.input_h * self.input_w * \
                                                               self.channel_zero + c_idx * self.input_h * \
                                                               self.input_w * self.channel_zero + \
                                                               (idx_oh * self.block_h + idx_bh - self.pads_t) * \
                                                               self.input_w * self.channel_zero + \
                                                               (idx_ow * self.block_w + idx_bw - self.pads_l) * \
                                                               self.channel_zero
                                                self.tik_instance.data_move(ub_a, self.input_gm[offset_gm_in], 0, 1,
                                                                            self.channel_zero // self.blk_ele, 0, 0)
                                            # move out
                                            offset_gm_out = (((idx_bd * self.block_h + idx_bh) * self.block_w + \
                                                            idx_bw) * self.input_b + idx_b) * self.output_d * \
                                                            self.channel_one * self.output_h * self.output_w * \
                                                            self.channel_zero + (core_idx * self.one_core_ele + \
                                                            ele_idx) * self.channel_one * self.output_h * \
                                                            self.output_w * self.channel_zero + c_idx * \
                                                            self.output_h * self.output_w * self.channel_zero + \
                                                            (idx_oh * self.output_w + idx_ow) * self.channel_zero
                                            self.tik_instance.data_move(self.output_gm[offset_gm_out], ub_a, 0, 1,
                                                                        self.channel_zero // self.blk_ele, 0, 0)

    def space_to_batch_nd_compute_tiling(self):
        """BatchToSpaceND compute tiling
        """
        with self.tik_instance.for_range(0, self.core_num_var, block_num=self.core_num_var) as core_idx:
            # define tiling ub and move tiling gm to tiling ub,then get tiling args
            self.tiling_ub = self.tik_instance.Tensor("int64", (Constant.TILING_ARG_NUM,),
                                                      name="tiling_ub",
                                                      scope=tik.scope_ubuf)
            self.tik_instance.data_move(self.tiling_ub, self.tiling_gm, 0, 1, 6, 0, 0)
            self.tiling_args()

            # call select tiling mode function
            core_ele = self.tik_instance.Scalar("int64", name="core_ele")
            with self.tik_instance.if_scope(core_idx <= self.act_core_num - 1):
                with self.tik_instance.if_scope(core_idx < self.act_core_num - 1):
                    core_ele.set_as(self.one_core_ele)
                with self.tik_instance.else_scope():
                    core_ele.set_as(self.last_core_ele)
                # when format is NC1HWC0, can copy output_h * output_w * block_w * c0, open double buffer
                with self.tik_instance.if_scope(self.tiling_mode == 0):
                    self.run_block_h_open_db_5hd(core_idx, core_ele)
                # when format is NC1HWC0, can copy output_h * output_w * block_w * c0, close double buffer
                with self.tik_instance.if_scope(self.tiling_mode == 1):
                    self.run_block_h_close_db_5hd(core_idx, core_ele)
                # when format is NC1HWC0, can copy output_w * block_w * c0, open double buffer
                with self.tik_instance.if_scope(self.tiling_mode == 2):
                    self.run_output_h_open_db_5hd(core_idx, core_ele)
                # when format is NC1HWC0, can copy output_w * block_w * c0, close double buffer
                with self.tik_instance.if_scope(self.tiling_mode == 3):
                    self.run_output_h_close_db_5hd(core_idx, core_ele)
                # when format is NC1HWC0, can copy output_w * c0, no double buffer
                with self.tik_instance.if_scope(self.tiling_mode == 4):
                    self.run_block_w_5hd(core_idx, core_ele)
                # when format is NC1HWC0, can copy c0, no double buffer
                with self.tik_instance.if_scope(self.tiling_mode == 5):
                    self.run_output_w_5hd(core_idx, core_ele)
                # when format is NHC1HWC0, can copy output_h * output_w * block_w * c0, open double buffer
                with self.tik_instance.if_scope(self.tiling_mode == 6):
                    self.run_block_h_open_db_6hd(core_idx, core_ele)
                # when format is NHC1HWC0, can copy output_h * output_w * block_w * c0, close double buffer
                with self.tik_instance.if_scope(self.tiling_mode == 7):
                    self.run_block_h_close_db_6hd(core_idx, core_ele)
                # when format is NHC1HWC0, can copy output_w * block_w * c0, open double buffer
                with self.tik_instance.if_scope(self.tiling_mode == 8):
                    self.run_output_h_open_db_6hd(core_idx, core_ele)
                # when format is NHC1HWC0, can copy output_w * block_w * c0, close double buffer
                with self.tik_instance.if_scope(self.tiling_mode == 9):
                    self.run_output_h_close_db_6hd(core_idx, core_ele)
                # when format is NHC1HWC0, can copy output_w * c0, no double buffer
                with self.tik_instance.if_scope(self.tiling_mode == 10):
                    self.run_block_w_6hd(core_idx, core_ele)
                # when format is NHC1HWC0, can copy c0, no double buffer
                with self.tik_instance.if_scope(self.tiling_mode == 11):
                    self.run_output_w_6hd(core_idx, core_ele)
                # when format is NC1HWC0, can copy input_w * block_w * c0, open double buffer, core at input_h
                with self.tik_instance.if_scope(self.tiling_mode == 12):
                    self.split_output_h_open_db_5hd(core_idx, core_ele)
                # when format is NC1HWC0, can copy input_w * block_w * c0, close double buffer, core at input_h
                with self.tik_instance.if_scope(self.tiling_mode == 13):
                    self.split_output_h_close_db_5hd(core_idx, core_ele)

    def space_to_batch_nd_operator(self):
        """SpaceToBatchND operator
        """
        self.space_to_batch_nd_compute_tiling()
        opt_config = {"out_of_bound_sync_check": True, "enable_const_fold": True}

        tbe_context.get_context().add_compile_info("vars", {
            "ub_ele": self.ub_ele,
            "core_num": self.core_num,
            "op_type": "SpaceToBatchND"
        })
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[self.input_gm, self.block_gm, self.paddings_gm],
                                   outputs=[self.output_gm],
                                   flowtable=[self.tiling_gm],
                                   config=opt_config)

        return self.tik_instance


@register_operator("SpaceToBatchND")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def space_to_batch_nd(x, block_shape, paddings, y, kernel_name="space_to_batch_nd"):
    """SpaceToBatchND for tensor.

    Parameters
    ----------
    x: dict
        the dict of input tensor.
    block_shape: dict
        the dict of block_shape tensor.
    paddings: dict
        the dict of paddings tensor.
    y: dict
        the dict of output tensor.
    kernel_name: str
        cce kernel name, default value is "space_to_batch_nd".

    Returns
    -------
    None.
    """
    # get input shape, format and dtype
    input_shape = x.get("shape")
    input_dtype = x.get("dtype").lower()
    input_format = x.get("format")

    # check input shape, format and dtype
    para_check.check_shape(input_shape, param_name="x")
    para_check.check_dtype(input_dtype, ("float16", "float32", "bfloat16"), param_name="x")
    if input_format not in ("NC1HWC0", "NDC1HWC0"):
        error_manager_vector.raise_err_input_format_invalid(kernel_name, "x", "NC1HWC0,NDC1HWC0", input_format)

    # run tik
    obj = SpaceToBatchND(input_dtype, 0, kernel_name)
    obj.space_to_batch_nd_operator()
