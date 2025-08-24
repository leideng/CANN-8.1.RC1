# -*- coding:utf-8 -*-
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
max_pool_with_argmaxv2
"""

from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import shape_util
from impl.dynamic import max_pool_with_argmax_v2_resnet50 as resnet50
from tbe.common.platform import get_bit_len
from impl.util.util_common import is_unknown_rank_input


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """

    def __init__(self):
        pass

    DT_INT32 = 3
    # max int32
    MAX_INT32 = 2 ** 31 - 1
    # max int16
    MAX_INT16 = 65535
    # tiling param num
    TILING_ARG_NUM = 32
    # reserved ub size
    RESERVED_UB_SIZE = 8 * 1024
    # 8 bit
    EIGHT_BIT = 8
    # bytes of one block
    BLOCK_BYTES = 32
    # repeat limit
    REPEAT_LIMIT = 255
    # min float16
    MIN_FL16 = -65504.0
    # min float32
    MIN_FL32 = -3.40282346638e38
    # gather len
    GATHER_LEN = 256
    # UB_SRC_SIZE
    UB_SRC_SIZE = 256 * 16
    WINDOW_AXES = "WINDOW_AXES"
    ATTR_AXES = "ATTR_AXES"
    WINDOW_DIMENSIONS = "WINDOW_DIMENSIONS"
    WINDOW_STRIDES = "WINDOW_STRIDES"
    WINDOW_PADDINGS = "WINDOW_PADDINGS"
    WINDOW_DILATIONS = "WINDOW_DILATIONS"
    CEIL_MODE = "CEIL_MODE"


# 'pylint: disable=too-many-lines,invalid-name,too-many-arguments,consider-using-in
# 'pylint: disable=too-many-branches,too-many-instance-attributes,too-many-locals
# 'pylint: disable=too-many-statements,no-self-use,too-few-public-methods
# 'pylint: disable=unused-argument
def check_supported(x, y, argmax, ksize, strides, pads, dtype=Constant.DT_INT32, dilation=(1, 1, 1, 1),
                    ceil_mode=False, kernel_name="max_pool_with_argmax_v2"):
    """
    check whether ai_core is supported
    """

    ori_format = x.get("origin_format")
    if ori_format in ("NCHW", "NC1HWC0"):
        dim_h = 2
        dim_w = 3
    elif ori_format in ("NHWC",):
        dim_h = 1
        dim_w = 2
    if ksize[dim_h] * ksize[dim_w] > Constant.REPEAT_LIMIT:
        reason = "ksize is too large, ksize is %s" % (str(ksize),)
        return False, reason

    return True, ""


# 'pylint: disable=too-many-lines,invalid-name,too-many-arguments,consider-using-in
# 'pylint: disable=too-many-branches,too-many-instance-attributes,too-many-locals
# 'pylint: disable=too-many-statements,no-self-use,too-many-public-methods
# 'pylint: disable=unused-argument
class MaxPoolWithargmaxPytorch:
    """
    Function: use to finish MaxPoolWithargmax main functions
    """

    def __init__(self, input_shape, ksize, strides, pads, dtype, dilation, ceil_mode, kernel_name):
        """
        init MaxPoolWithargmax parameters

        Parameters
        ----------
        ksize: list or tuple
            The size of the window for each dimension of the input tensor.
        strides: list or tuple
            The stride of the sliding window of the input tensor.
        pads: list int
            The value of padding in all dimention, (1, padh, padw, 1).
        dtype: int
            The output indices data type, only support int32 or int64.
        dilation: list int
            A parameter that controls the stride of elements in the window.
        ceil_mode: Bool
            If True, will use ceil instead of floor to compute the output
            shape
        kernel_name: str
            The kernel's name
        Returns
        -------
        None
        """
        self.dtype = dtype
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        self.kernel_name = kernel_name
        self.mode_310p_fp32 = True if tbe_platform.get_soc_spec("SHORT_SOC_VERSION") == "Ascend310P" \
                              and dtype == "float32" else False
        if not self.mode_310p_fp32:
            self.ksize_h = ksize[1]
            self.ksize_w = ksize[2]
            self.strides_h = strides[1]
            self.strides_w = strides[2]
            self.pad_top = pads[1]
            self.pad_left = pads[2]
        self.tik_instance = tik.Tik()

        self.load3d_supported = tbe_platform.api_check_support("tik.load3dv1")
        self.core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.dtype_size = get_bit_len(self.dtype) // Constant.EIGHT_BIT

        self.ub_ele = (tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) - Constant.RESERVED_UB_SIZE) \
                      // self.dtype_size
        self.c_zero = 16
        # reduce need three buffer and open double buffer
        self.one_sixth_ub_ele = self.ub_ele // 6
        self.ub_avg_size = (self.ub_ele - Constant.GATHER_LEN * self.c_zero) // 6
        # using dtype_size to caculate mask
        self.mask = 256 // self.dtype_size
        # burst num of each c0, float16:1, float32:2
        self.burst_c0 = self.c_zero * self.dtype_size // Constant.BLOCK_BYTES
        # len of each block, float16:16, float32:8
        self.block_c0 = Constant.BLOCK_BYTES // self.dtype_size
        # compare times in each block
        self.block_process_num = self.block_c0 - 1

        if self.dtype == "float16":
            self.min_float = Constant.MIN_FL16
        else:
            self.min_float = Constant.MIN_FL32
        self.init_gm_tensor()
        if not self.mode_310p_fp32:
            self.resnet50_branch = resnet50.MaxPoolWithArgmaxV2Resnet50(input_shape, dtype, self.input_gm,
                                                                        self.max_output_gm, self.mask_output_gm,
                                                                        self.tik_instance)
        self.tiling_mode = None
        self.act_core_num = None
        self.one_core_ele = None
        self.last_core_ele = None
        self.input_h = None
        self.input_w = None
        self.output_h = None
        self.output_w = None
        self.pad_h = None
        self.pad_w = None
        self.pad_t = None
        self.pad_b = None
        self.pad_l = None
        self.pad_r = None
        self.c_factor = None
        self.h_factor = None
        self.w_factor = None
        self.one_core_loop_num = None
        self.one_core_loop_left = None
        self.last_core_loop_num = None
        self.last_core_loop_left = None
        self.n_c1 = None
        self.align_w_factor = None
        self.align_w_loop_left = None
        self.align_output_w = None
        self.align_output_hw = None
        self.ub_a = None
        self.ub_b = None
        self.ub_c = None
        self.ub_d = None
        self.ub_e = None
        self.ub_f = None
        self.core_ele = None
        self.loop_num = None
        self.loop_left = None
        self.offset_gm = None
        self.offset_ub = None
        self.nburst = None
        self.burst_len = None
        self.src_stride = None
        self.dst_stride = None
        self.size_1 = None
        self.offset_2 = None
        self.size_2 = None
        self.offset_3 = None
        self.repeat_3 = None
        self.size_3 = None
        self.before_h = None
        self.after_h = None
        self.before_w = None
        self.after_w = None
        self.len_h = None
        self.len_w = None
        self.Scalar_zero = None
        self.max_ele = None
        self.tiling_ub = None
        self.data_repeat = None
        self.data_repeat_left = None
        self.ub_src = None
        self.align_ele_w = None
        self.core_num_var = None
        self.ksize_h_var = None
        self.ksize_w_var = None
        self.strides_h_var = None
        self.strides_w_var = None

    def tiling_args(self):
        """Get runtime params from tiling
        """
        self.tiling_mode = self.tik_instance.Scalar("int32", name="tiling_mode")
        self.tiling_mode.set_as(self.tiling_ub[0])
        self.act_core_num = self.tik_instance.Scalar("int32", name="act_core_num")
        self.act_core_num.set_as(self.tiling_ub[1])
        self.one_core_ele = self.tik_instance.Scalar("int32", name="one_core_ele")
        self.one_core_ele.set_as(self.tiling_ub[2])
        self.last_core_ele = self.tik_instance.Scalar("int32", name="last_core_ele")
        self.last_core_ele.set_as(self.tiling_ub[3])
        self.input_h = self.tik_instance.Scalar("int32", name="input_h")
        self.input_h.set_as(self.tiling_ub[4])
        self.input_w = self.tik_instance.Scalar("int32", name="input_w")
        self.input_w.set_as(self.tiling_ub[5])
        self.output_h = self.tik_instance.Scalar("int32", name="output_h")
        self.output_h.set_as(self.tiling_ub[6])
        self.output_w = self.tik_instance.Scalar("int32", name="output_w")
        self.output_w.set_as(self.tiling_ub[7])
        self.pad_h = self.tik_instance.Scalar("int32", name="pad_h")
        self.pad_h.set_as(self.tiling_ub[8])
        self.pad_w = self.tik_instance.Scalar("int32", name="pad_w")
        self.pad_w.set_as(self.tiling_ub[9])
        self.pad_t = self.tik_instance.Scalar("int32", name="pad_t")
        self.pad_t.set_as(self.tiling_ub[10])
        self.pad_b = self.tik_instance.Scalar("int32", name="pad_b")
        self.pad_b.set_as(self.tiling_ub[11])
        self.pad_l = self.tik_instance.Scalar("int32", name="pad_l")
        self.pad_l.set_as(self.tiling_ub[12])
        self.pad_r = self.tik_instance.Scalar("int32", name="pad_r")
        self.pad_r.set_as(self.tiling_ub[13])
        self.c_factor = self.tik_instance.Scalar("int32", name="c_factor")
        self.c_factor.set_as(self.tiling_ub[14])
        self.h_factor = self.tik_instance.Scalar("int32", name="h_factor")
        self.h_factor.set_as(self.tiling_ub[15])
        self.w_factor = self.tik_instance.Scalar("int32", name="w_factor")
        self.w_factor.set_as(self.tiling_ub[16])
        self.one_core_loop_num = self.tik_instance.Scalar("int32", name="one_core_loop_num")
        self.one_core_loop_num.set_as(self.tiling_ub[17])
        self.one_core_loop_left = self.tik_instance.Scalar("int32", name="one_core_loop_left")
        self.one_core_loop_left.set_as(self.tiling_ub[18])
        self.last_core_loop_num = self.tik_instance.Scalar("int32", name="last_core_loop_num")
        self.last_core_loop_num.set_as(self.tiling_ub[19])
        self.last_core_loop_left = self.tik_instance.Scalar("int32", name="last_core_loop_left")
        self.last_core_loop_left.set_as(self.tiling_ub[20])
        self.n_c1 = self.tik_instance.Scalar("int32", name="n_c1")
        self.n_c1.set_as(self.tiling_ub[21])
        self.align_w_factor = self.tik_instance.Scalar("int32", name="align_w_factor")
        self.align_w_factor.set_as(self.tiling_ub[22])
        self.align_w_loop_left = self.tik_instance.Scalar("int32", name="align_w_loop_left")
        self.align_w_loop_left.set_as(self.tiling_ub[23])
        self.align_output_w = self.tik_instance.Scalar("int32", name="align_output_w")
        self.align_output_w.set_as(self.tiling_ub[24])
        self.align_output_hw = self.tik_instance.Scalar("int32", name="align_output_hw")
        self.align_output_hw.set_as(self.tiling_ub[25])
        self.core_num_var = self.tik_instance.Scalar("int32", name="core_num_var")
        self.core_num_var.set_as(self.tiling_ub[26])
        # for gather tensor
        self.align_ele_w = self.tik_instance.Scalar("int32", name="align_ele_w")
        self.data_repeat = self.tik_instance.Scalar("int32", name="data_repeat")
        self.data_repeat.set_as(self.output_w * self.output_h // Constant.GATHER_LEN)
        self.data_repeat_left = self.tik_instance.Scalar("int32", name="data_repeat_left")
        self.data_repeat_left.set_as(self.output_w * self.output_h % Constant.GATHER_LEN)
        self.ksize_h_var = self.tik_instance.Scalar("int32", name="ksize_h_var")
        self.ksize_w_var = self.tik_instance.Scalar("int32", name="ksize_w_var")
        self.strides_h_var = self.tik_instance.Scalar("int32", name="strides_h_var")
        self.strides_w_var = self.tik_instance.Scalar("int32", name="strides_w_var")
        if self.mode_310p_fp32:
            self.ksize_h_var.set_as(self.tiling_ub[27])
            self.ksize_w_var.set_as(self.tiling_ub[28])
            self.strides_h_var.set_as(self.tiling_ub[29])
            self.strides_w_var.set_as(self.tiling_ub[30])
        else:
            self.ksize_h_var.set_as(self.ksize_h)
            self.ksize_w_var.set_as(self.ksize_w)
            self.strides_h_var.set_as(self.strides_h)
            self.strides_w_var.set_as(self.strides_w)

    def init_gm_tensor(self):
        """Init gm tensor
        """
        self.tiling_gm = self.tik_instance.Tensor("int32", (Constant.TILING_ARG_NUM,), name="tiling_gm",
                                                  scope=tik.scope_gm)
        self.input_gm = self.tik_instance.Tensor(self.dtype, (Constant.MAX_INT32,), name="input_gm",
                                                 scope=tik.scope_gm)
        self.max_output_gm = self.tik_instance.Tensor(self.dtype, (Constant.MAX_INT32,), name="max_output_gm",
                                                      scope=tik.scope_gm)
        self.mask_output_gm = self.tik_instance.Tensor("uint16", (Constant.MAX_INT32,), name="mask_output_gm",
                                                       scope=tik.scope_gm)

    def init_ub_tensor(self):
        """Init ub tensor
        """
        self.ub_a = self.tik_instance.Tensor(self.dtype, (self.one_sixth_ub_ele,), name="ub_a", scope=tik.scope_ubuf)
        self.ub_b = self.tik_instance.Tensor(self.dtype, (self.one_sixth_ub_ele,), name="ub_b", scope=tik.scope_ubuf)
        self.ub_c = self.tik_instance.Tensor(self.dtype, (self.one_sixth_ub_ele,), name="ub_c", scope=tik.scope_ubuf)
        self.ub_d = self.tik_instance.Tensor(self.dtype, (self.one_sixth_ub_ele,), name="ub_d", scope=tik.scope_ubuf)
        self.ub_e = self.tik_instance.Tensor(self.dtype, (self.one_sixth_ub_ele,), name="ub_e", scope=tik.scope_ubuf)
        self.ub_f = self.tik_instance.Tensor(self.dtype, (self.one_sixth_ub_ele,), name="ub_f", scope=tik.scope_ubuf)

    def dynamic_ub_tensor(self):
        """Init ub tensor
        """
        self.ub_src = self.tik_instance.Tensor(self.dtype, (Constant.UB_SRC_SIZE,), name="ub_src", scope=tik.scope_ubuf)

        self.ub_a = self.tik_instance.Tensor(self.dtype, (self.ub_avg_size,), name="ub_a", scope=tik.scope_ubuf)
        self.ub_b = self.tik_instance.Tensor(self.dtype, (self.ub_avg_size,), name="ub_b", scope=tik.scope_ubuf)
        self.ub_c = self.tik_instance.Tensor(self.dtype, (self.ub_avg_size,), name="ub_c", scope=tik.scope_ubuf)
        self.ub_d = self.tik_instance.Tensor(self.dtype, (self.ub_avg_size,), name="ub_d", scope=tik.scope_ubuf)
        self.ub_e = self.tik_instance.Tensor(self.dtype, (self.ub_avg_size,), name="ub_e", scope=tik.scope_ubuf)
        self.ub_f = self.tik_instance.Tensor(self.dtype, (self.ub_avg_size,), name="ub_f", scope=tik.scope_ubuf)

    def init_ub_scalar(self):
        """Init ub scalar
        """
        # first core and last core use the same scalar
        self.core_ele = self.tik_instance.Scalar("int32", name="core_ele")
        self.loop_num = self.tik_instance.Scalar("int32", name="loop_num")
        self.loop_left = self.tik_instance.Scalar("int32", name="loop_left")
        # only for data_move in
        self.offset_gm = self.tik_instance.Scalar("int32", name="offset_gm")
        self.offset_ub = self.tik_instance.Scalar("int32", name="offset_ub")
        self.nburst = self.tik_instance.Scalar("int32", name="nburst")
        self.burst_len = self.tik_instance.Scalar("int32", name="burst_len_in")
        self.src_stride = self.tik_instance.Scalar("int32", name="src_stride")
        self.dst_stride = self.tik_instance.Scalar("int32", name="dst_stride")
        # only for vector dup, offset_1 is zero
        self.size_1 = self.tik_instance.Scalar("int32", name="size_1")
        self.offset_2 = self.tik_instance.Scalar("int32", name="offset_2")
        self.size_2 = self.tik_instance.Scalar("int32", name="size_2")
        self.offset_3 = self.tik_instance.Scalar("int32", name="offset_3")
        self.repeat_3 = self.tik_instance.Scalar("int32", name="repeat_3")
        self.size_3 = self.tik_instance.Scalar("int32", name="size_3")
        # only for reduce max
        self.before_h = self.tik_instance.Scalar("int32", name="before_h")
        self.after_h = self.tik_instance.Scalar("int32", name="after_h")
        self.before_w = self.tik_instance.Scalar("int32", name="before_w")
        self.after_w = self.tik_instance.Scalar("int32", name="after_w")
        self.len_h = self.tik_instance.Scalar("int32", name="len_h")
        self.len_w = self.tik_instance.Scalar("int32", name="len_w")
        self.Scalar_zero = self.tik_instance.Scalar(dtype='uint16', name='mask_zero', init_value=0)

    def copy_only(self, core_idx, loop_num, loop_left):
        """Only execute move in and move out
        """
        # max ele of 'n*c1*h*w' can move to ub
        align_max_ele = self.tik_instance.Scalar("int32", name="align_max_ele")
        self.max_ele = self.ub_ele // self.c_zero
        with self.tik_instance.if_scope(self.max_ele % self.c_zero != 0):
            align_max_ele.set_as((self.max_ele // self.c_zero + 1) * self.c_zero)
        with self.tik_instance.else_scope():
            align_max_ele.set_as(self.max_ele)

        align_loop_left = self.tik_instance.Scalar("int32", name="align_loop_left")
        with self.tik_instance.if_scope(loop_left % self.c_zero != 0):
            align_loop_left.set_as((loop_left // self.c_zero + 1) * self.c_zero)
        with self.tik_instance.else_scope():
            align_loop_left.set_as(loop_left)

        with self.tik_instance.for_range(0, loop_num) as loop_idx:
            with self.tik_instance.if_scope(tik.all(core_idx == self.act_core_num - 1,
                                                    loop_left == 0, loop_idx == loop_num - 1)):
                self.copy_only_process(core_idx, loop_idx, self.max_ele, align_max_ele, 1)
            with self.tik_instance.else_scope():
                self.copy_only_process(core_idx, loop_idx, self.max_ele, align_max_ele, 0)

        with self.tik_instance.if_scope(loop_left > 0):
            with self.tik_instance.if_scope(core_idx == self.act_core_num - 1):
                self.copy_only_process(core_idx, loop_num, loop_left, align_loop_left, 1)
            with self.tik_instance.else_scope():
                self.copy_only_process(core_idx, loop_num, loop_left, align_loop_left, 0)

    def copy_only_process(self, core_idx, loop_idx, ele, align_ele, last_flag):
        """Only execute move in and move out
        """
        offset = (core_idx * self.one_core_ele + loop_idx * self.max_ele) * self.c_zero
        ub_tensor = self.tik_instance.Tensor(self.dtype, (self.max_ele * self.c_zero,),
                                             name="ub_tensor",
                                             scope=tik.scope_ubuf)
        self.tik_instance.data_move(ub_tensor, self.input_gm[offset], 0, 1, ele * self.burst_c0, 0, 0)
        self.tik_instance.data_move(self.max_output_gm[offset], ub_tensor, 0, 1, ele * self.burst_c0, 0, 0)

        mask_offset = core_idx * self.one_core_ele + loop_idx * self.max_ele
        ub_tensor = ub_tensor.reinterpret_cast_to("uint16")
        repeat_time = align_ele // self.mask
        repeat_left = align_ele % self.mask
        repeat_loop = repeat_time // Constant.REPEAT_LIMIT
        loop_left = repeat_time % Constant.REPEAT_LIMIT
        with self.tik_instance.for_range(0, repeat_loop) as rep_i:
            offset = self.mask * rep_i * Constant.REPEAT_LIMIT
            self.tik_instance.vec_dup(self.mask, ub_tensor[offset],
                                      Constant.MAX_INT16, Constant.REPEAT_LIMIT, 8)
        with self.tik_instance.if_scope(loop_left != 0):
            offset = self.mask * repeat_loop * Constant.REPEAT_LIMIT
            self.tik_instance.vec_dup(self.mask, ub_tensor[offset], Constant.MAX_INT16, loop_left, 8)
        with self.tik_instance.if_scope(repeat_left != 0):
            offset = self.mask * repeat_loop * Constant.REPEAT_LIMIT + self.mask * loop_left
            self.tik_instance.vec_dup(repeat_left, ub_tensor[offset], Constant.MAX_INT16, 1, 8)

        self.tik_instance.data_move(self.mask_output_gm[mask_offset], ub_tensor, 0, 1, align_ele // self.c_zero, 0, 0)
        with self.tik_instance.if_scope(last_flag == 1):
            mask_left = self.align_output_hw * self.n_c1 - (mask_offset + align_ele)
            last_offset = mask_offset + align_ele
            with self.tik_instance.if_scope(tik.all(mask_left < align_ele, mask_left >= self.c_zero)):
                self.tik_instance.data_move(self.mask_output_gm[last_offset], ub_tensor,
                                            0, 1, mask_left // self.c_zero, 0, 0)
            with self.tik_instance.else_scope():
                with self.tik_instance.for_range(0, mask_left // self.c_zero) as n_idx:
                    last_offset = last_offset + n_idx * self.c_zero
                    self.tik_instance.data_move(self.mask_output_gm[last_offset], ub_tensor,
                                                0, 1, 1, 0, 0)

    def vector_dup_continuous(self, src, size):
        """vector_dup continuous function, set ubuf to -65504.0
        """
        with self.tik_instance.if_scope(size > 0):
            size_loop = size // self.mask
            size_left = size % self.mask
            repeat_loop = size_loop // Constant.REPEAT_LIMIT
            repeat_left = size_loop % Constant.REPEAT_LIMIT

            with self.tik_instance.if_scope(repeat_left > 0):
                repeat_offset = repeat_loop * Constant.REPEAT_LIMIT * self.mask
                self.tik_instance.vector_dup(self.mask, src[repeat_offset], self.min_float, repeat_left, 1, 8)
            with self.tik_instance.if_scope(size_left > 0):
                size_offset = size_loop * self.mask
                self.tik_instance.vector_dup(size_left, src[size_offset], self.min_float, 1, 1, 8)

    def vector_dup_discrete(self, src, repeat, size, dst_blk=1, dst_rep=8):
        """vector_dup discrete function, set ubuf to -65504.0, dst_rep is pad_w or len_w (cut w)
        """
        with self.tik_instance.if_scope(size > 0):
            # dst_rep less and equal to 255
            with self.tik_instance.if_scope(dst_rep * self.burst_c0 <= 255):
                size_loop = size // self.mask
                size_left = size % self.mask
                repeat_loop = repeat // Constant.REPEAT_LIMIT
                repeat_left = repeat % Constant.REPEAT_LIMIT

                def _inner(src, mask_len):
                    """exec repeat
                    """
                    with self.tik_instance.for_range(0, repeat_loop) as repeat_loop_idx:
                        repeat_offset = repeat_loop_idx * Constant.REPEAT_LIMIT * dst_rep * self.c_zero
                        self.tik_instance.vector_dup(mask_len, src[repeat_offset], self.min_float,
                                                     Constant.REPEAT_LIMIT,
                                                     dst_blk, dst_rep * self.burst_c0)
                    with self.tik_instance.if_scope(repeat_left > 0):
                        repeat_offset = repeat_loop * Constant.REPEAT_LIMIT * dst_rep * self.c_zero
                        self.tik_instance.vector_dup(mask_len, src[repeat_offset], self.min_float, repeat_left,
                                                     dst_blk,
                                                     dst_rep * self.burst_c0)

                with self.tik_instance.for_range(0, size_loop) as size_loop_idx:
                    size_offset = size_loop_idx * self.mask
                    _inner(src[size_offset:], self.mask)
                with self.tik_instance.if_scope(size_left > 0):
                    size_offset = size_loop * self.mask
                    _inner(src[size_offset:], size_left)
            # dst_rep greater to 255
            with self.tik_instance.else_scope():
                with self.tik_instance.for_range(0, repeat) as repeat_loop_idx:
                    repeat_offset = repeat_loop_idx * dst_rep * self.c_zero
                    self.vector_dup_continuous(src[repeat_offset:], size)

    def reduce_max_repeat_width(self, dst, src0, src1, size, src0_blk=1, src1_blk=1):
        """reduce max for width and height with repeat width, src0_blk/src1_blk is strides_w or one
        """
        # strides_w less and equal to 31
        src0_blk = src0_blk * self.burst_c0
        src1_blk = src1_blk * self.burst_c0
        size = size // self.burst_c0
        with self.tik_instance.if_scope(self.strides_w_var * self.burst_c0 <= 31):
            with self.tik_instance.if_scope(size > 0):
                size_loop = size // self.mask
                size_left = size % self.mask
                repeat_loop = size_loop // Constant.REPEAT_LIMIT
                repeat_left = size_loop % Constant.REPEAT_LIMIT

                with self.tik_instance.if_scope(repeat_left > 0):
                    with self.tik_instance.for_range(0, self.burst_c0) as burst_c0_i:
                        c0_bias = self.block_c0 * burst_c0_i
                        repeat_offset = repeat_loop * Constant.REPEAT_LIMIT * self.burst_c0 * self.mask + c0_bias
                        repeat_offset_src0 = repeat_loop * Constant.REPEAT_LIMIT * src0_blk * self.mask + c0_bias
                        repeat_offset_src1 = repeat_loop * Constant.REPEAT_LIMIT * src1_blk * self.mask + c0_bias
                        self.tik_instance.vmax(self.mask, dst[repeat_offset], src0[repeat_offset_src0],
                                               src1[repeat_offset_src1], repeat_left, self.burst_c0,
                                               src0_blk, src1_blk, 8 * self.burst_c0, src0_blk * 8, src1_blk * 8)

                with self.tik_instance.if_scope(size_left > 0):
                    with self.tik_instance.for_range(0, self.burst_c0) as burst_c0_i:
                        c0_bias = self.block_c0 * burst_c0_i
                        size_offset = size_loop * self.burst_c0 * self.mask + c0_bias
                        size_offset_src0 = size_loop * src0_blk * self.mask + c0_bias
                        size_offset_src1 = size_loop * src1_blk * self.mask + c0_bias
                        self.tik_instance.vmax(size_left, dst[size_offset], src0[size_offset_src0],
                                               src1[size_offset_src1], 1, self.burst_c0, src0_blk,
                                               src1_blk, 8 * self.burst_c0, src0_blk * 8, src1_blk * 8)
        # strides_w greater to 31
        with self.tik_instance.else_scope():
            with self.tik_instance.if_scope(size > 0):
                size_loop = size // self.c_zero
                with self.tik_instance.for_range(0, size_loop) as size_loop_idx:
                    with self.tik_instance.for_range(0, self.burst_c0) as burst_c0_i:
                        c0_bias = self.block_c0 * burst_c0_i
                        size_offset = size_loop_idx * self.burst_c0 * self.c_zero + c0_bias
                        size_offset_src0 = size_loop_idx * src0_blk * self.c_zero + c0_bias
                        size_offset_src1 = size_loop_idx * src1_blk * self.c_zero + c0_bias
                        self.tik_instance.vmax(self.c_zero, dst[size_offset], src0[size_offset_src0],
                                               src1[size_offset_src1], 1, self.burst_c0, self.burst_c0,
                                               self.burst_c0, 8 * self.burst_c0, 8 * self.burst_c0, 8 * self.burst_c0)

    def reduce_max_repeat_width_ksize_one_width(self, ub_x, ub_y, repeat_ph, size_ow, size_pw):
        """Reduce max width with repeat width when ksize equal to one
        """
        with self.tik_instance.if_scope(self.strides_w_var == self.ksize_w_var):
            self.reduce_max_repeat_width(ub_y, ub_x, ub_x, repeat_ph * size_ow, self.strides_w_var, self.strides_w_var)
        with self.tik_instance.else_scope():
            with self.tik_instance.for_range(0, repeat_ph) as p_idx:
                offset_dst = p_idx * size_ow
                offset_src = p_idx * size_pw
                self.reduce_max_repeat_width(ub_y[offset_dst:], ub_x[offset_src:], ub_x[offset_src:], size_ow,
                                            self.strides_w_var, self.strides_w_var)

    def reduce_max_repeat_width_ksize_more_width(self, ub_x, ub_y, repeat_ph, size_ow, size_pw):
        """Reduce max width with repeat width when ksize not equal to one
        """
        with self.tik_instance.if_scope(self.strides_w_var == self.ksize_w_var):
            self.reduce_max_repeat_width(ub_y, ub_x, ub_x[self.c_zero:], repeat_ph * size_ow,
                                         self.strides_w_var, self.strides_w_var)
            with self.tik_instance.for_range(0, self.ksize_w_var - 2) as idx:
                offset_w = (idx + 2) * self.c_zero
                self.reduce_max_repeat_width(ub_y, ub_x[offset_w:], ub_y, repeat_ph * size_ow, self.strides_w_var, 1)
        with self.tik_instance.else_scope():
            with self.tik_instance.for_range(0, repeat_ph) as p_idx:
                offset_dst = p_idx * size_ow
                offset_src0 = p_idx * size_pw
                offset_src1 = offset_src0 + self.c_zero
                self.reduce_max_repeat_width(ub_y[offset_dst:], ub_x[offset_src0:], ub_x[offset_src1:], size_ow,
                                             self.strides_w_var, self.strides_w_var)
                with self.tik_instance.for_range(0, self.ksize_w_var - 2) as idx:
                    offset_w = offset_src0 + (idx + 2) * self.c_zero
                    self.reduce_max_repeat_width(ub_y[offset_dst:], ub_x[offset_w:], ub_y[offset_dst:], size_ow,
                                                 self.strides_w_var, 1)

    def reduce_max_repeat_width_ksize_one_height(self, ub_y, ub_z, repeat_oh, size_ow):
        """Reduce max height with repeat width when ksize equal to one
        """
        with self.tik_instance.if_scope(self.strides_h_var == 1):
            self.reduce_max_repeat_width(ub_z, ub_y, ub_y, repeat_oh * size_ow, 1, 1)
        with self.tik_instance.else_scope():
            with self.tik_instance.for_range(0, repeat_oh) as o_idx:
                offset_dst = o_idx * size_ow
                offset_src = o_idx * self.strides_h_var * size_ow
                self.reduce_max_repeat_width(ub_z[offset_dst:], ub_y[offset_src:], ub_y[offset_src:], size_ow, 1, 1)

    def reduce_max_repeat_width_ksize_more_height(self, ub_y, ub_z, repeat_oh, size_ow):
        """Reduce max height with repeat width when ksize not equal to one
        """
        with self.tik_instance.if_scope(self.strides_h_var == 1):
            self.reduce_max_repeat_width(ub_z, ub_y, ub_y[size_ow:], repeat_oh * size_ow, 1, 1)
            with self.tik_instance.for_range(0, self.ksize_h_var - 2) as idx:
                offset_h = (idx + 2) * size_ow
                self.reduce_max_repeat_width(ub_z, ub_y[offset_h:], ub_z, repeat_oh * size_ow, 1, 1)
        with self.tik_instance.else_scope():
            with self.tik_instance.for_range(0, repeat_oh) as o_idx:
                offset_dst = o_idx * size_ow
                offset_src0 = o_idx * self.strides_h_var * size_ow
                offset_src1 = offset_src0 + size_ow
                self.reduce_max_repeat_width(ub_z[offset_dst:], ub_y[offset_src0:], ub_y[offset_src1:], size_ow, 1, 1)
                with self.tik_instance.for_range(0, self.ksize_h_var - 2) as idx:
                    offset_h = offset_src0 + (idx + 2) * size_ow
                    self.reduce_max_repeat_width(ub_z[offset_dst:], ub_y[offset_h:], ub_z[offset_dst:], size_ow, 1, 1)

    def reduce_max_repeat_height(self, dst, src0, src1, repeat, size, src0_blk=1,
                                 src1_blk=1, dst_rep=8, src0_rep=8, src1_rep=8):
        """reduce max for width and height with repeat height,
        repeat is pad_h and output_h, size is output_w * self.c_zero,
        src0_blk/src1_blk is strides_w(<=255), dst_rep is output_w(<=255),
        src0_rep/src1_rep is pad_w or ouput_w or output_w * strides_h(<=255)
        """
        with self.tik_instance.if_scope(size > 0):
            src0_blk = src0_blk * self.burst_c0
            src1_blk = src1_blk * self.burst_c0
            dst_rep = dst_rep * self.burst_c0
            src0_rep = src0_rep * self.burst_c0
            src1_rep = src1_rep * self.burst_c0
            size = size // self.burst_c0
            size_loop = size // self.mask
            size_left = size % self.mask
            repeat_loop = repeat // Constant.REPEAT_LIMIT
            repeat_left = repeat % Constant.REPEAT_LIMIT

            def _inner(dst, src0, src1, mask_len):
                """exec repeat
                """
                with self.tik_instance.for_range(0, repeat_loop) as repeat_loop_idx:
                    with self.tik_instance.for_range(0, self.burst_c0) as burst_c0_i:
                        c0_bias = self.block_c0 * burst_c0_i
                        # dst_rep may not equal to 8, discrete emissions
                        repeat_offset = repeat_loop_idx * Constant.REPEAT_LIMIT * dst_rep * self.c_zero + c0_bias
                        # src_rep may not equal to 8, discrete emissions
                        repeat_offset_src0 = repeat_loop_idx * Constant.REPEAT_LIMIT * src0_rep * self.c_zero + c0_bias
                        repeat_offset_src1 = repeat_loop_idx * Constant.REPEAT_LIMIT * src1_rep * self.c_zero + c0_bias
                        self.tik_instance.vmax(mask_len, dst[repeat_offset], src0[repeat_offset_src0],
                                               src1[repeat_offset_src1], Constant.REPEAT_LIMIT, self.burst_c0,
                                               src0_blk,
                                               src1_blk, dst_rep, src0_rep, src1_rep)

                with self.tik_instance.if_scope(repeat_left > 0):
                    with self.tik_instance.for_range(0, self.burst_c0) as burst_c0_i:
                        c0_bias = self.block_c0 * burst_c0_i
                        repeat_offset = repeat_loop * Constant.REPEAT_LIMIT * dst_rep * self.c_zero + c0_bias
                        repeat_offset_src0 = repeat_loop * Constant.REPEAT_LIMIT * src0_rep * self.c_zero + c0_bias
                        repeat_offset_src1 = repeat_loop * Constant.REPEAT_LIMIT * src1_rep * self.c_zero + c0_bias
                        self.tik_instance.vmax(mask_len, dst[repeat_offset], src0[repeat_offset_src0],
                                               src1[repeat_offset_src1], repeat_left, self.burst_c0,
                                               src0_blk, src1_blk, dst_rep, src0_rep, src1_rep)

            # exec size
            with self.tik_instance.for_range(0, size_loop) as size_loop_idx:
                # dst_blk must be equal to 1, continuous emissions
                size_offset = size_loop_idx * self.burst_c0 * self.mask
                # src_blk may not equal to 1, discrete emissions
                size_offset_src0 = size_loop_idx * src0_blk * self.mask
                size_offset_src1 = size_loop_idx * src1_blk * self.mask
                _inner(dst[size_offset:], src0[size_offset_src0:], src1[size_offset_src1:], self.mask)

            with self.tik_instance.if_scope(size_left > 0):
                size_offset = size_loop * self.burst_c0 * self.mask
                size_offset_src0 = size_loop * src0_blk * self.mask
                size_offset_src1 = size_loop * src1_blk * self.mask
                _inner(dst[size_offset:], src0[size_offset_src0:], src1[size_offset_src1:], size_left)

    def caculate_mask(self, ub_x, ub_y, ub_z, repeat_ph, size_ow, align_w, size_pw, ksize_h, ksize_w):
        """caculate mask, ub_x:input, ub_y:mask, ub_z:max
        """
        with self.tik_instance.for_range(0, ksize_h) as ks_h:
            with self.tik_instance.for_range(0, ksize_w) as ks_w:
                with self.tik_instance.for_range(0, repeat_ph) as p_idx:
                    offset_dst = ks_h * ksize_w * repeat_ph * align_w + \
                                 ks_w * repeat_ph * align_w + p_idx * align_w
                    offset_src0 = p_idx * size_pw * self.strides_h_var + ks_h * size_pw + ks_w * self.c_zero
                    offset_src1 = p_idx * size_ow
                    self.mask_repeat_width(ub_y[offset_dst:], ub_x[offset_src0:], ub_z[offset_src1:],
                                           size_ow, self.strides_w_var, 1)

    def gather_tensor_w(self, raw_src, src_len, src_offset):
        first_row = self.output_w - src_offset
        with self.tik_instance.if_scope(first_row < src_len):
            with self.tik_instance.if_scope(first_row > 0):
                self.tik_instance.data_move(self.ub_src, raw_src, 0, first_row, 1, self.strides_w_var - 1, 0)

            mid_repeat = (src_len - first_row) // self.output_w
            last_left = (src_len - first_row) % self.output_w

            with self.tik_instance.for_range(0, mid_repeat) as mid_idx:
                src_idx = first_row * self.c_zero + self.output_w * mid_idx * self.c_zero
                raw_src_idx = ((first_row - 1) * self.strides_w_var + self.ksize_w_var) * self.c_zero + \
                              (self.strides_h_var - 1) * self.pad_w * self.c_zero + \
                              mid_idx * self.strides_h_var * self.pad_w * self.c_zero
                self.tik_instance.data_move(self.ub_src[src_idx], raw_src[raw_src_idx], 0, self.output_w, 1,
                                            self.strides_w_var - 1, 0)

            with self.tik_instance.if_scope(last_left > 0):
                src_idx = first_row * self.c_zero + self.output_w * mid_repeat * self.c_zero
                raw_src_idx = ((first_row - 1) * self.strides_w_var + self.ksize_w_var) * self.c_zero + \
                              (self.strides_h_var - 1) * self.pad_w * self.c_zero + \
                              mid_repeat * self.strides_h_var * self.pad_w * self.c_zero
                self.tik_instance.data_move(self.ub_src[src_idx], raw_src[raw_src_idx], 0, last_left, 1,
                                            self.strides_w_var - 1, 0)
        with self.tik_instance.else_scope():
            self.tik_instance.data_move(self.ub_src, raw_src, 0, src_len, 1, self.strides_w_var - 1, 0)

    def caculate_mask_gather(self, ub_x, ub_y, ub_z, size_pw, align_w, ksize_h, ksize_w):
        """caculate mask gather, ub_x:input, ub_y:mask, ub_z:max
        """
        with self.tik_instance.for_range(0, ksize_h) as ks_h:
            with self.tik_instance.for_range(0, ksize_w) as ks_w:
                with self.tik_instance.for_range(0, self.data_repeat) as p_idx:
                    src_h = (p_idx * Constant.GATHER_LEN // self.output_w) * self.strides_h_var + ks_h
                    src_w = (p_idx * Constant.GATHER_LEN % self.output_w) * self.strides_w_var + ks_w
                    src_offset = src_h * size_pw + src_w * self.c_zero
                    max_offset = p_idx * Constant.GATHER_LEN * self.c_zero
                    offset_dst = ks_h * ksize_w * align_w + \
                                 ks_w * align_w + p_idx * Constant.GATHER_LEN
                    self.gather_tensor_w(ub_x[src_offset:], Constant.GATHER_LEN,
                                         p_idx * Constant.GATHER_LEN % self.output_w)
                    self.mask_repeat_width(ub_y[offset_dst:], self.ub_src, ub_z[max_offset:],
                                           Constant.GATHER_LEN * self.c_zero, 1, 1)

                with self.tik_instance.if_scope(self.data_repeat_left > 0):
                    src_h = (self.data_repeat * Constant.GATHER_LEN // self.output_w) * self.strides_h_var + ks_h
                    src_w = (self.data_repeat * Constant.GATHER_LEN % self.output_w) * self.strides_w_var + ks_w
                    src_offset = src_h * size_pw + src_w * self.c_zero
                    max_offset = self.data_repeat * Constant.GATHER_LEN * self.c_zero
                    offset_dst = ks_h * ksize_w * align_w + \
                                 ks_w * align_w + self.data_repeat * Constant.GATHER_LEN
                    self.gather_tensor_w(ub_x[src_offset:], self.data_repeat_left,
                                         self.data_repeat * Constant.GATHER_LEN % self.output_w)
                    self.mask_repeat_width(ub_y[offset_dst:], self.ub_src, ub_z[max_offset:],
                                           self.data_repeat_left * self.c_zero, 1, 1)

    def mask_repeat_width(self, dst, src0, src1, size, src0_blk=1, src1_blk=1):
        """caculate mask
        """
        # strides_w less and equal to 31
        with self.tik_instance.if_scope(self.strides_w_var <= 31):
            with self.tik_instance.if_scope(size > 0):
                size_loop = self.tik_instance.Scalar(dtype="uint64")
                with self.tik_instance.if_scope(size % self.mask != 0):
                    size_loop.set_as(size // self.mask + 1)
                with self.tik_instance.else_scope():
                    size_loop.set_as(size // self.mask)
                self.tik_instance.vcmpv_eq(dst, src0, src1,
                                           size_loop, src0_blk, src1_blk, src0_blk * 8, src1_blk * 8)
        # strides_w greater to 31
        with self.tik_instance.else_scope():
            with self.tik_instance.if_scope(size > 0):
                size_loop = size // self.c_zero
                with self.tik_instance.for_range(0, size_loop) as size_loop_idx:
                    size_offset = size_loop_idx * self.c_zero
                    size_offset_src0 = size_loop_idx * src0_blk * self.c_zero
                    size_offset_src1 = size_loop_idx * src1_blk * self.c_zero
                    self.tik_instance.vcmpv_eq(dst[size_offset], src0[size_offset_src0], src1[size_offset_src1],
                                               1, 1, 1, 1, 1)
                    dst[size_loop_idx].set_as(dst[size_offset])

    def remove_repeated_mask(self, ub_x, ub_y, ub_z, size_owh, ksize_h, ksize_w):
        """remove repeated mask, ub_x:mask_or, ub_y:mask, ub_z:mask_not
        """
        with self.tik_instance.for_range(0, ksize_h) as ks_h:
            with self.tik_instance.for_range(0, ksize_w) as ks_w:
                offset = ks_h * ksize_w * size_owh + ks_w * size_owh
                self.remove_repeated_mask_kernel(ub_y[offset:], ub_x, ub_z, size_owh, ks_h, ks_w)

    def remove_repeated_mask_kernel(self, mask, mask_or, mask_not, size, index_h, index_w):
        """remove repeated mask
        """
        with self.tik_instance.if_scope(tik.all(index_h == 0, index_w == 0)):
            repeat_time = self.tik_instance.Scalar(dtype="uint64")
            repeat_left = self.tik_instance.Scalar(dtype="uint64")
            repeat_time.set_as(size // self.mask)
            repeat_left.set_as(size % self.mask)
            self.tik_instance.data_move(mask_or, mask, 0, 1, size // self.c_zero, 0, 0)
            with self.tik_instance.if_scope(repeat_time != 0):
                self.tik_instance.vnot(self.mask, mask_not, mask, repeat_time, 1, 1, 8, 8)
            with self.tik_instance.if_scope(repeat_left != 0):
                offset = self.mask * repeat_time
                self.tik_instance.vnot(repeat_left, mask_not[offset:], mask[offset:], 1, 1, 1, 1, 1)

        with self.tik_instance.else_scope():
            repeat_time = self.tik_instance.Scalar(dtype="uint64")
            repeat_left = self.tik_instance.Scalar(dtype="uint64")
            repeat_time.set_as(size // self.mask)
            repeat_left.set_as(size % self.mask)
            with self.tik_instance.if_scope(repeat_time != 0):
                self.tik_instance.vand(self.mask, mask, mask_not, mask, repeat_time, 1, 1, 1, 8, 8, 8)
                self.tik_instance.vor(self.mask, mask_or, mask_or, mask, repeat_time, 1, 1, 1, 8, 8, 8)
                self.tik_instance.vnot(self.mask, mask_not, mask_or, repeat_time, 1, 1, 8, 8)
            with self.tik_instance.if_scope(repeat_left != 0):
                offset = self.mask * repeat_time
                self.tik_instance.vand(repeat_left, mask[offset:], mask_not[offset:], mask[offset:],
                                       1, 1, 1, 1, 1, 1, 1)
                self.tik_instance.vor(repeat_left, mask_or[offset:], mask_or[offset:], mask[offset:],
                                      1, 1, 1, 1, 1, 1, 1)
                self.tik_instance.vnot(repeat_left, mask_not[offset:], mask_or[offset:],
                                       1, 1, 1, 1, 1)

    def tiling_c_dim_core_nc(self, core_idx, loop_num, loop_left):
        """Tiling c1 dim when core num at nc1
        """
        # move in and vector dup params
        self.size_1.set_as(self.pad_t * self.pad_w * self.c_zero + self.pad_l * self.c_zero)
        self.offset_2.set_as((self.pad_h - self.pad_b) * self.pad_w * self.c_zero - self.pad_r * self.c_zero)
        self.size_2.set_as(self.pad_b * self.pad_w * self.c_zero + self.pad_r * self.c_zero)
        self.offset_3.set_as(self.size_1 + self.input_w * self.c_zero)
        self.size_3.set_as((self.pad_r + self.pad_l) * self.c_zero)
        # when pad and cut at same time
        with self.tik_instance.if_scope((self.pad_t > 0) & ((self.pad_h - self.pad_t) < self.input_h)):
            self.nburst.set_as(self.pad_h - self.pad_t)
        with self.tik_instance.else_scope():
            with self.tik_instance.if_scope(self.pad_h <= self.input_h):
                self.nburst.set_as(self.pad_h)
            with self.tik_instance.else_scope():
                self.nburst.set_as(self.input_h)
        with self.tik_instance.if_scope((self.pad_l > 0) & ((self.pad_w - self.pad_l) < self.input_w)):
            self.offset_3.set_as(self.size_1 + (self.pad_w - self.pad_l - self.pad_r) * self.c_zero)
            self.src_stride.set_as(self.input_w - self.pad_w + self.pad_l)
            self.dst_stride.set_as(self.pad_l)
            self.burst_len.set_as(self.pad_w - self.pad_l)
            self.repeat_3.set_as(self.pad_h - 1 - self.pad_t)
        with self.tik_instance.else_scope():
            with self.tik_instance.if_scope(self.pad_w <= self.input_w):
                self.repeat_3.set_as(0)
                self.src_stride.set_as(self.input_w - self.pad_w)
                self.dst_stride.set_as(0)
                with self.tik_instance.if_scope(self.pad_w < self.input_w):
                    self.burst_len.set_as(self.pad_w)
                with self.tik_instance.else_scope():
                    self.burst_len.set_as(self.nburst * self.pad_w)
                    self.nburst.set_as(1)
            with self.tik_instance.else_scope():
                self.repeat_3.set_as(self.pad_h - 1)
                self.burst_len.set_as(self.input_w)
                self.src_stride.set_as(0)
                self.dst_stride.set_as(self.pad_r + self.pad_l)

        # run loop
        self.dynamic_ub_tensor()
        with self.tik_instance.for_range(0, loop_num // 2) as loop_idx:
            self.tiling_c_dim_core_nc_process(self.ub_a, self.ub_b, self.ub_c, core_idx, loop_idx * 2, self.c_factor)
            self.tiling_c_dim_core_nc_process(self.ub_d, self.ub_e, self.ub_f, core_idx, loop_idx * 2 + 1,
                                              self.c_factor)
        with self.tik_instance.if_scope(loop_num % 2 == 1):
            self.tiling_c_dim_core_nc_process(self.ub_a, self.ub_b, self.ub_c, core_idx, loop_num - 1, self.c_factor)
        with self.tik_instance.if_scope(loop_left > 0):
            self.tiling_c_dim_core_nc_process(self.ub_d, self.ub_e, self.ub_f, core_idx, loop_num, loop_left)

    def tiling_c_dim_core_nc_process(self, ub_x, ub_y, ub_z, core_idx, loop_idx, ele):
        """Tiling c1 dim process when core num at nc1
        """
        # vector dup and move in
        with self.tik_instance.new_stmt_scope(disable_sync=True):
            with self.tik_instance.for_range(0, ele) as ele_idx:
                offset_in = (core_idx * self.one_core_ele + loop_idx * self.c_factor +
                             ele_idx) * self.input_h * self.input_w * self.c_zero
                offset_a = ele_idx * self.pad_h * self.pad_w * self.c_zero
                self.tik_instance.data_move(ub_x[offset_a:][self.size_1], self.input_gm[offset_in], 0, self.nburst,
                                            self.burst_len * self.burst_c0, self.src_stride * self.burst_c0,
                                            self.dst_stride * self.burst_c0)
                self.vector_dup_continuous(ub_x[offset_a:], self.size_1)
                self.vector_dup_continuous(ub_x[offset_a:][self.offset_2:], self.size_2)
                self.vector_dup_discrete(ub_x[offset_a:][self.offset_3:], self.repeat_3, self.size_3, 1, self.pad_w)

        with self.tik_instance.for_range(0, ele) as ele_idx:
            offset_a = ele_idx * self.pad_h * self.pad_w * self.c_zero
            offset_b = ele_idx * self.output_h * self.pad_w * self.c_zero
            offset_c = ele_idx * self.output_h * self.output_w * self.c_zero
            # reduce max width
            with self.tik_instance.if_scope(self.ksize_w_var == 1):
                self.reduce_max_repeat_width_ksize_one_width(ub_x[offset_a:], ub_y[offset_b:], self.pad_h,
                                                             self.output_w * self.c_zero, self.pad_w * self.c_zero)
            with self.tik_instance.else_scope():
                self.reduce_max_repeat_width_ksize_more_width(ub_x[offset_a:], ub_y[offset_b:], self.pad_h,
                                                              self.output_w * self.c_zero, self.pad_w * self.c_zero)
            with self.tik_instance.if_scope(self.ksize_h_var == 1):
                self.reduce_max_repeat_width_ksize_one_height(ub_y[offset_b:], ub_z[offset_c:], self.output_h,
                                                              self.output_w * self.c_zero)
            with self.tik_instance.else_scope():
                self.reduce_max_repeat_width_ksize_more_height(ub_y[offset_b:], ub_z[offset_c:], self.output_h,
                                                               self.output_w * self.c_zero)
        # move out
        offset_out = (core_idx * self.one_core_ele + loop_idx * self.c_factor) * \
                     self.output_h * self.output_w * self.c_zero
        self.tik_instance.data_move(self.max_output_gm[offset_out], ub_z, 0, 1,
                                    ele * self.output_h * self.output_w * self.burst_c0, 0, 0)

        # caculate mask, ub_x:input, ub_y:mask, ub_z:max
        ub_y = ub_y.reinterpret_cast_to("uint16")
        with self.tik_instance.for_range(0, ele) as ele_idx:
            offset_a = ele_idx * self.pad_h * self.pad_w * self.c_zero
            offset_c = ele_idx * self.output_h * self.output_w * self.c_zero
            offset_b = ele_idx * self.align_output_hw * self.ksize_w_var * self.ksize_h_var
            self.caculate_mask_gather(ub_x[offset_a:], ub_y[offset_b:], ub_z[offset_c:], self.pad_w * self.c_zero,
                                      self.align_output_hw, self.ksize_h_var, self.ksize_w_var)

        # remove repeated mask, ub_x:mask_or, ub_y:mask, ub_z:mask_and
        ub_x = ub_x.reinterpret_cast_to("uint16")
        ub_z = ub_z.reinterpret_cast_to("uint16")
        with self.tik_instance.for_range(0, ele) as ele_idx:
            offset = ele_idx * self.align_output_hw * self.ksize_w_var * self.ksize_h_var
            self.remove_repeated_mask(ub_x[offset:], ub_y[offset:], ub_z[offset:], self.align_output_hw,
                                      self.ksize_h_var, self.ksize_w_var)
            offset_mask_out = (core_idx * self.one_core_ele + loop_idx * self.c_factor) \
                              * self.ksize_w_var * self.ksize_h_var * self.align_output_hw
            self.tik_instance.data_move(self.mask_output_gm[offset_mask_out], ub_y, 0, 1,
                                        ele * self.ksize_w_var * self.ksize_h_var * self.align_output_hw // 16, 0, 0)

    def tiling_h_dim_core_nc(self, core_idx, core_ele, loop_num, loop_left):
        """Tiling h dim when core num at nc1
        """
        # run loop
        with self.tik_instance.for_range(0, loop_num) as loop_idx:
            self.tiling_h_dim_core_nc_process(core_idx, core_ele, loop_idx, self.h_factor)
        with self.tik_instance.if_scope(loop_left > 0):
            self.tiling_h_dim_core_nc_process(core_idx, core_ele, loop_num, loop_left)

    def tiling_h_dim_core_nc_gather(self, core_idx, core_ele, loop_num, loop_left):
        """Tiling h dim when core num at nc1
        """
        # run loop
        with self.tik_instance.for_range(0, loop_num) as loop_idx:
            self.tiling_h_dim_core_nc_gather_process(core_idx, core_ele, loop_idx, self.h_factor)
        with self.tik_instance.if_scope(loop_left > 0):
            self.tiling_h_dim_core_nc_gather_process(core_idx, core_ele, loop_num, loop_left)

    def tiling_h_pad_cut_data(self):
        # when pad and cut at same time
        with self.tik_instance.if_scope((self.pad_l > 0) & ((self.pad_w - self.pad_l) < self.input_w)):
            self.offset_3.set_as(self.size_1 + (self.pad_w - self.pad_l - self.pad_r) * self.c_zero)
            self.src_stride.set_as(self.input_w - self.pad_w + self.pad_l)
            self.dst_stride.set_as(self.pad_l)
            self.burst_len.set_as(self.pad_w - self.pad_l)
        with self.tik_instance.else_scope():
            with self.tik_instance.if_scope(self.pad_w <= self.input_w):
                self.src_stride.set_as(self.input_w - self.pad_w)
                self.dst_stride.set_as(0)
                with self.tik_instance.if_scope(self.pad_w < self.input_w):
                    self.burst_len.set_as(self.pad_w)
                with self.tik_instance.else_scope():
                    self.burst_len.set_as(self.nburst * self.pad_w)
                    self.nburst.set_as(1)
            with self.tik_instance.else_scope():
                self.burst_len.set_as(self.input_w)
                self.src_stride.set_as(0)
                self.dst_stride.set_as(self.pad_r + self.pad_l)

    def tiling_h_load_input_data(self, loop_idx, ele):
        # move in and vector dup params
        self.before_h.set_as(loop_idx * self.h_factor * self.strides_h_var)
        self.after_h.set_as((loop_idx * self.h_factor + ele - 1) * self.strides_h_var + self.ksize_h_var)
        self.len_h.set_as(self.after_h - self.before_h)
        with self.tik_instance.if_scope(self.before_h < self.pad_t):
            self.size_1.set_as((self.pad_t - self.before_h) * self.pad_w * self.c_zero + self.pad_l * self.c_zero)
            self.offset_3.set_as(self.size_1 + self.input_w * self.c_zero)
            self.offset_gm.set_as(0)
            with self.tik_instance.if_scope(self.after_h <= self.pad_h - self.pad_b):
                self.offset_2.set_as(self.len_h * self.pad_w * self.c_zero - self.pad_r * self.c_zero)
                self.size_2.set_as(self.pad_r * self.c_zero)
                self.repeat_3.set_as(self.after_h - self.pad_t - 1)
                self.size_3.set_as((self.pad_r + self.pad_l) * self.c_zero)
                self.nburst.set_as(self.after_h - self.pad_t)
            with self.tik_instance.else_scope():
                self.offset_2.set_as((self.pad_h - self.pad_b - self.before_h) * self.pad_w * self.c_zero -
                                     self.pad_r * self.c_zero)
                self.size_2.set_as((self.after_h - (self.pad_h - self.pad_b)) * self.pad_w * self.c_zero +
                                   self.pad_r * self.c_zero)
                self.repeat_3.set_as(self.input_h - 1)
                self.size_3.set_as((self.pad_r + self.pad_l) * self.c_zero)
                self.nburst.set_as(self.input_h)
        with self.tik_instance.else_scope():
            self.size_1.set_as(self.pad_l * self.c_zero)
            self.offset_3.set_as(self.size_1 + self.input_w * self.c_zero)
            self.offset_gm.set_as((self.before_h - self.pad_t) * self.input_w * self.c_zero)
            with self.tik_instance.if_scope(self.after_h <= self.pad_h - self.pad_b):
                self.offset_2.set_as(self.len_h * self.pad_w * self.c_zero - self.pad_r * self.c_zero)
                self.size_2.set_as(self.pad_r * self.c_zero)
                self.repeat_3.set_as(self.len_h - 1)
                self.size_3.set_as((self.pad_r + self.pad_l) * self.c_zero)
                self.nburst.set_as(self.len_h)
            with self.tik_instance.else_scope():
                self.offset_2.set_as((self.pad_h - self.pad_b - self.before_h) * self.pad_w * self.c_zero -
                                     self.pad_r * self.c_zero)
                self.size_2.set_as((self.after_h - (self.pad_h - self.pad_b)) * self.pad_w * self.c_zero +
                                   self.pad_r * self.c_zero)
                self.repeat_3.set_as(self.pad_h - self.pad_b - self.before_h - 1)
                self.size_3.set_as((self.pad_r + self.pad_l) * self.c_zero)
                self.nburst.set_as(self.pad_h - self.pad_b - self.before_h)
        self.tiling_h_pad_cut_data()

    def tiling_h_caculate_max(self, ub_x, ub_y, ub_z, ele_idx, core_idx, loop_idx, ele):
        offset_in = (core_idx * self.one_core_ele + ele_idx) * self.input_h * self.input_w * self.c_zero + \
                    self.offset_gm
        # vector dup and move in
        with self.tik_instance.if_scope(self.before_h <= self.pad_h - self.pad_b):
            with self.tik_instance.new_stmt_scope(disable_sync=True):
                self.tik_instance.data_move(ub_x[self.size_1], self.input_gm[offset_in], 0, self.nburst,
                                            self.burst_len * self.burst_c0, self.src_stride * self.burst_c0,
                                            self.dst_stride * self.burst_c0)
                self.vector_dup_continuous(ub_x, self.size_1)
                self.vector_dup_continuous(ub_x[self.offset_2:], self.size_2)
                self.vector_dup_discrete(ub_x[self.offset_3:], self.repeat_3, self.size_3, 1, self.pad_w)
        with self.tik_instance.else_scope():
            self.vector_dup_continuous(ub_x, self.len_h * self.pad_w * self.c_zero)
        # reduce max width
        with self.tik_instance.if_scope(self.ksize_w_var == 1):
            self.reduce_max_repeat_width_ksize_one_width(ub_x, ub_y, self.len_h, self.output_w * self.c_zero,
                                                         self.pad_w * self.c_zero)
        with self.tik_instance.else_scope():
            self.reduce_max_repeat_width_ksize_more_width(ub_x, ub_y, self.len_h, self.output_w * self.c_zero,
                                                          self.pad_w * self.c_zero)
        with self.tik_instance.if_scope(self.ksize_h_var == 1):
            self.reduce_max_repeat_width_ksize_one_height(ub_y, ub_z, ele, self.output_w * self.c_zero)
        with self.tik_instance.else_scope():
            self.reduce_max_repeat_width_ksize_more_height(ub_y, ub_z, ele, self.output_w * self.c_zero)
        # move out
        offset_out = (core_idx * self.one_core_ele + ele_idx) * self.output_h * self.output_w * self.c_zero + \
                     loop_idx * self.h_factor * self.output_w * self.c_zero
        self.tik_instance.data_move(self.max_output_gm[offset_out], ub_z, 0, 1,
                                    ele * self.output_w * self.burst_c0, 0, 0)

    def tiling_h_dim_core_nc_gather_process(self, core_idx, core_ele, loop_idx, ele):
        """Tiling h dim process when core num at nc1
        """
        self.tiling_h_load_input_data(loop_idx, ele)

        def _inner(ub_x, ub_y, ub_z, ele_idx):
            self.tiling_h_caculate_max(ub_x, ub_y, ub_z, ele_idx, core_idx, loop_idx, ele)

            self.align_ele_w.set_as(ele * self.output_w)
            with self.tik_instance.if_scope(self.align_ele_w % self.c_zero != 0):
                self.align_ele_w.set_as((self.align_ele_w // self.c_zero + 1) * self.c_zero)
            self.data_repeat.set_as(ele * self.output_w // Constant.GATHER_LEN)
            self.data_repeat_left.set_as(ele * self.output_w % Constant.GATHER_LEN)

            # caculate mask, ub_x:input, ub_y:mask, ub_z:max
            ub_y = ub_y.reinterpret_cast_to("uint16")
            self.caculate_mask_gather(ub_x, ub_y, ub_z, self.pad_w * self.c_zero,
                                      self.align_ele_w, self.ksize_h_var, self.ksize_w_var)

            # remove repeated mask, ub_x:mask_or, ub_y:mask, ub_z:mask_and
            ub_x = ub_x.reinterpret_cast_to("uint16")
            ub_z = ub_z.reinterpret_cast_to("uint16")
            self.remove_repeated_mask(ub_x, ub_y, ub_z, self.align_ele_w,
                                      self.ksize_h_var, self.ksize_w_var)

            # move out the mask and gather the mask in gm at the same time
            base_dst = (core_idx * self.one_core_ele + ele_idx) * self.ksize_w_var * self.ksize_h_var \
                        * self.align_output_hw + loop_idx * self.h_factor * self.output_w
            with self.tik_instance.for_range(0, self.ksize_h_var * self.ksize_w_var) as k_idx:
                offset_dst = base_dst + k_idx * self.align_output_hw
                offset_src = k_idx * self.align_ele_w
                self.tik_instance.data_move(self.mask_output_gm[offset_dst], ub_y[offset_src], 0, 1,
                                            self.align_ele_w // self.c_zero, 0, 0)

        self.dynamic_ub_tensor()
        with self.tik_instance.for_range(0, core_ele // 2) as ele_idx:
            _inner(self.ub_a, self.ub_b, self.ub_c, ele_idx * 2)
            _inner(self.ub_d, self.ub_e, self.ub_f, ele_idx * 2 + 1)
        with self.tik_instance.if_scope(core_ele % 2 == 1):
            _inner(self.ub_a, self.ub_b, self.ub_c, core_ele - 1)

    def tiling_h_dim_core_nc_process(self, core_idx, core_ele, loop_idx, ele):
        """Tiling h dim process when core num at nc1
        """
        self.tiling_h_load_input_data(loop_idx, ele)

        def _inner(ub_x, ub_y, ub_z, ele_idx):
            self.tiling_h_caculate_max(ub_x, ub_y, ub_z, ele_idx, core_idx, loop_idx, ele)
            # caculate mask, ub_x:input, ub_y:mask, ub_z:max
            ub_y = ub_y.reinterpret_cast_to("uint16")
            self.caculate_mask(ub_x, ub_y, ub_z, ele, self.output_w * self.c_zero,
                               self.align_output_w, self.pad_w * self.c_zero, self.ksize_h_var, self.ksize_w_var)

            # remove repeated mask, ub_x:mask_or, ub_y:mask, ub_z:mask_and
            ub_x = ub_x.reinterpret_cast_to("uint16")
            ub_z = ub_z.reinterpret_cast_to("uint16")
            self.remove_repeated_mask(ub_x, ub_y, ub_z, ele * self.align_output_w,
                                      self.ksize_h_var, self.ksize_w_var)

            # move out the mask and gather the mask in gm at the same time
            base_dst = (core_idx * self.one_core_ele + ele_idx) * self.ksize_w_var * self.ksize_h_var \
                       * self.align_output_hw + loop_idx * self.h_factor * self.output_w
            with self.tik_instance.for_range(0, self.ksize_h_var * self.ksize_w_var) as k_idx:
                with self.tik_instance.for_range(0, ele) as h_idx:
                    offset_dst = base_dst + k_idx * self.align_output_hw + h_idx * self.output_w
                    offset_src = k_idx * ele * self.align_output_w + h_idx * self.align_output_w
                    self.tik_instance.data_move(self.mask_output_gm[offset_dst], ub_y[offset_src], 0, 1,
                                                self.align_output_w // self.c_zero, 0, 0)

        self.init_ub_tensor()
        with self.tik_instance.for_range(0, core_ele // 2) as ele_idx:
            _inner(self.ub_a, self.ub_b, self.ub_c, ele_idx * 2)
            _inner(self.ub_d, self.ub_e, self.ub_f, ele_idx * 2 + 1)
        with self.tik_instance.if_scope(core_ele % 2 == 1):
            _inner(self.ub_a, self.ub_b, self.ub_c, core_ele - 1)

    def tiling_w_dim_core_nc(self, core_idx, core_ele, loop_num, loop_left):
        """Tiling w dim when core num at nc1
        """
        with self.tik_instance.for_range(0, self.output_h) as h_idx:
            with self.tik_instance.for_range(0, loop_num) as loop_idx:
                self.tiling_w_dim_core_nc_process(core_idx, core_ele, h_idx, loop_idx, self.w_factor,
                                                  self.align_w_factor)
            with self.tik_instance.if_scope(loop_left > 0):
                self.tiling_w_dim_core_nc_process(core_idx, core_ele, h_idx, loop_num, loop_left,
                                                  self.align_w_loop_left)

    def tiling_w_dim_core_nc_process(self, core_idx, core_ele, h_idx, loop_idx, ele, align_ele):
        """Tiling w dim process when core num at nc1
        """
        # move in and vector dup params
        self.before_w.set_as(loop_idx * self.w_factor * self.strides_w_var)
        self.after_w.set_as((loop_idx * self.w_factor + ele - 1) * self.strides_w_var + self.ksize_w_var)
        self.len_w.set_as(self.after_w - self.before_w)
        self.before_h.set_as(h_idx * self.strides_h_var)
        self.after_h.set_as(self.before_h + self.ksize_h_var)
        with self.tik_instance.if_scope(self.before_h < self.pad_t):
            self.size_1.set_as((self.pad_t - self.before_h) * self.len_w * self.c_zero)
            self.offset_2.set_as(0)
            self.size_2.set_as(0)
            self.repeat_3.set_as(self.after_h - self.pad_t)
            self.nburst.set_as(self.after_h - self.pad_t)
            with self.tik_instance.if_scope(self.after_h > self.pad_t + self.input_h):
                self.offset_2.set_as((self.pad_t - self.before_h + self.input_h) * self.len_w * self.c_zero)
                self.size_2.set_as((self.after_h - self.pad_t - self.input_h) * self.len_w * self.c_zero)
                self.repeat_3.set_as(self.input_h)
                self.nburst.set_as(self.input_h)
            with self.tik_instance.if_scope(self.before_w < self.pad_l):
                with self.tik_instance.if_scope(self.after_w <= self.pad_w - self.pad_r):
                    self.offset_3.set_as(self.size_1)
                    self.size_3.set_as((self.pad_l - self.before_w) * self.c_zero)
                    self.offset_gm.set_as(0)
                    self.offset_ub.set_as(self.size_1 + (self.pad_l - self.before_w) * self.c_zero)
                    self.burst_len.set_as(self.after_w - self.pad_l)
                    self.src_stride.set_as(self.input_w - (self.after_w - self.pad_l))
                    self.dst_stride.set_as(self.pad_l - self.before_w)
            with self.tik_instance.else_scope():
                self.offset_gm.set_as((self.before_w - self.pad_l) * self.c_zero)
                self.offset_ub.set_as(self.size_1)
                with self.tik_instance.if_scope(self.after_w <= self.pad_w - self.pad_r):
                    self.offset_3.set_as(0)
                    self.size_3.set_as(0)
                    self.burst_len.set_as(self.len_w)
                    self.src_stride.set_as(self.input_w - self.len_w)
                    self.dst_stride.set_as(0)
                with self.tik_instance.else_scope():
                    self.offset_3.set_as(self.size_1 + (self.pad_w - self.pad_r - self.before_w) * self.c_zero)
                    self.size_3.set_as((self.after_w - (self.pad_w - self.pad_r)) * self.c_zero)
                    self.burst_len.set_as(self.pad_w - self.pad_r - self.before_w)
                    self.src_stride.set_as(self.before_w - self.pad_l)
                    self.dst_stride.set_as(self.after_w - (self.pad_w - self.pad_r))
        with self.tik_instance.else_scope():
            self.size_1.set_as(0)
            with self.tik_instance.if_scope(self.after_h <= self.pad_h - self.pad_b):
                self.offset_2.set_as(0)
                self.size_2.set_as(0)
                self.repeat_3.set_as(self.ksize_h_var)
                self.nburst.set_as(self.ksize_h_var)
            with self.tik_instance.else_scope():
                self.offset_2.set_as((self.pad_h - self.pad_b - self.before_h) * self.len_w * self.c_zero)
                self.size_2.set_as((self.after_h - (self.pad_h - self.pad_b)) * self.len_w * self.c_zero)
                self.repeat_3.set_as(self.pad_h - self.pad_b - self.before_h)
                self.nburst.set_as(self.pad_h - self.pad_b - self.before_h)
            with self.tik_instance.if_scope(self.before_w < self.pad_l):
                with self.tik_instance.if_scope(self.after_w <= self.pad_w - self.pad_r):
                    self.offset_3.set_as(0)
                    self.size_3.set_as((self.pad_l - self.before_w) * self.c_zero)
                    self.offset_gm.set_as((self.before_h - self.pad_t) * self.input_w * self.c_zero)
                    self.offset_ub.set_as(self.size_3)
                    self.burst_len.set_as(self.after_w - self.pad_l)
                    self.src_stride.set_as(self.input_w - (self.after_w - self.pad_l))
                    self.dst_stride.set_as(self.pad_l - self.before_w)
            with self.tik_instance.else_scope():
                self.offset_gm.set_as((self.before_h - self.pad_t) * self.input_w * self.c_zero +
                                      (self.before_w - self.pad_l) * self.c_zero)
                self.offset_ub.set_as(0)
                with self.tik_instance.if_scope(self.after_w <= self.pad_w - self.pad_r):
                    self.offset_3.set_as(0)
                    self.size_3.set_as(0)
                    self.burst_len.set_as(self.len_w)
                    self.src_stride.set_as(self.input_w - self.len_w)
                    self.dst_stride.set_as(0)
                with self.tik_instance.else_scope():
                    self.offset_3.set_as((self.pad_w - self.pad_r - self.before_w) * self.c_zero)
                    self.size_3.set_as((self.after_w - (self.pad_w - self.pad_r)) * self.c_zero)
                    self.burst_len.set_as(self.pad_w - self.pad_r - self.before_w)
                    self.src_stride.set_as(self.before_w - self.pad_l)
                    self.dst_stride.set_as(self.after_w - (self.pad_w - self.pad_r))

        def _inner(ub_x, ub_y, ub_z, ele_idx):
            offset_in = (core_idx * self.one_core_ele + ele_idx) * self.input_h * self.input_w * self.c_zero + \
                        self.offset_gm
            # vector dup and move in
            with self.tik_instance.if_scope((self.before_h <= (self.pad_h - self.pad_b)) &
                                            (self.before_w <= (self.pad_w - self.pad_r))):
                with self.tik_instance.new_stmt_scope(disable_sync=True):
                    self.tik_instance.data_move(ub_x[self.offset_ub], self.input_gm[offset_in], 0, self.nburst,
                                                self.burst_len * self.burst_c0, self.src_stride * self.burst_c0,
                                                self.dst_stride * self.burst_c0)
                    self.vector_dup_continuous(ub_x, self.size_1)
                    self.vector_dup_continuous(ub_x[self.offset_2:], self.size_2)
                    self.vector_dup_discrete(ub_x[self.offset_3:], self.repeat_3, self.size_3, 1, self.len_w)
            with self.tik_instance.else_scope():
                self.vector_dup_continuous(ub_x, (self.after_h - self.before_h) * self.len_w * self.c_zero)
            # reduce max width
            with self.tik_instance.if_scope(self.ksize_w_var == 1):
                self.reduce_max_repeat_width_ksize_one_width(ub_x, ub_y, self.ksize_h_var, ele * self.c_zero,
                                                             self.len_w * self.c_zero)
            with self.tik_instance.else_scope():
                self.reduce_max_repeat_width_ksize_more_width(ub_x, ub_y, self.ksize_h_var, ele * self.c_zero,
                                                              self.len_w * self.c_zero)
            with self.tik_instance.if_scope(self.ksize_h_var == 1):
                self.reduce_max_repeat_width_ksize_one_height(ub_y, ub_z, 1, ele * self.c_zero)
            with self.tik_instance.else_scope():
                self.reduce_max_repeat_width_ksize_more_height(ub_y, ub_z, 1, ele * self.c_zero)
            # move out
            offset_out = (core_idx * self.one_core_ele + ele_idx) * self.output_h * self.output_w * self.c_zero + \
                         h_idx * self.output_w * self.c_zero + loop_idx * self.w_factor * self.c_zero
            self.tik_instance.data_move(self.max_output_gm[offset_out], ub_z, 0, 1, ele * self.burst_c0, 0, 0)

            # caculate mask, ub_x:input, ub_y:mask, ub_z:max
            ub_y = ub_y.reinterpret_cast_to("uint16")
            self.caculate_mask(ub_x, ub_y, ub_z, 1, ele * self.c_zero,
                               align_ele, self.len_w * self.c_zero, self.ksize_h_var, self.ksize_w_var)

            # remove repeated mask, ub_x:mask_or, ub_y:mask, ub_z:mask_and
            ub_x = ub_x.reinterpret_cast_to("uint16")
            ub_z = ub_z.reinterpret_cast_to("uint16")
            self.remove_repeated_mask(ub_x, ub_y, ub_z, align_ele,
                                      self.ksize_h_var, self.ksize_w_var)

            # move out the mask and gather the mask in gm at the same time
            base_dst = (core_idx * self.one_core_ele + ele_idx) * self.ksize_w_var * self.ksize_h_var * \
                       self.align_output_hw + h_idx * self.output_w + loop_idx * self.w_factor
            with self.tik_instance.for_range(0, self.ksize_h_var * self.ksize_w_var) as k_idx:
                offset_dst = base_dst + k_idx * self.align_output_hw
                offset_src = k_idx * align_ele
                self.tik_instance.data_move(self.mask_output_gm[offset_dst], ub_y[offset_src],
                                            0, 1, align_ele // self.c_zero, 0, 0)

        self.init_ub_tensor()
        with self.tik_instance.for_range(0, core_ele // 2) as ele_idx:
            _inner(self.ub_a, self.ub_b, self.ub_c, ele_idx * 2)
            _inner(self.ub_d, self.ub_e, self.ub_f, ele_idx * 2 + 1)
        with self.tik_instance.if_scope(core_ele % 2 == 1):
            _inner(self.ub_a, self.ub_b, self.ub_c, core_ele - 1)

    def max_pool_compute_tiling(self):
        """Maxpool compute tiling
        """
        self.tiling_ub = self.tik_instance.Tensor("int32", (Constant.TILING_ARG_NUM,),
                                                  name="tiling_ub",
                                                  scope=tik.scope_ubuf)
        self.tik_instance.data_move(self.tiling_ub, self.tiling_gm, 0, 1, 4, 0, 0)
        self.tiling_args()
        with self.tik_instance.for_range(0, self.core_num_var, block_num=self.core_num_var) as core_idx:
            # define tiling ub and move tiling gm to tiling ub,then get tiling args
            # call select tiling mode function
            self.init_ub_scalar()
            with self.tik_instance.if_scope(core_idx <= self.act_core_num - 1):
                with self.tik_instance.if_scope(core_idx < self.act_core_num - 1):
                    self.core_ele.set_as(self.one_core_ele)
                    self.loop_num.set_as(self.one_core_loop_num)
                    self.loop_left.set_as(self.one_core_loop_left)
                with self.tik_instance.if_scope(core_idx == self.act_core_num - 1):
                    self.core_ele.set_as(self.last_core_ele)
                    self.loop_num.set_as(self.last_core_loop_num)
                    self.loop_left.set_as(self.last_core_loop_left)
                # only move in and move out
                with self.tik_instance.if_scope(self.tiling_mode == 0):
                    with self.tik_instance.new_stmt_scope():
                        self.copy_only(core_idx, self.loop_num, self.loop_left)
                # tiling c1 dim when core num at nc1
                with self.tik_instance.if_scope(self.tiling_mode == 1):
                    with self.tik_instance.new_stmt_scope():
                        self.tiling_c_dim_core_nc(core_idx, self.loop_num, self.loop_left)
                # gather tensor
                with self.tik_instance.if_scope(self.tiling_mode == 4):
                    with self.tik_instance.new_stmt_scope():
                        self.tiling_h_dim_core_nc_gather(core_idx, self.core_ele, self.loop_num, self.loop_left)
                with self.tik_instance.if_scope(self.tiling_mode == 2):
                    with self.tik_instance.new_stmt_scope():
                        self.tiling_h_dim_core_nc(core_idx, self.core_ele, self.loop_num, self.loop_left)
                # tiling w dim when core num at nc1
                with self.tik_instance.if_scope(self.tiling_mode == 3):
                    with self.tik_instance.new_stmt_scope():
                        self.tiling_w_dim_core_nc(core_idx, self.core_ele, self.loop_num, self.loop_left)
                # resnet50 branch
                if not self.mode_310p_fp32:
                    with self.tik_instance.if_scope(self.tiling_mode == 6):
                        with self.tik_instance.new_stmt_scope():
                            self.resnet50_branch.maxpool_resnet50(core_idx, self.core_ele, self.one_core_ele)

    def max_pool_operator(self):
        """Maxpool operator
        """
        self.max_pool_compute_tiling()
        opt_config = {"out_of_bound_sync_check": True, "enable_const_fold": True}
        tbe_context.get_context().add_compile_info(
            "vars", {
                "ub_ele": self.ub_ele,
                "core_num": self.core_num,
                "fp32_310p": self.mode_310p_fp32,
                "ksize_h": self.ksize_h if not self.mode_310p_fp32 else 1,
                "ksize_w": self.ksize_w if not self.mode_310p_fp32 else 1,
                "strides_h": self.strides_h if not self.mode_310p_fp32 else 1,
                "strides_w": self.strides_w if not self.mode_310p_fp32 else 1,
                "padding": 0,
                "ceil_mode": self.ceil_mode,
                "pad_top": self.pad_top if not self.mode_310p_fp32 else 1,
                "pad_bottom": 0,
                "pad_left": self.pad_left if not self.mode_310p_fp32 else 1,
                "pad_right": 0,
                "load3d_supported": self.load3d_supported
            })
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[self.input_gm],
                                   outputs=[self.max_output_gm, self.mask_output_gm],
                                   flowtable=[self.tiling_gm],
                                   config=opt_config)

        return self.tik_instance


# 'pylint: disable=too-many-lines,invalid-name,too-many-arguments,consider-using-in
# 'pylint: disable=too-many-branches,too-many-instance-attributes,too-many-locals
# 'pylint: disable=too-many-statements,no-self-use,too-few-public-methods
# 'pylint: disable=unused-argument
def _check_param(x, ksize, strides, pads, dtype, dilation, ceil_mode, kernel_name):
    """
    check parameters, if one is invalid, then raise error
    Parameters
    ----------
    x: dict
        shape and datatype
    ksize: list or tuple
        the size of the window
    strides: list or tuple
        the stride of the sliding window
    padding: list or tuple
    kernel_name: str
    Returns
    -------
    None
    """
    input_shape = x.get("shape")
    input_dtype = x.get("dtype").lower()
    input_format = x.get("format")
    ori_format = x.get("ori_format")

    if len(input_shape) != 5:
        raise RuntimeError("invalid shape params, input feature map must be "
                           "5D format in kernel.")

    if input_dtype != "float16" and input_dtype != "float32":
        raise RuntimeError("Only support float16, float32")

    if input_format not in ("NC1HWC0",):
        error_manager_vector.raise_err_input_format_invalid(kernel_name, "input_data", "NC1HWC0", input_format)

    if ori_format in ("NCHW", "NC1HWC0"):
        dim_n = 0
        dim_c = 1
        dim_h = 2
        dim_w = 3
    elif ori_format in ("NHWC",):
        dim_n = 0
        dim_h = 1
        dim_w = 2
        dim_c = 3

    if input_shape[4] != 16:
        raise RuntimeError("invalid featur map shape params, "
                           "C0 must be equal to 16")

    if len(ksize) != 4:
        raise RuntimeError("Invalid ksize params, ksize dim must be 4.")

    if ksize[dim_n] != 1 or ksize[dim_c] != 1:
        raise RuntimeError("MaxPoolWithArgmax only supports pooling across width/height, and other ksize "
                           "dimension should be one")

    if (ksize[dim_h] // 2 < pads[dim_h]) or (ksize[dim_w] // 2 < pads[dim_w]):
        expected_value = "pads should be smaller than half of kernel size"
        real_value = "ksize // 2 < pads"
        error_manager_vector.raise_err_input_value_invalid(kernel_name, "ksize and pads", expected_value, real_value)

    if ksize[dim_h] < 1:
        expected_value = "greater than zero"
        real_value = ksize[dim_h]
        error_manager_vector.raise_err_input_value_invalid(kernel_name, "ksize_h", expected_value, real_value)

    if ksize[dim_w] < 1:
        expected_value = "greater than zero"
        real_value = ksize[dim_w]
        error_manager_vector.raise_err_input_value_invalid(kernel_name, "ksize_w", expected_value, real_value)

    if len(strides) != 4:
        raise RuntimeError("Invalid strides params, strides dim must be 4.")

    if strides[dim_n] != 1 or strides[dim_c] != 1:
        raise RuntimeError("MaxPoolWithArgmax only supports pooling across width/height, and other strides dimension "
                           "should be one")
    if strides[dim_h] < 1:
        expected_value = "greater than zero"
        real_value = strides[dim_h]
        error_manager_vector.raise_err_input_value_invalid(kernel_name, "strides_h", expected_value, real_value)

    if strides[dim_w] < 1:
        expected_value = "greater than zero"
        real_value = strides[dim_w]
        error_manager_vector.raise_err_input_value_invalid(kernel_name, "strides_w", expected_value, real_value)

    if strides[dim_h] > 2048:
        raise RuntimeError("strides h too large")

    if strides[dim_w] > 2048:
        raise RuntimeError("strides w too large")

    if len(pads) != 4:
        raise RuntimeError("Invalid padding params, padding dim must be 4.")

    if pads[dim_n] != 1 or pads[dim_c] != 1:
        raise RuntimeError(
            "MaxPoolWithArgmax only supports pooling across width or height, and other padding dimension "
            "should be one")

    if len(dilation) != 4:
        raise RuntimeError("Invalid dilation params, dilation dim must be 4.")

    if dilation[dim_n] != 1 or dilation[dim_c] != 1:
        raise RuntimeError(
            "MaxPoolWithArgmax only supports pooling across width or height, and other dilation dimension "
            "should be one")

    if dilation[dim_h] < 1:
        expected_value = "greater than zero"
        real_value = dilation[dim_h]
        error_manager_vector.raise_err_input_value_invalid(kernel_name, "strides_h", expected_value, real_value)

    if dilation[dim_w] < 1:
        expected_value = "greater than zero"
        real_value = dilation[dim_w]
        error_manager_vector.raise_err_input_value_invalid(kernel_name, "strides_w", expected_value, real_value)

    if ceil_mode is not True and ceil_mode is not False:
        raise RuntimeError("MaxPoolWithArgmax only supports ceil_mode across "
                           "True/False, and other string not support!")
    if dtype != Constant.DT_INT32:
        raise RuntimeError("MaxPoolWithArgmax only supports output indices data type: "
                           "int32, and other data type not support!")
    return [dim_n, dim_c, dim_h, dim_w]


def max_pool_with_argmax_v2_tik(x, y, argmax, ksize, strides, pads, dtype, dilation, ceil_mode, kernel_name):
    """
    max_pool_with_argmax_v2 interface for tik
    :param x: dict of shape and dtype of the input x
    :param y: dict of shape and dtype of the output y
    :param argmax: dict of shape and dtype of the output argmax
    :param ksize: the size of the window to take a max over
    :param strides: the stride of the window
    :param pads: implicit zero padding to be added on both sides
    :param dtype: input data type, only support int32 or int64
    :param dilation: a parameter that controls the stride of elements \
                     in the window
    :param ceil_mode: when True, will use ceil instead of floor to compute \
                      the output shape
    :param kernel_name: the kernel's name
    :return tik_instance
    """
    [_, _, dim_h, dim_w] = _check_param(x, ksize, strides, pads, dtype, dilation, ceil_mode, kernel_name)
    obj = MaxPoolWithargmaxPytorch(x.get("shape"), [1, ksize[dim_h], ksize[dim_w], 1],
                                   [1, strides[dim_h], strides[dim_w], 1],
                                   [1, pads[dim_h], pads[dim_w], 1], x.get("dtype").lower(),
                                   dilation, ceil_mode, kernel_name)
    return obj.max_pool_operator()


def max_pool_with_argmax_v2_compute(x, y, argmax, ksize, strides, pads, dtype, dilation, ceil_mode=False,
                                    kernel_name="max_pool_with_argmax_v1"):
    """
    max_pool_with_argmax_v2 compute for dsl
    """
    round_mode = "CEIL" if ceil_mode else "FLOOR"
    return tbe.reduce_window(x, "MAX", (2, 3), ksize, strides, dilation, "CALCULATED", pads, round_mode, True)


def max_pool_with_argmax_v2_dsl(x, y, argmax, ksize, strides, pads, dtype, dilation, ceil_mode, kernel_name):
    """
    max_pool_with_argmax_v2 interface for dsl
    :param x: dict of shape and dtype of the input x
    :param y: dict of shape and dtype of the output y
    :param argmax: dict of shape and dtype of the output argmax
    :param ksize: the size of the window to take a max over
    :param strides: the stride of the window
    :param pads: implicit zero padding to be added on both sides
    :param dtype: input data type, only support int32 or int64
    :param dilation: a parameter that controls the stride of elements \
                     in the window
    :param ceil_mode: when True, will use ceil instead of floor to compute \
                      the output shape
    :param kernel_name: the kernel's name
    """
    dtype_x = x.get("dtype")
    dtype_lower = dtype_x.lower()
    check_list = ("float16",)
    ori_format = x.get("ori_format")
    para_check.check_dtype(dtype_lower, check_list, param_name="x")
    windows_axes = [2, 3]
    attr_axes = [2, 3]
    if is_unknown_rank_input(x):
        x["shape"] = (-1, -1, -1, -1, 16)
        x["range"] = ((1, None), (1, None), (1, None), (1, None), (16, 16))
        extra_params = {
            Constant.WINDOW_AXES: windows_axes,
            Constant.ATTR_AXES: attr_axes,
            Constant.WINDOW_DIMENSIONS: [None, None, None, None],
            Constant.WINDOW_STRIDES: [None, None, None, None],
            Constant.WINDOW_PADDINGS: [[None, None], [None, None], [None, None], [None, None]],
            Constant.WINDOW_DILATIONS: [1, 1, 1, 1],
            Constant.CEIL_MODE: ceil_mode
        }
    else:
        extra_params = {
            Constant.WINDOW_AXES: windows_axes,
            Constant.ATTR_AXES: attr_axes,
            Constant.WINDOW_DIMENSIONS: ksize,
            Constant.WINDOW_STRIDES: strides,
            Constant.WINDOW_PADDINGS: [[pads[0], pads[0]], [pads[1], pads[1]], [pads[2], pads[2]], [pads[3], pads[3]]],
            Constant.WINDOW_DILATIONS: [1, 1, 1, 1],
            Constant.CEIL_MODE: ceil_mode
        }

    ins = classify([x], "pooling_with_arg", extra_params)
    schedules = []
    tensors = []
    for _x, _axes, _ksize, _strides, _paddings, _dilations, _mode in ins:
        with tbe.compute():
            shape_var, window_axes, window_dimensions, window_strides, padding_dimensions, window_dilations \
                = shape_util.variable_shape([_x, _axes, _ksize, _strides, _paddings, _dilations, _mode],
                                            op_mode="pooling_with_arg")

            data_input = tvm.placeholder(shape_var, name="data_input", dtype=dtype_lower)
            res = max_pool_with_argmax_v2_compute(data_input, y, argmax, window_dimensions, window_strides,
                                                  padding_dimensions, 3, window_dilations, _mode)
            tensors.append([data_input] + res)

        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    tbe_context.get_context().add_compile_info("dimensions_attr_idx", 0)
    tbe_context.get_context().add_compile_info("strides_attr_idx", 1)
    tbe_context.get_context().add_compile_info("pads_attr_idx", 2)
    tbe_context.get_context().add_compile_info("dilations_attr_idx", 4)
    tbe_context.get_context().add_compile_info("ceil_mode_idx", 5)
    tbe_context.get_context().add_compile_info("dimensions_attr_name", "ksize")
    tbe_context.get_context().add_compile_info("strides_attr_name", "strides")
    tbe_context.get_context().add_compile_info("pads_attr_name", "pads")
    tbe_context.get_context().add_compile_info("dilations_attr_name", "dilation")
    tbe_context.get_context().add_compile_info("ceil_mode_name", "ceil_mode")
    # It is used to distinguish between Tik implementation and DSL implementation in the tilling phase
    tbe_context.get_context().add_compile_info("is_dsl", True)
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)


# 'pylint: disable=unused-argument
@register_operator("MaxPoolWithArgmaxV2")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_LIST_INT,
                            para_check.OPTION_ATTR_LIST_INT,
                            para_check.OPTION_ATTR_LIST_INT,
                            para_check.OPTION_ATTR_INT,
                            para_check.OPTION_ATTR_LIST_INT,
                            para_check.OPTION_ATTR_BOOL,
                            para_check.KERNEL_NAME)
def max_pool_with_argmax_v2(x, y, argmax, ksize, strides, pads, dtype=Constant.DT_INT32, dilation=(1, 1, 1, 1),
                            ceil_mode=False, kernel_name="max_pool_with_argmax_v2"):
    """
    implementation of max_pool_with_argmax for pytorch and return the \
    tik instance
    :param x: dict of shape and dtype of the input x
    :param y: dict of shape and dtype of the output y
    :param argmax: dict of shape and dtype of the output argmax
    :param ksize: the size of the window to take a max over
    :param strides: the stride of the window
    :param pads: implicit zero padding to be added on both sides
    :param dtype: input data type, only support int32 or int64
    :param dilation: a parameter that controls the stride of elements \
                     in the window
    :param ceil_mode: when True, will use ceil instead of floor to compute \
                      the output shape
    :param kernel_name: the kernel's name
    :return: tik_instance
    """
    if is_unknown_rank_input(x):
        max_pool_with_argmax_v2_dsl(x, y, argmax, ksize, strides, pads, dtype, dilation, ceil_mode, kernel_name)
    else:
        max_pool_with_argmax_v2_tik(x, y, argmax, ksize, strides, pads, dtype, dilation, ceil_mode, kernel_name)
