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
max_pool_grad_with_argmaxv2
"""
# 'pylint: disable=too-many-lines
import math
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from impl.dynamic import max_pool_grad_with_argmax_v2_resnet50 as resnet50
from tbe.common.platform import get_bit_len


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    def __init__(self):
        pass
    # # tiling param num
    TILING_ARG_NUM = 64
    # MIN VALUE OF FP16
    MIN_VALUE_FP16 = -65500.0
    # VALID MASK BITS FOR 128
    MASK128_VALUE = 128
    # VALID MASK BITS FOR 64
    MASK64_VALUE = 64
    # MAX_VECTOR_REPEATE_TIME
    MAX_VECTOR_REPEATE_TIME = 255
    # VECTOR FP16 SIZE
    VECTOR_FP16_SIZE = 128
    # VECTOR FP32 SIZE
    VECTOR_FP32_SIZE = 64
    # BLOCK SIZE(32B)
    BLOCK_SIZE = 32
    # UB SIZE
    UB_SIZE = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
    # RESERVED UB SIZE
    RESERVED_UB_SIZE = 8 * 1024
    C0 = 16
    # BLOCK NUMS
    CORE_NUM = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
    DT_INT32 = 3
    # max int32
    MAX_INT32 = 2 ** 31 - 1
    # workspace size
    WORKSPACE_ONE_CORE = 1048576
    # element nums of each process
    ELEMENT_NUMS_EACH_TIME = 160


# 'pylint: disable=too-many-arguments,too-many-statements,too-many-branches
# 'pylint: disable=invalid-name,unused-argument
def check_param(x, argmax, grad, ksize, strides, data_format, dilation, pads, kernel_name):
    """
    check parameters, if one is invalid, then raise error
    Parameters
    ----------
    x: dict
        shape and data type of ori_input
    argmax: dict
        shape and data type of argmax
    grad: dict
        shape and data type of grad
    ksize: list or tuple
        the size of the window
    strides: list or tuple
        the stride of the sliding window
    dilation: list or tuple
        dilation of kernel
    pads: list or tuple
        padding size of height and width
    kernel_name: str

    Returns
    -------
    None
    """
    ori_input_shape = x.get("shape")
    ori_input_dtype = x.get("dtype").lower()
    argmax_shape = argmax.get("shape")
    argmax_dtype = argmax.get("dtype").lower()
    grad_shape = grad.get("shape")
    grad_dtype = grad.get("dtype").lower()
    para_check.check_shape(ori_input_shape, param_name="x")
    para_check.check_dtype(ori_input_dtype, ("float16",), param_name="x")
    para_check.check_shape(grad_shape, param_name="grad")
    para_check.check_dtype(grad_dtype, ("float16",), param_name="grad")
    para_check.check_shape(argmax_shape, param_name="argmax")
    para_check.check_dtype(argmax_dtype, ("uint16",), param_name="argmax")
    # the format of input_x must be NC1HWC0
    if len(ori_input_shape) != 5:
        error_manager_vector.raise_err_specific_reson("max_pool_grad_with_argmax_v2", "input feature map \
                                                      must be 5D format in kernel.")
    if len(grad_shape) != 5:
        error_manager_vector.raise_err_specific_reson("max_pool_grad_with_argmax_v2", "update grad \
                                                      must be 5D format in kernel.")

    if grad_shape[-1] != 16 or ori_input_shape[-1] != 16:
        error_manager_vector.raise_err_specific_reson("max_pool_grad_with_argmax_v2", "C0 must be equal to 16.")

    if len(dilation) != 4:
        error_manager_vector.raise_err_input_param_not_in_range("max_pool_grad_with_argmax_v2", "dilation",
                                                                "4", "4", str(len(dilation)))
    if dilation[0] != 1 or dilation[1] != 1 or dilation[2] != 1 or dilation[3] != 1:
        error_manager_vector.raise_err_specific_reson("max_pool_grad_with_argmax_v2", "can not support dilation now.")

    if len(pads) != 4:
        error_manager_vector.raise_err_input_param_not_in_range("max_pool_grad_with_argmax_v2", "pads", "4", "4",
                                                                str(len(pads)))
    if data_format in ("NHWC", "NC1HWC0", "NCHW"):
        if len(ksize) != 4:
            error_manager_vector.raise_err_input_param_not_in_range("max_pool_grad_with_argmax_v2", "ksize", "4", "4",
                                                                    str(len(ksize)))
        if ksize[0] != 1 or ksize[3] != 1:
            error_manager_vector.raise_err_input_value_invalid("max_pool_grad_with_argmax_v2", "ksize[0], ksize[3]",
                                                               "1", str(ksize[0]) + ", " + str(ksize[3]))
        if len(strides) != 4:
            error_manager_vector.raise_err_input_param_not_in_range("max_pool_grad_with_argmax_v2", "strides",
                                                                    "4", "4", str(len(strides)))
        if strides[0] != 1 or strides[3] != 1:
            error_manager_vector.raise_err_input_value_invalid("max_pool_grad_with_argmax_v2", "strides[0], strides[3]",
                                                               "1", str(strides[0]) + ", " + str(strides[3]))

        if strides[1] > 127 or strides[2] > 127:
            error_manager_vector.raise_err_specific_reson("max_pool_grad_with_argmax_v2",
                                                          "only support stride_h and stride_w be smaller than 127")
        if strides[1] * strides[2] >= 7932:
            error_manager_vector.raise_err_specific_reson("max_pool_grad_with_argmax_v2",
                                                          "strides is too large and not support")
        if ksize[1] * ksize[2] >= 1033:
            error_manager_vector.raise_err_specific_reson("max_pool_grad_with_argmax_v2",
                                                          "kernel size is too large and not support")
        if ksize[1] < 2 * pads[1]:
            error_manager_vector.raise_err_specific_reson("max_pool_grad_with_argmax_v2",
                                                          "pad H should be smaller than half of kernel H")
        if ksize[2] < 2 * pads[2]:
            error_manager_vector.raise_err_specific_reson("max_pool_grad_with_argmax_v2",
                                                          "pad W should be smaller than half of kernel W")
    else:
        error_manager_vector.raise_err_input_format_invalid("max_pool_grad_with_argmax_v2", "x", "NC1HWC0, NCHW, NHWC",
                                                            str(data_format))


# 'pylint: disable = too-many-instance-attributes,too-few-public-methods
class MaxpoolGrad:
    """
    MaxpoolGrad  Object include all fuction and paras
    """

    def __init__(self, dtype, ksize, strides, pads, dilation, ceil_mode, kernel_name):
        self.dtype = dtype
        self.ksize = ksize
        self.strides = strides
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        self.stride_h = strides[1]
        self.stride_w = strides[2]
        self.kh = ksize[1]
        self.kw = ksize[2]
        self.pads = pads
        self.kernel_name = kernel_name

        self.tik_instance = tik.Tik()
        self.core_num = Constant.CORE_NUM
        self.dtype_size = get_bit_len(self.dtype.lower()) // 8
        self.ub_ele = (Constant.UB_SIZE - Constant.RESERVED_UB_SIZE) // self.dtype_size
        # need seven buffers
        self.one_seventh_ub_ele = self.ub_ele // 7
        self.init_gm_tensor()

        # define some scalar
        self.scalar_zero = self.tik_instance.Scalar(dtype='float32', name='scalar_zero')
        self.scalar_zero_fp16 = self.tik_instance.Scalar(dtype='float16', name='scalar_zero_fp16')
        self.offset_gm = self.tik_instance.Scalar(dtype='int64', name='offset_gm')
        self.temp_scalar_int64 = self.tik_instance.Scalar(dtype='int64', name='temp_scalar_int64')

        self.scalar_zero_fp16.set_as(0)
        self.scalar_zero.set_as(0)

        self.core_num_var = None
        # get tiling params and init workspace
        self.init_tiling_param()
        self.get_tiling_args()
        self.init_workspace()

        self.resnet50_branch = resnet50.MaxpoolGradV2Resnet50(self.dtype,
                                                              self.ori_input_gm, self.grad_gm, self.argmax_gm,
                                                              self.res_gm, self.tik_instance)

        self.total_repeate_time = None
        self.remain_ele = None
        self.repeate_max_time = None
        self.remain_repeate_time = None
        self.ele_num = None
        self.ub_a = None
        self.ub_b = None
        self.ub_c = None
        self.ub_d = None
        self.ub_e = None
        self.pad = None
        self.pad_value = None

    def init_tiling_param(self):
        """init tiling params
        """
        self.tiling_mode = self.tik_instance.Scalar("int32", name="tiling_mode")
        self.real_block = self.tik_instance.Scalar("int32", name="real_block")
        self.block_cycle = self.tik_instance.Scalar("int32", name="block_cycle")
        self.ho_wo_16 = self.tik_instance.Scalar("int32", name="ho_wo_16")
        self.mask_shape_128 = self.tik_instance.Scalar("int32", name="mask_shape_128")
        self.pad_left = self.tik_instance.Scalar("int64", name="pad_left")
        self.pad_right = self.tik_instance.Scalar("int32", name="pad_right")
        self.pad_top = self.tik_instance.Scalar("int64", name="pad_top")
        self.pad_bottom = self.tik_instance.Scalar("int32", name="pad_bottom")
        self.each_process_wo = self.tik_instance.Scalar("int32", name="each_process_wo")
        self.each_process_ho = self.tik_instance.Scalar("int32", name="each_process_ho")
        self.each_process_wi = self.tik_instance.Scalar("int32", name="each_process_wi")
        self.each_process_hi = self.tik_instance.Scalar("int32", name="each_process_hi")
        self.c1 = self.tik_instance.Scalar("int32", name="c1")
        self.ho = self.tik_instance.Scalar("int32", name="ho")
        self.wo = self.tik_instance.Scalar("int32", name="wo")
        self.hi = self.tik_instance.Scalar("int32", name="hi")
        self.wi = self.tik_instance.Scalar("int32", name="wi")
        self.nc1 = self.tik_instance.Scalar("int32", name="n_c1")
        self.block_num = self.tik_instance.Scalar("int32", name="block_num")
        self.block_num_inner = self.tik_instance.Scalar("int32", name="block_num_inner")
        self.block_num_outer = self.tik_instance.Scalar("int32", name="block_num_outer")
        self.ho_inner = self.tik_instance.Scalar("int32", name="ho_inner")
        self.ho_outer = self.tik_instance.Scalar("int32", name="ho_outer")
        self.block = self.tik_instance.Scalar("int32", name="block")
        self.act_core_num = self.tik_instance.Scalar("int32", name="act_core_num")
        self.tile_h_to_block = self.tik_instance.Scalar("int32", name="tile_h_to_block")
        self.if_block = self.tik_instance.Scalar("int32", name="if_block")
        self.shape_ho = self.tik_instance.Scalar("int32", name="shape_ho")
        self.shape_hi = self.tik_instance.Scalar("int32", name="shape_hi")
        self.one_window_size = self.tik_instance.Scalar("int32", name="one_window_size")
        self.workspace_shape = self.tik_instance.Scalar("int32", name="workspace_shape")
        self.core_num_var = self.tik_instance.Scalar("int32", name="core_num_var")

    def get_tiling_args(self):
        """Get runtime params from tiling
        """
        with self.tik_instance.new_stmt_scope():
            tiling_ub = self.tik_instance.Tensor("int32", (Constant.TILING_ARG_NUM,),
                                                 name="tiling_ub",
                                                 scope=tik.scope_ubuf)
            self.tik_instance.data_move(tiling_ub, self.tiling_gm, 0, 1, 8, 0, 0)

            self.tiling_mode.set_as(tiling_ub[0])
            self.real_block.set_as(tiling_ub[1])
            self.block_cycle.set_as(tiling_ub[2])
            self.ho_wo_16.set_as(tiling_ub[3])
            self.mask_shape_128.set_as(tiling_ub[4])
            _pad_left = self.tik_instance.Scalar("int32", name="_pad_left")
            _pad_left.set_as(tiling_ub[5])
            self.pad_left.set_as(_pad_left)
            self.pad_right.set_as(tiling_ub[6])
            _pad_top = self.tik_instance.Scalar("int32", name="_pad_top")
            _pad_top.set_as(tiling_ub[7])
            self.pad_top.set_as(_pad_top)
            self.pad_bottom.set_as(tiling_ub[8])
            self.each_process_wo.set_as(tiling_ub[9])
            self.each_process_ho.set_as(tiling_ub[10])
            self.each_process_wi.set_as(tiling_ub[11])
            self.each_process_hi.set_as(tiling_ub[12])
            self.c1.set_as(tiling_ub[13])
            self.ho.set_as(tiling_ub[14])
            self.wo.set_as(tiling_ub[15])
            self.hi.set_as(tiling_ub[16])
            self.wi.set_as(tiling_ub[17])
            self.nc1.set_as(tiling_ub[18])
            self.block_num.set_as(tiling_ub[19])
            self.block_num_inner.set_as(tiling_ub[20])
            self.block_num_outer.set_as(tiling_ub[21])
            self.ho_inner.set_as(tiling_ub[22])
            self.ho_outer.set_as(tiling_ub[23])
            self.block.set_as(tiling_ub[24])
            self.act_core_num.set_as(tiling_ub[25])
            self.tile_h_to_block.set_as(tiling_ub[26])
            self.if_block.set_as(tiling_ub[27])
            self.shape_ho.set_as(tiling_ub[28])
            self.shape_hi.set_as(tiling_ub[29])
            self.one_window_size.set_as(tiling_ub[30])
            self.workspace_shape.set_as(tiling_ub[31])
            self.core_num_var.set_as(tiling_ub[32])

    def init_gm_tensor(self):
        """Init gm tensor
        """
        self.tiling_gm = self.tik_instance.Tensor("int32", (Constant.TILING_ARG_NUM,),
                                                  name="tiling_gm",
                                                  scope=tik.scope_gm)
        self.ori_input_gm = self.tik_instance.Tensor(self.dtype, (Constant.MAX_INT32,),
                                                     name="ori_input_gm",
                                                     scope=tik.scope_gm)
        self.argmax_gm = self.tik_instance.Tensor("uint16", (Constant.MAX_INT32,),
                                                  name="argmax_gm",
                                                  scope=tik.scope_gm)
        self.grad_gm = self.tik_instance.Tensor(self.dtype, (Constant.MAX_INT32,),
                                                name="grad_gm",
                                                scope=tik.scope_gm)
        self.res_gm = self.tik_instance.Tensor(self.dtype, (Constant.MAX_INT32,),
                                               name="res_gm",
                                               scope=tik.scope_gm)

    def init_workspace(self):
        """Init temporary storage of overlap in workspace
        """
        self.overlap_gm = self.tik_instance.Tensor("float32", (Constant.CORE_NUM * Constant.WORKSPACE_ONE_CORE // 4,), 
                                                   name="overlap_gm", scope=tik.scope_gm, is_workspace=True)

    def init_ub_tensor(self):
        """Init ub tensor
        """
        # grad_ub and temp_zero
        self.ub_a = self.tik_instance.Tensor(self.dtype, (self.one_seventh_ub_ele,), name="ub_a", scope=tik.scope_ubuf)
        # grad_sel_ub_fp16 and col2img_ub_fp16
        self.ub_b = self.tik_instance.Tensor(self.dtype, (self.one_seventh_ub_ele,), name="ub_b", scope=tik.scope_ubuf)
        # mask_ub
        self.ub_c = self.tik_instance.Tensor("uint16", (self.one_seventh_ub_ele,), name="ub_c", scope=tik.scope_ubuf)
        # col2img_ub_fp32
        self.ub_d = self.tik_instance.Tensor("float32", (self.one_seventh_ub_ele,), name="ub_d", scope=tik.scope_ubuf)
        # grad_sel_ub_fp32 and temp_tensor_ub
        self.ub_e = self.tik_instance.Tensor("float32", (self.one_seventh_ub_ele,), name="ub_e", scope=tik.scope_ubuf)

    def init_ub_scalar(self):
        """Init ub scalar
        """
        self.total_repeate_time = self.tik_instance.Scalar(dtype="int64", name="total_repeate_time")
        self.remain_ele = self.tik_instance.Scalar(dtype="int64", name="remain_ele")
        self.repeate_max_time = self.tik_instance.Scalar(dtype="int64", name="repeate_max_time")
        self.remain_repeate_time = self.tik_instance.Scalar(dtype="int64", name="remain_repeate_time")
        self.ele_num = self.tik_instance.Scalar(dtype="int64", name="ele_num")

    def _vector_dup(self, src, src_start, shape, dup_reg, dtype):
        if len(shape) == 3:
            self.ele_num.set_as(shape[0] * shape[1] * shape[2])
        elif len(shape) == 2:
            self.ele_num.set_as(shape[0] * shape[1])
        else:
            self.ele_num.set_as(shape[0])

        if dtype == "float16":
            self.total_repeate_time.set_as(self.ele_num // Constant.VECTOR_FP16_SIZE)
            self.remain_ele.set_as(self.ele_num % Constant.VECTOR_FP16_SIZE)
            mask_value = Constant.VECTOR_FP16_SIZE
        elif dtype == "float32":
            self.total_repeate_time.set_as(self.ele_num // Constant.VECTOR_FP32_SIZE)
            self.remain_ele.set_as(self.ele_num % Constant.VECTOR_FP32_SIZE)
            mask_value = Constant.VECTOR_FP32_SIZE
        else:
            error_manager_vector.raise_err_input_dtype_not_supported("max_pool_grad_with_argmax_v2", "dtype",
                                                                     "float16, float32", str(dtype))
        self.repeate_max_time.set_as(self.total_repeate_time // Constant.MAX_VECTOR_REPEATE_TIME)
        self.remain_repeate_time.set_as(self.total_repeate_time % Constant.MAX_VECTOR_REPEATE_TIME)

        src_addr = self.tik_instance.Scalar(dtype="int64", name="src_addr")
        with self.tik_instance.if_scope(self.repeate_max_time > 0):
            with self.tik_instance.for_range(0, self.repeate_max_time) as loop1:
                src_addr.set_as(src_start + loop1 * Constant.MAX_VECTOR_REPEATE_TIME * mask_value)
                self.tik_instance.vector_dup(mask_value, src[src_addr],
                                             dup_reg, Constant.MAX_VECTOR_REPEATE_TIME,
                                             1, 8)
        with self.tik_instance.if_scope(self.remain_repeate_time > 0):
            src_addr.set_as(src_start + self.repeate_max_time * Constant.MAX_VECTOR_REPEATE_TIME * mask_value)
            self.tik_instance.vector_dup(mask_value, src[src_addr],
                                         dup_reg, self.remain_repeate_time, 1, 8)
        with self.tik_instance.if_scope(self.remain_ele > 0):
            src_addr.set_as(src_start + self.repeate_max_time * Constant.MAX_VECTOR_REPEATE_TIME * mask_value +
                            self.remain_repeate_time * mask_value)
            self.tik_instance.vector_dup(self.remain_ele, src[src_addr], dup_reg, 1, 1, 8)

    def _vconv(self, src, src_start, dst, dst_start, ele_num, src_dtype):
        self.total_repeate_time.set_as(ele_num // Constant.VECTOR_FP32_SIZE)
        self.remain_ele.set_as(ele_num % Constant.VECTOR_FP32_SIZE)
        mask_value = Constant.VECTOR_FP32_SIZE

        self.repeate_max_time.set_as(self.total_repeate_time // Constant.MAX_VECTOR_REPEATE_TIME)
        self.remain_repeate_time.set_as(self.total_repeate_time % Constant.MAX_VECTOR_REPEATE_TIME)

        if src_dtype == 'float16':
            src_stride, dst_stride = 4, 8
            with self.tik_instance.if_scope(self.repeate_max_time > 0):
                with self.tik_instance.for_range(0, self.repeate_max_time) as loop1:
                    self.tik_instance.vconv(
                        Constant.MASK64_VALUE, "",
                        dst[
                            dst_start + loop1 * Constant.MAX_VECTOR_REPEATE_TIME * mask_value],
                        src[
                            src_start + loop1 * Constant.MAX_VECTOR_REPEATE_TIME * mask_value],
                        Constant.MAX_VECTOR_REPEATE_TIME, 1, 1, dst_stride, src_stride)
            with self.tik_instance.if_scope(self.remain_repeate_time > 0):
                self.tik_instance.vconv(
                    Constant.MASK64_VALUE, "",
                    dst[
                        dst_start + self.repeate_max_time * Constant.MAX_VECTOR_REPEATE_TIME * mask_value],
                    src[
                        src_start + self.repeate_max_time * Constant.MAX_VECTOR_REPEATE_TIME * mask_value],
                    self.remain_repeate_time, 1, 1, dst_stride, src_stride)
            with self.tik_instance.if_scope(self.remain_ele > 0):
                self.tik_instance.vconv(
                    self.remain_ele, "",
                    dst[dst_start + self.repeate_max_time * Constant.MAX_VECTOR_REPEATE_TIME *
                        mask_value + self.remain_repeate_time * mask_value],
                    src[src_start + self.repeate_max_time * Constant.MAX_VECTOR_REPEATE_TIME *
                        mask_value + self.remain_repeate_time * mask_value],
                    1, 1, 1, dst_stride, src_stride)
        else:
            src_stride, dst_stride = 8, 4
            with self.tik_instance.if_scope(self.repeate_max_time > 0):
                with self.tik_instance.for_range(0, self.repeate_max_time) as loop1:
                    self.tik_instance.vconv(
                        Constant.MASK64_VALUE, "",
                        dst[
                            dst_start + loop1 * Constant.MAX_VECTOR_REPEATE_TIME * mask_value],
                        src[
                            src_start + loop1 * Constant.MAX_VECTOR_REPEATE_TIME * mask_value],
                        Constant.MAX_VECTOR_REPEATE_TIME, 1, 1, dst_stride, src_stride)
            with self.tik_instance.if_scope(self.remain_repeate_time > 0):
                self.tik_instance.vconv(
                    Constant.MASK64_VALUE, "",
                    dst[
                        dst_start + self.repeate_max_time * Constant.MAX_VECTOR_REPEATE_TIME * mask_value],
                    src[
                        src_start + self.repeate_max_time * Constant.MAX_VECTOR_REPEATE_TIME * mask_value],
                    self.remain_repeate_time, 1, 1, dst_stride, src_stride)
            with self.tik_instance.if_scope(self.remain_ele > 0):
                self.tik_instance.vconv(
                    self.remain_ele, "",
                    dst[dst_start + self.repeate_max_time * Constant.MAX_VECTOR_REPEATE_TIME *
                        mask_value + self.remain_repeate_time * mask_value],
                    src[src_start + self.repeate_max_time * Constant.MAX_VECTOR_REPEATE_TIME *
                        mask_value + self.remain_repeate_time * mask_value],
                    1, 1, 1, dst_stride, src_stride)

    # 'pylint: disable=unused-variable,too-many-locals
    def _vector_op(self, operator, src1, src2, dst, dtype, ele_num, stride_cofig=None, offset=None):
        repeate_times = self.tik_instance.Scalar(dtype="int64", name="repeate_times")
        remain_ele = self.tik_instance.Scalar(dtype="int64", name="remain_ele")
        repeat_max_loop = self.tik_instance.Scalar(dtype="int64", name="repeat_max_loop")
        remain_max_loop = self.tik_instance.Scalar(dtype="int64", name="remain_max_loop")

        if dtype == "float16":
            repeate_times.set_as(ele_num // Constant.VECTOR_FP16_SIZE)
            remain_ele.set_as(ele_num % Constant.VECTOR_FP16_SIZE)
            mask = Constant.VECTOR_FP16_SIZE
        else:
            repeate_times.set_as(ele_num // Constant.VECTOR_FP32_SIZE)
            remain_ele.set_as(ele_num % Constant.VECTOR_FP32_SIZE)
            mask = Constant.VECTOR_FP32_SIZE
        repeat_max_loop.set_as(repeate_times // Constant.MAX_VECTOR_REPEATE_TIME)
        remain_max_loop.set_as(repeate_times % Constant.MAX_VECTOR_REPEATE_TIME)
        if operator == "vmuls":
            if offset:
                dst_start = offset[0]
                src1_start = offset[1]
            else:
                dst_start = 0
                src1_start = 0
            if stride_cofig is None:
                stride_cofig = 1, 1, 8, 8

            dst_offset = self.tik_instance.Scalar(dtype="int64", name="dst_offset")
            src1_offset = self.tik_instance.Scalar(dtype="int64", name="src1_offset")
            block_length = Constant.BLOCK_SIZE // (get_bit_len(dst.dtype.lower()) // 8)

            with self.tik_instance.for_range(0, repeat_max_loop) as repeat_idx:
                dst_offset.set_as(dst_start + block_length * stride_cofig[2] * 255 * repeat_idx)
                src1_offset.set_as(src1_start + block_length * stride_cofig[3] * 255 * repeat_idx)
                self.tik_instance.vmuls(mask, dst[dst_offset], src1[src1_offset],
                                        src2, 255,
                                        stride_cofig[0], stride_cofig[1],
                                        stride_cofig[2], stride_cofig[3])
            with self.tik_instance.if_scope(remain_max_loop > 0):
                dst_offset.set_as(dst_start + block_length * stride_cofig[2] * 255 * repeat_max_loop)
                src1_offset.set_as(src1_start + block_length * stride_cofig[3] * 255 * repeat_max_loop)
                self.tik_instance.vmuls(mask, dst[dst_offset], src1[src1_offset],
                                        src2, remain_max_loop,
                                        stride_cofig[0], stride_cofig[1],
                                        stride_cofig[2], stride_cofig[3])
            with self.tik_instance.if_scope(remain_ele > 0):
                dst_offset.set_as(
                    dst_start + block_length * stride_cofig[2] * (255 * repeat_max_loop + remain_max_loop))
                src1_offset.set_as(
                    src1_start + block_length * stride_cofig[3] * (255 * repeat_max_loop + remain_max_loop))
                self.tik_instance.vmuls(remain_ele, dst[dst_offset], src1[src1_offset],
                                        src2, 1,
                                        stride_cofig[0], stride_cofig[1],
                                        stride_cofig[2], stride_cofig[3])
        if operator == "vadd":
            if stride_cofig is None:
                stride_cofig = 1, 1, 1, 8, 8, 8
            block_length = Constant.BLOCK_SIZE // (get_bit_len(dst.dtype.lower()) // 8)
            dst_offset = self.tik_instance.Scalar(dtype="int64", name="dst_offset")
            src1_offset = self.tik_instance.Scalar(dtype="int64", name="src1_offset")
            src2_offset = self.tik_instance.Scalar(dtype="int64", name="src2_offset")
            # stride w is smaller than 15
            if stride_cofig[3] <= 255:
                with self.tik_instance.for_range(0, repeat_max_loop) as repeat_idx:
                    dst_offset.set_as(block_length * stride_cofig[3] * 255 * repeat_idx)
                    src1_offset.set_as(block_length * stride_cofig[4] * 255 * repeat_idx)
                    src2_offset.set_as(block_length * stride_cofig[5] * 255 * repeat_idx)
                    self.tik_instance.vadd(mask, dst[dst_offset:], src1[src1_offset:], src2[src2_offset:],
                                           255,
                                           stride_cofig[0], stride_cofig[1],
                                           stride_cofig[2], stride_cofig[3],
                                           stride_cofig[4], stride_cofig[5])
                with self.tik_instance.if_scope(remain_max_loop > 0):
                    dst_offset.set_as(block_length * stride_cofig[3] * 255 * repeat_max_loop)
                    src1_offset.set_as(block_length * stride_cofig[4] * 255 * repeat_max_loop)
                    src2_offset.set_as(block_length * stride_cofig[5] * 255 * repeat_max_loop)
                    self.tik_instance.vadd(mask, dst[dst_offset:], src1[src1_offset:], src2[src2_offset:],
                                           remain_max_loop,
                                           stride_cofig[0], stride_cofig[1],
                                           stride_cofig[2], stride_cofig[3],
                                           stride_cofig[4], stride_cofig[5])
                with self.tik_instance.if_scope(remain_ele > 0):
                    dst_offset.set_as(block_length * stride_cofig[3] * (255 * repeat_max_loop + remain_max_loop))
                    src1_offset.set_as(block_length * stride_cofig[4] * (255 * repeat_max_loop + remain_max_loop))
                    src2_offset.set_as(block_length * stride_cofig[5] * (255 * repeat_max_loop + remain_max_loop))
                    self.tik_instance.vadd(remain_ele, dst[dst_offset:], src1[src1_offset:],
                                           src2[src2_offset:],
                                           1,
                                           stride_cofig[0], stride_cofig[1],
                                           stride_cofig[2],
                                           0, 0, 0)
            # stride w is greater than 15
            else:
                with self.tik_instance.for_range(0, repeate_times) as _repeat_idx:
                    dst_offset.set_as(block_length * stride_cofig[3] * _repeat_idx)
                    src1_offset.set_as(block_length * stride_cofig[4] * _repeat_idx)
                    src2_offset.set_as(block_length * stride_cofig[5] * _repeat_idx)
                    self.tik_instance.vadd(mask, dst[dst_offset:], src1[src1_offset:], src2[src2_offset:],
                                           1,
                                           stride_cofig[0], stride_cofig[1],
                                           stride_cofig[2],
                                           0, 0, 0)
                with self.tik_instance.if_scope(remain_ele > 0):
                    dst_offset.set_as(block_length * stride_cofig[3] * (255 * repeat_max_loop + remain_max_loop))
                    src1_offset.set_as(block_length * stride_cofig[4] * (255 * repeat_max_loop + remain_max_loop))
                    src2_offset.set_as(block_length * stride_cofig[5] * (255 * repeat_max_loop + remain_max_loop))
                    self.tik_instance.vadd(remain_ele, dst[dst_offset:], src1[src1_offset:],
                                           src2[src2_offset:],
                                           1,
                                           stride_cofig[0], stride_cofig[1],
                                           stride_cofig[2],
                                           0, 0, 0)

    def _clean_mask(self, src, src_start, shape):
        dup_reg = self.tik_instance.Scalar(dtype='uint16',
                                           name='dup_reg')
        dup_reg.set_as(0)

        if len(shape) == 3:
            self.ele_num.set_as(shape[0] * shape[1] * shape[2])
        elif len(shape) == 2:
            self.ele_num.set_as(shape[0] * shape[1])
        else:
            self.ele_num.set_as(shape[0])

        mask_value = 128
        self.total_repeate_time.set_as(self.ele_num // 128)
        self.remain_ele.set_as(self.ele_num % 128)
        self.repeate_max_time.set_as(self.total_repeate_time // Constant.MAX_VECTOR_REPEATE_TIME)
        self.remain_repeate_time.set_as(self.total_repeate_time % Constant.MAX_VECTOR_REPEATE_TIME)

        src_addr = self.tik_instance.Scalar(dtype="int64", name="src_addr")
        with self.tik_instance.if_scope(self.repeate_max_time > 0):
            with self.tik_instance.for_range(0, self.repeate_max_time) as loop1:
                src_addr.set_as(src_start + loop1 * Constant.MAX_VECTOR_REPEATE_TIME * mask_value)
                self.tik_instance.vector_dup(mask_value, src[src_addr],
                                             dup_reg, Constant.MAX_VECTOR_REPEATE_TIME,
                                             1, 8)
        with self.tik_instance.if_scope(self.remain_repeate_time > 0):
            src_addr.set_as(src_start + self.repeate_max_time * Constant.MAX_VECTOR_REPEATE_TIME * mask_value)
            self.tik_instance.vector_dup(mask_value, src[src_addr],
                                         dup_reg, self.remain_repeate_time,
                                         1, 8)
        with self.tik_instance.if_scope(self.remain_ele > 0):
            src_addr.set_as(src_start + self.repeate_max_time * Constant.MAX_VECTOR_REPEATE_TIME * mask_value +
                            self.remain_repeate_time * mask_value)
            self.tik_instance.vector_dup(self.remain_ele, src[src_addr], dup_reg, 1, 1, 8)

    def _vsel_grad_col(self, mask_ub, grad_ub, grad_shape_0):
        grad_sel_ub = self.ub_b
        temp_zero = self.tik_instance.Tensor("float16", (Constant.MASK128_VALUE,),
                                             name="temp_zero", scope=tik.scope_ubuf)
        self._vector_dup(temp_zero, 0, (Constant.MASK128_VALUE,), self.scalar_zero_fp16, "float16")

        # vsel
        with self.tik_instance.for_range(0, grad_shape_0) as mask_index:
            fractal_repeat = Constant.C0 * Constant.C0 // Constant.VECTOR_FP16_SIZE
            with self.tik_instance.for_range(0, fractal_repeat) as fractal_index:
                mask_type_bit_size = get_bit_len("uint16")
                mask_offset = (mask_index * fractal_repeat + fractal_index) * \
                              Constant.MASK128_VALUE // mask_type_bit_size

                cmpmask = self.tik_instance.mov_tensor_to_cmpmask(mask_ub[mask_offset])
                grad_ub_offset = (mask_index * fractal_repeat + fractal_index) * Constant.MASK128_VALUE
                self.tik_instance.vsel(Constant.MASK128_VALUE, 0, grad_sel_ub[grad_ub_offset], cmpmask,
                                       grad_ub[grad_ub_offset], temp_zero,
                                       1, 1, 1, 1, 8, 8, 8)

        return grad_sel_ub

    # 'pylint: disable=unused-variable,too-many-locals
    def _mov_func(self, cut_ho_nums_index, cut_ho_nums, remain_ho_nums, each_process_ho,
                  each_process_hi, each_valid_ho, col2img_fp32_ub, temp_tensor_ub, pad,
                  col2img_ub_shape, temp_size):
        pad_left, pad_right, pad_top, pad_bottom = pad
        wi = self.wi + pad_left + pad_right
        pad_top_rows = self.tik_instance.Scalar(dtype="int64", name='pad_top_rows')
        pad_top_rows.set_as(pad_top - cut_ho_nums_index * each_process_ho * self.stride_h)
        self.tik_instance.scalar_max(pad_top_rows, pad_top_rows, 0)
        each_valid_hi = each_valid_ho * self.stride_h - pad_top_rows

        col2img_fp16_ub = self.ub_b
        self.ele_num.set_as(col2img_ub_shape[0] * col2img_ub_shape[1] * col2img_ub_shape[2])
        self._vconv(col2img_fp32_ub, 0, col2img_fp16_ub, 0,
                    self.ele_num, "float32")

        with self.tik_instance.if_scope(
                tik.all(cut_ho_nums_index < cut_ho_nums - 1, each_valid_hi > 0)):
            self.tik_instance.data_move(self.res_gm[self.offset_gm],
                                        col2img_fp16_ub[pad_top_rows * wi * Constant.C0 + pad_left * Constant.C0], 0,
                                        each_valid_hi, self.wi * Constant.C0 // 16,
                                        pad_left + pad_right, 0)

            self.offset_gm.set_as(self.offset_gm + each_valid_hi * self.wi * Constant.C0)

        last_valid_hi = self.tik_instance.Scalar(dtype="int64", name="last_valid_hi")
        remain_hi = self.tik_instance.Scalar(dtype="int64", name="remain_hi")
        with self.tik_instance.if_scope(remain_ho_nums == 0):
            with self.tik_instance.if_scope(cut_ho_nums_index == cut_ho_nums - 1):
                with self.tik_instance.if_scope(cut_ho_nums - 1 == 0):
                    last_valid_hi.set_as(self.hi)
                with self.tik_instance.else_scope():
                    last_valid_hi.set_as(self.hi - ((cut_ho_nums - 1) * each_process_ho * self.stride_h - pad_top))
                with self.tik_instance.if_scope(last_valid_hi <= each_process_hi):
                    with self.tik_instance.if_scope(last_valid_hi > 0):
                        self.tik_instance.data_move(self.res_gm[self.offset_gm],
                                                    col2img_fp16_ub[pad_top_rows * wi * Constant.C0 + pad_left *
                                                    Constant.C0], 0, last_valid_hi, self.wi * Constant.C0 // 16,
                                                    pad_left + pad_right, 0)
                with self.tik_instance.else_scope():
                    self.tik_instance.data_move(self.res_gm[self.offset_gm],
                                                col2img_fp16_ub[pad_top_rows * wi * Constant.C0 + pad_left *
                                                Constant.C0], 0, each_process_hi, self.wi * Constant.C0 // 16,
                                                pad_left + pad_right, 0)
                    remain_hi.set_as(last_valid_hi - each_process_hi)
                    temp_zero = self.ub_a
                    temp_zero_shape = (remain_hi, self.wi * Constant.C0)
                    self._vector_dup(temp_zero, 0, temp_zero_shape,
                                     self.scalar_zero_fp16, "float16")
                    self.tik_instance.data_move(
                        self.res_gm[self.offset_gm + each_process_hi * self.wi * Constant.C0],
                        temp_zero,
                        0,
                        1, remain_hi * self.wi * Constant.C0 // 16,
                        0, 0)
                self.offset_gm.set_as(self.offset_gm + last_valid_hi * self.wi * Constant.C0)
        with self.tik_instance.else_scope():
            with self.tik_instance.if_scope(cut_ho_nums_index == cut_ho_nums - 1):
                self.tik_instance.data_move(self.res_gm[self.offset_gm],
                                            col2img_fp16_ub[pad_top_rows * wi * Constant.C0 + pad_left * Constant.C0],
                                            0,
                                            each_valid_hi, self.wi * Constant.C0 // 16,
                                            pad_left + pad_right, 0)
                self.offset_gm.set_as(self.offset_gm + each_valid_hi * self.wi * Constant.C0)

            with self.tik_instance.if_scope(cut_ho_nums_index == cut_ho_nums):
                with self.tik_instance.if_scope(cut_ho_nums == 0):
                    last_valid_hi.set_as(self.hi)
                with self.tik_instance.else_scope():
                    last_valid_hi.set_as(self.hi - (cut_ho_nums * each_process_ho * self.stride_h - pad_top))
                with self.tik_instance.if_scope(last_valid_hi <= each_process_hi):
                    self.tik_instance.data_move(self.res_gm[self.offset_gm],
                                                col2img_fp16_ub[
                                                    pad_top_rows * wi * Constant.C0 + pad_left * Constant.C0],
                                                0,
                                                last_valid_hi, self.wi * Constant.C0 // 16,
                                                pad_left + pad_right, 0)
                with self.tik_instance.else_scope():
                    self.tik_instance.data_move(self.res_gm[self.offset_gm],
                                                col2img_fp16_ub[
                                                    pad_top_rows * wi * Constant.C0 + pad_left * Constant.C0],
                                                0,
                                                each_process_hi, self.wi * Constant.C0 // 16,
                                                pad_left + pad_right, 0)
                    remain_hi.set_as(last_valid_hi - each_process_hi)
                    temp_zero = self.ub_a
                    temp_zero_shape = (remain_hi, self.wi * Constant.C0)
                    self._vector_dup(temp_zero, 0, temp_zero_shape,
                                     self.scalar_zero_fp16, "float16")
                    self.tik_instance.data_move(
                        self.res_gm[self.offset_gm + each_process_hi * self.wi * Constant.C0],
                        temp_zero,
                        0,
                        1, remain_hi * self.wi * Constant.C0 // 16,
                        0, 0)
                self.offset_gm.set_as(self.offset_gm + last_valid_hi * self.wi * Constant.C0)

        with self.tik_instance.if_scope(self.kh > self.stride_h):
            self.ele_num.set_as(temp_size[0] * temp_size[1] * temp_size[2])
            self._vector_op("vmuls", col2img_fp32_ub, 1.0, temp_tensor_ub, "float32",
                            self.ele_num,
                            None, [0, each_process_ho * self.stride_h * wi * Constant.C0])

    # 'pylint: disable=unused-variable,too-many-locals
    def _move_func_block(self, cut_ho_nums_index, cut_ho_nums, start_h, end_h, each_process_ho,
                         valid_hi_block, col2img_fp32_ub, temp_tensor_ub,
                         remained_hi, remain, pad, col2img_ub_shape, temp_size):
        pad_left, pad_right, pad_top, pad_bottom = pad
        wi = self.wi + pad_left + pad_right
        mov_len_h = self.tik_instance.Scalar(dtype='int64', name='mov_len_h')
        hi_max = self.tik_instance.Scalar(dtype='int64', name='hi_max')
        hi_min = self.tik_instance.Scalar(dtype='int64', name='hi_min')
        self.tik_instance.scalar_max(hi_min, pad_top, start_h)
        hi_max.set_as(valid_hi_block + pad_top)
        self.tik_instance.scalar_min(hi_max, hi_max, end_h)
        mov_len_h.set_as(hi_max - hi_min)

        col2img_fp16_ub = self.ub_b
        self.ele_num.set_as(col2img_ub_shape[0] * col2img_ub_shape[1] * col2img_ub_shape[2])
        self._vconv(col2img_fp32_ub, 0, col2img_fp16_ub, 0,
                    self.ele_num, "float32")
        with self.tik_instance.if_scope(end_h > pad_top):
            with self.tik_instance.if_scope(start_h < pad_top + valid_hi_block):
                with self.tik_instance.if_scope(mov_len_h > 0):
                    self.tik_instance.data_move(self.res_gm[self.offset_gm],
                                                col2img_fp16_ub[(hi_min - start_h) *
                                                                wi * Constant.C0 +
                                                                pad_left * Constant.C0],
                                                0, mov_len_h, self.wi * Constant.C0 // 16,
                                                pad_left + pad_right, 0)
                    self.offset_gm.set_as(self.offset_gm + mov_len_h * self.wi * Constant.C0)
                    remained_hi.set_as(remained_hi - mov_len_h)

                with self.tik_instance.if_scope(cut_ho_nums_index == cut_ho_nums - 1):
                    with self.tik_instance.if_scope(remain == 0):
                        with self.tik_instance.if_scope(remained_hi > 0):
                            temp_zero = self.ub_a
                            temp_zero_shape = (1, self.wi, Constant.C0)
                            temp_zero_dtype = "float16"
                            self._vector_dup(temp_zero,
                                             0,
                                             temp_zero_shape,
                                             self.scalar_zero_fp16,
                                             temp_zero_dtype)

                            with self.tik_instance.for_range(0, remained_hi) as index_0:
                                self.tik_instance.data_move(self.res_gm[self.offset_gm],
                                                            temp_zero, 0,
                                                            1, self.wi * Constant.C0 // 16, 0, 0)
                                self.offset_gm.set_as(self.offset_gm + self.wi * Constant.C0)
                with self.tik_instance.if_scope(cut_ho_nums_index == cut_ho_nums):
                    with self.tik_instance.if_scope(remained_hi > 0):
                        temp_zero = self.ub_a
                        temp_zero_shape = (1, self.wi, Constant.C0)
                        temp_zero_dtype = "float16"
                        self._vector_dup(temp_zero,
                                         0,
                                         temp_zero_shape,
                                         self.scalar_zero_fp16,
                                         temp_zero_dtype)
                        with self.tik_instance.for_range(0, remained_hi) as index_0:
                            self.tik_instance.data_move(self.res_gm[self.offset_gm],
                                                        temp_zero, 0,
                                                        1, self.wi * Constant.C0 // 16, 0, 0)
                            self.offset_gm.set_as(self.offset_gm + self.wi * Constant.C0)

        with self.tik_instance.if_scope(self.kh > self.stride_h):
            self.ele_num.set_as(temp_size[0] * temp_size[1] * temp_size[2])
            self._vector_op("vmuls", col2img_fp32_ub, 1.0, temp_tensor_ub, "float32",
                            self.ele_num,
                            None, [0, each_process_ho * self.stride_h * wi * Constant.C0])
        return remained_hi

    # 'pylint: disable=unused-variable,too-many-locals
    def _not_tilling(self, n_index, c1_index,
                     each_process_ho_block, each_process_hi_block,
                     mov_len_ho, mov_len_hi,
                     start_ho_index, start_hi_index,
                     start_threshold,
                     offset_gm_block, shape, pad):

        pad_left, pad_right, pad_top, pad_bottom = pad
        shape_ho, shape_wo, shape_hi, shape_wi = shape

        howo_ceil16 = self.ho_wo_16

        wi = self.tik_instance.Scalar(dtype="int64", name="wi")
        wi.set_as(self.wi + self.pad_left + self.pad_right)
        hi = self.tik_instance.Scalar(dtype="int64", name="hi")
        hi.set_as(shape_hi + self.pad_top + self.pad_bottom)
        col_index = self.tik_instance.Scalar(dtype="int64", name="col_index")
        mask_index = self.tik_instance.Scalar(dtype="int64", name="mask_index")

        # define col res
        col2img_ub_shape = (hi, wi, Constant.C0)
        col2img_fp32_ub = self.ub_d

        ori_output_shape = (howo_ceil16, 16, Constant.C0)
        output_data_nums = self.tik_instance.Scalar(dtype="int64", name="output_data_nums")
        output_data_nums.set_as(mov_len_ho * self.wo * Constant.C0)
        output_data_burst = self.tik_instance.Scalar(dtype="int64", name="output_data_burst")
        output_data_burst.set_as(output_data_nums // 16)

        src_output_offset = self.tik_instance.Scalar(dtype="int64", name="src_output_offset")
        src_output_offset.set_as(((n_index * self.c1 + c1_index) * self.ho + start_ho_index) * \
                                 self.wo * Constant.C0)

        # mov ori grad to ub, shape is ori_output_shape
        grad_ub = self.ub_a
        self._vector_dup(grad_ub, 0, ori_output_shape,
                         self.scalar_zero_fp16, "float16")
        self.tik_instance.data_move(grad_ub[0],
                                    self.grad_gm[src_output_offset],
                                    0, 1, output_data_burst, 0, 0)

        # init col2img_fp32_ub, if not the first one and have overlap, dump
        # the overlap part to col2img_fp32_ub, here we process whole ho, so
        # no need to move overlap part
        self._vector_dup(col2img_fp32_ub, 0, col2img_ub_shape,
                         self.scalar_zero, "float32")

        with self.tik_instance.for_range(0, self.kh, thread_num=1) as index_h:
            with self.tik_instance.for_range(0, self.kw, thread_num=1) as index_w:
                # calculate mask here
                mask_shape = (self.mask_shape_128,)
                mask_ub = self.ub_c
                self._clean_mask(mask_ub, 0, mask_shape)

                mask_offset_h = self.tik_instance.Scalar(dtype='int64', name='mask_offset_h')
                mask_offset_h.set_as(start_ho_index * shape_wo)
                argmax_offset = self.tik_instance.Scalar(dtype='int64', name='argmax_offset')
                argmax_offset.set_as(((n_index * self.c1 + c1_index) * self.kh * self.kw +
                                      index_h * self.kw + index_w) *
                                     self.one_window_size +
                                     mask_offset_h)

                mask_data_num = self.tik_instance.Scalar(dtype='int64', name='mask_data_num')
                with self.tik_instance.if_scope((mov_len_ho * shape_wo) % 16 != 0):
                    mask_data_num.set_as((mov_len_ho * shape_wo // 16 + 1) * 16)
                with self.tik_instance.else_scope():
                    mask_data_num.set_as((mov_len_ho * shape_wo // 16) * 16)

                mask_data_burst = self.tik_instance.Scalar(dtype='int64', name='mask_data_burst')
                mask_data_burst.set_as(mask_data_num // 16)

                self.tik_instance.data_move(mask_ub[0],
                                            self.argmax_gm[argmax_offset], 0,
                                            1,
                                            mask_data_burst,
                                            0, 0)

                grad_sel_ub = self._vsel_grad_col(mask_ub, grad_ub, ori_output_shape[0])
                grad_sel_ub_fp32 = self.ub_e

                self.ele_num.set_as(ori_output_shape[0] * ori_output_shape[1] * ori_output_shape[2])
                self._vconv(grad_sel_ub, 0, grad_sel_ub_fp32, 0,
                            self.ele_num, "float16")
                with self.tik_instance.for_range(0, mov_len_ho) as ho_idx:
                    # process ELEMENT_NUMS_EACH_TIME to improve dynamic shape performence
                    loop_ = (self.wo * Constant.C0 // 2) // Constant.ELEMENT_NUMS_EACH_TIME
                    remain_ = (self.wo * Constant.C0 // 2) % Constant.ELEMENT_NUMS_EACH_TIME

                    with self.tik_instance.for_range(0, loop_) as loopi_:
                        col_index.set_as(index_h * wi * Constant.C0 + 
                                        index_w * Constant.C0 + 
                                        wi * Constant.C0 * self.stride_h * ho_idx + 
                                        Constant.ELEMENT_NUMS_EACH_TIME * 2 * self.stride_w * loopi_)
                        mask_index.set_as(self.wo * Constant.C0 * ho_idx + 
                                        Constant.ELEMENT_NUMS_EACH_TIME * 2 * loopi_)

                        self._vector_op("vadd", col2img_fp32_ub[col_index:], grad_sel_ub_fp32[mask_index:],
                                        col2img_fp32_ub[col_index:], "float32", Constant.ELEMENT_NUMS_EACH_TIME,
                                        stride_cofig=(self.stride_w * 2, self.stride_w * 2, 2,
                                                    self.stride_w * 16, self.stride_w * 16, 16))
                        self._vector_op("vadd", col2img_fp32_ub[col_index + 8:],
                                        grad_sel_ub_fp32[mask_index + 8:],
                                        col2img_fp32_ub[col_index + 8:], "float32", Constant.ELEMENT_NUMS_EACH_TIME,
                                        stride_cofig=(self.stride_w * 2, self.stride_w * 2, 2,
                                                    self.stride_w * 16, self.stride_w * 16, 16))

                    # deal with the remain part
                    with self.tik_instance.if_scope(remain_ > 0):
                        col_index.set_as(index_h * wi * Constant.C0 + 
                                        index_w * Constant.C0 + 
                                        wi * Constant.C0 * self.stride_h * ho_idx + 
                                        Constant.ELEMENT_NUMS_EACH_TIME * 2 * self.stride_w * loop_)
                        mask_index.set_as(self.wo * Constant.C0 * ho_idx + 
                                        Constant.ELEMENT_NUMS_EACH_TIME * 2 * loop_)

                        self.ele_num.set_as(remain_)

                        self._vector_op("vadd", col2img_fp32_ub[col_index:], grad_sel_ub_fp32[mask_index:],
                                        col2img_fp32_ub[col_index:], "float32", self.ele_num,
                                        stride_cofig=(self.stride_w * 2, self.stride_w * 2, 2,
                                                    self.stride_w * 16, self.stride_w * 16, 16))
                        self._vector_op("vadd", col2img_fp32_ub[col_index + 8:],
                                        grad_sel_ub_fp32[mask_index + 8:],
                                        col2img_fp32_ub[col_index + 8:], "float32", self.ele_num,
                                        stride_cofig=(self.stride_w * 2, self.stride_w * 2, 2,
                                                    self.stride_w * 16, self.stride_w * 16, 16))

        col2img_fp16_ub = self.ub_b
        self.ele_num.set_as(col2img_ub_shape[0] * col2img_ub_shape[1] * col2img_ub_shape[2])
        self._vconv(col2img_fp32_ub, 0, col2img_fp16_ub, 0,
                    self.ele_num, "float32")

        pad_top_offset = pad_top * wi * Constant.C0
        self.tik_instance.data_move(self.res_gm[self.offset_gm],
                                    col2img_fp16_ub[pad_top_offset + pad_left * Constant.C0],
                                    0, self.hi, self.wi * Constant.C0 // 16,
                                    pad_left + pad_right, 0)

    # 'pylint: disable=unused-variable,too-many-locals
    def _not_tilling_nc1h(self, n_index, c1_index,
                          each_process_ho_block, each_process_hi_block,
                          mov_len_ho, mov_len_hi,
                          start_ho_index, start_hi_index,
                          start_threshold,
                          offset_gm_block, shape, pad):

        pad_left, pad_right, pad_top, pad_bottom = pad
        shape_ho, shape_wo, shape_hi, shape_wi = shape

        howo_ceil16 = self.ho_wo_16

        wi = self.tik_instance.Scalar(dtype="int64", name="wi")
        wi.set_as(self.wi + self.pad_left + self.pad_right)
        hi = self.tik_instance.Scalar(dtype="int64", name="hi")
        hi.set_as(shape_hi + self.pad_top + self.pad_bottom)
        col_index = self.tik_instance.Scalar(dtype="int64", name="col_index")
        mask_index = self.tik_instance.Scalar(dtype="int64", name="mask_index")

        # define col res
        col2img_ub_shape = (hi, wi, Constant.C0)
        col2img_fp32_ub = self.ub_d

        ori_output_shape = (howo_ceil16, 16, Constant.C0)
        output_data_nums = self.tik_instance.Scalar(dtype="int64", name="output_data_nums")
        output_data_nums.set_as(mov_len_ho * self.wo * Constant.C0)
        output_data_burst = self.tik_instance.Scalar(dtype="int64", name="output_data_burst")
        output_data_burst.set_as(output_data_nums // 16)

        src_output_offset = self.tik_instance.Scalar(dtype="int64", name="src_output_offset")
        src_output_offset.set_as(((n_index * self.c1 + c1_index) * self.ho + start_ho_index) * \
                                 self.wo * Constant.C0)

        # mov ori grad to ub, shape is ori_output_shape
        grad_ub = self.ub_a
        self._vector_dup(grad_ub, 0, ori_output_shape,
                         self.scalar_zero_fp16, "float16")
        self.tik_instance.data_move(grad_ub[0],
                                    self.grad_gm[src_output_offset],
                                    0, 1, output_data_burst, 0, 0)

        # init col2img_fp32_ub, if not the first one and have overlap, dump
        # the overlap part to col2img_fp32_ub, here we process whole ho, so
        # no need to move overlap part
        self._vector_dup(col2img_fp32_ub, 0, col2img_ub_shape,
                         self.scalar_zero, "float32")

        with self.tik_instance.for_range(0, self.kh, thread_num=1) as index_h:
            with self.tik_instance.for_range(0, self.kw, thread_num=1) as index_w:
                # calculate mask here
                mask_shape = (self.mask_shape_128,)
                mask_ub = self.ub_c
                self._clean_mask(mask_ub, 0, mask_shape)

                mask_offset_h = self.tik_instance.Scalar(dtype='int64', name='mask_offset_h')
                mask_offset_h.set_as(start_ho_index * shape_wo)
                argmax_offset = self.tik_instance.Scalar(dtype='int64', name='argmax_offset')
                argmax_offset.set_as(((n_index * self.c1 + c1_index) * self.kh * self.kw +
                                      index_h * self.kw + index_w) *
                                     self.one_window_size +
                                     mask_offset_h)

                mask_data_num = self.tik_instance.Scalar(dtype='int64', name='mask_data_num')
                with self.tik_instance.if_scope((mov_len_ho * shape_wo) % 16 != 0):
                    mask_data_num.set_as((mov_len_ho * shape_wo // 16 + 1) * 16)
                with self.tik_instance.else_scope():
                    mask_data_num.set_as((mov_len_ho * shape_wo // 16) * 16)

                mask_data_burst = self.tik_instance.Scalar(dtype='int64', name='mask_data_burst')
                mask_data_burst.set_as(mask_data_num // 16)

                self.tik_instance.data_move(mask_ub[0],
                                            self.argmax_gm[argmax_offset], 0,
                                            1,
                                            mask_data_burst,
                                            0, 0)

                grad_sel_ub = self._vsel_grad_col(mask_ub, grad_ub, ori_output_shape[0])
                grad_sel_ub_fp32 = self.ub_e

                self.ele_num.set_as(ori_output_shape[0] * ori_output_shape[1] * ori_output_shape[2])
                self._vconv(grad_sel_ub, 0, grad_sel_ub_fp32, 0,
                            self.ele_num, "float16")
                with self.tik_instance.for_range(0, mov_len_ho) as ho_idx:
                    col_index.set_as(index_h * wi * Constant.C0 + index_w * Constant.C0 + wi * Constant.C0 * \
                    self.stride_h * ho_idx)
                    mask_index.set_as(self.wo * Constant.C0 * ho_idx)
                    self.ele_num.set_as(self.wo * Constant.C0 // 2)
                    self._vector_op("vadd", col2img_fp32_ub[col_index:], grad_sel_ub_fp32[mask_index:],
                                    col2img_fp32_ub[col_index:], "float32", self.ele_num,
                                    stride_cofig=(self.stride_w * 2, self.stride_w * 2, 2,
                                                  self.stride_w * 16, self.stride_w * 16, 16))
                    self._vector_op("vadd", col2img_fp32_ub[col_index + 8:],
                                    grad_sel_ub_fp32[mask_index + 8:],
                                    col2img_fp32_ub[col_index + 8:], "float32", self.ele_num,
                                    stride_cofig=(self.stride_w * 2, self.stride_w * 2, 2,
                                                  self.stride_w * 16, self.stride_w * 16, 16))

        col2img_fp16_ub = self.ub_b
        self.ele_num.set_as(col2img_ub_shape[0] * col2img_ub_shape[1] * col2img_ub_shape[2])
        self._vconv(col2img_fp32_ub, 0, col2img_fp16_ub, 0,
                    self.ele_num, "float32")

        with self.tik_instance.if_scope(each_process_hi_block > 0):
            with self.tik_instance.if_scope(start_threshold > pad_top):
                self.tik_instance.data_move(self.res_gm[offset_gm_block],
                                            col2img_fp16_ub[
                                                start_threshold * wi * Constant.C0 + pad_left * Constant.C0],
                                            0, each_process_hi_block, self.wi * Constant.C0 // 16,
                                            pad_left + pad_right, 0)
            with self.tik_instance.else_scope():
                self.tik_instance.data_move(self.res_gm[offset_gm_block],
                                            col2img_fp16_ub[pad_top * wi * Constant.C0 + pad_left * Constant.C0],
                                            0, each_process_hi_block, self.wi * Constant.C0 // 16,
                                            pad_left + pad_right, 0)

    # 'pylint: disable=unused-variable,too-many-locals,unused-variable
    def _tilling_ho(self, each_process_ho, n_index, c1_index,
                    each_process_ho_block, each_process_hi_block,
                    mov_len_ho, mov_len_hi,
                    start_ho_index, start_hi_index,
                    start_threshold,
                    offset_gm_block, shape, pad):

        # 'pylint: disable=too-many-locals
        pad_left, pad_right, _, _ = pad
        shape_wo = shape[1]

        cut_ho_nums = self.tik_instance.Scalar(dtype='int64', name='cut_ho_nums')
        remain_ho_nums = self.tik_instance.Scalar(dtype='int64', name='remain_ho_nums')
        wi = self.tik_instance.Scalar(dtype='int64', name='wi')
        cut_ho_nums.set_as(mov_len_ho // each_process_ho)
        remain_ho_nums.set_as(mov_len_ho % each_process_ho)
        wi.set_as(self.wi + pad_left + pad_right)

        temp_size_h = self.tik_instance.Scalar(dtype='int64', name='temp_size_h')
        temp_size_w = self.tik_instance.Scalar(dtype='int64', name='temp_size_w')
        with self.tik_instance.if_scope(self.kh > self.stride_h):
            self.each_process_hi.set_as((each_process_ho - 1) * self.stride_h + self.kh)
            temp_size_h.set_as(self.kh - self.stride_h)
            temp_size_w.set_as(wi)
        with self.tik_instance.else_scope():
            self.each_process_hi.set_as(each_process_ho * self.stride_h)
            temp_size_h.set_as(1)
            temp_size_w.set_as(16)
        temp_size = (temp_size_h, temp_size_w, Constant.C0)

        temp_tensor_ub = self.ub_e

        each_process_ho_wo_div16 = self.ho_wo_16
        ori_output_shape = (each_process_ho_wo_div16, 16, Constant.C0)

        col2img_ub_shape = (self.each_process_hi, wi, Constant.C0)
        col2img_fp32_ub = self.ub_d

        # 'pylint: disable=unused-variable,too-many-locals
        def process_ho(output_data_nums, cut_ho_nums_index, each_valid_ho):
            """
            :param output_data_nums:
            :param cut_ho_nums_index:
            :param each_valid_ho:
            :return:
            """
            self._vector_dup(col2img_fp32_ub, 0, col2img_ub_shape,
                             self.scalar_zero, "float32")
            with self.tik_instance.if_scope(self.kh > self.stride_h):
                with self.tik_instance.if_scope(cut_ho_nums_index > 0):
                    self.ele_num.set_as(temp_size[0] * temp_size[1] * temp_size[2])
                    self._vector_op("vmuls", temp_tensor_ub, 1.0, col2img_fp32_ub,
                                    temp_tensor_ub.dtype, self.ele_num)

            start_h = self.tik_instance.Scalar(dtype='int64', name='start_h')
            end_h = self.tik_instance.Scalar(dtype='int64', name='end_h')
            start_h.set_as(cut_ho_nums_index * each_process_ho * self.stride_h)
            end_h.set_as(cut_ho_nums_index * each_process_ho * self.stride_h + self.each_process_hi)

            src_output_offset = self.tik_instance.Scalar(dtype='int64', name='src_output_offset')
            src_output_offset.set_as(((n_index * self.c1 + c1_index) * self.ho + start_ho_index +
                                      each_process_ho * cut_ho_nums_index) * self.wo * Constant.C0)

            # mov ori grad to ub
            grad_ub = self.ub_a
            self._vector_dup(grad_ub, 0, ori_output_shape,
                             self.scalar_zero_fp16, "float16")
            self.tik_instance.data_move(grad_ub[0],
                                        self.grad_gm[src_output_offset],
                                        0, 1, output_data_nums // 16, 0, 0)

            mask_shape = (self.mask_shape_128,)

            with self.tik_instance.for_range(0, self.kh, thread_num=1) as index_h:
                with self.tik_instance.for_range(0, self.kw, thread_num=1) as index_w:
                    mask_ub = self.ub_c
                    self._clean_mask(mask_ub, 0, mask_shape)

                    # use each_process_ho to calc offset
                    mask_offset_h = self.tik_instance.Scalar(dtype='int64', name='mask_offset_h')
                    mask_offset_h.set_as((start_ho_index + each_process_ho * cut_ho_nums_index)
                                         * shape_wo)
                    argmax_offset = self.tik_instance.Scalar(dtype='int64', name='argmax_offset')
                    argmax_offset.set_as(((n_index * self.c1 + c1_index) * self.kh * self.kw +
                                          index_h * self.kw + index_w) *
                                         self.one_window_size +
                                         mask_offset_h)

                    # use each_valid_ho to calc how much data should move
                    mask_data_num = self.tik_instance.Scalar(dtype='int64', name='mask_data_num')
                    with self.tik_instance.if_scope((each_valid_ho * shape_wo) % 16 != 0):
                        mask_data_num.set_as((each_valid_ho * shape_wo // 16 + 1) * 16)
                    with self.tik_instance.else_scope():
                        mask_data_num.set_as((each_valid_ho * shape_wo // 16) * 16)

                    mask_data_burst = self.tik_instance.Scalar(dtype='int64', name='mask_data_burst')
                    mask_data_burst.set_as(mask_data_num // 16)

                    self.tik_instance.data_move(mask_ub[0],
                                                self.argmax_gm[argmax_offset], 0,
                                                1,
                                                mask_data_burst,
                                                0, 0)

                    grad_sel_ub = self._vsel_grad_col(mask_ub, grad_ub, ori_output_shape[0])
                    grad_sel_ub_fp32 = self.ub_e

                    self.ele_num.set_as(ori_output_shape[0] * ori_output_shape[1] * ori_output_shape[2])
                    self._vconv(grad_sel_ub, 0, grad_sel_ub_fp32, 0,
                                self.ele_num, "float16")

                    col_index = self.tik_instance.Scalar(dtype="int64", name="col_index")
                    mask_idx = self.tik_instance.Scalar(dtype="int64", name="mask_idx")
                    with self.tik_instance.for_range(0, each_valid_ho) as h_idx:
                        col_index.set_as(index_h * wi * Constant.C0 + index_w * Constant.C0 + wi * Constant.C0 * \
                        self.stride_h * h_idx)
                        mask_idx.set_as(self.wo * Constant.C0 * h_idx)
                        self.ele_num.set_as(self.wo * Constant.C0 // 2)
                        self._vector_op("vadd", col2img_fp32_ub[col_index:],
                                        grad_sel_ub_fp32[mask_idx:],
                                        col2img_fp32_ub[col_index:], "float32", self.ele_num,
                                        stride_cofig=(self.stride_w * 2, self.stride_w * 2, 2,
                                                      self.stride_w * 16, self.stride_w * 16, 16))
                        self._vector_op("vadd", col2img_fp32_ub[col_index + 8:],
                                        grad_sel_ub_fp32[mask_idx + 8:],
                                        col2img_fp32_ub[col_index + 8:], "float32", self.ele_num,
                                        stride_cofig=(self.stride_w * 2, self.stride_w * 2, 2,
                                                      self.stride_w * 16, self.stride_w * 16, 16))

            self._mov_func(cut_ho_nums_index, cut_ho_nums, remain_ho_nums, each_process_ho,
                           self.each_process_hi, each_valid_ho, col2img_fp32_ub, temp_tensor_ub, pad,
                           col2img_ub_shape, temp_size)

        output_data_nums = self.tik_instance.Scalar(dtype="int64", name="output_data_nums")
        with self.tik_instance.for_range(0, cut_ho_nums) as cut_ho_nums_index:
            output_data_nums.set_as(each_process_ho * self.wo * Constant.C0)
            process_ho(output_data_nums, cut_ho_nums_index, each_process_ho)

        with self.tik_instance.if_scope(remain_ho_nums > 0):
            output_data_nums.set_as(remain_ho_nums * self.wo * Constant.C0)
            process_ho(output_data_nums, cut_ho_nums, remain_ho_nums)

    # 'pylint: disable=unused-variable,too-many-locals
    def _tilling_ho_nc1h(self, each_process_ho, n_index, c1_index,
                         each_process_ho_block, each_process_hi_block,
                         mov_len_ho, mov_len_hi,
                         start_ho_index, start_hi_index,
                         start_threshold,
                         offset_gm_block, shape, pad):

        # 'pylint: disable=too-many-locals
        pad_left, pad_right, _, pad_bottom = pad
        shape_wo = shape[1]

        cut_ho_nums = self.tik_instance.Scalar(dtype='int64', name='cut_ho_nums')
        remain_ho_nums = self.tik_instance.Scalar(dtype='int64', name='remain_ho_nums')
        wi = self.tik_instance.Scalar(dtype='int64', name='wi')
        cut_ho_nums.set_as(mov_len_ho // each_process_ho)
        remain_ho_nums.set_as(mov_len_ho % each_process_ho)
        wi.set_as(self.wi + pad_left + pad_right)

        temp_size_h = self.tik_instance.Scalar(dtype='int64', name='temp_size_h')
        temp_size_w = self.tik_instance.Scalar(dtype='int64', name='temp_size_w')
        with self.tik_instance.if_scope(self.kh > self.stride_h):
            self.each_process_hi.set_as((each_process_ho - 1) * self.stride_h + self.kh)
            temp_size_h.set_as(self.kh - self.stride_h)
            temp_size_w.set_as(wi)
        with self.tik_instance.else_scope():
            self.each_process_hi.set_as(each_process_ho * self.stride_h)
            temp_size_h.set_as(1)
            temp_size_w.set_as(16)
        temp_size = (temp_size_h, temp_size_w, Constant.C0)

        temp_tensor_ub = self.ub_e

        each_process_ho_wo_div16 = self.ho_wo_16
        ori_output_shape = (each_process_ho_wo_div16, 16, Constant.C0)

        col2img_ub_shape = (self.each_process_hi, wi, Constant.C0)
        col2img_fp32_ub = self.ub_d

        self.offset_gm.set_as(offset_gm_block)
        remained_hi = self.tik_instance.Scalar(dtype='int64', name='remained_hi')
        remained_hi.set_as(each_process_hi_block)

        # 'pylint: disable=unused-variable,too-many-locals
        def process_ho(output_data_nums, cut_ho_nums_index, each_valid_ho, remained_hi):
            """

            :param output_data_nums:
            :param cut_ho_nums_index:
            :param each_valid_ho:
            :param remained_hi:
            :return:
            """
            self._vector_dup(col2img_fp32_ub, 0, col2img_ub_shape,
                             self.scalar_zero, "float32")
            with self.tik_instance.if_scope(self.kh > self.stride_h):
                with self.tik_instance.if_scope(cut_ho_nums_index > 0):
                    self.ele_num.set_as(temp_size[0] * temp_size[1] * temp_size[2])
                    self._vector_op("vmuls", temp_tensor_ub, 1.0, col2img_fp32_ub,
                                    temp_tensor_ub.dtype, self.ele_num)

            start_h = self.tik_instance.Scalar(dtype='int64', name='start_h')
            end_h = self.tik_instance.Scalar(dtype='int64', name='end_h')
            start_h.set_as(cut_ho_nums_index * each_process_ho * self.stride_h)
            end_h.set_as(cut_ho_nums_index * each_process_ho * self.stride_h + self.each_process_hi)

            src_output_offset = ((n_index * self.c1 + c1_index) * self.ho + start_ho_index +
                                 each_process_ho * cut_ho_nums_index) * self.wo * Constant.C0

            # mov ori grad to ub
            grad_ub = self.ub_a
            self._vector_dup(grad_ub, 0, ori_output_shape,
                             self.scalar_zero_fp16, "float16")
            self.tik_instance.data_move(grad_ub[0],
                                        self.grad_gm[src_output_offset],
                                        0, 1, output_data_nums // 16, 0, 0)

            mask_shape = (self.mask_shape_128,)

            with self.tik_instance.for_range(0, self.kh, thread_num=1) as index_h:
                with self.tik_instance.for_range(0, self.kw, thread_num=1) as index_w:
                    mask_ub = self.ub_c
                    self._clean_mask(mask_ub, 0, mask_shape)

                    # use each_process_ho to calc offset
                    mask_offset_h = self.tik_instance.Scalar(dtype='int64', name='mask_offset_h')
                    mask_offset_h.set_as((start_ho_index + each_process_ho * cut_ho_nums_index)
                                         * shape_wo)
                    argmax_offset = self.tik_instance.Scalar(dtype='int64', name='argmax_offset')
                    argmax_offset.set_as(((n_index * self.c1 + c1_index) * self.kh * self.kw +
                                          index_h * self.kw + index_w) *
                                         self.one_window_size +
                                         mask_offset_h)

                    # use each_valid_ho to calc how much data should move
                    mask_data_num = self.tik_instance.Scalar(dtype='int64', name='mask_data_num')
                    with self.tik_instance.if_scope((each_valid_ho * shape_wo) % 16 != 0):
                        mask_data_num.set_as((each_valid_ho * shape_wo // 16 + 1) * 16)
                    with self.tik_instance.else_scope():
                        mask_data_num.set_as((each_valid_ho * shape_wo // 16) * 16)

                    mask_data_burst = self.tik_instance.Scalar(dtype='int64', name='mask_data_burst')
                    mask_data_burst.set_as(mask_data_num // 16)

                    self.tik_instance.data_move(mask_ub[0],
                                                self.argmax_gm[argmax_offset], 0,
                                                1,
                                                mask_data_burst,
                                                0, 0)

                    grad_sel_ub = self._vsel_grad_col(mask_ub, grad_ub, ori_output_shape[0])
                    grad_sel_ub_fp32 = self.ub_e

                    self.ele_num.set_as(ori_output_shape[0] * ori_output_shape[1] * ori_output_shape[2])
                    self._vconv(grad_sel_ub, 0, grad_sel_ub_fp32, 0,
                                self.ele_num, "float16")

                    with self.tik_instance.for_range(0, each_valid_ho) as h_idx:
                        col_index = index_h * wi * Constant.C0 + index_w * Constant.C0 + \
                                    wi * Constant.C0 * self.stride_h * h_idx
                        mask_idx = self.wo * Constant.C0 * h_idx
                        self._vector_op("vadd", col2img_fp32_ub[col_index:],
                                        grad_sel_ub_fp32[mask_idx:],
                                        col2img_fp32_ub[col_index:], "float32", self.wo * Constant.C0 // 2,
                                        stride_cofig=(self.stride_w * 2, self.stride_w * 2, 2,
                                                      self.stride_w * 16, self.stride_w * 16, 16))
                        self._vector_op("vadd", col2img_fp32_ub[col_index + 8:],
                                        grad_sel_ub_fp32[mask_idx + 8:],
                                        col2img_fp32_ub[col_index + 8:], "float32", self.wo * Constant.C0 // 2,
                                        stride_cofig=(self.stride_w * 2, self.stride_w * 2, 2,
                                                      self.stride_w * 16, self.stride_w * 16, 16))

            start_h.set_as(cut_ho_nums_index * each_process_ho * self.stride_h)
            end_h.set_as(cut_ho_nums_index * each_process_ho * self.stride_h +
                         each_process_ho * self.stride_h)
            with self.tik_instance.if_scope(remain_ho_nums == 0):
                with self.tik_instance.if_scope(cut_ho_nums_index == cut_ho_nums - 1):
                    end_h.set_as(cut_ho_nums_index * each_process_ho * self.stride_h + (
                            each_process_ho - 1) * self.stride_h + self.kh)
            with self.tik_instance.else_scope():
                with self.tik_instance.if_scope(cut_ho_nums_index == cut_ho_nums):
                    end_h.set_as(cut_ho_nums_index * each_process_ho * self.stride_h + (
                            each_process_ho - 1) * self.stride_h + self.kh)

            remained_hi = self._move_func_block(cut_ho_nums_index, cut_ho_nums,
                                                start_h, end_h, each_process_ho,
                                                each_process_hi_block, col2img_fp32_ub,
                                                temp_tensor_ub,
                                                remained_hi, remain_ho_nums,
                                                (pad_left, pad_right, start_threshold,
                                                 pad_bottom), col2img_ub_shape, temp_size)

            return remained_hi

        output_data_nums = self.tik_instance.Scalar(dtype='int64', name='output_data_nums')
        with self.tik_instance.for_range(0, cut_ho_nums) as cut_ho_nums_index:
            output_data_nums.set_as(each_process_ho * self.wo * Constant.C0)
            remained_hi = process_ho(output_data_nums, cut_ho_nums_index, each_process_ho,
                                     remained_hi)

        with self.tik_instance.if_scope(remain_ho_nums > 0):
            output_data_nums.set_as(remain_ho_nums * self.wo * Constant.C0)
            process_ho(output_data_nums, cut_ho_nums, remain_ho_nums, remained_hi)

    # 'pylint: disable=unused-variable,too-many-locals
    def _tilling_ho_wo(self, each_process_wo, n_index, c1_index,
                       each_process_ho_block, each_process_hi_block,
                       mov_len_ho, mov_len_hi,
                       start_ho_index, start_hi_index,
                       start_threshold,
                       offset_gm_block, shape, pad, core_idx):
        pad_left, pad_right, pad_top, pad_bottom = pad
        start_h = self.tik_instance.Scalar(dtype='int64', name='start_h')
        end_h = self.tik_instance.Scalar(dtype='int64', name='end_h')
        hi_max = self.tik_instance.Scalar(dtype='int64', name='hi_max')
        hi_min = self.tik_instance.Scalar(dtype='int64', name='hi_min')
        mov_len_h = self.tik_instance.Scalar(dtype='int64', name='mov_len_h')
        start_pos_h = self.tik_instance.Scalar(dtype='int64', name='start_pos_h')
        start_w = self.tik_instance.Scalar(dtype='int64', name='start_h')
        end_w = self.tik_instance.Scalar(dtype='int64', name='end_w')
        wi_max = self.tik_instance.Scalar(dtype='int64', name='wi_max')
        wi_min = self.tik_instance.Scalar(dtype='int64', name='wi_min')
        mov_len_w = self.tik_instance.Scalar(dtype='int64', name='mov_len_w')
        start_pos_w = self.tik_instance.Scalar(dtype='int64', name='start_pos_w')
        overlap_burst = self.tik_instance.Scalar(dtype='int64', name='overlap_burst')
        last_valid_wi = self.tik_instance.Scalar(dtype='int64', name='last_valid_wi')
        remain_wi = self.tik_instance.Scalar(dtype='int64', name='remain_wi')
        start_pos = self.tik_instance.Scalar(dtype='int64', name='start_pos')
        remained_hi = self.tik_instance.Scalar(dtype='int64', name='remained_hi')
        remained_hi.set_as(each_process_hi_block)
        cut_wo_nums = self.tik_instance.Scalar(dtype='int64', name='cut_wo_nums')
        cut_wo_nums.set_as(self.wo // each_process_wo)
        remain_wo_nums = self.tik_instance.Scalar(dtype='int64', name='remain_wo_nums')
        remain_wo_nums.set_as(self.wo % each_process_wo)

        each_process_wi = self.tik_instance.Scalar(dtype='int32', name='each_process_wi')
        each_process_hi = self.tik_instance.Scalar(dtype='int32', name='each_process_hi')

        with self.tik_instance.if_scope(self.stride_w >= self.kw):
            each_process_wi.set_as(each_process_wo * self.stride_w)
        with self.tik_instance.else_scope():
            each_process_wi.set_as((each_process_wo - 1) * self.stride_w + self.kw)

        overlap_h_nums = self.tik_instance.Scalar(dtype='int64', name='overlap_h_nums')
        with self.tik_instance.if_scope(self.stride_h >= self.kh):
            each_process_hi.set_as(self.stride_h)
            overlap_h_nums.set_as(0)
        with self.tik_instance.else_scope():
            each_process_hi.set_as(self.kh)
            overlap_h_nums.set_as(self.kh - self.stride_h)
        each_process_wo_div16 = self.ho_wo_16

        # define col res, init to zero
        col2img_ub_shape = (each_process_hi, each_process_wi, Constant.C0)
        col2img_fp32_ub = self.ub_d

        # when tile h to block, n*c1 is small and no need to calc offset for overlap
        if offset_gm_block is not None:
            self.offset_gm.set_as(offset_gm_block)

        overlap_shape_w = self.tik_instance.Scalar(dtype='int64', name='overlap_shape_w')
        overlap_offset = self.tik_instance.Scalar(dtype='int64', name='overlap_offset')
        with self.tik_instance.if_scope(self.stride_h < self.kh):
            overlap_shape_w.set_as((self.wi + pad_left + pad_right) * Constant.C0)
            overlap_offset.set_as(core_idx * Constant.WORKSPACE_ONE_CORE // 4)
            # save every h overlap on gm
            overlap_buffer = self.overlap_gm

        with self.tik_instance.for_range(0, mov_len_ho, thread_num=1) as ho_index:
            start_h.set_as(ho_index * self.stride_h)
            end_h.set_as(ho_index * self.stride_h + each_process_hi)

            self.tik_instance.scalar_max(hi_min, pad_top, start_h)
            hi_max.set_as(mov_len_hi + pad_top)
            self.tik_instance.scalar_min(hi_max, hi_max, end_h)
            mov_len_h.set_as(hi_max - hi_min)

            offset_gm_inside = self.tik_instance.Scalar(dtype='int64',
                                                        name='offset_gm_inside')
            offset_gm_inside.set_as(self.offset_gm)

            # init col2img after every looph
            self._vector_dup(col2img_fp32_ub, 0, col2img_ub_shape,
                             self.scalar_zero, "float32")

            with self.tik_instance.for_range(0, cut_wo_nums, thread_num=1) as cut_wo_nums_index:
                with self.tik_instance.if_scope(self.kh > self.stride_h):
                    with self.tik_instance.if_scope(ho_index != 0):
                        with self.tik_instance.if_scope(cut_wo_nums_index == 0):
                            with self.tik_instance.for_range(0, overlap_h_nums) as index_khs:
                                self.tik_instance.data_move(
                                    col2img_fp32_ub[index_khs * each_process_wi * Constant.C0],
                                    overlap_buffer[overlap_offset + index_khs * overlap_shape_w],
                                    0, 1, each_process_wi * Constant.C0 // 8,
                                    0, 0)

                        with self.tik_instance.else_scope():
                            start_pos.set_as((each_process_wi - self.stride_w * each_process_wo) * Constant.C0)
                            with self.tik_instance.for_range(0, overlap_h_nums) as index_khs:
                                self.tik_instance.data_move(
                                    col2img_fp32_ub[index_khs * each_process_wi * Constant.C0 + start_pos],
                                    overlap_buffer[overlap_offset + index_khs * overlap_shape_w +
                                                   cut_wo_nums_index * each_process_wo *
                                                   self.stride_w * Constant.C0 + start_pos],
                                    0, 1, self.stride_w * each_process_wo * Constant.C0 // 8,
                                    0, 0)

                ori_output_shape = (each_process_wo_div16, 16, Constant.C0)
                output_data_nums = self.tik_instance.Scalar(dtype='int64', name='output_data_nums')
                output_data_nums.set_as(each_process_wo * Constant.C0)
                src_output_offset = self.tik_instance.Scalar(dtype='int64', name='src_output_offset')
                src_output_offset.set_as(((n_index * self.c1 + c1_index) * self.ho + start_ho_index + ho_index) *
                                         self.wo * Constant.C0 + cut_wo_nums_index * each_process_wo * Constant.C0)

                # move ori grad to ub
                grad_ub = self.ub_a
                self._vector_dup(grad_ub, 0, ori_output_shape,
                                 self.scalar_zero_fp16, "float16")
                self.tik_instance.data_move(grad_ub[0],
                                            self.grad_gm[src_output_offset],
                                            0, 1, output_data_nums // 16, 0, 0)

                mask_shape = (self.mask_shape_128,)

                with self.tik_instance.for_range(0, self.kh,
                                                 thread_num=1) as index_h:
                    with self.tik_instance.for_range(0, self.kw,
                                                     thread_num=1) as index_w:
                        mask_ub = self.ub_c
                        self._clean_mask(mask_ub, 0, mask_shape)

                        # calc offset
                        mask_offset_h_w = self.tik_instance.Scalar(dtype='int64', name='mask_offset_h_w')
                        mask_offset_h_w.set_as((start_ho_index + ho_index) * self.wo +
                                               cut_wo_nums_index * each_process_wo)

                        argmax_offset = self.tik_instance.Scalar(dtype='int64', name='argmax_offset')
                        argmax_offset.set_as(((n_index * self.c1 + c1_index) * self.kh * self.kw +
                                              index_h * self.kw + index_w) *
                                             self.one_window_size +
                                             mask_offset_h_w)

                        # calc how much data should move
                        mask_data_num = self.tik_instance.Scalar(dtype='int64', name='mask_data_num')
                        with self.tik_instance.if_scope(each_process_wo % 16 != 0):
                            mask_data_num.set_as((each_process_wo // 16 + 1) * 16)
                        with self.tik_instance.else_scope():
                            mask_data_num.set_as((each_process_wo // 16) * 16)

                        mask_data_burst = self.tik_instance.Scalar(dtype='int64', name='mask_data_burst')
                        mask_data_burst.set_as(mask_data_num // 16)

                        self.tik_instance.data_move(mask_ub[0],
                                                    self.argmax_gm[argmax_offset], 0,
                                                    1,
                                                    mask_data_burst,
                                                    0, 0)

                        grad_sel_ub = self._vsel_grad_col(mask_ub, grad_ub, ori_output_shape[0])
                        grad_sel_ub_fp32 = self.ub_e
                        self.ele_num.set_as(ori_output_shape[0] * ori_output_shape[1] * ori_output_shape[2])
                        self._vconv(grad_sel_ub, 0, grad_sel_ub_fp32, 0,
                                    self.ele_num, "float16")

                        with self.tik_instance.for_range(0, 1) as h_idx:
                            col_index = index_h * each_process_wi * Constant.C0 + index_w * Constant.C0 + \
                                        each_process_wi * Constant.C0 * self.stride_h * h_idx
                            mask_idx = each_process_wo * Constant.C0 * h_idx

                            self._vector_op("vadd", col2img_fp32_ub[col_index:],
                                            grad_sel_ub_fp32[mask_idx:],
                                            col2img_fp32_ub[col_index:], "float32",
                                            each_process_wo * Constant.C0 // 2,
                                            stride_cofig=(self.stride_w * 2, self.stride_w * 2, 2,
                                                          self.stride_w * 16, self.stride_w * 16,
                                                          16))
                            self._vector_op("vadd", col2img_fp32_ub[col_index + 8:],
                                            grad_sel_ub_fp32[mask_idx + 8:],
                                            col2img_fp32_ub[col_index + 8:], "float32",
                                            each_process_wo * Constant.C0 // 2,
                                            stride_cofig=(self.stride_w * 2, self.stride_w * 2, 2,
                                                          self.stride_w * 16, self.stride_w * 16,
                                                          16))

                col2img_fp16_ub = self.ub_b
                self.ele_num.set_as(col2img_ub_shape[0] * col2img_ub_shape[1] * col2img_ub_shape[2])
                self._vconv(col2img_fp32_ub, 0, col2img_fp16_ub, 0,
                            self.ele_num, "float32")
                # set h direction's paras
                start_h.set_as(ho_index * self.stride_h)
                end_h.set_as(start_h + self.stride_h)
                with self.tik_instance.if_scope(ho_index == mov_len_ho - 1):
                    end_h.set_as(start_h + self.kh)
                if offset_gm_block is not None:
                    with self.tik_instance.if_scope(start_threshold > pad_top):
                        self.tik_instance.scalar_max(hi_min, start_threshold, start_h)
                        hi_max.set_as(each_process_hi_block + start_threshold)
                    with self.tik_instance.else_scope():
                        self.tik_instance.scalar_max(hi_min, pad_top, start_h)
                        hi_max.set_as(each_process_hi_block + pad_top)
                else:
                    self.tik_instance.scalar_max(hi_min, pad_top, start_h)
                    hi_max.set_as(each_process_hi_block + pad_top)
                self.tik_instance.scalar_min(hi_max, hi_max, end_h)
                mov_len_h.set_as(hi_max - hi_min)
                start_pos_h.set_as(hi_min - ho_index * self.stride_h)

                # set w direction's paras
                start_w.set_as(cut_wo_nums_index * each_process_wo * self.stride_w)
                end_w.set_as(cut_wo_nums_index * each_process_wo *
                             self.stride_w + each_process_wo * self.stride_w)
                self.tik_instance.scalar_max(wi_min, pad_left, start_w)
                self.temp_scalar_int64.set_as(self.wi + pad_left)
                self.tik_instance.scalar_min(wi_max, self.temp_scalar_int64, end_w)
                mov_len_w.set_as(wi_max - wi_min)
                start_pos_w.set_as(wi_min - cut_wo_nums_index * each_process_wo * self.stride_w)

                with self.tik_instance.if_scope(
                        tik.all(cut_wo_nums_index < cut_wo_nums - 1, mov_len_h > 0, mov_len_w > 0)):
                    self.tik_instance.data_move(self.res_gm[offset_gm_inside],
                                                col2img_fp16_ub[start_pos_h * each_process_wi *
                                                                Constant.C0 + start_pos_w * Constant.C0], 0,
                                                mov_len_h, mov_len_w * Constant.C0 // 16,
                                                each_process_wi - mov_len_w,
                                                self.wi - mov_len_w)
                    offset_gm_inside.set_as(offset_gm_inside + mov_len_w * Constant.C0)
                    self.offset_gm.set_as(self.offset_gm + mov_len_h * mov_len_w * Constant.C0)

                with self.tik_instance.if_scope(tik.all(mov_len_h > 0, mov_len_w > 0)):
                    with self.tik_instance.if_scope(remain_wo_nums == 0):
                        with self.tik_instance.if_scope(cut_wo_nums_index == cut_wo_nums - 1):
                            last_valid_wi.set_as(self.wi - ((cut_wo_nums - 1) * each_process_wo *
                                                            self.stride_w - pad_left))
                            with self.tik_instance.if_scope(self.wi < last_valid_wi):
                                last_valid_wi.set_as(self.wi)

                            with self.tik_instance.if_scope(last_valid_wi <= each_process_wi):
                                self.tik_instance.data_move(
                                    self.res_gm[offset_gm_inside],
                                    col2img_fp16_ub[start_pos_h * each_process_wi *
                                                    Constant.C0 + start_pos_w * Constant.C0],
                                    0, mov_len_h,
                                    last_valid_wi * Constant.C0 // 16,
                                    each_process_wi - last_valid_wi, self.wi - last_valid_wi)
                                offset_gm_inside.set_as(offset_gm_inside + last_valid_wi * Constant.C0)
                                self.offset_gm.set_as(
                                    self.offset_gm + mov_len_h * last_valid_wi * Constant.C0)
                            with self.tik_instance.else_scope():
                                self.tik_instance.data_move(self.res_gm[offset_gm_inside],
                                                            col2img_fp16_ub[
                                                                start_pos_h * each_process_wi *
                                                                Constant.C0 + start_pos_w * Constant.C0], 0,
                                                            mov_len_h, each_process_wi * Constant.C0 // 16,
                                                            0,
                                                            self.wi - each_process_wi)
                                offset_gm_inside.set_as(offset_gm_inside + each_process_wi * Constant.C0)

                                remain_wi.set_as(last_valid_wi - each_process_wi)
                                temp_zero_shape = (remain_wi * Constant.C0,)
                                temp_zero = self.ub_a
                                self._vector_dup(temp_zero, 0, temp_zero_shape,
                                                 self.scalar_zero_fp16, temp_zero.dtype)
                                with self.tik_instance.for_range(0, mov_len_h) as index_0:
                                    self.ele_num.set_as(remain_wi * Constant.C0)
                                    self.tik_instance.data_move(
                                        self.res_gm[offset_gm_inside + index_0 * self.wi * Constant.C0],
                                        temp_zero, 0, 1,
                                        self.ele_num // 16, 0, 0)
                                offset_gm_inside.set_as(offset_gm_inside + remain_wi * Constant.C0)
                                self.offset_gm.set_as(
                                    self.offset_gm + mov_len_h * last_valid_wi * Constant.C0)
                    with self.tik_instance.else_scope():
                        with self.tik_instance.if_scope(cut_wo_nums_index == cut_wo_nums - 1):
                            self.tik_instance.data_move(self.res_gm[offset_gm_inside],
                                                        col2img_fp16_ub[
                                                            start_pos_h * each_process_wi *
                                                            Constant.C0 + start_pos_w * Constant.C0], 0,
                                                        mov_len_h, mov_len_w * Constant.C0 // 16,
                                                        each_process_wi - mov_len_w,
                                                        self.wi - mov_len_w)
                            offset_gm_inside.set_as(offset_gm_inside + mov_len_w * Constant.C0)
                            self.offset_gm.set_as(self.offset_gm + mov_len_h * mov_len_w * Constant.C0)

                # move back to init col2img_fp16 tensor
                with self.tik_instance.if_scope(cut_wo_nums_index < cut_wo_nums - 1):
                    # save h overlap
                    with self.tik_instance.if_scope(self.kh > self.stride_h):
                        with self.tik_instance.for_range(0, overlap_h_nums) as index_s:
                            self.tik_instance.data_move(
                                overlap_buffer[overlap_offset + index_s * overlap_shape_w +
                                               cut_wo_nums_index * each_process_wo *
                                               self.stride_w * Constant.C0],
                                col2img_fp32_ub[self.stride_h *
                                                each_process_wi * Constant.C0 + each_process_wi *
                                                Constant.C0 * index_s],
                                0, 1, self.stride_w * each_process_wo * Constant.C0 // 8, 0, 0)

                    with self.tik_instance.if_scope(self.kw > self.stride_w):
                        with self.tik_instance.for_range(0, self.kh) as index_kh:
                            offset = [index_kh * each_process_wi * Constant.C0, index_kh * each_process_wi * \
                            Constant.C0 + self.stride_w * each_process_wo * Constant.C0]
                            self._vector_op("vmuls", col2img_fp32_ub, 1.0, col2img_fp32_ub,
                                            col2img_fp32_ub.dtype,
                                            (each_process_wi - each_process_wo * self.stride_w) * Constant.C0,
                                            None, offset)
                        with self.tik_instance.for_range(0, self.kh) as index_kh:
                            self._vector_dup(col2img_fp32_ub,
                                             index_kh * each_process_wi * Constant.C0 +
                                             (each_process_wi - self.stride_w *
                                              each_process_wo) * Constant.C0,
                                             (self.stride_w * each_process_wo * Constant.C0,),
                                             self.scalar_zero,
                                             col2img_fp32_ub.dtype)

                    with self.tik_instance.else_scope():
                        self._vector_dup(col2img_fp32_ub, 0, col2img_ub_shape,
                                         self.scalar_zero, col2img_fp32_ub.dtype)

                with self.tik_instance.else_scope():
                    with self.tik_instance.if_scope(remain_wo_nums > 0):
                        # save h overlap
                        with self.tik_instance.if_scope(self.kh > self.stride_h):
                            with self.tik_instance.for_range(0, overlap_h_nums) as index_s:
                                self.tik_instance.data_move(
                                    overlap_buffer[overlap_offset + index_s * overlap_shape_w +
                                                   cut_wo_nums_index * each_process_wo *
                                                   self.stride_w * Constant.C0],
                                    col2img_fp32_ub[self.stride_h *
                                                    each_process_wi * Constant.C0 + each_process_wi *
                                                    Constant.C0 * index_s],
                                    0, 1, self.stride_w * each_process_wo * Constant.C0 // 8, 0, 0)

                        with self.tik_instance.if_scope(self.kw > self.stride_w):
                            with self.tik_instance.for_range(0, self.kh) as index_kh:
                                offset = [index_kh * each_process_wi * Constant.C0,
                                          index_kh * each_process_wi * Constant.C0 +
                                          self.stride_w * each_process_wo * Constant.C0]
                                self._vector_op("vmuls", col2img_fp32_ub, 1.0, col2img_fp32_ub,
                                                col2img_fp32_ub.dtype,
                                                (each_process_wi - each_process_wo *
                                                 self.stride_w) * Constant.C0,
                                                None, offset)
                            with self.tik_instance.for_range(0, self.kh) as index_kh:
                                self._vector_dup(col2img_fp32_ub,
                                                 index_kh * each_process_wi * Constant.C0 +
                                                 (each_process_wi - self.stride_w *
                                                  each_process_wo) * Constant.C0,
                                                 (self.stride_w * each_process_wo * Constant.C0,),
                                                 self.scalar_zero,
                                                 col2img_fp32_ub.dtype)
                        with self.tik_instance.else_scope():
                            self._vector_dup(col2img_fp32_ub, 0, col2img_ub_shape,
                                             self.scalar_zero, col2img_fp32_ub.dtype)
                    with self.tik_instance.else_scope():
                        with self.tik_instance.if_scope(self.kh > self.stride_h):
                            with self.tik_instance.for_range(0, overlap_h_nums) as index_s:
                                self.tik_instance.data_move(
                                    overlap_buffer[overlap_offset + index_s * overlap_shape_w + (cut_wo_nums - 1) *
                                                   each_process_wo * self.stride_w * Constant.C0],
                                    col2img_fp32_ub[self.stride_h * each_process_wi * Constant.C0 +
                                                    each_process_wi * Constant.C0 * index_s],
                                    0, 1, each_process_wi * Constant.C0 // 8, 0, 0)

            with self.tik_instance.if_scope(remain_wo_nums > 0):
                each_process_remain_wi = self.tik_instance.Scalar(dtype="int64", name="each_process_remain_wi")
                with self.tik_instance.if_scope(self.kw > self.stride_w):
                    each_process_remain_wi.set_as((remain_wo_nums - 1) * self.stride_w + self.kw)
                with self.tik_instance.else_scope():
                    each_process_remain_wi.set_as(remain_wo_nums * self.stride_w)

                start_h.set_as(ho_index * self.stride_h)
                end_h.set_as(ho_index * self.stride_h + each_process_hi)

                # move overlap to init the col2img_fp16 tensor. if stride < kernel, there has
                # overlap between each ho.
                with self.tik_instance.if_scope(self.kh > self.stride_h):
                    with self.tik_instance.if_scope(ho_index != 0):
                        with self.tik_instance.if_scope(cut_wo_nums == 0):
                            with self.tik_instance.for_range(0, overlap_h_nums) as index_khs:
                                self.tik_instance.data_move(col2img_fp32_ub[index_khs * each_process_wi * Constant.C0],
                                                            overlap_buffer[
                                                                overlap_offset + index_khs * overlap_shape_w],
                                                            0, 1,
                                                            each_process_remain_wi * Constant.C0 // 8,
                                                            0, 0)
                        with self.tik_instance.else_scope():
                            start_pos.set_as((each_process_wi - self.stride_w * each_process_wo) * Constant.C0)
                            overlap_burst.set_as((self.wi + pad_left + pad_right -
                                                  (each_process_wi + self.stride_w * each_process_wo * (
                                                              cut_wo_nums - 1))) * Constant.C0)
                            with self.tik_instance.if_scope(overlap_burst > (self.stride_w * \
                            remain_wo_nums * Constant.C0)):
                                overlap_burst.set_as(self.stride_w * remain_wo_nums * Constant.C0)

                            with self.tik_instance.for_range(0, overlap_h_nums) as index_khs:
                                self.tik_instance.data_move(
                                    col2img_fp32_ub[index_khs * each_process_wi * Constant.C0 + start_pos],
                                    overlap_buffer[overlap_offset + index_khs * overlap_shape_w +
                                                   cut_wo_nums * each_process_wo *
                                                   self.stride_w * Constant.C0 + start_pos],
                                    0, 1, overlap_burst // 8, 0, 0)

                # mov forward output and grad to UB
                ori_output_ub_shape = (each_process_wo_div16, 16, Constant.C0)

                grad_ub = self.ub_a
                self.tik_instance.data_move(
                    grad_ub, self.grad_gm[((n_index * self.c1 + c1_index) *
                                           self.ho + start_ho_index + ho_index) * self.wo * Constant.C0 +
                                          cut_wo_nums * each_process_wo * Constant.C0],
                    0, 1, remain_wo_nums * Constant.C0 // 16, 0, 0)

                mask_shape = (self.mask_shape_128,)

                # calculate grad_x, here we loop kh and kw, so each loop
                # it process one row of output, image output as a window slide on kernel window
                with self.tik_instance.for_range(0, self.kh) as index_h:
                    with self.tik_instance.for_range(0, self.kw) as index_w:
                        mask_ub = self.ub_c
                        self._clean_mask(mask_ub, 0, mask_shape)

                        # calc offset
                        mask_offset_h_w = self.tik_instance.Scalar(dtype='int64', name='mask_offset_h_w')
                        mask_offset_h_w.set_as((start_ho_index + ho_index) * self.wo +
                                               cut_wo_nums * each_process_wo)

                        argmax_offset = self.tik_instance.Scalar(dtype='int64', name='argmax_offset')
                        argmax_offset.set_as(((n_index * self.c1 + c1_index) * self.kh * self.kw +
                                              index_h * self.kw + index_w) *
                                             self.one_window_size +
                                             mask_offset_h_w)

                        # calc how much data should move
                        mask_data_num = self.tik_instance.Scalar(dtype='int64', name='mask_data_num')
                        with self.tik_instance.if_scope(remain_wo_nums % 16 != 0):
                            mask_data_num.set_as((remain_wo_nums // 16 + 1) * 16)
                        with self.tik_instance.else_scope():
                            mask_data_num.set_as((remain_wo_nums // 16) * 16)

                        mask_data_burst = self.tik_instance.Scalar(dtype='int64', name='mask_data_burst')
                        mask_data_burst.set_as(mask_data_num // 16)

                        self.tik_instance.data_move(mask_ub[0],
                                                    self.argmax_gm[argmax_offset], 0,
                                                    1,
                                                    mask_data_burst,
                                                    0, 0)

                        grad_sel_ub = self._vsel_grad_col(mask_ub, grad_ub, ori_output_ub_shape[0])
                        grad_sel_ub_fp32 = self.ub_e
                        self.ele_num.set_as(ori_output_ub_shape[0] * ori_output_ub_shape[1] * ori_output_ub_shape[2])
                        self._vconv(grad_sel_ub, 0, grad_sel_ub_fp32, 0,
                                    self.ele_num, "float16")

                        # each procee ho is 1, so here loop value is 1
                        with self.tik_instance.for_range(0, 1) as h_idx:
                            col_index = index_h * each_process_wi * Constant.C0 + index_w * Constant.C0 + \
                                        each_process_wi * Constant.C0 * self.stride_h * h_idx
                            mask_idx = each_process_wo * Constant.C0 * h_idx
                            self._vector_op("vadd", col2img_fp32_ub[col_index:],
                                            grad_sel_ub_fp32[mask_idx:],
                                            col2img_fp32_ub[col_index:], "float32",
                                            remain_wo_nums * Constant.C0 // 2,
                                            stride_cofig=(self.stride_w * 2, self.stride_w * 2, 2,
                                                          self.stride_w * 16, self.stride_w * 16,
                                                          16))
                            self._vector_op("vadd", col2img_fp32_ub[col_index + 8:],
                                            grad_sel_ub_fp32[mask_idx + 8:],
                                            col2img_fp32_ub[col_index + 8:], "float32",
                                            remain_wo_nums * Constant.C0 // 2,
                                            stride_cofig=(self.stride_w * 2, self.stride_w * 2, 2,
                                                          self.stride_w * 16, self.stride_w * 16,
                                                          16))

                col2img_fp16_ub = self.ub_b
                self.ele_num.set_as(col2img_ub_shape[0] * col2img_ub_shape[1] * col2img_ub_shape[2])
                self._vconv(col2img_fp32_ub, 0, col2img_fp16_ub, 0,
                            self.ele_num, "float32")

                # set h direction's paras
                start_h.set_as(ho_index * self.stride_h)
                end_h.set_as(start_h + self.stride_h)
                with self.tik_instance.if_scope(ho_index == mov_len_ho - 1):
                    end_h.set_as(start_h + self.kh)

                if offset_gm_block is not None:
                    with self.tik_instance.if_scope(start_threshold > pad_top):
                        self.tik_instance.scalar_max(hi_min, start_threshold, start_h)
                        hi_max.set_as(each_process_hi_block + start_threshold)
                    with self.tik_instance.else_scope():
                        self.tik_instance.scalar_max(hi_min, pad_top, start_h)
                        hi_max.set_as(each_process_hi_block + pad_top)
                else:
                    self.tik_instance.scalar_max(hi_min, pad_top, start_h)
                    hi_max.set_as(each_process_hi_block + pad_top)
                self.tik_instance.scalar_min(hi_max, hi_max, end_h)
                mov_len_h.set_as(hi_max - hi_min)
                start_pos_h.set_as(hi_min - ho_index * self.stride_h)

                # set w direction's paras
                start_w.set_as(cut_wo_nums * each_process_wo * self.stride_w)
                end_w.set_as(cut_wo_nums * each_process_wo *
                             self.stride_w + remain_wo_nums * self.stride_w)
                self.tik_instance.scalar_max(wi_min, pad_left, start_w)
                self.temp_scalar_int64.set_as(self.wi + pad_left)
                self.tik_instance.scalar_min(wi_max, self.temp_scalar_int64, end_w)
                mov_len_w.set_as(wi_max - wi_min)
                start_pos_w.set_as(wi_min - cut_wo_nums * each_process_wo * self.stride_w)

                with self.tik_instance.if_scope(mov_len_h > 0):
                    last_valid_wi.set_as(self.wi - (
                            cut_wo_nums * each_process_wo * self.stride_w - pad_left))
                    with self.tik_instance.if_scope(last_valid_wi > self.wi):
                        last_valid_wi.set_as(self.wi)
                    with self.tik_instance.if_scope(last_valid_wi <= each_process_wi):
                        self.tik_instance.data_move(
                            self.res_gm[offset_gm_inside],
                            col2img_fp16_ub[start_pos_h * each_process_wi * Constant.C0 + start_pos_w * Constant.C0],
                            0, mov_len_h,
                            last_valid_wi * Constant.C0 // 16,
                            each_process_wi - last_valid_wi,
                            self.wi - last_valid_wi)
                        offset_gm_inside.set_as(offset_gm_inside + last_valid_wi * Constant.C0)
                        self.offset_gm.set_as(self.offset_gm + mov_len_h * last_valid_wi * Constant.C0)
                    with self.tik_instance.else_scope():
                        self.tik_instance.data_move(self.res_gm[offset_gm_inside],
                                                    col2img_fp16_ub[start_pos_h * each_process_wi *
                                                                    Constant.C0 + start_pos_w * Constant.C0], 0,
                                                    mov_len_h,
                                                    each_process_wi * Constant.C0 // 16,
                                                    0,
                                                    self.wi - each_process_wi)
                        offset_gm_inside.set_as(offset_gm_inside + each_process_wi * Constant.C0)

                        remain_wi.set_as(last_valid_wi - each_process_wi)
                        temp_zero_shape = (remain_wi * Constant.C0,)
                        temp_zero = self.ub_a
                        self._vector_dup(temp_zero, 0, temp_zero_shape,
                                         self.scalar_zero_fp16, temp_zero.dtype)
                        with self.tik_instance.for_range(0, mov_len_h) as index_0:
                            self.tik_instance.data_move(
                                self.res_gm[offset_gm_inside + index_0 * self.wi * Constant.C0],
                                temp_zero, 0, 1,
                                temp_zero_shape[0] // 16, 0, 0)
                        offset_gm_inside.set_as(offset_gm_inside + remain_wi * Constant.C0)
                        self.offset_gm.set_as(self.offset_gm + mov_len_h * last_valid_wi * Constant.C0)

                with self.tik_instance.if_scope(self.kh > self.stride_h):
                    overlap_burst.set_as(
                        (self.wi + pad_left + pad_right - self.stride_w * each_process_wo * cut_wo_nums) * Constant.C0)
                    with self.tik_instance.if_scope(overlap_burst > each_process_remain_wi * Constant.C0):
                        overlap_burst.set_as(each_process_remain_wi * Constant.C0)
                    with self.tik_instance.for_range(0, overlap_h_nums) as index_s:
                        self.tik_instance.data_move(
                            overlap_buffer[overlap_offset + index_s * overlap_shape_w + cut_wo_nums *
                                           each_process_wo * self.stride_w * Constant.C0],
                            col2img_fp32_ub[self.stride_h * each_process_wi *
                                            Constant.C0 + each_process_wi * Constant.C0 * index_s],
                            0, 1, overlap_burst // 8, 0, 0)
                with self.tik_instance.if_scope(self.kw <= self.stride_w):
                    self._vector_dup(col2img_fp32_ub, 0, col2img_ub_shape,
                                     self.scalar_zero, col2img_fp32_ub.dtype)

            with self.tik_instance.if_scope(mov_len_h > 0):
                remained_hi.set_as(remained_hi - mov_len_h)

            with self.tik_instance.if_scope(tik.all(ho_index == mov_len_ho - 1, remained_hi > 0)):
                # `one_h = (self.wi, C0)`
                temp_zero = self.ub_a
                w_ele_num = self.one_seventh_ub_ele // Constant.C0
                temp_zero_shape = (w_ele_num, Constant.C0)
                self._vector_dup(temp_zero, 0,
                                 temp_zero_shape,
                                 self.scalar_zero_fp16,
                                 temp_zero.dtype)
                with self.tik_instance.for_range(0, remained_hi) as repeate_index:
                    total_repeate_time = self.wi // w_ele_num
                    remain_ele = self.wi % w_ele_num
                    with self.tik_instance.for_range(0, total_repeate_time) as time_index:
                        self.tik_instance.data_move(self.res_gm[self.offset_gm],
                                                    temp_zero, 0,
                                                    1, w_ele_num * Constant.C0 // 16, 0, 0)
                        self.offset_gm.set_as(self.offset_gm + w_ele_num * Constant.C0)
                    with self.tik_instance.if_scope(remain_ele > 0):
                        self.tik_instance.data_move(self.res_gm[self.offset_gm],
                                                    temp_zero, 0,
                                                    1, remain_ele * Constant.C0 // 16, 0, 0)
                        self.offset_gm.set_as(self.offset_gm + remain_ele * Constant.C0)

    # 'pylint: disable=unused-variable,too-many-locals
    def max_pool_grad_compute_tiling(self):
        """maxpoolgrad compute tiling
        """
        if self.dtype == "float16":
            self.pad_value = Constant.MIN_VALUE_FP16
        else:
            error_manager_vector.raise_err_input_dtype_not_supported("max_pool_grad_with_argmaxv2", "dtype",
                                                                     "float16", str(self.dtype))

        with self.tik_instance.for_range(0, self.core_num_var, block_num=self.core_num_var) as block_index:
            with self.tik_instance.if_scope(self.tiling_mode == 3):
                # Instead of defining new tiling parameters for resnet50,
                # using block_cycle represents one_core_ele
                # using real_block represents last_core_ele
                one_core_ele = self.block_cycle
                last_core_ele = self.real_block
                with self.tik_instance.if_scope(block_index < self.act_core_num - 1):
                    self.resnet50_branch.maxpool_resnet50(block_index, one_core_ele, one_core_ele)
                with self.tik_instance.if_scope(block_index == self.act_core_num - 1):
                    self.resnet50_branch.maxpool_resnet50(block_index, last_core_ele, one_core_ele)

            with self.tik_instance.else_scope():
                self.init_ub_scalar()
                self.init_ub_tensor()
                # call select tiling mode function
                with self.tik_instance.if_scope(block_index <= self.act_core_num - 1):
                    self.pad = (self.pad_left, self.pad_right, self.pad_top, self.pad_bottom)
                    # `real_block == 32`
                    with self.tik_instance.if_scope(self.real_block == self.core_num_var):
                        with self.tik_instance.for_range(0, self.block_cycle) as cycle_index:
                            n_index = self.tik_instance.Scalar(dtype='int64', name='n_axis')
                            c1_index = self.tik_instance.Scalar(dtype='int64', name='c1_index')
                            index_sum = self.tik_instance.Scalar(dtype='int64', name='index_sum')
                            index_sum.set_as(block_index * self.block_cycle + cycle_index)
                            with self.tik_instance.if_scope(index_sum < self.block_num):
                                n_index.set_as(index_sum // self.c1)
                                c1_index.set_as(index_sum % self.c1)
                                shape = (self.ho, self.wo, self.hi, self.wi)
                                self.offset_gm.set_as(
                                    (n_index * self.c1 + c1_index) * self.hi * self.wi * Constant.C0)
                                with self.tik_instance.if_scope(self.tiling_mode == 2):
                                    with self.tik_instance.new_stmt_scope():
                                        self._tilling_ho_wo(self.each_process_wo, n_index, c1_index,
                                                            self.ho, self.hi,
                                                            self.ho, self.hi,
                                                            0, 0,
                                                            0,
                                                            None, shape, self.pad, block_index)
                                with self.tik_instance.if_scope(self.tiling_mode == 1):
                                    with self.tik_instance.new_stmt_scope():
                                        self._tilling_ho(self.each_process_ho, n_index, c1_index,
                                                        self.ho, self.hi,
                                                        self.ho, self.hi,
                                                        0, 0,
                                                        0,
                                                        None, shape, self.pad)
                                with self.tik_instance.if_scope(self.tiling_mode == 0):
                                    with self.tik_instance.new_stmt_scope():
                                        self._not_tilling(n_index, c1_index,
                                                        self.ho, self.hi,
                                                        self.ho, self.hi,
                                                        0, 0,
                                                        0, None, shape, self.pad)
                    with self.tik_instance.else_scope():
                        with self.tik_instance.if_scope(self.if_block == 1):
                            block_outer_index = self.tik_instance.Scalar(dtype='int64', name='block_outer_index')
                            block_outer_index.set_as(block_index)
                            with self.tik_instance.for_range(0, self.block_num_inner) as block_innner_index:
                                nc1_index = self.tik_instance.Scalar(dtype='int64', name='nc1_index')
                                nc1_index.set_as(
                                    block_outer_index // self.ho_outer * self.block_num_inner + block_innner_index)
                                with self.tik_instance.if_scope(nc1_index < self.block_num):
                                    n_index = self.tik_instance.Scalar(dtype='int64', name='n_index')
                                    c1_index = self.tik_instance.Scalar(dtype='int64', name='c1_index')
                                    ho_outer_index = self.tik_instance.Scalar(dtype='int64',
                                                                            name='ho_outer_index')
                                    offset_gm_block = self.tik_instance.Scalar(dtype='int64',
                                                                            name='offset_gm_block')
                                    n_index.set_as(nc1_index // self.c1)
                                    c1_index.set_as(nc1_index % self.c1)
                                    ho_outer_index.set_as(block_outer_index % self.ho_outer)

                                    start_hi_index = self.tik_instance.Scalar(dtype='int64',
                                                                            name='start_hi_index')
                                    start_ho_index = self.tik_instance.Scalar(dtype='int64',
                                                                            name='start_ho_index')
                                    actual_start_ho_index = self.tik_instance.Scalar(
                                        dtype='int64', name='actual_start_ho_index')
                                    actual_start_hi_index = self.tik_instance.Scalar(
                                        dtype='int64', name='actual_start_hi_index')
                                    each_process_ho_block = self.tik_instance.Scalar(
                                        dtype='int64', name='each_process_ho_block')
                                    each_process_hi_block = self.tik_instance.Scalar(
                                        dtype='int64', name='each_process_hi_block')
                                    pad_top_block = self.tik_instance.Scalar(dtype='int64',
                                                                            name='pad_top_block')
                                    pad_bottom_block = self.tik_instance.Scalar(dtype='int64',
                                                                                name='pad_bottom_block')
                                    start_threshold = self.tik_instance.Scalar(dtype='int64',
                                                                            name='start_threshold')
                                    mov_len_ho = self.tik_instance.Scalar(dtype='int64', name='mov_len_ho')
                                    mov_len_hi = self.tik_instance.Scalar(dtype='int64', name='mov_len_hi')

                                    # each block's start ho pos and hi pos
                                    # calculate the offset gm
                                    start_ho_index.set_as(ho_outer_index * self.ho_inner)
                                    start_hi_index.set_as(start_ho_index * self.stride_h)
                                    actual_start_ho_index.set_as(start_ho_index)
                                    actual_start_hi_index.set_as(start_hi_index)
                                    start_threshold.set_as(0)

                                    with self.tik_instance.if_scope(start_hi_index <= self.pad_top):
                                        offset_gm_block.set_as(
                                            (n_index * self.c1 + c1_index) * self.hi * self.wi * Constant.C0)
                                        pad_top_block.set_as(self.pad_top - start_hi_index)
                                        self.tik_instance.scalar_max(start_threshold, start_threshold,
                                                                    pad_top_block)
                                        actual_start_hi_index.set_as(0)
                                    with self.tik_instance.else_scope():
                                        offset_gm_block.set_as(((n_index * self.c1 + c1_index) * self.hi +
                                                                start_hi_index - self.pad_top) * self.wi * Constant.C0)
                                        pad_top_block.set_as(0)
                                        actual_start_hi_index.set_as(actual_start_hi_index - self.pad_top)

                                    with self.tik_instance.if_scope(ho_outer_index != self.ho_outer - 1):
                                        each_process_ho_block.set_as(self.ho_inner)
                                    with self.tik_instance.else_scope():
                                        each_process_ho_block.set_as(self.ho - self.ho_inner * (self.ho_outer - 1))
                                    mov_len_ho.set_as(each_process_ho_block)
                                    mov_len_hi.set_as(each_process_ho_block * self.stride_h)

                                    if self.stride_h < self.kh:
                                        overlap = self.kh - self.stride_h
                                        overlap_num = int(math.ceil(overlap * 1.0 / self.stride_h))

                                        actual_start_hi_index.set_as(
                                            (start_ho_index - overlap_num) * self.stride_h)

                                        with self.tik_instance.if_scope(actual_start_hi_index <= 0):
                                            actual_start_hi_index.set_as(0)
                                            actual_start_ho_index.set_as(0)
                                            pad_top_block.set_as(self.pad_top)
                                            mov_len_ho.set_as(start_ho_index + each_process_ho_block)
                                            start_threshold.set_as(start_ho_index * self.stride_h)
                                            self.tik_instance.scalar_max(start_threshold, start_threshold,
                                                                        pad_top_block)

                                        with self.tik_instance.else_scope():
                                            pad_top_block.set_as(self.pad_top - actual_start_hi_index)
                                            self.tik_instance.scalar_max(pad_top_block, pad_top_block, 0)
                                            actual_start_ho_index.set_as(start_ho_index - overlap_num)
                                            with self.tik_instance.if_scope(
                                                    actual_start_hi_index <= self.pad_top):
                                                actual_start_hi_index.set_as(0)
                                            with self.tik_instance.else_scope():
                                                actual_start_hi_index.set_as(
                                                    actual_start_hi_index - self.pad_top)
                                            mov_len_ho.set_as(overlap_num + each_process_ho_block)
                                            start_threshold.set_as(overlap_num * self.stride_h)
                                        mov_len_hi.set_as(
                                            (mov_len_ho - 1) * self.stride_h + self.kh)

                                    with self.tik_instance.if_scope(start_hi_index < self.pad_top):
                                        each_process_hi_block.set_as(
                                            each_process_ho_block * self.stride_h - (
                                                    self.pad_top - start_hi_index))
                                    with self.tik_instance.else_scope():
                                        each_process_hi_block.set_as(
                                            each_process_ho_block * self.stride_h)

                                    with self.tik_instance.if_scope(
                                            actual_start_ho_index + mov_len_ho > self.ho):
                                        mov_len_ho.set_as(self.ho - actual_start_ho_index)

                                    with self.tik_instance.if_scope(
                                            actual_start_hi_index + mov_len_hi < self.hi):
                                        pad_bottom_block.set_as(0)
                                    with self.tik_instance.else_scope():
                                        pad_bottom_block.set_as(
                                            actual_start_hi_index + mov_len_hi - self.hi)
                                        mov_len_hi.set_as(self.hi - actual_start_hi_index)

                                    with self.tik_instance.if_scope(ho_outer_index == self.ho_outer - 1):
                                        each_process_hi_block.set_as(self.hi + self.pad_top - start_hi_index)
                                    with self.tik_instance.if_scope(
                                            start_hi_index + each_process_hi_block > self.hi + self.pad_top):
                                        each_process_hi_block.set_as(self.hi + self.pad_top - start_hi_index)

                                    pad = (self.pad_left, self.pad_right, pad_top_block, pad_bottom_block)
                                    shape = (self.shape_ho, self.wo, self.shape_hi, self.wi)

                                    with self.tik_instance.if_scope(self.tiling_mode == 2):
                                        with self.tik_instance.new_stmt_scope():
                                            self._tilling_ho_wo(self.each_process_wo, n_index, c1_index,
                                                                each_process_ho_block, each_process_hi_block,
                                                                mov_len_ho, mov_len_hi,
                                                                actual_start_ho_index, actual_start_hi_index,
                                                                start_threshold,
                                                                offset_gm_block, shape, pad, block_index)
                                    with self.tik_instance.if_scope(self.tiling_mode == 1):
                                        with self.tik_instance.new_stmt_scope():
                                            self._tilling_ho_nc1h(self.each_process_ho, n_index, c1_index,
                                                                each_process_ho_block, each_process_hi_block,
                                                                mov_len_ho, mov_len_hi,
                                                                actual_start_ho_index, actual_start_hi_index,
                                                                start_threshold,
                                                                offset_gm_block, shape, pad)
                                    with self.tik_instance.if_scope(self.tiling_mode == 0):
                                        with self.tik_instance.new_stmt_scope():
                                            self._not_tilling_nc1h(n_index, c1_index,
                                                                each_process_ho_block, each_process_hi_block,
                                                                mov_len_ho, mov_len_hi,
                                                                actual_start_ho_index, actual_start_hi_index,
                                                                start_threshold,
                                                                offset_gm_block, shape, pad)

                        with self.tik_instance.else_scope():
                            with self.tik_instance.for_range(0, self.nc1, thread_num=1) as nc1_index:
                                self.offset_gm.set_as((block_index * self.nc1 + nc1_index) *
                                                    self.hi * self.wi * Constant.C0)
                                n_index = (block_index * self.nc1 + nc1_index) // self.c1
                                c1_index = (block_index * self.nc1 + nc1_index) % self.c1

                                shape = (self.ho, self.wo, self.hi, self.wi)

                                with self.tik_instance.if_scope(self.tiling_mode == 2):
                                    with self.tik_instance.new_stmt_scope():
                                        self._tilling_ho_wo(self.each_process_wo, n_index, c1_index,
                                                            self.ho, self.hi,
                                                            self.ho, self.hi,
                                                            0, 0,
                                                            0,
                                                            None, shape, self.pad, block_index)
                                with self.tik_instance.if_scope(self.tiling_mode == 1):
                                    with self.tik_instance.new_stmt_scope():
                                        self._tilling_ho(self.each_process_ho, n_index, c1_index,
                                                        self.ho, self.hi,
                                                        self.ho, self.hi,
                                                        0, 0,
                                                        0,
                                                        None, shape, self.pad)
                                with self.tik_instance.if_scope(self.tiling_mode == 0):
                                    with self.tik_instance.new_stmt_scope():
                                        self._not_tilling(n_index, c1_index,
                                                        self.ho, self.hi,
                                                        self.ho, self.hi,
                                                        0, 0,
                                                        0, None, shape, self.pad)

    # 'pylint: disable=unused-variable
    def tik_instance_function(self):
        """
        main function of tik_instance
        """
        self.max_pool_grad_compute_tiling()
        # config and build CCE
        opt_config = {"out_of_bound_sync_check": True, "enable_const_fold": True}
        tbe_context.get_context().add_compile_info(
            "vars", {
                "ub_ele": self.ub_ele,
                "core_num": self.core_num,
                "kh": self.kh,
                "kw": self.kw,
                "stride_h": self.stride_h,
                "stride_w": self.stride_w,
                "pad_h": self.pads[1],
                "pad_w": self.pads[2],
                "dilation_h": self.dilation[1],
                "dilation_w": self.dilation[2],
                "ceil_mode": self.ceil_mode
            })
        tbe_context.get_context().add_compile_info("is_tik", True)
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[self.ori_input_gm, self.grad_gm, self.argmax_gm],
                                   outputs=[self.res_gm],
                                   flowtable=[self.tiling_gm],
                                   config=opt_config)
        return self.tik_instance


# 'pylint: disable=dangerous-default-value,too-many-locals,
# 'pylint: disable=too-many-arguments,,unused-argument,invalid-name
@register_operator("MaxPoolGardWithArgmaxV2")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_LIST_INT,
                            para_check.REQUIRED_ATTR_LIST_INT,
                            para_check.REQUIRED_ATTR_LIST_INT,
                            para_check.OPTION_ATTR_INT,
                            para_check.OPTION_ATTR_LIST_INT,
                            para_check.OPTION_ATTR_BOOL,
                            para_check.KERNEL_NAME)
def max_pool_grad_with_argmax_v2(x,
                                 grad,
                                 argmax,
                                 y,
                                 ksize,
                                 strides,
                                 pads=[0, 0, 0, 0],
                                 dtype=Constant.DT_INT32,
                                 dilation=(1, 1, 1, 1),
                                 ceil_mode=False,
                                 kernel_name="max_pool_grad_with_argmax_v2"):
    """
    main function of max_pool_grad_with_argmax_v2

    Parameters
    ----------
    x: dict
        shape and data type of ori_input
    argmax: dict
        shape and data type of argmax
    grad: dict
        shape and data type of grad
    y: dict
        shape and data type of y
    ksize: list or tuple
        the size of the window
    strides: list or tuple
        the stride of the sliding window
    pads: list or tuple
        padding size of height and width
    dtype: int
    dilation: list or tuple
        dilation of kernel
    ceil_mode: bool
        whether use ceil fuction to calculate the output of height and width
    kernel_name: str

    Returns
    -------
    return the tik api function
    """
    ori_format = x.get("ori_format")
    if ori_format in ("NCHW", "NC1HWC0"):
        ksize = (ksize[0], ksize[2], ksize[3], ksize[1])
        strides = (strides[0], strides[2], strides[3], strides[1])
        pads = (pads[0], pads[2], pads[3], pads[1])

    check_param(x, argmax, grad, ksize, strides, ori_format, dilation, pads,
                kernel_name)

    dtype = x.get("dtype").lower()
    maxpoolgrad = MaxpoolGrad(dtype, ksize, strides, pads, dilation, ceil_mode, kernel_name)
    return maxpoolgrad.tik_instance_function()
