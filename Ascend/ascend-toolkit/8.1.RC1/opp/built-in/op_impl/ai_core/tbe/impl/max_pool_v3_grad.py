#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright 2020 Huawei Technologies Co., Ltd
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
max_pool_v3_grad
"""
# 'pylint: disable=too-many-lines
import math
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import error_manager_vector
from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json
from impl.max_pool_grad import MaxpoolGradAtomic
from impl.max_pool_grad import MaxpoolGrad
from impl.load3d_common_func import img2col

# MIN VALUE OF FP16
MIN_VALUE_FP16 = -65500.0
# MIN VALUE OF POSIRIVE FP32
MIN_VALUE_FP32 = 1.18e-38
# VALID MASK BITS FOR 128
MASK128_VALUE = 128
# VALID MASK BITS FOR 64
MASK64_VALUE = 64
# REPEAT ONE TIMES
REPEAT_1 = 1
# REPEAT TWO TIMES
REPEAT_2 = 2
# DSTSTRIDEM0
DSTSTRIDEM0 = 1
# SRC0STRIDEM0
SRC0STRIDEM0 = 1
# SRC1STRIDEM0
SRC1STRIDEM0 = 1
# DSTSTRIDEM1
DSTSTRIDEM1 = 8
# SRC0STRIDEM1
SRC0STRIDEM1 = 8
# SRC1STRIDEM1
SRC1STRIDEM1 = 8
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
# L1 SIZE
L1_SIZE = tbe_platform.get_soc_spec(tbe_platform.L1_SIZE)
C0 = 16
# BLOCK NUMS
CORE_NUM = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
# maximum dma_copy stride
MAX_STRIDE = 65535


def ceil_div(num, divisor):
    """calcu use ceil_div

    Parameters
    ----------
    num: int
        input num
    divisor: int
        input divisor

    returns
    -------
    result: int
        num // divisor
    """
    if num % divisor != 0:
        return num // divisor + 1
    return num // divisor


def cal_shape_ele(shape):
    """calcu element nums

    Parameters
    ----------
    shape: list
         input shape list

    returns
    -------
    result: int
        the total num of shape
    """
    reduce_ = 1
    for i in shape:
        reduce_ *= int(i)
    return reduce_


def cal_byte_size(shape, dtype):
    """calcu tensor size

    Parameters
    ----------
    shape: list
        input shape list
    dtype: str
        input data dtype
    returns
    -------
    result: int
        the total num of shape
    """
    if dtype == "float16":
        return cal_shape_ele(shape) * 2
    if dtype == "float32":
        return cal_shape_ele(shape) * 4
    error_manager_vector.raise_err_specific_reson("max_pool_v3_grad", "Not support shape now")


def prod(values):
    """
    Prod the input values by multiply.
    """
    res = 1
    for value in values:
        res *= value

    return res


def cal_core(tik_instance, total_core_loop_num, num_core, core_number):
    """
    calculate the loop number on each core
    """
    if total_core_loop_num % core_number == 0:
        core_loop = total_core_loop_num // core_number
        sum_core = core_loop * num_core
    else:
        core_loop = tik_instance.Scalar("uint64")
        sum_core = tik_instance.Scalar("uint64")
        with tik_instance.if_scope(num_core < total_core_loop_num %
                                   CORE_NUM):
            core_loop.set_as((total_core_loop_num + core_number - 1) //
                             core_number)
            sum_core.set_as(core_loop * num_core)
        with tik_instance.else_scope():
            core_loop.set_as(total_core_loop_num // core_number)
            sum_core.set_as((core_loop + 1) * (total_core_loop_num % CORE_NUM) +
                            core_loop * (num_core - total_core_loop_num %
                                         CORE_NUM))
    return core_loop, sum_core


def init_coordinate(tik_instance, pad_x_top, xi_coordinate):
    """
    init_coordinate
    """
    # actual xi_coord
    if pad_x_top != 0:
        xi_coord = tik_instance.Scalar(dtype='int64', name='xi_coord')
        with tik_instance.if_scope(xi_coordinate < 0):
            xi_coord.set_as(0)
        with tik_instance.else_scope():
            xi_coord.set_as(xi_coordinate)
    else:
        xi_coord = xi_coordinate

    return xi_coord


# 'pylint: disable=too-many-arguments
def calc_pad(tik_instance, pad_top, pad_bottom,
             xi_coord, xi_value, boundary):
    """
    calc_pad
    """
    # return pad_value in different axis
    top = pad_top
    bottom = pad_bottom

    if pad_top != 0:
        top = tik_instance.Scalar(dtype='int64', name='top')
        with tik_instance.if_scope(xi_coord < 0):
            top.set_as(0 - xi_coord)
        with tik_instance.else_scope():
            top.set_as(0)

    if pad_bottom != 0:
        bottom = tik_instance.Scalar(dtype='int64', name='bottom')
        with tik_instance.if_scope(xi_coord + xi_value > boundary):
            bottom.set_as(xi_coord + xi_value - boundary)
        with tik_instance.else_scope():
            bottom.set_as(0)
    return top, bottom


def check_config(config):
    """
    check configuration

    Parameters
    ----------
    config: tuple
        configurations to be checked

    Returns
    -------
    mark: boolean
        pass check or not
    """
    config = list(config)
    mark = True
    for i in config:
        if i > 255:
            mark = False
            break

    return mark


# 'pylint: disable=too-many-arguments,too-many-statements,too-many-branches
# 'pylint: disable=invalid-name,unused-argument
def check_param(ori_input, ori_output, grad, ksize, strides, padding, data_format, global_pooling, pads, kernel_name):
    """
    check parameters, if one is invalid, then raise error
    Parameters
    ----------
    ori_input: dict
        shape and data type of ori_input
    ori_output: dict
        shape and data type of ori_output
    grad: dict
        shape and data type of grad
    ksize: list or tuple
        the size of the window
    strides: list or tuple
        the stride of the sliding window
    padding: str
        value from `SAME`, `VALID`, 'CALCULATED'
    pads: list or tuple
        padding size of height and width while padding is CALCULATED
    kernel_name: str

    Returns
    -------
    None
    """
    ori_input_shape = ori_input.get("shape")
    ori_input_dtype = ori_input.get("dtype").lower()
    ori_output_shape = ori_output.get("shape")
    grad_shape = grad.get("shape")
    para_check.check_shape(ori_input_shape, param_name="ori_input")
    para_check.check_dtype(ori_input_dtype, ("float16",), param_name="ori_input")
    # the format of input_x must be NC1HWC0
    if len(ori_input_shape) != 5:
        error_manager_vector.raise_err_specific_reson("max_pool_v3_grad", "input feature map \
                                                      must be 5D format in kernel.")
    if len(ori_output_shape) != 5:
        error_manager_vector.raise_err_specific_reson("max_pool_v3_grad", "forward output \
                                                      must be 5D format in kernel.")
    if len(grad_shape) != 5:
        error_manager_vector.raise_err_specific_reson("max_pool_v3_grad", "update grad \
                                                      must be 5D format in kernel.")

    if grad_shape != ori_output_shape:
        error_manager_vector.raise_err_two_input_shape_invalid("max_pool_v3_grad", "grad", "ori_output",
                                                               "update grad must be same shape as forward output")

    if grad_shape[-1] != 16 or ori_input_shape[-1] != 16:
        error_manager_vector.raise_err_specific_reson("max_pool_v3_grad", "C0 must be equal to 16.")

    if ori_output_shape[:2] != ori_input_shape[:2]:
        error_manager_vector.raise_err_specific_reson("max_pool_v3_grad", "N axis and C1 axis should be same.")

    if global_pooling:
        error_manager_vector.raise_err_specific_reson('max_pool_v3', 'can not support global now.')

    if len(pads) != 4:
        error_manager_vector.raise_err_input_param_not_in_range("max_pool_v3", "pads", "4", "4", str(len(pads)))
    if data_format in ("NHWC", "NC1HWC0", "NCHW"):
        if len(ksize) != 4:
            error_manager_vector.raise_err_input_param_not_in_range("max_pool_v3", "ksize", "4", "4", str(len(ksize)))

        if ksize[0] != 1 or ksize[3] != 1:
            error_manager_vector.raise_err_input_value_invalid("max_pool_v3", "ksize[0], ksize[3]",
                                                               "1", str(ksize[0]) + ", " + str(ksize[3]))
        if len(strides) != 4:
            error_manager_vector.raise_err_input_param_not_in_range("max_pool_v3", "strides",
                                                                    "4", "4", str(len(strides)))
        if strides[0] != 1 or strides[3] != 1:
            error_manager_vector.raise_err_input_value_invalid("max_pool_v3", "strides[0], strides[3]",
                                                               "1", str(strides[0]) + ", " + str(strides[3]))
        # not global mode, limit by load3d instruction.
        if ksize[1] > 255 or ksize[2] > 255 or ksize[1] < 1 or ksize[2] < 1:
            error_manager_vector.raise_err_input_param_not_in_range('max_pool_v3_grad', "ksize", 1, 255, ksize)
        if strides[1] > 63 or strides[2] > 63 or strides[1] < 1 or strides[2] < 1:
            error_manager_vector.raise_err_input_param_not_in_range('max_pool_v3_grad', "strides", 1, 63, strides)
    else:
        error_manager_vector.raise_err_input_format_invalid("max_pool_v3", "x", "NC1HWC0, NCHW, NHWC", str(data_format))


# 'pylint: disable=too-many-locals,too-many-return-statements
def branch_choice(ori_input_shape, ksize, strides, padding, pads, data_format, ceil_mode):
    """
    choose branch

    Parameters
    ----------
    ori_input_shape: list or tuple
        shape and data type of ori_input_shape
    ksize: list or tuple
        the size of the window
    strides: list or tuple
        the stride of the sliding window
    padding: str
        value from `SAME`, `VALID`, 'CALCULATED'
    pads: list or tuple
        padding size of height and width while padding is CALCULATED
    data_format: str
        format of input
    ceil_mode: boolean

    Returns
    -------
        atomic_flag
    """
    atomic_flag = False
    if data_format == "NCHW":
        ksize = (ksize[0], ksize[2], ksize[3], ksize[1])
        strides = (strides[0], strides[2], strides[3], strides[1])
    if len(ori_input_shape) == 5:
        _, _, fmap_h, fmap_w, _ = ori_input_shape
    elif data_format == "NHWC":
        _, fmap_h, fmap_w, _ = ori_input_shape
        ori_input_shape = ori_input_shape[0], (ori_input_shape[3] + 15) // 16, fmap_h, fmap_w, 16
    else:
        _, _, fmap_h, fmap_w = ori_input_shape
        ori_input_shape = ori_input_shape[0], (ori_input_shape[1] + 15) // 16, fmap_h, fmap_w, 16
    _, kernel_h, kernel_w, _ = ksize
    _, stride_h, stride_w, _ = strides
    if padding == 'VALID':
        ho = int(math.ceil((fmap_h - kernel_h + 1) * 1.0 / stride_h))
        wo = int(math.ceil((fmap_w - kernel_w + 1) * 1.0 / stride_w))
        pad_top = pad_left = pad_bottom = pad_right = 0

    if padding == 'SAME':
        ho = (fmap_h + stride_h - 1) // stride_h
        wo = (fmap_w + stride_w - 1) // stride_w
        pad_h = max((ho - 1) * stride_h + kernel_h - fmap_h, 0)
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_w = max((wo - 1) * stride_w + kernel_w - fmap_w, 0)
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

    if padding == 'CALCULATED':
        pad_top, pad_bottom, pad_left, pad_right = pads
        if ceil_mode:
            ho = (fmap_h - kernel_h + pad_top + pad_bottom + stride_h - 1) // stride_h + 1
            wo = (fmap_w - kernel_w + pad_left + pad_right + stride_w - 1) // stride_w + 1
            pad_bottom = (ho - 1) * stride_h + kernel_h - fmap_h - pad_top
            pad_right = (wo - 1) * stride_w + kernel_w - fmap_w - pad_left
        else:
            ho = (fmap_h - kernel_h + pad_top + pad_bottom) // stride_h + 1
            wo = (fmap_w - kernel_w + pad_left + pad_right) // stride_w + 1
            pad_bottom = max((ho - 1) * stride_h + kernel_h - fmap_h - pad_top, 0)
            pad_right = max((wo - 1) * stride_w + kernel_w - fmap_w - pad_left, 0)

    #large H support
    if fmap_h >= 200000:
        return True
    c0_local = ori_input_shape[-1]
    # each type of buffer's bit size
    fp16_data_size = tbe_platform.get_bit_len("float16") // 8
    fp32_data_size = tbe_platform.get_bit_len("float32") // 8
    uint16_data_size = tbe_platform.get_bit_len("uint16") // 8

    if ho != 1 and wo == 1:
        col2img_ub_shape = (max(kernel_h, stride_h), fmap_w, C0)
        if kernel_h > stride_h:
            temp_size = ((kernel_h - stride_h), fmap_w, C0)
        else:
            temp_size = (1, 16, C0)
        total_used_ub = cal_shape_ele(col2img_ub_shape) * (fp16_data_size + fp32_data_size) + cal_shape_ele(
            temp_size) * fp32_data_size
        remain_hi = (fmap_h - (ho - 1) * stride_h - pad_top) - max(kernel_h, stride_h)
        if remain_hi > 0:
            total_used_ub += cal_shape_ele((remain_hi, fmap_w * C0)) * fp16_data_size
        if total_used_ub > UB_SIZE:
            atomic_flag = True
        return atomic_flag

    wi_temp = fmap_w + pad_left + pad_right
    hi_temp = fmap_h + pad_top + pad_bottom
    if kernel_h > stride_h or stride_h * wi_temp * c0_local * (fp16_data_size + fp32_data_size) > L1_SIZE:
        each_process_hi = kernel_h
    else:
        each_process_hi = stride_h

    # calculate col ub size
    # There are two col ub, one is fp32, other is fp16, shape is (each_hi, each_wi, c0_local)
    # Here we need calculate each_wi to judge if need cut wo or cut ho.
    # `self.kw > self.stride_w, each_wi = (each_process_wo - 1) * self.stride_w + self.kw`
    # `self.kw <= self.stride_w, each_wi = each_process_wo * self.stride_w`
    col_size_times = each_process_hi * stride_w * c0_local * (fp16_data_size + fp32_data_size)
    col_size_const = each_process_hi * max(0,
                                            kernel_w - stride_w) * c0_local * (fp16_data_size + fp32_data_size)

    # calculate mask ub size
    # There are for mask buffer on ub, each is (math.ceil(each_wo_16 * c0_local // 128) * 128, )
    # Here each_wo_16 is ceil to 16 times.
    # Since it is not evenly divisible, consider the maximum possible value
    mask_size_times = uint16_data_size * 4
    mask_size_const = (MASK128_VALUE - 1) * uint16_data_size * 4

    # calculate tensor ub size
    # There are five tensor buffer on UB, each is (each_wo_16, 16, c0_local), one is fp32, others are fp16.
    # Here each_wo_16 is ceil to 16 times.
    # Since it is not evenly divisible, consider the maximum possible value
    tensor_size_times = c0_local * (4 * fp16_data_size + fp32_data_size)
    tensor_size_const = (c0_local - 1) * c0_local * (4 * fp16_data_size + fp32_data_size)

    # some temp size
    # At most have 3 temp buffer on UB, one is (128, ), one is (wi, c0_local), dtype is fp16
    temp_ub_size = fp16_data_size * fmap_w * c0_local + MASK128_VALUE * fp16_data_size
    # Tail block data may need dump 0 when last_valid_wi > each_process_wi
    # shape is ((last_valid_wi - each_process_wi) * c0_local, )
    temp_remain_size_const = (fmap_w - max(0, kernel_w - stride_w)) * c0_local * fp16_data_size

    # mode1: last_valid_wi > each_process_wi, need dump 0
    const_remain = UB_SIZE - temp_ub_size - temp_remain_size_const - tensor_size_const - mask_size_const - \
        col_size_const
    each_process_wo_mode1 = const_remain * 1.0 / (col_size_times + mask_size_times + tensor_size_times -
                                                    stride_w * c0_local * fp16_data_size)

    wo_mode1_effect = False
    if each_process_wo_mode1 == 0:
        wo_mode1_effect = False
    elif min(fmap_w - ((wo - 1) // each_process_wo_mode1 * each_process_wo_mode1 * stride_w - pad_left),
                fmap_w) > (stride_w * each_process_wo_mode1 + max(0, kernel_w - stride_w)):
        wo_mode1_effect = True
        if each_process_wo_mode1 >= 16:
            each_process_wo_mode1 = int(each_process_wo_mode1 // 16 * 16)
        else:
            each_process_wo_mode1 = int(each_process_wo_mode1)

    # mode2: last_valid_wi <= each_process_wi, no need to dump 0
    const_remain = UB_SIZE - temp_ub_size - tensor_size_const - mask_size_const - col_size_const
    each_process_wo_mode2 = const_remain * 1.0 / (col_size_times + mask_size_times + tensor_size_times)

    wo_mode2_effect = False
    if each_process_wo_mode2 == 0:
        wo_mode2_effect = False
    elif min(fmap_w - ((wo - 1) // each_process_wo_mode2 * each_process_wo_mode2 * stride_w - pad_left),
                fmap_w) <= (stride_w * each_process_wo_mode2 + max(0, kernel_w - stride_w)):
        wo_mode2_effect = True
        if each_process_wo_mode2 >= 16:
            each_process_wo_mode2 = int(each_process_wo_mode2 // 16 * 16)
        else:
            each_process_wo_mode2 = int(each_process_wo_mode2)

    each_process_wo_min = 0
    each_process_wo_max = 0

    if wo_mode1_effect and wo_mode2_effect:
        each_process_wo_min = min(each_process_wo_mode1, each_process_wo_mode2)
        each_process_wo_max = max(each_process_wo_mode1, each_process_wo_mode2)
    else:
        if wo_mode1_effect:
            each_process_wo_min = each_process_wo_mode1
            each_process_wo_max = each_process_wo_mode1
        if wo_mode2_effect:
            each_process_wo_min = each_process_wo_mode2
            each_process_wo_max = each_process_wo_mode2

    if each_process_wo_min < 1 or each_process_wo_max < 1:
        atomic_flag = True
        return atomic_flag

    if stride_h < kernel_h:
        overlap_l1_shape = (kernel_h - stride_h, (fmap_w + pad_left + pad_right)*C0)
        each_process_wi = max(kernel_w, stride_w)
        ori_l1_shape = (kernel_h, each_process_wi, C0)
        if cal_shape_ele(overlap_l1_shape) * fp32_data_size + cal_shape_ele(ori_l1_shape) * fp16_data_size > L1_SIZE:
            atomic_flag = True
            return atomic_flag

    if each_process_wo_min >= wo:
        wi = fmap_w + pad_left + pad_right

        # calculate col ub size
        # There are two col ub, one is fp32, other is fp16, shape is (each_hi, wi, c0_local)
        # `kernel_h > stride_h, each_hi = (each_process_ho - 1) * stride_h + kernel_h`
        # `kernel_h <= stride_h, each_hi = each_process_ho * stride_h`
        if stride_h * wi_temp * c0_local * (fp16_data_size + fp32_data_size) > L1_SIZE:
            col_size_times = kernel_h * wi * c0_local * (fp16_data_size + fp32_data_size)
        else:
            col_size_times = stride_h * wi * c0_local * (fp16_data_size + fp32_data_size)
        col_size_const = max(0, kernel_h - stride_h) * wi * c0_local * (fp16_data_size + fp32_data_size)

        # calculate mask ub size
        # There are for mask buffer on UB, each is (math.ceil(each_process_ho_wo_div16 * c0_local // 128) * 128, )
        # Here each_process_ho_wo_div16 is (each_process_ho * wo) ceil to 16 times.
        # Since it is not evenly divisible, consider the maximum possible value
        mask_size_times = wo * uint16_data_size * 4
        mask_size_const = (MASK128_VALUE - 1) * uint16_data_size * 4

        # calculate tensor ub size
        # There are five tensor buffer on UB, each is (each_process_ho_wo_div16, 16, c0_local), one is fp32,
        # others are fp16.
        # Here each_process_ho_wo_div16 is (each_process_ho * wo) ceil to 16 times.
        # Since it is not evenly divisible, consider the maximum possible value
        tensor_size_times = wo * c0_local * (4 * fp16_data_size + fp32_data_size)
        tensor_size_const = (c0_local - 1) * c0_local * (4 * fp16_data_size + fp32_data_size)

        # calculate temp tensor size
        # There is one temp tensor on UB, dtype is fp32
        # if kernel_h > stride_h, the shape is ((kernel_h - stride_h), wi, c0_local)
        # if kernel_h <= stride_h, the shape is (1, 16, c0_local)
        if kernel_h > stride_h:
            temp_tensor_size = (kernel_h - stride_h) * wi * c0_local * fp32_data_size
        else:
            temp_tensor_size = c0_local * c0_local * fp32_data_size

        # some temp size
        # one fixed temp buffer on UB, shape is (128, ), dtype is float16
        temp_ub_size = MASK128_VALUE * fp16_data_size

        each_process_ho = 0

        def _judge_last_and_process(each_process):
            if (ho - 1) // each_process == 0:
                return fmap_h > (stride_h * each_process + max(0, kernel_h - stride_h))
            return (fmap_h - ((ho - 1) // each_process * each_process * stride_h - pad_top)) > (
                stride_h * each_process + max(0, kernel_h - stride_h))

        # when there is no need to tile h to block, tail block data may need dump 0
        # mode1: last_valid_hi > each_process_hi
        # dump tensor shape is ((last_valid_hi - each_process_hi), fmap_w * c0_local)
        temp_remain_size_const = (fmap_h -
                                    max(0, kernel_h - stride_h)) * fmap_w * c0_local * fp16_data_size
        const_remain = UB_SIZE - temp_remain_size_const - temp_ub_size - temp_tensor_size - mask_size_const - \
            tensor_size_const - col_size_const
        each_process_ho_mode1 = const_remain // (col_size_times + mask_size_times + tensor_size_times -
                                                    stride_h * fmap_w * c0_local * fp16_data_size)

        ho_mode1_effect = False
        if each_process_ho_mode1 == 0:
            ho_mode1_effect = False
        elif _judge_last_and_process(each_process_ho_mode1):
            if each_process_ho_mode1 * stride_h >= pad_top or (ho - 1) // each_process_ho_mode1 == 0:
                ho_mode1_effect = True
        if not ho_mode1_effect:
            # when ((ho - 1) // each_process * each_process * stride_h - pad_top) < 0,
            # last_valid_hi > fmap_h
            # Since the value is uncertain, consider the maximum possible value
            temp_remain_size_const = (fmap_h + pad_top -
                                        max(0, kernel_h - stride_h)) * fmap_w * c0_local * fp16_data_size
            const_remain = UB_SIZE - temp_remain_size_const - temp_ub_size - temp_tensor_size - \
                mask_size_const - tensor_size_const - col_size_const
            each_process_ho_mode1 = const_remain // (col_size_times + mask_size_times + tensor_size_times -
                                                        stride_h * 2 * fmap_w * c0_local * fp16_data_size)
            if each_process_ho_mode1 == 0:
                ho_mode1_effect = False
            elif _judge_last_and_process(each_process_ho_mode1):
                if each_process_ho_mode1 * stride_h < pad_top and (ho -
                                                                   1) // each_process_ho_mode1 > 0:
                    ho_mode1_effect = True

        # mode2: last_valid_hi <= each_process_hi, no need to dump 0
        const_remain = UB_SIZE - temp_ub_size - temp_tensor_size - mask_size_const - tensor_size_const - \
            col_size_const
        each_process_ho_mode2 = const_remain // (col_size_times + mask_size_times + tensor_size_times)

        ho_mode2_effect = False
        if each_process_ho_mode2 == 0:
            ho_mode2_effect = False
        elif not _judge_last_and_process(each_process_ho_mode2):
            ho_mode2_effect = True

        if ho_mode1_effect and ho_mode2_effect:
            each_process_ho = max(each_process_ho_mode1, each_process_ho_mode2)
        else:
            if ho_mode1_effect:
                each_process_ho = each_process_ho_mode1
            if ho_mode2_effect:
                each_process_ho = each_process_ho_mode2

        if each_process_ho <= 0:
            each_process_ho = 1
        each_process_wo = wo
    else:
        each_process_ho = 0
        each_process_wo = each_process_wo_max

    if each_process_ho >= ho:
        l1_input_ori = (fmap_h, fmap_w, C0)
        col2img_ub_shape = (kernel_h, wi_temp, C0) if stride_h * fmap_w * C0 * 6 > L1_SIZE else (hi_temp, fmap_w, C0)
        ori_output_shape = (ceil_div(ho * wo, 16), 16, C0)
        bank_used_ub = 512 * 4 + 128 * 2
        if cal_shape_ele(l1_input_ori) * fp16_data_size > L1_SIZE:
            atomic_flag = True
        if cal_shape_ele(col2img_ub_shape) * (fp16_data_size * 2 + fp32_data_size) + cal_shape_ele(
                ori_output_shape) * fp16_data_size * 3 + bank_used_ub > UB_SIZE:
            atomic_flag = True
        return atomic_flag

    if each_process_ho > 1:
        if kernel_h > stride_h:
            each_process_hi = (each_process_ho - 1) * stride_h + kernel_h
        else:
            each_process_hi = each_process_ho * stride_h
        col2img_ub_shape = (each_process_hi, fmap_w, C0)
        cut_ho_nums = ho // each_process_ho
        last_valid_hi = hi_temp - (cut_ho_nums * each_process_ho * stride_h - pad_top)
        extra_ub = 512 * 3 + 256
        temp_process_hi = kernel_h
        if last_valid_hi > temp_process_hi:
            remain_hi = last_valid_hi - temp_process_hi
            extra_ub += cal_shape_ele((remain_hi, wi_temp * C0)) * fp16_data_size
        if kernel_h > stride_h:
            temp_size = ((kernel_h - stride_h), wi_temp, C0)
            extra_ub += cal_shape_ele(temp_size) * fp32_data_size
        if cal_shape_ele(col2img_ub_shape) * (fp16_data_size + fp32_data_size) + extra_ub > UB_SIZE:
            atomic_flag = True
        return atomic_flag

    if stride_w >= kernel_w:
        col_w = each_process_wo * stride_w
    else:
        col_w = (each_process_wo - 1) * stride_w + kernel_w

    col_ub_shape = (max(stride_h, kernel_h), col_w, C0)
    if stride_h < kernel_h:
        overlap_l1_shape = (kernel_h - stride_h, (fmap_w + pad_left + pad_right) * C0)
        if cal_shape_ele(overlap_l1_shape) * fp32_data_size + cal_shape_ele(col_ub_shape) * fp16_data_size > L1_SIZE:
            atomic_flag = True
    elif cal_shape_ele(col_ub_shape) * fp16_data_size > L1_SIZE:
        atomic_flag = True
    if cal_shape_ele(col_ub_shape) * (fp16_data_size + fp32_data_size) + cal_shape_ele(
            (fmap_w, C0)) * fp16_data_size > UB_SIZE:
        atomic_flag = True
    return atomic_flag


# 'pylint: disable=dangerous-default-value,too-many-locals,
# 'pylint: disable=too-many-arguments,,unused-argument,invalid-name
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_LIST_INT,
                            para_check.REQUIRED_ATTR_LIST_INT,
                            para_check.OPTION_ATTR_STR,
                            para_check.OPTION_ATTR_LIST_INT,
                            para_check.OPTION_ATTR_STR,
                            para_check.OPTION_ATTR_BOOL,
                            para_check.OPTION_ATTR_BOOL,
                            para_check.KERNEL_NAME)
def max_pool_v3_grad(ori_input,
                     ori_output,
                     grad,
                     y,
                     ksize,
                     strides,
                     padding="CALCULATED",
                     pads=[0, 0, 0, 0],
                     data_format="NCHW",
                     global_pooling=False,
                     ceil_mode=False,
                     kernel_name="max_pool_v3_grad"):
    """
    main function of max_pool_v3_grad

    Parameters
    ----------
    ori_input: dict
        shape and data type of ori_input
    ori_output: dict
        shape and data type of ori_output
    grad: dict
        shape and data type of grad
    y: dict
        shape and data type of y
    ksize: list or tuple
        the size of the window
    strides: list or tuple
        the stride of the sliding window
    padding: str
        value from `SAME`, `VALID`, 'CALCULATED'
    pads: list or tuple
        padding size of height and width while padding is CALCULATED
    data_format: str
        value from `NCHW`, `NHWC`
    global_pooling: bool
        if true, ksize and padding will be invalid
    ceil_mode: bool
        whether use ceil fuction to calculate the output of height and width
    kernel_name: str

    Returns
    -------
    return the tik api function
    """
    ori_input_shape = ori_input.get("shape")
    ori_format = ori_input.get("ori_format")
    atomic_flag = branch_choice(ori_input_shape, ksize, strides, padding, pads, ori_format, ceil_mode)
    if atomic_flag:
        forward_in_shape = list(ori_input.get("shape"))
        forward_ou_shape = list(ori_output.get("shape"))
        grad_shape = list(grad.get("shape"))
        if ori_format == "NCHW":
            ksize = (ksize[0], 1, ksize[2], ksize[3], ksize[1])
            strides = (strides[0], 1, strides[2], strides[3], strides[1])
        else:
            ksize = (ksize[0], 1, ksize[1], ksize[2], ksize[3])
            strides = (strides[0], 1, strides[1], strides[2], strides[3])
        check_param(ori_input, ori_output, grad, [ksize[0], ksize[2], ksize[3], ksize[4]], [
                     strides[0], strides[2], strides[3], strides[4]], padding,
                     data_format, global_pooling, pads, kernel_name)
        forward_in_shape.insert(1, 1)
        forward_ou_shape.insert(1, 1)
        grad_shape.insert(1, 1)
        dtype = ori_input.get("dtype").lower()
        ksize = list(ksize)
        strides = list(strides)
        shape_list = [forward_in_shape, forward_ou_shape, grad_shape, forward_in_shape]
        params = [ksize, strides, [0, 0, 0, 0, 0, 0], dtype, kernel_name, padding]
        if tbe_platform.api_check_support("tik.load3dv1"):
            result = MaxpoolV3GradAtomic(shape_list, params)
            return result.get_tik_instance()
        else:
            result = MaxpoolGradAtomic(shape_list, params)
            return result.get_tik_instance()
    else:
        if ori_format == "NCHW":
            ksize = (ksize[0], ksize[2], ksize[3], ksize[1])
            strides = (strides[0], strides[2], strides[3], strides[1])
        check_param(ori_input, ori_output, grad, ksize, strides, padding,
                    data_format, global_pooling, pads, kernel_name)
        ori_input_shape = ori_input.get("shape")
        ori_output_shape = ori_output.get("shape")
        grad_shape = grad.get("shape")
        dtype = ori_input.get("dtype").lower()
        if tbe_platform.api_check_support("tik.load3dv1"):
            maxpoolv3grad = MaxpoolV3Grad(ori_input_shape, ori_output_shape, grad_shape,
                                          dtype, ksize, strides, padding, pads, global_pooling, ceil_mode)
            return maxpoolv3grad.tik_instance_function(kernel_name)
        else:
            maxpoolv3grad = MaxpoolGrad(ori_input_shape, ori_output_shape, grad_shape,
                                        dtype, ksize, strides, padding, pads, global_pooling, ceil_mode)
            return maxpoolv3grad.tik_instance_function(kernel_name)


# 'pylint: disable = too-many-instance-attributes,too-few-public-methods
class MaxpoolV3Grad():
    """
    MaxpoolV3Grad  Object include all fuction and paras
    """

    def __init__(self, ori_input, ori_output, grad, dtype, ksize, strides,
                 padding, pads, global_pooling, ceil_mode):
        self.ori_input_shape = ori_input
        self.ori_output_shape = ori_output
        self.grad_shape = grad
        self.dtype = dtype
        self.ksize = ksize
        self.strides = strides
        self.padding = padding
        self.pads = pads
        self.global_pooling = global_pooling
        self.ceil_mode = ceil_mode
        self.n = ori_input[0]
        self.c1 = ori_input[1]
        self.hi = ori_input[2]
        self.wi = ori_input[3]

        self.ho = grad[2]
        self.wo = grad[3]
        self.stride_h = strides[1]
        self.stride_w = strides[2]
        self.kh = ksize[1]
        self.kw = ksize[2]
        self.pad = None
        self.pad_top = None
        self.pad_bottom = None
        self.pad_left = None
        self.pad_right = None
        self.pad_value = None
        self.ho_block = None
        self.hi_block = None
        self.tile_h_to_block = False

        self.tik_instance = tik.Tik()
        self.ori_input_gm = self.tik_instance.Tensor(dtype,
                                                     self.ori_input_shape,
                                                     name='ori_input_gm',
                                                     scope=tik.scope_gm)
        self.ori_output_gm = self.tik_instance.Tensor(dtype,
                                                      self.ori_output_shape,
                                                      name='ori_output_gm',
                                                      scope=tik.scope_gm)
        self.grad_gm = self.tik_instance.Tensor(dtype, self.grad_shape,
                                                name='grad_gm',
                                                scope=tik.scope_gm)
        self.res_gm = self.tik_instance.Tensor(dtype, (
            self.n, self.c1, self.hi, self.wi, C0), name='res_gm',
                                               scope=tik.scope_gm)

        self.scalar_esp = self.tik_instance.Scalar(dtype='float32',
                                                   name='scalar_esp')
        self.scalar_one = self.tik_instance.Scalar(dtype='float32',
                                                   name='scalar_one')
        self.scalar_zero = self.tik_instance.Scalar(dtype='float32',
                                                    name='scalar_zero')
        self.scalar_zero_fp16 = self.tik_instance.Scalar(dtype='float16',
                                                         name='scalar_zero_fp16')
        self.offset_gm = self.tik_instance.Scalar(dtype='int64',
                                                  name='offset_gm')
        self.actual_pad_top = self.tik_instance.Scalar(dtype='int64',
                                                       name='actual_pad_top')
        self.actual_pad_bottom = self.tik_instance.Scalar(dtype='int64',
                                                          name='actual_pad_bottom')
        self.row_effective = self.tik_instance.Scalar(dtype='int64',
                                                      name='row_effective')
        # define some sclar
        self.scalar_zero_fp16.set_as(0)
        self.scalar_zero.set_as(0)
        self.scalar_esp.set_as(1.18e-38)
        self.scalar_one.set_as(1)

        self.check_load3d_support = tbe_platform.api_check_support("tik.load3dv1")

    # 'pylint: disable = too-many-locals
    def _padding_mode(self, ori_input, ksize, strides, padding, pads, global_pooling, ceil_mod):
        _, _, fmap_h, fmap_w, _ = ori_input
        _, kernel_h, kernel_w, _ = ksize
        _, self.stride_h, self.stride_w, _ = strides
        if global_pooling:
            kernel_h = fmap_h
            kernel_w = fmap_w
            padding = 'VALID'
        if padding == 'VALID':
            ho = int(math.ceil((fmap_h - kernel_h + 1) * 1.0 / self.stride_h))
            wo = int(math.ceil((fmap_w - kernel_w + 1) * 1.0 / self.stride_w))
            pad_top = pad_left = pad_bottom = pad_right = 0

        if padding == 'SAME':
            ho = (fmap_h + self.stride_h - 1) // self.stride_h
            wo = (fmap_w + self.stride_w - 1) // self.stride_w
            pad_h = max((ho - 1) * self.stride_h + kernel_h - fmap_h, 0)
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_w = max((wo - 1) * self.stride_w + kernel_w - fmap_w, 0)
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left

        if padding == 'CALCULATED':
            pad_top, pad_bottom, pad_left, pad_right = pads
            if ceil_mod:
                ho = (fmap_h - kernel_h + pad_top + pad_bottom + self.stride_h - 1) // self.stride_h + 1
                wo = (fmap_w - kernel_w + pad_left + pad_right + self.stride_w - 1) // self.stride_w + 1
                pad_bottom = (ho - 1) * self.stride_h + kernel_h - fmap_h - pad_top
                pad_right = (wo - 1) * self.stride_w + kernel_w - fmap_w - pad_left
            else:
                ho = (fmap_h - kernel_h + pad_top + pad_bottom) // self.stride_h + 1
                wo = (fmap_w - kernel_w + pad_left + pad_right) // self.stride_w + 1
                pad_bottom = max((ho - 1) * self.stride_h + kernel_h - fmap_h - pad_top, 0)
                pad_right = max((wo - 1) * self.stride_w + kernel_w - fmap_w - pad_left, 0)
        pads_result = (pad_left, pad_right, pad_top, pad_bottom)

        return wo, ho, pads_result

    def _vector_dup(self, src, src_start, shape, dup_reg, dtype):
        ele_num = cal_shape_ele(shape)
        if dtype == "float16":
            total_repeate_time = ele_num // VECTOR_FP16_SIZE
            remain_ele = ele_num % VECTOR_FP16_SIZE
            mask_value = VECTOR_FP16_SIZE
        elif dtype == "float32":
            total_repeate_time = ele_num // VECTOR_FP32_SIZE
            remain_ele = ele_num % VECTOR_FP32_SIZE
            mask_value = VECTOR_FP32_SIZE
        else:
            error_manager_vector.raise_err_input_dtype_not_supported("max_pool_v3", "dtype",
                                                                     "float16, float32", str(dtype))
        repeate_max_time = total_repeate_time // MAX_VECTOR_REPEATE_TIME
        remain_repeate_time = total_repeate_time % MAX_VECTOR_REPEATE_TIME

        if repeate_max_time > 0:
            with self.tik_instance.for_range(0, repeate_max_time) as loop1:
                self.tik_instance.vector_dup(mask_value, src[
                    src_start + loop1 * MAX_VECTOR_REPEATE_TIME * mask_value],
                                             dup_reg, MAX_VECTOR_REPEATE_TIME,
                                             1, 8)
        if remain_repeate_time > 0:
            self.tik_instance.vector_dup(mask_value, src[
                src_start + repeate_max_time * MAX_VECTOR_REPEATE_TIME * mask_value],
                                         dup_reg, remain_repeate_time, 1, 8)
        if remain_ele > 0:
            self.tik_instance.vector_dup(remain_ele, src[
                src_start + repeate_max_time * MAX_VECTOR_REPEATE_TIME * mask_value +
                remain_repeate_time * mask_value], dup_reg, 1, 1, 8)

    def _vconv(self, src, src_start, dst, dst_start, ele_num, src_dtype):
        total_repeate_time = ele_num // VECTOR_FP32_SIZE
        remain_ele = ele_num % VECTOR_FP32_SIZE
        mask_value = VECTOR_FP32_SIZE

        repeate_max_time = total_repeate_time // MAX_VECTOR_REPEATE_TIME
        remain_repeate_time = total_repeate_time % MAX_VECTOR_REPEATE_TIME

        if src_dtype == 'float16':
            src_stride, dst_stride = 4, 8
            if repeate_max_time > 0:
                with self.tik_instance.for_range(0, repeate_max_time) as loop1:
                    self.tik_instance.vconv(
                        MASK64_VALUE, "",
                        dst[
                            dst_start + loop1 * MAX_VECTOR_REPEATE_TIME * mask_value],
                        src[
                            src_start + loop1 * MAX_VECTOR_REPEATE_TIME * mask_value],
                        MAX_VECTOR_REPEATE_TIME, 1, 1, dst_stride, src_stride)
            if remain_repeate_time > 0:
                self.tik_instance.vconv(
                    MASK64_VALUE, "",
                    dst[
                        dst_start + repeate_max_time * MAX_VECTOR_REPEATE_TIME * mask_value],
                    src[
                        src_start + repeate_max_time * MAX_VECTOR_REPEATE_TIME * mask_value],
                    remain_repeate_time, 1, 1, dst_stride, src_stride)
            if remain_ele > 0:
                self.tik_instance.vconv(
                    remain_ele, "",
                    dst[dst_start + repeate_max_time * MAX_VECTOR_REPEATE_TIME *
                        mask_value + remain_repeate_time * mask_value],
                    src[src_start + repeate_max_time * MAX_VECTOR_REPEATE_TIME *
                        mask_value + remain_repeate_time * mask_value],
                    1, 1, 1, dst_stride, src_stride)
        else:
            src_stride, dst_stride = 8, 4
            if repeate_max_time > 0:
                with self.tik_instance.for_range(0, repeate_max_time) as loop1:
                    self.tik_instance.vconv(
                        MASK64_VALUE, "",
                        dst[
                            dst_start + loop1 * MAX_VECTOR_REPEATE_TIME * mask_value],
                        src[
                            src_start + loop1 * MAX_VECTOR_REPEATE_TIME * mask_value],
                        MAX_VECTOR_REPEATE_TIME, 1, 1, dst_stride, src_stride)
            if remain_repeate_time > 0:
                self.tik_instance.vconv(
                    MASK64_VALUE, "",
                    dst[
                        dst_start + repeate_max_time * MAX_VECTOR_REPEATE_TIME * mask_value],
                    src[
                        src_start + repeate_max_time * MAX_VECTOR_REPEATE_TIME * mask_value],
                    remain_repeate_time, 1, 1, dst_stride, src_stride)
            if remain_ele > 0:
                self.tik_instance.vconv(
                    remain_ele, "",
                    dst[dst_start + repeate_max_time * MAX_VECTOR_REPEATE_TIME *
                        mask_value + remain_repeate_time * mask_value],
                    src[src_start + repeate_max_time * MAX_VECTOR_REPEATE_TIME *
                        mask_value + remain_repeate_time * mask_value],
                    1, 1, 1, dst_stride, src_stride)

    def _vector_op(self, operator, src1, src2, dst, dtype, ele_num, stride_cofig=None, offset=None):
        if dtype == "float16":
            repeate_times = ele_num // VECTOR_FP16_SIZE
            remain_ele = ele_num % VECTOR_FP16_SIZE
            mask = VECTOR_FP16_SIZE
        else:
            repeate_times = ele_num // VECTOR_FP32_SIZE
            remain_ele = ele_num % VECTOR_FP32_SIZE
            mask = VECTOR_FP32_SIZE
        repeat_max_loop = repeate_times // MAX_VECTOR_REPEATE_TIME
        remain_max_loop = repeate_times % MAX_VECTOR_REPEATE_TIME
        if operator == "vmuls":
            if offset:
                dst_offset = offset[0]
                src1_offset = offset[1]
            else:
                dst_offset = 0
                src1_offset = 0
            if stride_cofig is None:
                stride_cofig = 1, 1, 8, 8
            if repeat_max_loop > 0:
                self.tik_instance.vmuls(mask, dst[dst_offset], src1[src1_offset],
                                        src2, 255,
                                        stride_cofig[0], stride_cofig[1],
                                        stride_cofig[2], stride_cofig[3])
                dst_offset += BLOCK_SIZE // (tbe_platform.get_bit_len(
                    dst.dtype.lower()) // 8) * stride_cofig[2] * 255
                src1_offset += BLOCK_SIZE // (tbe_platform.get_bit_len(
                    src1.dtype.lower()) // 8) * stride_cofig[3] * 255
            if remain_max_loop > 0:
                self.tik_instance.vmuls(mask, dst[dst_offset], src1[src1_offset],
                                        src2, remain_max_loop,
                                        stride_cofig[0], stride_cofig[1],
                                        stride_cofig[2], stride_cofig[3])
                dst_offset += BLOCK_SIZE // (tbe_platform.get_bit_len(
                    dst.dtype.lower()) // 8) * stride_cofig[2] * remain_max_loop
                src1_offset += BLOCK_SIZE // (tbe_platform.get_bit_len(
                    src1.dtype.lower()) // 8) * stride_cofig[3] * remain_max_loop
            if remain_ele > 0:
                self.tik_instance.vmuls(remain_ele, dst[dst_offset], src1[src1_offset],
                                        src2, 1,
                                        stride_cofig[0], stride_cofig[1],
                                        stride_cofig[2], stride_cofig[3])
        if operator == "vadd":
            if stride_cofig is None:
                stride_cofig = 1, 1, 1, 8, 8, 8
            dst_offset = 0
            src1_offset = 0
            src2_offset = 0
            if stride_cofig[3] > 255 or stride_cofig[4] > 255 or stride_cofig[5] > 255:
                dst_offset_stride = BLOCK_SIZE // (tbe_platform.get_bit_len(dst.dtype.lower()) // 8) * stride_cofig[3]
                src1_offset_stride = BLOCK_SIZE // (tbe_platform.get_bit_len(src1.dtype.lower()) // 8) * stride_cofig[4]
                src2_offset_stride = BLOCK_SIZE // (tbe_platform.get_bit_len(src2.dtype.lower()) // 8) * stride_cofig[5]
                if repeat_max_loop > 0:
                    with self.tik_instance.for_range(0, 255, thread_num=1) as loop_idx:
                        self.tik_instance.vadd(mask, dst[dst_offset + dst_offset_stride * loop_idx],
                                               src1[src1_offset + src1_offset_stride * loop_idx],
                                               src2[src2_offset + src2_offset_stride * loop_idx], 1,
                                               stride_cofig[0], stride_cofig[1], stride_cofig[2], 8, 8, 8)
                    dst_offset += dst_offset_stride * 255
                    src1_offset += src1_offset_stride * 255
                    src2_offset += src2_offset_stride * 255
                if remain_max_loop > 0:
                    with self.tik_instance.for_range(0, remain_max_loop, thread_num=1) as loop_idx:
                        self.tik_instance.vadd(mask, dst[dst_offset + dst_offset_stride * loop_idx],
                                               src1[src1_offset + src1_offset_stride * loop_idx],
                                               src2[src2_offset + src2_offset_stride * loop_idx], 1,
                                               stride_cofig[0], stride_cofig[1], stride_cofig[2], 8, 8, 8)
                    dst_offset += dst_offset_stride * remain_max_loop
                    src1_offset += src1_offset_stride * remain_max_loop
                    src2_offset += src2_offset_stride * remain_max_loop
                if remain_ele > 0:
                    self.tik_instance.vadd(remain_ele, dst[dst_offset], src1[src1_offset], src2[src2_offset], 1,
                                           stride_cofig[0], stride_cofig[1], stride_cofig[2], 8, 8, 8)
            else:
                if repeat_max_loop > 0:
                    self.tik_instance.vadd(mask, dst[dst_offset], src1[src1_offset], src2[src2_offset],
                                          255,
                                          stride_cofig[0], stride_cofig[1],
                                          stride_cofig[2], stride_cofig[3],
                                          stride_cofig[4], stride_cofig[5])
                    dst_offset += BLOCK_SIZE // (tbe_platform.get_bit_len(
                        dst.dtype.lower()) // 8) * stride_cofig[3] * 255
                    src1_offset += BLOCK_SIZE // (tbe_platform.get_bit_len(
                        src1.dtype.lower()) // 8) * stride_cofig[4] * 255
                    src2_offset += BLOCK_SIZE // (tbe_platform.get_bit_len(
                        src2.dtype.lower()) // 8) * stride_cofig[5] * 255
                if remain_max_loop > 0:
                    self.tik_instance.vadd(mask, dst[dst_offset], src1[src1_offset], src2[src2_offset],
                                          remain_max_loop,
                                          stride_cofig[0], stride_cofig[1],
                                          stride_cofig[2], stride_cofig[3],
                                          stride_cofig[4], stride_cofig[5])
                    dst_offset += BLOCK_SIZE // (tbe_platform.get_bit_len(
                        dst.dtype.lower()) // 8) * stride_cofig[3] * remain_max_loop
                    src1_offset += BLOCK_SIZE // (tbe_platform.get_bit_len(
                        src1.dtype.lower()) // 8) * stride_cofig[4] * remain_max_loop
                    src2_offset += BLOCK_SIZE // (tbe_platform.get_bit_len(
                        src2.dtype.lower()) // 8) * stride_cofig[5] * remain_max_loop
                if remain_ele > 0:
                    self.tik_instance.vadd(remain_ele, dst[dst_offset], src1[src1_offset],
                                          src2[src2_offset], 1,
                                          stride_cofig[0], stride_cofig[1],
                                          stride_cofig[2], stride_cofig[3],
                                          stride_cofig[4], stride_cofig[5])

    def _calc_mask(self, index_h, index_w, mask_shape, ori_output_ub, ori_input_col_ub,
                   mask_or, mask_not):
        mask_ori = self.tik_instance.Tensor('uint16', mask_shape,
                                            name='mask_ori',
                                            scope=tik.scope_ubuf)
        mask_ub = self.tik_instance.Tensor('uint16', mask_shape,
                                           name='mask_ori',
                                           scope=tik.scope_ubuf)

        with self.tik_instance.if_scope(tik.all(index_h == 0, index_w == 0)):
            self.tik_instance.vcmpv_eq(mask_ub, ori_output_ub, ori_input_col_ub,
                                       cal_shape_ele(ori_output_ub.shape) // VECTOR_FP16_SIZE,
                                       1, 1, 8, 8)

            self.tik_instance.data_move(mask_or[0],
                                        mask_ub[0], 0, 1,
                                        cal_shape_ele(mask_ub.shape) // 16, 0, 0)

            self.tik_instance.vnot(MASK128_VALUE, mask_not, mask_ub,
                                   cal_shape_ele(mask_ub.shape) // VECTOR_FP16_SIZE,
                                   1, 1, 8, 8)

        with self.tik_instance.else_scope():
            self.tik_instance.vcmpv_eq(mask_ori, ori_output_ub, ori_input_col_ub,
                                       cal_shape_ele(ori_output_ub.shape) // VECTOR_FP16_SIZE,
                                       1, 1, 8, 8)

            mask_ori = mask_ori.reinterpret_cast_to("uint16")
            self.tik_instance.vand(MASK128_VALUE, mask_ub, mask_not, mask_ori,
                                   cal_shape_ele(mask_ub.shape) // VECTOR_FP16_SIZE,
                                   1, 1, 1, 8, 8, 8)

            mask_or = mask_or.reinterpret_cast_to("uint16")
            self.tik_instance.vor(MASK128_VALUE, mask_or, mask_or, mask_ub,
                                  cal_shape_ele(mask_ub.shape) // VECTOR_FP16_SIZE,
                                  1, 1, 1, 8, 8, 8)
            self.tik_instance.vnot(MASK128_VALUE, mask_not, mask_or,
                                   cal_shape_ele(mask_ub.shape) // VECTOR_FP16_SIZE,
                                   1, 1, 8, 8)

        return mask_ub

    def _vsel_grad_col(self, mask_ub, grad_ub):
        grad_sel_ub = self.tik_instance.Tensor("float16", grad_ub.shape,
                                               name="col2img_fp16_ub", scope=tik.scope_ubuf)
        temp_zero = self.tik_instance.Tensor("float16", (MASK128_VALUE,),
                                             name="temp_zero", scope=tik.scope_ubuf)
        self._vector_dup(temp_zero, 0, temp_zero.shape, self.scalar_zero_fp16, "float16")

        # vsel
        with self.tik_instance.for_range(0, grad_ub.shape[0]) as mask_index:
            fractal_repeat = C0 * C0 // VECTOR_FP16_SIZE
            with self.tik_instance.for_range(0, fractal_repeat) as fractal_index:
                mask_type_bit_size = tbe_platform.get_bit_len("uint16")
                mask_offset = (mask_index * fractal_repeat + fractal_index) * \
                              MASK128_VALUE // mask_type_bit_size

                cmpmask = self.tik_instance.mov_tensor_to_cmpmask(mask_ub[mask_offset])
                grad_ub_offset = (mask_index * fractal_repeat + fractal_index) * MASK128_VALUE
                self.tik_instance.vsel(MASK128_VALUE, 0, grad_sel_ub[grad_ub_offset], cmpmask,
                                       grad_ub[grad_ub_offset], temp_zero,
                                       1, 1, 1, 1, 8, 8, 8)

        return grad_sel_ub

    # 'pylint: disable=unused-variable
    def _load3d(self, index_h, index_w, start_h, end_h, ori_input_col_ub,
                ori_input_l1, start_pos_h, each_process_hi, each_process_wi,
                repeat_times, pad, pad_value, wo_offset, each_process_hi_block):
        """
        load3d function

        Parameters
        index_h: scalar
            start pos of kernel
        index_w: scalar
            start pos of kernel
        start_h: scalar
            start row number of Ho
        end_h: scalar
            end row number of Ho
        ori_input_col_ub: tensor
            col ub
        ori_input_l1: tensor
            ori input tensor on L1
        start_pos_h:
            start pos of row number on Ho
        each_process_hi: int
            number of rows on Hi processed each loop
        each_process_wi: int
            number of rows on Wi processed each loop
        repeat_times: int
            repeate times of doing load3d
        pad: list
            pad value
        wo_offset: scalar
            offset on W direction
        ----------
        Returns
        ori_input_col_ub: tensor
            load3d's result
        -------
        """
        pad_left, pad_right, pad_top, pad_bottom = pad
        if not self.check_load3d_support:
            self._vector_dup(ori_input_col_ub, 0, ori_input_col_ub.shape, MIN_VALUE_FP16, "float16")
        # load3d
        with self.tik_instance.if_scope(start_h <= pad_top):
            self.actual_pad_top.set_as(pad_top - start_h)
            with self.tik_instance.if_scope(end_h < each_process_hi_block + pad_top):
                if self.check_load3d_support:
                    self.tik_instance.load3dv1(ori_input_col_ub[0],
                                               ori_input_l1[0],
                                               (pad_left, pad_right,
                                               self.actual_pad_top, 0),
                                               each_process_hi - self.actual_pad_top,
                                               each_process_wi, 0, index_w, index_h,
                                               wo_offset, -self.actual_pad_top,
                                               self.stride_w, self.stride_h,
                                               self.kw, self.kh, 1, 1, 1, 1,
                                               repeat_times, 0, pad_value)
                else:
                    img2col(self.tik_instance, ori_input_l1, ori_input_col_ub, 0, 0, index_h, index_w,
                            -self.actual_pad_top, wo_offset, each_process_hi - self.actual_pad_top, each_process_wi,
                            self.kh, self.kw, self.stride_h, self.stride_w, repeat_times, 1,
                            (pad_left, pad_right, self.actual_pad_top, 0))
            with self.tik_instance.else_scope():
                self.actual_pad_bottom.set_as(end_h - each_process_hi_block - pad_top)
                if self.check_load3d_support:
                    self.tik_instance.load3dv1(ori_input_col_ub[0],
                                               ori_input_l1[0],
                                               (pad_left, pad_right,
                                               self.actual_pad_top,
                                               self.actual_pad_bottom),
                                               each_process_hi -
                                               self.actual_pad_top -
                                               self.actual_pad_bottom,
                                               each_process_wi, 0, index_w, index_h,
                                               wo_offset, -self.actual_pad_top,
                                               self.stride_w, self.stride_h,
                                               self.kw, self.kh, 1, 1, 1, 1,
                                               repeat_times,
                                               0, pad_value)
                else:
                    img2col(self.tik_instance, ori_input_l1, ori_input_col_ub, 0, 0, index_h, index_w,
                            -self.actual_pad_top, wo_offset,
                            each_process_hi - self.actual_pad_top - self.actual_pad_bottom, each_process_wi,
                            self.kh, self.kw, self.stride_h, self.stride_w, repeat_times, 1,
                            (pad_left, pad_right, self.actual_pad_top, self.actual_pad_bottom))
        with self.tik_instance.else_scope():
            with self.tik_instance.if_scope(end_h < each_process_hi_block + pad_top):
                if self.check_load3d_support:
                    self.tik_instance.load3dv1(ori_input_col_ub[0],
                                               ori_input_l1[
                                                   start_pos_h * self.wi * C0],
                                               (pad_left, pad_right, 0, 0),
                                               each_process_hi, each_process_wi, 0,
                                               index_w, index_h, wo_offset, 0,
                                               self.stride_w, self.stride_h,
                                               self.kw, self.kh, 1, 1, 1, 1,
                                               repeat_times, 0, pad_value)
                else:
                    img2col(self.tik_instance, ori_input_l1, ori_input_col_ub, start_pos_h * self.wi * C0, 0,
                            index_h, index_w, 0, wo_offset, each_process_hi, each_process_wi,
                            self.kh, self.kw, self.stride_h, self.stride_w, repeat_times, 1,
                            (pad_left, pad_right, 0, 0))

            with self.tik_instance.else_scope():
                self.actual_pad_bottom.set_as(end_h - each_process_hi_block - pad_top)
                if self.check_load3d_support:
                    self.tik_instance.load3dv1(ori_input_col_ub[0],
                                               ori_input_l1[
                                                   start_pos_h * self.wi * C0],
                                               (pad_left, pad_right, 0,
                                                self.actual_pad_bottom),
                                               each_process_hi - self.actual_pad_bottom,
                                               each_process_wi, 0,
                                               index_w, index_h, wo_offset, 0,
                                               self.stride_w, self.stride_h,
                                               self.kw, self.kh, 1, 1, 1, 1,
                                               repeat_times, 0, pad_value)
                else:
                    img2col(self.tik_instance, ori_input_l1, ori_input_col_ub, start_pos_h * self.wi * C0, 0,
                            index_h, index_w, 0, wo_offset, each_process_hi - self.actual_pad_bottom, each_process_wi,
                            self.kh, self.kw, self.stride_h, self.stride_w, repeat_times, 1,
                            (pad_left, pad_right, 0, self.actual_pad_bottom))

        return ori_input_col_ub

    def _data_move_ub(self, ori_input_shape, ori_output_shape, input_data_num,
                      output_data_nums, src_input_offset, src_output_offset):
        if ori_input_shape:
            ori_input_l1 = self.tik_instance.Tensor(self.dtype, ori_input_shape,
                                                    name='ori_input_l1',
                                                    scope=tik.scope_cbuf)
            self.tik_instance.data_move(ori_input_l1[0],
                                        self.ori_input_gm[src_input_offset], 0, 1,
                                        input_data_num // 16, 0, 0)
        else:
            ori_input_l1 = None

        # mov actual ori output to ub

        ori_output_ub = self.tik_instance.Tensor(self.dtype, ori_output_shape,
                                                 name='ori_output_ub',
                                                 scope=tik.scope_ubuf)
        self._vector_dup(ori_output_ub, 0, ori_output_shape,
                         self.scalar_zero_fp16, "float16")
        self.tik_instance.data_move(ori_output_ub[0],
                                    self.ori_output_gm[src_output_offset],
                                    0, 1, output_data_nums // 16, 0, 0)

        # mov ori grad to ub
        grad_ub = self.tik_instance.Tensor(self.dtype, ori_output_shape,
                                           name='grad_ub', scope=tik.scope_ubuf)
        self._vector_dup(grad_ub, 0, ori_output_shape,
                         self.scalar_zero_fp16, "float16")
        self.tik_instance.data_move(grad_ub[0],
                                    self.grad_gm[src_output_offset],
                                    0, 1, output_data_nums // 16, 0, 0)

        return ori_input_l1, ori_output_ub, grad_ub

    # 'pylint: disable=unused-variable
    def _mov_func(self, cut_ho_nums_index, cut_ho_nums, remain_ho_nums, each_process_ho,
                  each_process_hi, each_valid_ho, col2img_fp32_ub, temp_tensor_ub, pad):
        pad_left, pad_right, pad_top, pad_bottom = pad
        wi = self.wi + pad_left + pad_right
        pad_top_rows = self.tik_instance.Scalar(dtype="int64", name='pad_top_rows')
        pad_top_rows.set_as(pad_top - cut_ho_nums_index * each_process_ho * self.stride_h)
        self.tik_instance.scalar_max(pad_top_rows, pad_top_rows, 0)
        each_valid_hi = each_valid_ho * self.stride_h - pad_top_rows

        col2img_fp16_ub = self.tik_instance.Tensor("float16",
                                                   col2img_fp32_ub.shape,
                                                   name="col2img_fp16_ub",
                                                   scope=tik.scope_ubuf)
        self._vconv(col2img_fp32_ub, 0, col2img_fp16_ub, 0,
                    cal_shape_ele(col2img_fp32_ub.shape), "float32")

        with self.tik_instance.if_scope(
                tik.all(cut_ho_nums_index < cut_ho_nums - 1, each_valid_hi > 0)):
            self.tik_instance.data_move(self.res_gm[self.offset_gm],
                                        col2img_fp16_ub[pad_top_rows * wi * C0 + pad_left * C0], 0,
                                        each_valid_hi, self.wi * C0 // 16,
                                        pad_left + pad_right, 0)

            self.offset_gm.set_as(self.offset_gm + each_valid_hi * self.wi * C0)

        if remain_ho_nums == 0:
            with self.tik_instance.if_scope(cut_ho_nums_index == cut_ho_nums - 1):
                if cut_ho_nums - 1 == 0:
                    last_valid_hi = self.hi
                else:
                    last_valid_hi = self.hi - (
                            (cut_ho_nums - 1) * each_process_ho * self.stride_h - pad_top)
                if last_valid_hi <= each_process_hi:
                    self.tik_instance.data_move(self.res_gm[self.offset_gm],
                                                col2img_fp16_ub[
                                                    pad_top_rows * wi * C0 + pad_left * C0],
                                                0,
                                                last_valid_hi, self.wi * C0 // 16,
                                                pad_left + pad_right, 0)
                else:
                    self.tik_instance.data_move(self.res_gm[self.offset_gm],
                                                col2img_fp16_ub[
                                                    pad_top_rows * wi * C0 + pad_left * C0],
                                                0,
                                                each_process_hi, self.wi * C0 // 16,
                                                pad_left + pad_right, 0)
                    remain_hi = last_valid_hi - each_process_hi
                    temp_zero = self.tik_instance.Tensor("float16",
                                                         (remain_hi, self.wi * C0),
                                                         name='temp_zero',
                                                         scope=tik.scope_ubuf)
                    self._vector_dup(temp_zero, 0, temp_zero.shape,
                                     self.scalar_zero_fp16, "float16")
                    self.tik_instance.data_move(
                        self.res_gm[self.offset_gm + each_process_hi * self.wi * C0],
                        temp_zero,
                        0,
                        1, remain_hi * self.wi * C0 // 16,
                        0, 0)
                self.offset_gm.set_as(self.offset_gm + last_valid_hi * self.wi * C0)
        else:
            with self.tik_instance.if_scope(cut_ho_nums_index == cut_ho_nums - 1):
                self.tik_instance.data_move(self.res_gm[self.offset_gm],
                                            col2img_fp16_ub[pad_top_rows * wi * C0 + pad_left * C0],
                                            0,
                                            each_valid_hi, self.wi * C0 // 16,
                                            pad_left + pad_right, 0)
                self.offset_gm.set_as(self.offset_gm + each_valid_hi * self.wi * C0)
            if isinstance(cut_ho_nums_index, int):
                if cut_ho_nums == 0:
                    last_valid_hi = self.hi
                else:
                    last_valid_hi = self.hi - (
                            cut_ho_nums * each_process_ho * self.stride_h - pad_top)
                if last_valid_hi <= each_process_hi:
                    self.tik_instance.data_move(self.res_gm[self.offset_gm],
                                                col2img_fp16_ub[
                                                    pad_top_rows * wi * C0 + pad_left * C0],
                                                0,
                                                last_valid_hi, self.wi * C0 // 16,
                                                pad_left + pad_right, 0)
                else:
                    self.tik_instance.data_move(self.res_gm[self.offset_gm],
                                                col2img_fp16_ub[
                                                    pad_top_rows * wi * C0 + pad_left * C0],
                                                0,
                                                each_process_hi, self.wi * C0 // 16,
                                                pad_left + pad_right, 0)
                    remain_hi = last_valid_hi - each_process_hi
                    temp_zero = self.tik_instance.Tensor("float16",
                                                         (remain_hi, self.wi * C0),
                                                         name='temp_zero',
                                                         scope=tik.scope_ubuf)
                    self._vector_dup(temp_zero, 0, temp_zero.shape,
                                     self.scalar_zero_fp16, "float16")
                    self.tik_instance.data_move(
                        self.res_gm[self.offset_gm + each_process_hi * self.wi * C0],
                        temp_zero,
                        0,
                        1, remain_hi * self.wi * C0 // 16,
                        0, 0)
                self.offset_gm.set_as(self.offset_gm + last_valid_hi * self.wi * C0)

        if self.kh > self.stride_h:
            self._vector_op("vmuls", col2img_fp32_ub, 1.0, temp_tensor_ub, "float32",
                            cal_shape_ele(temp_tensor_ub.shape),
                            None, [0, each_process_ho * self.stride_h * wi * C0])

    # 'pylint: disable=unused-variable
    def _move_func_block(self, cut_ho_nums_index, cut_ho_nums, start_h, end_h, each_process_ho,
                         valid_hi_block, col2img_fp32_ub, temp_tensor_ub,
                         remained_hi, remain, pad):
        pad_left, pad_right, pad_top, pad_bottom = pad
        wi = self.wi + pad_left + pad_right
        mov_len_h = self.tik_instance.Scalar(dtype='int64', name='mov_len_h')
        hi_max = self.tik_instance.Scalar(dtype='int64', name='hi_max')
        hi_min = self.tik_instance.Scalar(dtype='int64', name='hi_min')
        self.tik_instance.scalar_max(hi_min, pad_top, start_h)
        hi_max.set_as(valid_hi_block + pad_top)
        self.tik_instance.scalar_min(hi_max, hi_max, end_h)
        mov_len_h.set_as(hi_max - hi_min)

        col2img_fp16_ub = self.tik_instance.Tensor("float16", col2img_fp32_ub.shape,
                                                   name="col2img_fp16_ub", scope=tik.scope_ubuf)
        self._vconv(col2img_fp32_ub, 0, col2img_fp16_ub, 0,
                    cal_shape_ele(col2img_fp32_ub.shape), "float32")
        with self.tik_instance.if_scope(end_h > pad_top):
            with self.tik_instance.if_scope(start_h < pad_top + valid_hi_block):
                with self.tik_instance.if_scope(mov_len_h > 0):
                    self.tik_instance.data_move(self.res_gm[self.offset_gm],
                                                col2img_fp16_ub[(hi_min - start_h) *
                                                                wi * C0 +
                                                                pad_left * C0],
                                                0, mov_len_h, self.wi * C0 // 16,
                                                pad_left + pad_right, 0)
                    self.offset_gm.set_as(self.offset_gm + mov_len_h * self.wi * C0)
                    remained_hi.set_as(remained_hi - mov_len_h)

                with self.tik_instance.if_scope(cut_ho_nums_index == cut_ho_nums - 1):
                    with self.tik_instance.if_scope(remain == 0):
                        with self.tik_instance.if_scope(remained_hi > 0):
                            temp_zero = self.tik_instance.Tensor(
                                "float16", (1, self.wi, C0), name="temp_zero",
                                scope=tik.scope_ubuf)
                            self._vector_dup(temp_zero,
                                             0,
                                             temp_zero.shape,
                                             self.scalar_zero_fp16,
                                             temp_zero.dtype)

                            with self.tik_instance.for_range(0, remained_hi) as index_0:
                                self.tik_instance.data_move(self.res_gm[self.offset_gm],
                                                            temp_zero, 0,
                                                            1, self.wi * C0 // 16, 0, 0)
                                self.offset_gm.set_as(self.offset_gm + self.wi * C0)
                with self.tik_instance.if_scope(cut_ho_nums_index == cut_ho_nums):
                    with self.tik_instance.if_scope(remained_hi > 0):
                        temp_zero = self.tik_instance.Tensor(
                            "float16", (1, self.wi, C0), name="temp_zero",
                            scope=tik.scope_ubuf)
                        self._vector_dup(temp_zero,
                                         0,
                                         temp_zero.shape,
                                         self.scalar_zero_fp16,
                                         temp_zero.dtype)
                        with self.tik_instance.for_range(0, remained_hi) as index_0:
                            self.tik_instance.data_move(self.res_gm[self.offset_gm],
                                                        temp_zero, 0,
                                                        1, self.wi * C0 // 16, 0, 0)
                            self.offset_gm.set_as(self.offset_gm + self.wi * C0)

        if self.kh > self.stride_h:
            self._vector_op("vmuls", col2img_fp32_ub, 1.0, temp_tensor_ub, "float32",
                            cal_shape_ele(temp_tensor_ub.shape),
                            None, [0, each_process_ho * self.stride_h * wi * C0])
        return remained_hi

    # 'pylint: disable=redefined-outer-name,simplifiable-if-statement,self-assigning-variable
    def _tilling_factor(self, ori_input_shape, pad):
        pad_left, pad_right, _, _ = pad
        C0 = ori_input_shape[-1]
        input_l1_size = cal_byte_size(ori_input_shape, self.dtype)
        # each type of buffer's bit size
        fp16_data_size = tbe_platform.get_bit_len("float16") // 8
        fp32_data_size = tbe_platform.get_bit_len("float32") // 8
        uint16_data_size = tbe_platform.get_bit_len("uint16") // 8

        if input_l1_size >= L1_SIZE:
            need_cut_L1 = True
        else:
            need_cut_L1 = False

        if self.kh > self.stride_h:
            each_process_hi = self.kh
        else:
            each_process_hi = self.stride_h

        # define eaxtra_size for some case
        # if each_process_wo is not 32B alined and 256B alined
        extra_size = (VECTOR_FP16_SIZE * C0 + C0) * fp16_data_size
        if self.kw > self.stride_w:
            extra_size += each_process_hi * (self.kw - self.stride_w) * C0 * \
                          (fp16_data_size + fp32_data_size)

        # calculate col ub size
        # There are two col ub, one is fp32, other is fp16, shape is (each_hi, each_wi, C0)
        # Here we need calculate each_wi to judge if need cut wo or cut ho.
        # `self.kw > self.stride_w, each_wi = (each_process_wo - 1) * self.stride_w + self.kw`
        # `self.kw <= self.stride_w, each_wi = each_process_wo * self.stride_w`

        col_ub_size_times = each_process_hi * self.stride_w * C0 * (fp16_data_size + fp32_data_size)

        # calculate mask ub size
        # There are for mask buffer on ub, each is (math.ceil(each_wo_16 * C0 // 128) * 128, )
        # Here each_wo_16 is ceil to 16 times.
        mask_ub_size_times = uint16_data_size * 4

        # calculate tensor ub siez
        # There are five tensor buffer on UB, each is (each_wo_16, 16, C0), one is fp32, others
        # are fp16. Here each_wo_16 is ceil to 16 times.
        tensor_ub_size_time = C0 * (4 * fp16_data_size + fp32_data_size)

        # some temp size
        # at most have 4 temp buffer on ub, each is (128, ), dtype is float16
        temp_ub_size = fp16_data_size * self.wi * C0

        each_process_wo = (UB_SIZE - temp_ub_size) * 1.0 / \
                          (col_ub_size_times + mask_ub_size_times + tensor_ub_size_time) - \
                          pad_left - pad_right
        each_process_wo = int(each_process_wo // 16 * 16)

        if each_process_wo >= self.wo:
            wi = self.wi + pad_left + pad_right
            extra_size = 0
            if self.kh > self.stride_h:
                extra_size += (self.kh - self.stride_h) * wi * C0 * (
                        fp16_data_size + 2 * fp32_data_size)
            # calculate col ub size
            # There are two col ub, one is fp32, other is fp16, shape is (each_hi, wi, C0)
            # `self.kh > self.stride_h, each_hi = (each_process_ho - 1) * self.stride_h + self.kh`
            # `self.kh <= self.stride_h, each_hi = each_process_ho * self.stride_h`
            col_ub_size_times = self.stride_h * wi * C0 * (fp16_data_size + fp32_data_size)

            mask_ub_size_times = self.wo * uint16_data_size * 4

            tensor_ub_size_time = self.wo * C0 * (4 * fp16_data_size + fp32_data_size)

            temp_ub_size = fp16_data_size * self.wo * C0
            each_process_ho = (UB_SIZE - temp_ub_size - extra_size) // (
                    col_ub_size_times + mask_ub_size_times +
                    tensor_ub_size_time)
            each_process_ho = each_process_ho
            if each_process_ho <= 0:
                each_process_ho = 1
                each_process_wo = self.wo
        else:
            each_process_ho = 0

        if each_process_ho >= self.ho:
            need_cut_Ho = False
            need_cut_Wo = False
            tiling_result = (need_cut_L1, need_cut_Ho, need_cut_Wo, self.ho, 0)
        elif each_process_ho > 1:
            need_cut_Ho = True
            need_cut_Wo = False
            tiling_result = (need_cut_L1, need_cut_Ho, need_cut_Wo, each_process_ho, 0)
        else:
            need_cut_Ho = True
            need_cut_Wo = True
            tiling_result = (True, need_cut_Ho, need_cut_Wo, 1, each_process_wo)
        return tiling_result

    # 'pylint: disable=unused-variable
    def _not_tilling(self, n_index, c1_index,
                     each_process_ho_block, each_process_hi_block,
                     mov_len_ho, mov_len_hi,
                     start_ho_index, start_hi_index,
                     start_threshold,
                     offset_gm_block, shape, pad):

        pad_left, pad_right, pad_top, pad_bottom = pad
        shape_ho, shape_wo, shape_hi, shape_wi = shape

        howo_ceil16 = ceil_div(shape_ho * shape_wo, 16)

        wi = self.wi + self.pad_left + self.pad_right
        hi = shape_hi + self.pad_top + self.pad_bottom

        # define col res
        col2img_ub_shape = (hi, wi, C0)
        col2img_fp32_ub = self.tik_instance.Tensor("float32",
                                                   col2img_ub_shape,
                                                   name="col2img_fp32_ub",
                                                   scope=tik.scope_ubuf)
        ori_input_shape = (hi, self.wi, C0)
        ori_output_shape = (howo_ceil16, 16, C0)
        input_data_nums = mov_len_hi * self.wi * C0
        output_data_nums = mov_len_ho * self.wo * C0

        src_input_offset = ((n_index * self.c1 + c1_index) * self.hi + start_hi_index) * \
                           self.wi * C0
        src_output_offset = ((n_index * self.c1 + c1_index) * self.ho + start_ho_index) * \
                            self.wo * C0
        repeate_time = (mov_len_ho * self.wo + 15) // 16

        ori_input_l1, ori_output_ub, grad_ub = self._data_move_ub(
            ori_input_shape, ori_output_shape, input_data_nums,
            output_data_nums, src_input_offset, src_output_offset)
        ori_input_col_ub = self.tik_instance.Tensor(self.dtype,
                                                    ori_output_shape,
                                                    name='ori_input_col_ub',
                                                    scope=tik.scope_ubuf)
        mask_shape = (ceil_div(cal_shape_ele(ori_output_ub.shape[:2]), MASK128_VALUE) *
                      MASK128_VALUE,)
        mask_not = self.tik_instance.Tensor("uint16",
                                            mask_shape,
                                            name='mask_not',
                                            scope=tik.scope_ubuf)
        mask_or = self.tik_instance.Tensor("uint16",
                                           mask_shape,
                                           name='mask_or',
                                           scope=tik.scope_ubuf)

        # init col2img_fp32_ub, if not the first one and have overlap, dump
        # the overlap part to col2img_fp32_ub, here we process whole ho, so
        # no need to move overlap part
        self._vector_dup(col2img_fp32_ub, 0, col2img_fp32_ub.shape,
                         self.scalar_zero, "float32")

        with self.tik_instance.for_range(0, self.kh, thread_num=1) as index_h:
            with self.tik_instance.for_range(0, self.kw, thread_num=1) as index_w:
                if self.check_load3d_support:
                    self.tik_instance.load3dv1(ori_input_col_ub[0],
                                               ori_input_l1[0],
                                               (pad_left, pad_right, pad_top, pad_bottom),
                                               mov_len_hi, self.wi, 0,
                                               index_w, index_h,
                                               -pad_left, -pad_top,
                                               self.stride_w, self.stride_h,
                                               self.kw, self.kh, 1, 1, 1, 1,
                                               repeate_time, 0, self.pad_value)
                else:
                    self._vector_dup(ori_input_col_ub, 0, ori_input_col_ub.shape, MIN_VALUE_FP16, "float16")
                    img2col(self.tik_instance, ori_input_l1, ori_input_col_ub, 0, 0, index_h, index_w,
                            -pad_top, -pad_left, mov_len_hi, self.wi, self.kh, self.kw, self.stride_h, self.stride_w,
                            repeate_time, 1, (pad_left, pad_right, pad_top, pad_bottom))

                # calculate mask here
                mask_shape = (ceil_div(cal_shape_ele(ori_output_ub.shape[:2]), MASK128_VALUE) *
                              MASK128_VALUE,)
                mask_ub = self._calc_mask(index_h, index_w, mask_shape, ori_output_ub,
                                          ori_input_col_ub, mask_or, mask_not)
                grad_sel_ub = self._vsel_grad_col(mask_ub, grad_ub)
                grad_sel_ub_fp32 = self.tik_instance.Tensor("float32",
                                                            grad_sel_ub.shape,
                                                            name='grad_sel_ub_fp32',
                                                            scope=tik.scope_ubuf)
                self._vconv(grad_sel_ub, 0, grad_sel_ub_fp32, 0,
                            cal_shape_ele(grad_sel_ub.shape), "float16")
                with self.tik_instance.for_range(0, mov_len_ho) as ho_idx:
                    col_index = index_h * wi * C0 + index_w * C0 + wi * C0 * self.stride_h * ho_idx
                    mask_index = self.wo * C0 * ho_idx
                    self._vector_op("vadd", col2img_fp32_ub[col_index:], grad_sel_ub_fp32[mask_index:],
                                    col2img_fp32_ub[col_index:], "float32", self.wo * C0 // 2,
                                    stride_cofig=(self.stride_w * 2, self.stride_w * 2, 2,
                                                  self.stride_w * 16, self.stride_w * 16, 16))
                    self._vector_op("vadd", col2img_fp32_ub[col_index + 8:],
                                    grad_sel_ub_fp32[mask_index + 8:],
                                    col2img_fp32_ub[col_index + 8:], "float32", self.wo * C0 // 2,
                                    stride_cofig=(self.stride_w * 2, self.stride_w * 2, 2,
                                                  self.stride_w * 16, self.stride_w * 16, 16))

        col2img_fp16_ub = self.tik_instance.Tensor("float16",
                                                   col2img_ub_shape,
                                                   name="col2img_fp16_ub",
                                                   scope=tik.scope_ubuf)
        self._vconv(col2img_fp32_ub, 0, col2img_fp16_ub, 0,
                    cal_shape_ele(col2img_fp32_ub.shape), "float32")
        if offset_gm_block is None:
            pad_top_offset = pad_top * wi * C0
            self.tik_instance.data_move(self.res_gm[self.offset_gm],
                                        col2img_fp16_ub[pad_top_offset + pad_left * C0],
                                        0, self.hi, self.wi * C0 // 16,
                                        pad_left + pad_right, 0)
        else:
            with self.tik_instance.if_scope(start_threshold > pad_top):
                self.tik_instance.data_move(self.res_gm[offset_gm_block],
                                            col2img_fp16_ub[
                                                start_threshold * wi * C0 + pad_left * C0],
                                            0, each_process_hi_block, self.wi * C0 // 16,
                                            pad_left + pad_right, 0)

            with self.tik_instance.else_scope():
                self.tik_instance.data_move(self.res_gm[offset_gm_block],
                                            col2img_fp16_ub[pad_top * wi * C0 + pad_left * C0],
                                            0, each_process_hi_block, self.wi * C0 // 16,
                                            pad_left + pad_right, 0)

    # 'pylint: disable=unused-variable
    def _tilling_ho_only(self, each_process_ho, n_index, c1_index,
                         each_process_ho_block, each_process_hi_block,
                         mov_len_ho, mov_len_hi,
                         start_ho_index, start_hi_index,
                         start_threshold,
                         offset_gm_block, shape, pad):

        pad_left, pad_right, pad_top, pad_bottom = pad
        shape_ho, shape_wo, shape_hi, shape_wi = shape
        cut_ho_nums = mov_len_ho // each_process_ho
        remain_ho_nums = mov_len_ho % each_process_ho
        wi = self.wi + pad_left + pad_right

        if self.kh > self.stride_h:
            each_process_hi = (each_process_ho - 1) * self.stride_h + self.kh
            temp_size = ((self.kh - self.stride_h), wi, C0)
        else:
            each_process_hi = each_process_ho * self.stride_h
            temp_size = (1, 16, C0)
        temp_tensor_ub = self.tik_instance.Tensor("float32", temp_size,
                                                  name="temp_tensor_ub", scope=tik.scope_ubuf)

        each_process_ho_wo_div16 = ceil_div(each_process_ho * self.wo, 16)
        ori_input_shape = (shape_hi, self.wi, C0)
        ori_output_shape = (each_process_ho_wo_div16, 16, C0)
        ori_input_l1 = self.tik_instance.Tensor(self.dtype, ori_input_shape, name='ori_input_l1',
                                                scope=tik.scope_cbuf)

        col2img_ub_shape = (each_process_hi, wi, C0)
        col2img_fp32_ub = self.tik_instance.Tensor("float32", col2img_ub_shape,
                                                   name="col2img_fp32_ub", scope=tik.scope_ubuf)

        input_data_nums = mov_len_hi * self.wi * C0
        self.tik_instance.data_move(ori_input_l1[0],
                                    self.ori_input_gm[
                                        ((n_index * self.c1 + c1_index) * self.hi + start_hi_index
                                         ) * self.wi * C0],
                                    0, 1, input_data_nums // 16, 0, 0)

        if offset_gm_block is not None:
            self.offset_gm.set_as(offset_gm_block)
            remained_hi = self.tik_instance.Scalar(dtype='int64', name='remained_hi')
            remained_hi.set_as(each_process_hi_block)

        def process_ho(output_data_nums, cut_ho_nums_index, each_valid_ho, remained_hi):
            """

            :param output_data_nums:
            :param cut_ho_nums_index:
            :param each_valid_ho:
            :param remained_hi:
            :return:
            """
            self._vector_dup(col2img_fp32_ub, 0, col2img_fp32_ub.shape,
                             self.scalar_zero, "float32")
            if self.kh > self.stride_h:
                with self.tik_instance.if_scope(cut_ho_nums_index > 0):
                    self._vector_op("vmuls", temp_tensor_ub, 1.0, col2img_fp32_ub,
                                    temp_tensor_ub.dtype, cal_shape_ele(temp_tensor_ub.shape))

            start_h = self.tik_instance.Scalar(dtype='int64', name='start_h')
            end_h = self.tik_instance.Scalar(dtype='int64', name='end_h')
            start_h.set_as(cut_ho_nums_index * each_process_ho * self.stride_h)
            end_h.set_as(cut_ho_nums_index * each_process_ho * self.stride_h + each_process_hi)

            src_output_offset = ((n_index * self.c1 + c1_index) * self.ho + start_ho_index +
                                 each_process_ho * cut_ho_nums_index) * self.wo * C0
            _, ori_output_ub, grad_ub = self._data_move_ub(None,
                                                           ori_output_shape,
                                                           None,
                                                           output_data_nums,
                                                           None,
                                                           src_output_offset)

            ori_input_col_ub = self.tik_instance.Tensor(self.dtype,
                                                        ori_output_shape,
                                                        name='ori_input_col_ub',
                                                        scope=tik.scope_ubuf)
            mask_shape = (ceil_div(cal_shape_ele(ori_output_ub.shape[:2]), MASK128_VALUE) *
                          MASK128_VALUE,)
            mask_not = self.tik_instance.Tensor("uint16",
                                                mask_shape,
                                                name='mask_not',
                                                scope=tik.scope_ubuf)
            mask_or = self.tik_instance.Tensor("uint16",
                                               mask_shape,
                                               name='mask_or',
                                               scope=tik.scope_ubuf)
            with self.tik_instance.for_range(0, self.kh, thread_num=1) as index_h:
                with self.tik_instance.for_range(0, self.kw, thread_num=1) as index_w:
                    self._load3d(index_h, index_w, start_h, end_h,
                                 ori_input_col_ub,
                                 ori_input_l1, start_h - pad_top,
                                 each_process_hi, self.wi,
                                 each_process_ho_wo_div16,
                                 pad, self.pad_value, -pad_left, mov_len_hi)
                    mask_shape = (ceil_div(cal_shape_ele(ori_output_ub.shape[:2]), MASK128_VALUE) *
                                  MASK128_VALUE,)
                    mask_ub = self._calc_mask(index_h, index_w, mask_shape, ori_output_ub,
                                              ori_input_col_ub, mask_or, mask_not)
                    grad_sel_ub = self._vsel_grad_col(mask_ub, grad_ub)
                    grad_sel_ub_fp32 = self.tik_instance.Tensor("float32",
                                                                grad_sel_ub.shape,
                                                                name='grad_sel_ub_fp32',
                                                                scope=tik.scope_ubuf)
                    self._vconv(grad_sel_ub, 0, grad_sel_ub_fp32, 0,
                                cal_shape_ele(grad_sel_ub.shape), "float16")

                    with self.tik_instance.for_range(0, each_valid_ho) as h_idx:
                        col_index = index_h * wi * C0 + index_w * C0 + \
                                    wi * C0 * self.stride_h * h_idx
                        mask_idx = self.wo * C0 * h_idx
                        self._vector_op("vadd", col2img_fp32_ub[col_index:],
                                        grad_sel_ub_fp32[mask_idx:],
                                        col2img_fp32_ub[col_index:], "float32", self.wo * C0 // 2,
                                        stride_cofig=(self.stride_w * 2, self.stride_w * 2, 2,
                                                      self.stride_w * 16, self.stride_w * 16, 16))
                        self._vector_op("vadd", col2img_fp32_ub[col_index + 8:],
                                        grad_sel_ub_fp32[mask_idx + 8:],
                                        col2img_fp32_ub[col_index + 8:], "float32", self.wo * C0 // 2,
                                        stride_cofig=(self.stride_w * 2, self.stride_w * 2, 2,
                                                      self.stride_w * 16, self.stride_w * 16, 16))

            if self.tile_h_to_block:
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
                                                     pad_bottom))


            else:
                self._mov_func(cut_ho_nums_index, cut_ho_nums, remain_ho_nums, each_process_ho,
                               each_process_hi, each_valid_ho, col2img_fp32_ub, temp_tensor_ub, pad)
            return remained_hi

        with self.tik_instance.for_range(0, cut_ho_nums) as cut_ho_nums_index:
            output_data_nums = each_process_ho * self.wo * C0
            if self.tile_h_to_block:
                remained_hi = process_ho(output_data_nums, cut_ho_nums_index, each_process_ho,
                                         remained_hi)
            else:
                process_ho(output_data_nums, cut_ho_nums_index, each_process_ho, None)

        if not self.tile_h_to_block:
            if remain_ho_nums > 0:
                output_data_nums = remain_ho_nums * self.wo * C0
                process_ho(output_data_nums, cut_ho_nums, remain_ho_nums, None)
        else:
            with self.tik_instance.if_scope(remain_ho_nums > 0):
                output_data_nums = remain_ho_nums * self.wo * C0
                process_ho(output_data_nums, cut_ho_nums, remain_ho_nums, remained_hi)

    # 'pylint: disable=unused-variable
    def _tilling_l1_ho_only(self, each_process_ho, n_index, c1_index,
                            each_process_ho_block, each_process_hi_block,
                            mov_len_ho, mov_len_hi,
                            start_ho_index, start_hi_index,
                            start_threshold,
                            offset_gm_block, shape, pad):
        pad_left, pad_right, pad_top, pad_bottom = pad

        wi = self.wi + pad_left + pad_right
        # if cut self.ho, every time process each_process_ho * self.wo
        cut_ho_nums = mov_len_ho // each_process_ho
        remain_ho_nums = mov_len_ho % each_process_ho

        start_h = self.tik_instance.Scalar(dtype='int64', name='start_h')
        end_h = self.tik_instance.Scalar(dtype='int64', name='end_h')
        hi_max = self.tik_instance.Scalar(dtype='int64', name='hi_max')
        hi_min = self.tik_instance.Scalar(dtype='int64', name='hi_min')
        mov_len_h = self.tik_instance.Scalar(dtype='int64', name='mov_len_h')
        start_pos_h = self.tik_instance.Scalar(dtype='int64', name='start_pos_h')

        # each loop process each_process_ho * self.wo * CO
        if self.kh > self.stride_h:
            each_process_hi = (each_process_ho - 1) * self.stride_h + self.kh
            temp_size = ((self.kh - self.stride_h), wi, C0)

        else:
            each_process_hi = each_process_ho * self.stride_h
            temp_size = (1, 16, C0)
        temp_tensor_ub = self.tik_instance.Tensor("float32", temp_size,
                                                  name="temp_tensor_ub", scope=tik.scope_ubuf)

        each_process_ho_wo_div16 = ceil_div(each_process_ho * self.wo, 16)
        # define col res, init to zero
        col2img_ub_shape = (each_process_hi, wi, C0)
        col2img_fp32_ub = self.tik_instance.Tensor("float32", col2img_ub_shape,
                                                   name="col2img_fp32_ub",
                                                   scope=tik.scope_ubuf)
        self._vector_dup(col2img_fp32_ub, 0, col2img_fp32_ub.shape,
                         self.scalar_zero, "float32")

        # one times the number of (each_process_hi, self.wi, C0) can be storaged on L1
        n_each_process_hi_block = \
            (L1_SIZE // 2 - cal_shape_ele((each_process_hi, self.wi, C0))) // \
            (each_process_ho * self.stride_h * self.wi * C0) + 1
        # times of process all (each_process_hi, self.wi, C0) blocks
        n_hi_block = cut_ho_nums // n_each_process_hi_block
        # remains of (each_process_hi, self.wi, C0) blocks
        remain_hi_block = cut_ho_nums % n_each_process_hi_block
        if offset_gm_block is not None:
            self.offset_gm.set_as(offset_gm_block)
            remained_hi = self.tik_instance.Scalar(dtype='int64', name='remained_hi')
            remained_hi.set_as(each_process_hi_block)

        # 'pylint: disable=unused-variable
        def process_tiling_l1_ho(n_hi_block_index, n_each_process_hi_block_index, start_h, end_h,
                                 start_pos_h, src_output_offset,
                                 output_data_nums, each_valid_ho, remained_hi, remain):
            """

            :param n_hi_block_index: n_hi_block_index
            :param n_each_process_hi_block_index: n_each_process_hi_block_index
            :param start_h: start height
            :param end_h: end height
            :param start_pos_h: start position height
            :param src_output_offset: src_output_offset
            :param output_data_nums: output_data_nums
            :param each_valid_ho: each_valid_ho
            :param remained_hi: remained_hi
            :param remain: remain
            :return: process_tiling_l1_ho
            """
            # init col2img buffer each loop
            self._vector_dup(col2img_fp32_ub, 0, col2img_fp32_ub.shape,
                             self.scalar_zero, "float32")
            if self.kh > self.stride_h:
                with self.tik_instance.if_scope(
                        tik.any(n_hi_block_index > 0, n_each_process_hi_block_index > 0)):
                    self._vector_op("vmuls", temp_tensor_ub, 1.0, col2img_fp32_ub,
                                    temp_tensor_ub.dtype, cal_shape_ele(temp_tensor_ub.shape))
            ori_output_shape = (ceil_div(each_process_ho * self.wo, 16), 16, C0)
            _, ori_output_ub, grad_ub = self._data_move_ub(None,
                                                           ori_output_shape,
                                                           None,
                                                           output_data_nums,
                                                           None,
                                                           src_output_offset)

            ori_input_col_ub = self.tik_instance.Tensor(self.dtype,
                                                        ori_output_shape,
                                                        name='ori_input_col_ub',
                                                        scope=tik.scope_ubuf)
            mask_shape = (ceil_div(cal_shape_ele(ori_output_ub.shape[:2]), MASK128_VALUE) *
                          MASK128_VALUE,)
            mask_not = self.tik_instance.Tensor("uint16",
                                                mask_shape,
                                                name='mask_not',
                                                scope=tik.scope_ubuf)
            mask_or = self.tik_instance.Tensor("uint16",
                                               mask_shape,
                                               name='mask_or',
                                               scope=tik.scope_ubuf)

            with self.tik_instance.for_range(0, self.kh, thread_num=1) as index_h:
                with self.tik_instance.for_range(0, self.kw, thread_num=1) as index_w:
                    self._load3d(index_h, index_w, start_h, end_h,
                                 ori_input_col_ub,
                                 ori_input_l1, start_pos_h,
                                 each_process_hi, self.wi,
                                 each_process_ho_wo_div16,
                                 pad, self.pad_value,
                                 -pad_left, self.hi)

                    mask_ub = self._calc_mask(index_h, index_w, mask_shape, ori_output_ub,
                                              ori_input_col_ub, mask_or, mask_not)
                    grad_sel_ub = self._vsel_grad_col(mask_ub, grad_ub)
                    grad_sel_ub_fp32 = self.tik_instance.Tensor("float32",
                                                                grad_sel_ub.shape,
                                                                name='grad_sel_ub_fp32',
                                                                scope=tik.scope_ubuf)
                    self._vconv(grad_sel_ub, 0, grad_sel_ub_fp32, 0,
                                cal_shape_ele(grad_sel_ub.shape), "float16")
                    with self.tik_instance.for_range(0, each_valid_ho) as h_idx:
                        col_index = index_h * wi * C0 + index_w * C0 + \
                                    wi * C0 * self.stride_h * h_idx
                        mask_idx = self.wo * C0 * h_idx
                        self._vector_op("vadd", col2img_fp32_ub[col_index:],
                                        grad_sel_ub_fp32[mask_idx:],
                                        col2img_fp32_ub[col_index:], "float32", self.wo * C0 // 2,
                                        stride_cofig=(self.stride_w * 2, self.stride_w * 2, 2,
                                                      self.stride_w * 16, self.stride_w * 16, 16))
                        self._vector_op("vadd", col2img_fp32_ub[col_index + 8:],
                                        grad_sel_ub_fp32[mask_idx + 8:],
                                        col2img_fp32_ub[col_index + 8:], "float32", self.wo * C0 // 2,
                                        stride_cofig=(self.stride_w * 2, self.stride_w * 2, 2,
                                                      self.stride_w * 16, self.stride_w * 16, 16))

            cut_ho_nums_index = n_hi_block_index * n_each_process_hi_block + \
                                n_each_process_hi_block_index
            if self.tile_h_to_block:
                remained_hi = self._move_func_block(cut_ho_nums_index, cut_ho_nums,
                                                    start_h, end_h, each_process_ho,
                                                    each_process_hi_block, col2img_fp32_ub,
                                                    temp_tensor_ub,
                                                    remained_hi, remain,
                                                    (pad_left, pad_right, start_threshold,
                                                     pad_bottom))
            else:
                self._mov_func(cut_ho_nums_index, cut_ho_nums, remain_ho_nums, each_process_ho,
                               each_process_hi, each_valid_ho, col2img_fp32_ub, temp_tensor_ub, pad)
            return remained_hi

        with self.tik_instance.for_range(0, n_hi_block) as n_hi_block_index:
            start_h.set_as(
                n_hi_block_index * n_each_process_hi_block * each_process_ho * self.stride_h)
            end_h.set_as(n_hi_block_index * n_each_process_hi_block *
                         each_process_ho * self.stride_h + each_process_hi +
                         (n_each_process_hi_block - 1) * each_process_ho * self.stride_h)
            ori_input_shape = (
                each_process_hi + (n_each_process_hi_block - 1) * each_process_ho * self.stride_h,
                self.wi, C0)
            ori_input_l1 = self.tik_instance.Tensor(self.dtype, ori_input_shape,
                                                    name='ori_input_l1',
                                                    scope=tik.scope_cbuf)
            self.tik_instance.scalar_max(hi_min, pad_top, start_h)
            hi_max.set_as(self.hi + pad_top)
            self.tik_instance.scalar_min(hi_max, hi_max, end_h)
            mov_len_h.set_as(hi_max - hi_min)
            self.tik_instance.data_move(ori_input_l1[0],
                                        self.ori_input_gm[((n_index * self.c1 + c1_index) *
                                                           self.hi + start_hi_index +
                                                           hi_min - pad_top) *
                                                          self.wi * C0],
                                        0, mov_len_h, self.wi * C0 // 16, 0, 0)

            with self.tik_instance.for_range(0, n_each_process_hi_block) as \
                    n_each_process_hi_block_index:
                start_h.set_as((n_hi_block_index * n_each_process_hi_block +
                                n_each_process_hi_block_index) *
                               each_process_ho * self.stride_h)
                end_h.set_as((n_hi_block_index * n_each_process_hi_block +
                              n_each_process_hi_block_index) *
                             each_process_ho * self.stride_h +
                             each_process_ho * self.stride_h)

                start_pos_h.set_as(
                    n_hi_block_index * n_each_process_hi_block * each_process_ho * self.stride_h)
                with self.tik_instance.if_scope(start_pos_h > pad_top):
                    start_pos_h.set_as(n_each_process_hi_block_index * each_process_ho *
                                       self.stride_h)
                with self.tik_instance.else_scope():
                    start_pos_h.set_as(n_each_process_hi_block_index * each_process_ho *
                                       self.stride_h - pad_top)
                self.tik_instance.scalar_max(start_pos_h, start_pos_h, 0)

                output_data_nums = each_process_ho * self.wo * C0
                src_output_offset = ((n_index * self.c1 + c1_index) * self.ho + start_ho_index +
                                     each_process_ho * (n_hi_block_index *
                                                        n_each_process_hi_block +
                                                        n_each_process_hi_block_index)) * \
                                    self.wo * C0

                if self.tile_h_to_block:
                    remain0 = self.tik_instance.Scalar(dtype='int64', name='remain0')
                    remain1 = self.tik_instance.Scalar(dtype='int64', name='remain')
                    remain0.set_as(remain_hi_block)
                    remain1.set_as(remain_ho_nums)
                    self.tik_instance.scalar_max(remain0, remain0, remain1)
                    remained_hi = process_tiling_l1_ho(n_hi_block_index,
                                                       n_each_process_hi_block_index,
                                                       start_h, end_h, start_pos_h,
                                                       src_output_offset,
                                                       output_data_nums, each_process_ho,
                                                       remained_hi, remain0)
                else:
                    process_tiling_l1_ho(n_hi_block_index, n_each_process_hi_block_index,
                                         start_h, end_h, start_pos_h, src_output_offset,
                                         output_data_nums, each_process_ho, None, None)

        if offset_gm_block is None:
            if remain_hi_block != 0:
                start_h.set_as(
                    n_hi_block * n_each_process_hi_block * each_process_ho * self.stride_h)
                end_h.set_as(
                    n_hi_block * n_each_process_hi_block * each_process_ho * self.stride_h +
                    each_process_hi + (remain_hi_block - 1) * each_process_ho * self.stride_h)

                ori_input_shape = (each_process_hi + (remain_hi_block - 1) *
                                   each_process_ho * self.stride_h, self.wi, C0)
                ori_input_l1 = self.tik_instance.Tensor(self.dtype, ori_input_shape,
                                                        name='ori_input_l1',
                                                        scope=tik.scope_cbuf)

                self.tik_instance.scalar_max(hi_min, pad_top, start_h)
                hi_max.set_as(self.hi + pad_top)
                self.tik_instance.scalar_min(hi_max, hi_max, end_h)
                mov_len_h.set_as(hi_max - hi_min)
                self.tik_instance.data_move(ori_input_l1[0],
                                            self.ori_input_gm[
                                                ((n_index * self.c1 + c1_index) * self.hi +
                                                 hi_min - pad_top) * self.wi * C0],
                                            0, mov_len_h, self.wi * C0 // 16, 0, 0)

                with self.tik_instance.for_range(0, remain_hi_block) as remain_hi_n_index:
                    output_data_nums = each_process_ho * self.wo * C0
                    src_output_offset = ((n_index * self.c1 + c1_index) * self.ho +
                                         each_process_ho * (n_hi_block * n_each_process_hi_block +
                                                            remain_hi_n_index)) * self.wo * C0
                    start_h.set_as((n_hi_block * n_each_process_hi_block + remain_hi_n_index) *
                                   each_process_ho * self.stride_h)
                    end_h.set_as((n_hi_block * n_each_process_hi_block + remain_hi_n_index) *
                                 each_process_ho * self.stride_h + each_process_ho * self.stride_h)
                    if n_hi_block == 0:
                        start_pos_h.set_as(
                            remain_hi_n_index * each_process_ho * self.stride_h - pad_top)
                    else:
                        start_pos_h.set_as(remain_hi_n_index * each_process_ho * self.stride_h)

                    process_tiling_l1_ho(n_hi_block, remain_hi_n_index, start_h, end_h,
                                         start_pos_h, src_output_offset,
                                         output_data_nums, each_process_ho, None, None)
            if remain_ho_nums != 0:
                input_data_num = (self.hi + pad_top - cut_ho_nums * each_process_ho *
                                  self.stride_h) * self.wi * C0
                ori_input_shape = (
                    self.hi + pad_top - cut_ho_nums * each_process_ho * self.stride_h, self.wi, C0)
                ori_input_l1 = self.tik_instance.Tensor(self.dtype, ori_input_shape,
                                                        name='ori_input_l1', scope=tik.scope_cbuf)
                start_h.set_as((n_index * self.c1 + c1_index) * self.hi * self.wi *
                               C0 + (cut_ho_nums * each_process_ho *
                                     self.stride_h - pad_top) * self.wi * C0)

                self.tik_instance.data_move(ori_input_l1[0],
                                            self.ori_input_gm[(n_index * self.c1 + c1_index) *
                                                              self.hi * self.wi *
                                                              C0 + (cut_ho_nums * each_process_ho *
                                                                    self.stride_h - pad_top) *
                                                              self.wi * C0],
                                            0, 1, input_data_num // 16, 0, 0)

                each_process_ho_wo_div16 = (remain_ho_nums * self.wo + 15) // 16
                start_h.set_as(cut_ho_nums * each_process_ho * self.stride_h)
                if self.stride_h >= self.kh:
                    each_process_hi = remain_ho_nums * self.stride_h
                    end_h.set_as(cut_ho_nums * each_process_ho * self.stride_h + (
                            remain_ho_nums - 1) * self.stride_h + self.kh)

                else:
                    each_process_hi = (remain_ho_nums - 1) * self.stride_h + self.kh
                    end_h.set_as(
                        cut_ho_nums * each_process_ho * self.stride_h +
                        remain_ho_nums * self.stride_h)
                each_process_hi = (remain_ho_nums - 1) * self.stride_h + self.kh

                output_data_nums = remain_ho_nums * self.wo * C0
                src_output_offset = ((n_index * self.c1 + c1_index) * self.ho +
                                     each_process_ho * cut_ho_nums) * self.wo * C0

                process_tiling_l1_ho(n_hi_block, remain_hi_block, start_h, end_h,
                                     0, src_output_offset,
                                     output_data_nums, remain_ho_nums, None, None)
        else:
            with self.tik_instance.if_scope(remain_hi_block != 0):
                start_h.set_as(
                    n_hi_block * n_each_process_hi_block * each_process_ho * self.stride_h)
                end_h.set_as(
                    n_hi_block * n_each_process_hi_block * each_process_ho * self.stride_h +
                    each_process_hi + (remain_hi_block - 1) * each_process_ho * self.stride_h)

                ori_input_shape = (each_process_hi + (n_each_process_hi_block - 1) *
                                   each_process_ho * self.stride_h, self.wi, C0)
                ori_input_l1 = self.tik_instance.Tensor(self.dtype, ori_input_shape,
                                                        name='ori_input_l1',
                                                        scope=tik.scope_cbuf)

                self.tik_instance.scalar_max(hi_min, pad_top, start_h)
                hi_max.set_as(self.hi + pad_top)
                self.tik_instance.scalar_min(hi_max, hi_max, end_h)
                mov_len_h.set_as(hi_max - hi_min)
                self.tik_instance.data_move(ori_input_l1[0],
                                            self.ori_input_gm[
                                                ((n_index * self.c1 + c1_index) *
                                                 self.hi + start_hi_index +
                                                 hi_min - pad_top) * self.wi * C0],
                                            0, mov_len_h, self.wi * C0 // 16, 0, 0)

                with self.tik_instance.for_range(0, remain_hi_block) as remain_hi_n_index:
                    output_data_nums = each_process_ho * self.wo * C0
                    src_output_offset = ((n_index * self.c1 + c1_index) * self.ho + start_ho_index +
                                         each_process_ho * (n_hi_block * n_each_process_hi_block +
                                                            remain_hi_n_index)) * self.wo * C0
                    start_h.set_as((n_hi_block * n_each_process_hi_block + remain_hi_n_index) *
                                   each_process_ho * self.stride_h)
                    end_h.set_as((n_hi_block * n_each_process_hi_block + remain_hi_n_index) *
                                 each_process_ho * self.stride_h + each_process_ho * self.stride_h)
                    with self.tik_instance.if_scope(n_hi_block == 0):
                        start_pos_h.set_as(
                            remain_hi_n_index * each_process_ho * self.stride_h - pad_top)
                    with self.tik_instance.else_scope():
                        start_pos_h.set_as(remain_hi_n_index * each_process_ho * self.stride_h)

                    remained_hi = process_tiling_l1_ho(n_hi_block, remain_hi_n_index, start_h,
                                                       end_h,
                                                       start_pos_h, src_output_offset,
                                                       output_data_nums, each_process_ho,
                                                       remained_hi, remain_ho_nums)

            with self.tik_instance.if_scope(remain_ho_nums != 0):
                input_data_num = (mov_len_hi + pad_top - cut_ho_nums * each_process_ho *
                                  self.stride_h) * self.wi * C0
                ori_input_shape = (each_process_hi, self.wi, C0)
                ori_input_l1 = self.tik_instance.Tensor(self.dtype, ori_input_shape,
                                                        name='ori_input_l1', scope=tik.scope_cbuf)
                start_h.set_as((n_index * self.c1 + c1_index) * self.hi * self.wi *
                               C0 + (cut_ho_nums * each_process_ho *
                                     self.stride_h - pad_top) * self.wi * C0)

                self.tik_instance.data_move(ori_input_l1[0],
                                            self.ori_input_gm[((n_index * self.c1 + c1_index) *
                                                               self.hi + start_hi_index) * self.wi *
                                                              C0 + (cut_ho_nums * each_process_ho *
                                                                    self.stride_h - pad_top) *
                                                              self.wi * C0],
                                            0, 1, input_data_num // 16, 0, 0)

                each_process_ho_wo_div16 = (remain_ho_nums * self.wo + 15) // 16
                start_h.set_as(cut_ho_nums * each_process_ho * self.stride_h)
                if self.stride_h >= self.kh:
                    each_process_hi = remain_ho_nums * self.stride_h
                    end_h.set_as(cut_ho_nums * each_process_ho * self.stride_h + (
                            remain_ho_nums - 1) * self.stride_h + self.kh)

                else:
                    each_process_hi = (remain_ho_nums - 1) * self.stride_h + self.kh
                    end_h.set_as(
                        cut_ho_nums * each_process_ho * self.stride_h +
                        remain_ho_nums * self.stride_h)
                each_process_hi = (remain_ho_nums - 1) * self.stride_h + self.kh

                output_data_nums = remain_ho_nums * self.wo * C0
                src_output_offset = ((n_index * self.c1 + c1_index) * self.ho + start_ho_index +
                                     each_process_ho * cut_ho_nums) * self.wo * C0

                process_tiling_l1_ho(n_hi_block, remain_hi_block, start_h, end_h,
                                     0, src_output_offset,
                                     output_data_nums, remain_ho_nums, remained_hi, 0)

    # 'pylint: disable=unused-variable,no-self-use
    @staticmethod
    def _get_core_divlist():
        div_list = []
        for i in range(1, CORE_NUM + 1):
            if CORE_NUM % i == 0:
                if CORE_NUM // i not in div_list:
                    div_list.append(CORE_NUM // i)
        return div_list

    # 'pylint: disable=unused-variable
    def _tilling_l1_ho_wo(self, each_process_wo, n_index, c1_index,
                          each_process_ho_block, each_process_hi_block,
                          mov_len_ho, mov_len_hi,
                          start_ho_index, start_hi_index,
                          start_threshold,
                          offset_gm_block, shape, pad):
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
        remained_hi = self.tik_instance.Scalar(dtype='int64', name='remained_hi')
        remained_hi.set_as(each_process_hi_block)

        cut_wo_nums = self.wo // each_process_wo
        remain_wo_nums = self.wo % each_process_wo

        if self.stride_w >= self.kw:
            each_process_wi = each_process_wo * self.stride_w
        else:
            each_process_wi = (each_process_wo - 1) * self.stride_w + self.kw

        if self.stride_h >= self.kh:
            each_process_hi = self.stride_h
        else:
            each_process_hi = self.kh
        each_process_wo_div16 = ceil_div(each_process_wo, 16)

        # define col res, init to zero
        col2img_ub_shape = (each_process_hi, each_process_wi, C0)
        col2img_fp32_ub = self.tik_instance.Tensor("float32", col2img_ub_shape,
                                                   name="col2img_fp32_ub",
                                                   scope=tik.scope_ubuf)
        if offset_gm_block is not None:
            self.offset_gm.set_as(offset_gm_block)

        if self.stride_h < self.kh:
            overlap_l1_shape = (self.kh - self.stride_h, (self.wi + pad_left + pad_right) * C0)

            overlap_l1 = self.tik_instance.Tensor('float32', overlap_l1_shape,
                                                  name='overlap_l1',
                                                  scope=tik.scope_cbuf)
            overlap_l1_h, overlap_l1_w = overlap_l1_shape

        with self.tik_instance.for_range(0, mov_len_ho, thread_num=1) as ho_index:
            start_h.set_as(ho_index * self.stride_h)
            end_h.set_as(ho_index * self.stride_h + each_process_hi)
            # mov actual non pad ori input to L1
            ori_input_shape = (each_process_hi, self.wi + pad_left + pad_right, C0)
            ori_input_l1 = self.tik_instance.Tensor(self.dtype, ori_input_shape,
                                                    name='ori_input_l1',
                                                    scope=tik.scope_cbuf)

            self.tik_instance.scalar_max(hi_min, pad_top, start_h)
            hi_max.set_as(mov_len_hi + pad_top)
            self.tik_instance.scalar_min(hi_max, hi_max, end_h)
            mov_len_h.set_as(hi_max - hi_min)

            self.tik_instance.data_move(
                ori_input_l1[0],
                self.ori_input_gm[((n_index * self.c1 + c1_index) *
                                   self.hi + start_hi_index + hi_min - pad_top) * self.wi * C0],
                0, mov_len_h, self.wi * C0 // 16, 0, 0)

            offset_gm_inside = self.tik_instance.Scalar(dtype='int64',
                                                        name='offset_gm_inside')
            offset_gm_inside.set_as(self.offset_gm)

            # init col2img after every looph
            self._vector_dup(col2img_fp32_ub, 0, col2img_fp32_ub.shape,
                             self.scalar_zero, "float32")

            with self.tik_instance.for_range(0, cut_wo_nums, thread_num=1) as cut_wo_nums_index:
                if self.kh > self.stride_h:
                    with self.tik_instance.if_scope(ho_index != 0):
                        with self.tik_instance.if_scope(cut_wo_nums_index == 0):
                            with self.tik_instance.for_range(0,
                                                             self.kh - self.stride_h) as index_khs:
                                self.tik_instance.data_move(
                                    col2img_fp32_ub[index_khs * each_process_wi * C0],
                                    overlap_l1[index_khs * overlap_l1_w],
                                    0, 1, each_process_wi * C0 // 8,
                                    0, 0)

                        with self.tik_instance.else_scope():
                            start_pos = (each_process_wi - self.stride_w * each_process_wo) * C0
                            with self.tik_instance.for_range(0,
                                                             self.kh - self.stride_h) as index_khs:
                                self.tik_instance.data_move(
                                    col2img_fp32_ub[index_khs * each_process_wi * C0 + start_pos],
                                    overlap_l1[index_khs * overlap_l1_w +
                                               cut_wo_nums_index * each_process_wo *
                                               self.stride_w * C0 + start_pos],
                                    0, 1, self.stride_w * each_process_wo * C0 // 8,
                                    0, 0)

                ori_output_shape = (each_process_wo_div16, 16, C0)
                output_data_nums = each_process_wo * C0
                src_output_offset = ((n_index * self.c1 + c1_index) * self.ho +
                                     start_ho_index + ho_index) * self.wo * C0 + \
                                    cut_wo_nums_index * each_process_wo * C0
                _, ori_output_ub, grad_ub = self._data_move_ub(
                    None, ori_output_shape, None,
                    output_data_nums, None, src_output_offset)

                ori_input_col_ub = self.tik_instance.Tensor(self.dtype,
                                                            ori_output_shape,
                                                            name='ori_input_col_ub',
                                                            scope=tik.scope_ubuf)
                start_w.set_as(
                    cut_wo_nums_index * each_process_wo * self.stride_w)
                end_w.set_as(
                    cut_wo_nums_index * each_process_wo * self.stride_w + each_process_wi)

                # load3d to get col
                wo_offset = self.tik_instance.Scalar(dtype='int64',
                                                     name='wo_offset')
                wo_offset.set_as(
                    each_process_wo * cut_wo_nums_index * self.stride_w - pad_left)

                mask_shape = (ceil_div(cal_shape_ele(ori_output_ub.shape[:2]), MASK128_VALUE) *
                              MASK128_VALUE,)
                mask_not = self.tik_instance.Tensor("uint16",
                                                    mask_shape,
                                                    name='mask_not',
                                                    scope=tik.scope_ubuf)
                mask_or = self.tik_instance.Tensor("uint16",
                                                   mask_shape,
                                                   name='mask_or',
                                                   scope=tik.scope_ubuf)

                with self.tik_instance.for_range(0, self.kh,
                                                 thread_num=1) as index_h:
                    with self.tik_instance.for_range(0, self.kw,
                                                     thread_num=1) as index_w:
                        ori_input_col_ub = self._load3d(index_h, index_w,
                                                        start_h, end_h,
                                                        ori_input_col_ub,
                                                        ori_input_l1, 0,
                                                        each_process_hi, self.wi,
                                                        each_process_wo_div16,
                                                        pad, self.pad_value,
                                                        wo_offset, mov_len_hi)
                        mask_ub = self._calc_mask(index_h, index_w, mask_shape, ori_output_ub,
                                                  ori_input_col_ub, mask_or, mask_not)
                        grad_sel_ub = self._vsel_grad_col(mask_ub, grad_ub)
                        grad_sel_ub_fp32 = self.tik_instance.Tensor("float32",
                                                                    grad_sel_ub.shape,
                                                                    name='grad_sel_ub_fp32',
                                                                    scope=tik.scope_ubuf)
                        self._vconv(grad_sel_ub, 0, grad_sel_ub_fp32, 0,
                                    cal_shape_ele(grad_sel_ub.shape), "float16")

                        with self.tik_instance.for_range(0, 1) as h_idx:
                            col_index = index_h * each_process_wi * C0 + index_w * C0 + \
                                        each_process_wi * C0 * self.stride_h * h_idx
                            mask_idx = each_process_wo * C0 * h_idx

                            self._vector_op("vadd", col2img_fp32_ub[col_index:],
                                            grad_sel_ub_fp32[mask_idx:],
                                            col2img_fp32_ub[col_index:], "float32",
                                            each_process_wo * C0 // 2,
                                            stride_cofig=(self.stride_w * 2, self.stride_w * 2, 2,
                                                          self.stride_w * 16, self.stride_w * 16,
                                                          16))
                            self._vector_op("vadd", col2img_fp32_ub[col_index + 8:],
                                            grad_sel_ub_fp32[mask_idx + 8:],
                                            col2img_fp32_ub[col_index + 8:], "float32",
                                            each_process_wo * C0 // 2,
                                            stride_cofig=(self.stride_w * 2, self.stride_w * 2, 2,
                                                          self.stride_w * 16, self.stride_w * 16,
                                                          16))

                col2img_fp16_ub = self.tik_instance.Tensor("float16",
                                                           col2img_ub_shape,
                                                           name="col2img_fp16_ub",
                                                           scope=tik.scope_ubuf)
                self._vconv(col2img_fp32_ub, 0, col2img_fp16_ub, 0,
                            cal_shape_ele(col2img_fp32_ub.shape), "float32")
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
                self.tik_instance.scalar_min(wi_max, self.wi + pad_left, end_w)
                mov_len_w.set_as(wi_max - wi_min)
                start_pos_w.set_as(wi_min - cut_wo_nums_index *
                                   each_process_wo * self.stride_w)

                with self.tik_instance.if_scope(
                        tik.all(cut_wo_nums_index < cut_wo_nums - 1, mov_len_h > 0)):
                    self.tik_instance.data_move(self.res_gm[offset_gm_inside],
                                                col2img_fp16_ub[start_pos_h * each_process_wi *
                                                                C0 + start_pos_w * C0], 0,
                                                mov_len_h, mov_len_w * C0 // 16,
                                                each_process_wi - mov_len_w,
                                                self.wi - mov_len_w)
                    offset_gm_inside.set_as(offset_gm_inside + mov_len_w * C0)
                    self.offset_gm.set_as(self.offset_gm + mov_len_h * mov_len_w * C0)

                with self.tik_instance.if_scope(tik.all(mov_len_h > 0, mov_len_w > 0)):
                    if remain_wo_nums == 0:
                        with self.tik_instance.if_scope(cut_wo_nums_index == cut_wo_nums - 1):
                            last_valid_wi = self.wi - ((cut_wo_nums - 1) * each_process_wo *
                                                       self.stride_w - pad_left)
                            last_valid_wi = min(last_valid_wi, self.wi)

                            if last_valid_wi <= each_process_wi:
                                self.tik_instance.data_move(
                                    self.res_gm[offset_gm_inside],
                                    col2img_fp16_ub[start_pos_h * each_process_wi *
                                                    C0 + start_pos_w * C0],
                                    0, mov_len_h,
                                    last_valid_wi * C0 // 16,
                                    each_process_wi - last_valid_wi, self.wi - last_valid_wi)
                                offset_gm_inside.set_as(offset_gm_inside + last_valid_wi * C0)
                                self.offset_gm.set_as(
                                    self.offset_gm + mov_len_h * last_valid_wi * C0)
                            else:
                                self.tik_instance.data_move(self.res_gm[offset_gm_inside],
                                                            col2img_fp16_ub[
                                                                start_pos_h * each_process_wi *
                                                                C0 + start_pos_w * C0], 0,
                                                            mov_len_h, each_process_wi * C0 // 16,
                                                            0,
                                                            self.wi - each_process_wi)
                                offset_gm_inside.set_as(offset_gm_inside + each_process_wi * C0)

                                remain_wi = last_valid_wi - each_process_wi
                                temp_zero = self.tik_instance.Tensor("float16",
                                                                     (remain_wi * C0,),
                                                                     name='temp_zero',
                                                                     scope=tik.scope_ubuf)
                                self._vector_dup(temp_zero, 0, temp_zero.shape,
                                                 self.scalar_zero_fp16, temp_zero.dtype)
                                with self.tik_instance.for_range(0, mov_len_h) as index_0:
                                    self.tik_instance.data_move(
                                        self.res_gm[offset_gm_inside + index_0 * self.wi * C0],
                                        temp_zero, 0, 1,
                                        cal_shape_ele(temp_zero.shape) // 16, 0, 0)
                                offset_gm_inside.set_as(offset_gm_inside + remain_wi * C0)
                                self.offset_gm.set_as(
                                    self.offset_gm + mov_len_h * last_valid_wi * C0)
                    else:
                        with self.tik_instance.if_scope(cut_wo_nums_index == cut_wo_nums - 1):
                            self.tik_instance.data_move(self.res_gm[offset_gm_inside],
                                                        col2img_fp16_ub[
                                                            start_pos_h * each_process_wi *
                                                            C0 + start_pos_w * C0], 0,
                                                        mov_len_h, mov_len_w * C0 // 16,
                                                        each_process_wi - mov_len_w,
                                                        self.wi - mov_len_w)
                            offset_gm_inside.set_as(offset_gm_inside + mov_len_w * C0)
                            self.offset_gm.set_as(self.offset_gm + mov_len_h * mov_len_w * C0)

                # move back to init col2img_fp16 tensor
                with self.tik_instance.if_scope(cut_wo_nums_index < cut_wo_nums - 1):
                    # mov h overlap to L1
                    if self.kh > self.stride_h:
                        with self.tik_instance.for_range(0, self.kh - self.stride_h) as index_s:
                            self.tik_instance.data_move(
                                overlap_l1[index_s * overlap_l1_w +
                                           cut_wo_nums_index * each_process_wo *
                                           self.stride_w * C0],
                                col2img_fp32_ub[self.stride_h *
                                                each_process_wi * C0 + each_process_wi *
                                                C0 * index_s],
                                0, 1, self.stride_w * each_process_wo * C0 // 8, 0, 0)

                    if self.kw > self.stride_w:
                        with self.tik_instance.for_range(0, self.kh) as index_kh:
                            offset = [index_kh * each_process_wi * C0,
                                      index_kh * each_process_wi * C0 +
                                      self.stride_w * each_process_wo * C0]
                            self._vector_op("vmuls", col2img_fp32_ub, 1.0, col2img_fp32_ub,
                                            col2img_fp32_ub.dtype,
                                            (each_process_wi - each_process_wo * self.stride_w) * C0,
                                            None, offset)
                        with self.tik_instance.for_range(0, self.kh) as index_kh:
                            self._vector_dup(col2img_fp32_ub,
                                             index_kh * each_process_wi * C0 +
                                             (each_process_wi - self.stride_w *
                                              each_process_wo) * C0,
                                             (self.stride_w * each_process_wo * C0,),
                                             self.scalar_zero,
                                             col2img_fp32_ub.dtype)

                    else:
                        self._vector_dup(col2img_fp32_ub, 0, col2img_fp32_ub.shape,
                                         self.scalar_zero, col2img_fp32_ub.dtype)

                with self.tik_instance.else_scope():
                    if remain_wo_nums > 0:
                        # mov h overlap to L1
                        if self.kh > self.stride_h:
                            with self.tik_instance.for_range(0, self.kh - self.stride_h) as index_s:
                                self.tik_instance.data_move(
                                    overlap_l1[index_s * overlap_l1_w +
                                               cut_wo_nums_index * each_process_wo *
                                               self.stride_w * C0],
                                    col2img_fp32_ub[self.stride_h *
                                                    each_process_wi * C0 + each_process_wi *
                                                    C0 * index_s],
                                    0, 1, self.stride_w * each_process_wo * C0 // 8, 0, 0)

                        if self.kw > self.stride_w:
                            with self.tik_instance.for_range(0, self.kh) as index_kh:
                                offset = [index_kh * each_process_wi * C0,
                                          index_kh * each_process_wi * C0 +
                                          self.stride_w * each_process_wo * C0]
                                self._vector_op("vmuls", col2img_fp32_ub, 1.0, col2img_fp32_ub,
                                                col2img_fp32_ub.dtype,
                                                (each_process_wi - each_process_wo *
                                                 self.stride_w) * C0,
                                                None, offset)
                            with self.tik_instance.for_range(0, self.kh) as index_kh:
                                self._vector_dup(col2img_fp32_ub,
                                                 index_kh * each_process_wi * C0 +
                                                 (each_process_wi - self.stride_w *
                                                  each_process_wo) * C0,
                                                 (self.stride_w * each_process_wo * C0,),
                                                 self.scalar_zero,
                                                 col2img_fp32_ub.dtype)
                        else:
                            self._vector_dup(col2img_fp32_ub, 0, col2img_fp32_ub.shape,
                                             self.scalar_zero, col2img_fp32_ub.dtype)
                    else:
                        if self.kh > self.stride_h:
                            with self.tik_instance.for_range(
                                    0, self.kh - self.stride_h) as index_s:
                                self.tik_instance.data_move(
                                    overlap_l1[index_s * overlap_l1_w + (cut_wo_nums - 1) *
                                               each_process_wo * self.stride_w * C0],
                                    col2img_fp32_ub[self.stride_h * each_process_wi * C0 +
                                                    each_process_wi * C0 * index_s],
                                    0, 1, each_process_wi * C0 // 8, 0, 0)

            if remain_wo_nums:
                each_process_remain_div16 = ceil_div(remain_wo_nums, 16)
                if self.kw > self.stride_w:
                    each_process_remain_wi = (remain_wo_nums - 1) * self.stride_w + self.kw
                else:
                    each_process_remain_wi = remain_wo_nums * self.stride_w

                start_h.set_as(ho_index * self.stride_h)
                end_h.set_as(ho_index * self.stride_h + each_process_hi)

                # move l1 to UB to init the col2img_fp16 tensor. if stride < kernel, there has
                # overlap between each ho.
                if self.kh > self.stride_h:
                    with self.tik_instance.if_scope(ho_index != 0):
                        start_pos = (each_process_wi - self.stride_w * each_process_wo) * C0
                        with self.tik_instance.for_range(0, self.kh - self.stride_h) as index_khs:
                            self.tik_instance.data_move(
                                col2img_fp32_ub[index_khs * each_process_wi * C0 + start_pos],
                                overlap_l1[index_khs * overlap_l1_w +
                                           cut_wo_nums * each_process_wo *
                                           self.stride_w * C0 + start_pos],
                                0, 1, self.stride_w * remain_wo_nums * C0 // 8, 0, 0)

                # mov forward output and grad to UB
                ori_output_ub_shape = (each_process_wo_div16, 16, C0)
                ori_output_ub = self.tik_instance.Tensor(self.dtype,
                                                         ori_output_ub_shape,
                                                         name='ori_output_ub',
                                                         scope=tik.scope_ubuf)
                self.tik_instance.data_move(
                    ori_output_ub,
                    self.ori_output_gm[((n_index * self.c1 + c1_index) *
                                        self.ho + start_ho_index + ho_index) * self.wo * C0 +
                                       cut_wo_nums * each_process_wo * C0],
                    0, 1, remain_wo_nums * C0 // 16, 0, 0)

                grad_ub = self.tik_instance.Tensor(self.dtype, ori_output_ub_shape,
                                                   name='grad_ub',
                                                   scope=tik.scope_ubuf)

                self.tik_instance.data_move(
                    grad_ub, self.grad_gm[((n_index * self.c1 + c1_index) *
                                           self.ho + start_ho_index + ho_index) * self.wo * C0 +
                                          cut_wo_nums * each_process_wo * C0],
                    0, 1, remain_wo_nums * C0 // 16, 0, 0)

                ori_input_col_ub_shape = (each_process_wo_div16 * 16, C0)
                ori_input_col_ub = self.tik_instance.Tensor(self.dtype,
                                                            ori_input_col_ub_shape,
                                                            name='ori_input_col_ub',
                                                            scope=tik.scope_ubuf)

                wo_offset = self.tik_instance.Scalar(dtype='int64',
                                                     name='wo_offset')
                wo_offset.set_as(each_process_wo * cut_wo_nums * self.stride_w - pad_left)
                mask_shape = (ceil_div(cal_shape_ele(ori_output_ub.shape[:2]), MASK128_VALUE) *
                              MASK128_VALUE,)

                mask_not = self.tik_instance.Tensor("uint16",
                                                    mask_shape,
                                                    name='mask_not',
                                                    scope=tik.scope_ubuf)
                mask_or = self.tik_instance.Tensor("uint16",
                                                   mask_shape,
                                                   name='mask_or',
                                                   scope=tik.scope_ubuf)

                # do load3d, calculate mask and grad_x, here we loop kh and kw, so each loop
                # it process one row of output, image output as a window slide on kernel window
                with self.tik_instance.for_range(0, self.kh) as index_h:
                    with self.tik_instance.for_range(0, self.kw) as index_w:
                        self._load3d(index_h, index_w,
                                     start_h, end_h,
                                     ori_input_col_ub,
                                     ori_input_l1, 0, each_process_hi,
                                     self.wi,
                                     each_process_remain_div16,
                                     pad, self.pad_value,
                                     wo_offset, mov_len_hi)
                        mask_ub = self._calc_mask(index_h, index_w, mask_shape, ori_output_ub,
                                                  ori_input_col_ub, mask_or, mask_not)
                        grad_sel_ub = self._vsel_grad_col(mask_ub, grad_ub)
                        grad_sel_ub_fp32 = self.tik_instance.Tensor("float32",
                                                                    grad_sel_ub.shape,
                                                                    name='grad_sel_ub_fp32',
                                                                    scope=tik.scope_ubuf)
                        self._vconv(grad_sel_ub, 0, grad_sel_ub_fp32, 0,
                                    cal_shape_ele(grad_sel_ub.shape), "float16")

                        # each procee ho is 1, so here loop value is 1
                        with self.tik_instance.for_range(0, 1) as h_idx:
                            col_index = index_h * each_process_wi * C0 + index_w * C0 + \
                                        each_process_wi * C0 * self.stride_h * h_idx
                            mask_idx = each_process_wo * C0 * h_idx
                            self._vector_op("vadd", col2img_fp32_ub[col_index:],
                                            grad_sel_ub_fp32[mask_idx:],
                                            col2img_fp32_ub[col_index:], "float32",
                                            remain_wo_nums * C0 // 2,
                                            stride_cofig=(self.stride_w * 2, self.stride_w * 2, 2,
                                                          self.stride_w * 16, self.stride_w * 16,
                                                          16))
                            self._vector_op("vadd", col2img_fp32_ub[col_index + 8:],
                                            grad_sel_ub_fp32[mask_idx + 8:],
                                            col2img_fp32_ub[col_index + 8:], "float32",
                                            remain_wo_nums * C0 // 2,
                                            stride_cofig=(self.stride_w * 2, self.stride_w * 2, 2,
                                                          self.stride_w * 16, self.stride_w * 16,
                                                          16))

                col2img_fp16_ub = self.tik_instance.Tensor("float16",
                                                           col2img_ub_shape,
                                                           name="col2img_fp16_ub",
                                                           scope=tik.scope_ubuf)
                self._vconv(col2img_fp32_ub, 0, col2img_fp16_ub, 0,
                            cal_shape_ele(col2img_fp32_ub.shape), "float32")

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

                # set w direction's paras
                start_w.set_as(cut_wo_nums * each_process_wo * self.stride_w)
                end_w.set_as(cut_wo_nums * each_process_wo *
                             self.stride_w + remain_wo_nums * self.stride_w)
                self.tik_instance.scalar_max(wi_min, pad_left, start_w)
                self.tik_instance.scalar_min(wi_max, self.wi + pad_left, end_w)
                mov_len_w.set_as(wi_max - wi_min)

                with self.tik_instance.if_scope(mov_len_h > 0):
                    last_valid_wi = self.wi - (
                            cut_wo_nums * each_process_wo * self.stride_w - pad_left)
                    last_valid_wi = min(last_valid_wi, self.wi)
                    if last_valid_wi <= each_process_wi:
                        self.tik_instance.data_move(
                            self.res_gm[offset_gm_inside],
                            col2img_fp16_ub[start_pos_h * each_process_wi * C0],
                            0, mov_len_h,
                            last_valid_wi * C0 // 16,
                            each_process_wi - last_valid_wi, self.wi - last_valid_wi)
                        offset_gm_inside.set_as(offset_gm_inside + last_valid_wi * C0)
                        self.offset_gm.set_as(self.offset_gm + mov_len_h * last_valid_wi * C0)
                    else:
                        self.tik_instance.data_move(self.res_gm[offset_gm_inside],
                                                    col2img_fp16_ub[start_pos_h * each_process_wi *
                                                                    C0], 0,
                                                    mov_len_h, each_process_wi * C0 // 16,
                                                    0,
                                                    self.wi - each_process_wi)
                        offset_gm_inside.set_as(offset_gm_inside + each_process_wi * C0)

                        remain_wi = last_valid_wi - each_process_wi
                        temp_zero = self.tik_instance.Tensor("float16",
                                                             (remain_wi * C0,),
                                                             name='temp_zero',
                                                             scope=tik.scope_ubuf)
                        self._vector_dup(temp_zero, 0, temp_zero.shape,
                                         self.scalar_zero_fp16, temp_zero.dtype)
                        with self.tik_instance.for_range(0, mov_len_h) as index_0:
                            self.tik_instance.data_move(
                                self.res_gm[offset_gm_inside + index_0 * self.wi * C0],
                                temp_zero, 0, 1,
                                cal_shape_ele(temp_zero.shape) // 16, 0, 0)
                        offset_gm_inside.set_as(offset_gm_inside + remain_wi * C0)
                        self.offset_gm.set_as(self.offset_gm + mov_len_h * last_valid_wi * C0)

                if self.kh > self.stride_h:
                    with self.tik_instance.for_range(0, self.kh - self.stride_h) as index_s:
                        self.tik_instance.data_move(
                            overlap_l1[index_s * overlap_l1_w + cut_wo_nums *
                                       each_process_wo * self.stride_w * C0],
                            col2img_fp32_ub[self.stride_h * each_process_wi *
                                            C0 + each_process_wi * C0 * index_s],
                            0, 1, each_process_remain_wi * C0 // 8, 0, 0)
                if self.kw <= self.stride_w:
                    self._vector_dup(col2img_fp32_ub, 0, col2img_fp32_ub.shape,
                                     self.scalar_zero, col2img_fp32_ub.dtype)

            with self.tik_instance.if_scope(mov_len_h > 0):
                remained_hi.set_as(remained_hi - mov_len_h)

            with self.tik_instance.if_scope(tik.all(ho_index == mov_len_ho - 1, remained_hi > 0)):
                temp_zero = self.tik_instance.Tensor("float16",
                                                     (self.wi, C0),
                                                     name="temp_zero",
                                                     scope=tik.scope_ubuf)
                self._vector_dup(temp_zero, 0,
                                 temp_zero.shape,
                                 self.scalar_zero_fp16,
                                 temp_zero.dtype)
                with self.tik_instance.for_range(0, remained_hi) as repeate_index:
                    self.tik_instance.data_move(
                        self.res_gm[self.offset_gm],
                        temp_zero, 0,
                        1, cal_shape_ele(temp_zero.shape) // 16, 0, 0)
                    self.offset_gm.set_as(self.offset_gm +
                                          cal_shape_ele(
                                              temp_zero.shape))

    # 'pylint: disable=unused-variable,no-self-use
    @staticmethod
    def _get_block_num(block_num):
        if block_num > CORE_NUM:
            real_block = CORE_NUM
            block_cycle = (block_num + CORE_NUM - 1) // CORE_NUM
        else:
            real_block = block_num
            block_cycle = 1
        return real_block, block_cycle

    # 'pylint: disable=unused-variable,
    def _if_block(self, ho_outer, ho_inner):
        if ho_inner <= 1:
            return False
        if self.stride_h >= self.kh:
            return True
        overlap_num = math.ceil((self.kh - self.stride_h) * 1.0 / self.stride_h)
        if self.kh > self.stride_h:
            shape_hi = (ho_inner + overlap_num - 1) * self.stride_h + self.kh
        else:
            shape_hi = ho_inner * self.stride_h
        need_cut_L1, need_cut_Ho, need_cut_Wo, \
        each_process_ho, each_process_wo = \
            self._tilling_factor((shape_hi, self.wi, C0), self.pad)

        times = math.ceil(ho_inner * 1.0 / each_process_ho)
        overlaps = overlap_num * times
        if (overlaps + ho_inner) * 1.0 / ho_inner >= ho_outer:
            return False
        return True

    # 'pylint: disable=unused-variable
    def tik_instance_function(self, kernel_name):
        """
        main function of tik_instance
        """
        block_num = self.n * self.c1
        real_block, block_cycle = self._get_block_num(block_num)
        if self.dtype == "float16":
            self.pad_value = MIN_VALUE_FP16
        else:
            error_manager_vector.raise_err_input_dtype_not_supported("max_pool_v3", "dtype",
                                                                     "float16", str(self.dtype))
        pad_calc_wo, pad_calc_ho, pad_pads = \
            self._padding_mode(self.ori_input_shape, self.ksize, self.strides,
                               self.padding, self.pads, self.global_pooling, self.ceil_mode)
        self.pad = (int(pad_pads[0]), int(pad_pads[1]), int(pad_pads[2]), int(pad_pads[3]))
        self.pad_left, self.pad_right, self.pad_top, self.pad_bottom = self.pad
        if pad_calc_ho != self.ho or pad_calc_wo != self.wo:
            error_manager_vector.raise_err_specific_reson("max_pool_v3", "Wrong ori_output shape")

        # nc do block is not enough, but ncho is enough
        # real_block == 32 or block_num * self.ho < 32
        if real_block == CORE_NUM:
            need_cut_L1, need_cut_Ho, need_cut_Wo, \
            each_process_ho, each_process_wo = \
                self._tilling_factor((self.hi, self.wi, C0), self.pad)
            with self.tik_instance.for_range(0, real_block, block_num=real_block) as block_index:
                with self.tik_instance.for_range(0, block_cycle) as cycle_index:
                    n_index = self.tik_instance.Scalar(dtype='int64', name='n_axis')
                    c1_index = self.tik_instance.Scalar(dtype='int64', name='c1_index')
                    index_sum = self.tik_instance.Scalar(dtype='int64', name='index_sum')
                    index_sum.set_as(block_index * block_cycle + cycle_index)
                    with self.tik_instance.if_scope(index_sum < block_num):
                        n_index.set_as(index_sum // self.c1)
                        c1_index.set_as(index_sum % self.c1)
                        shape = (self.ho, self.wo, self.hi, self.wi)
                        self.offset_gm.set_as(
                            (n_index * self.c1 + c1_index) * self.hi * self.wi * C0)
                        if need_cut_L1 and need_cut_Ho and need_cut_Wo:
                            self._tilling_l1_ho_wo(each_process_wo, n_index, c1_index,
                                                   self.ho, self.hi,
                                                   self.ho, self.hi,
                                                   0, 0,
                                                   0,
                                                   None, shape, self.pad)
                        elif need_cut_L1 and need_cut_Ho:
                            self._tilling_l1_ho_only(each_process_ho, n_index, c1_index,
                                                     self.ho, self.hi,
                                                     self.ho, self.hi,
                                                     0, 0,
                                                     0,
                                                     None, shape, self.pad)

                        elif need_cut_Ho:
                            self._tilling_ho_only(each_process_ho, n_index, c1_index,
                                                  self.ho, self.hi,
                                                  self.ho, self.hi,
                                                  0, 0,
                                                  0,
                                                  None, shape, self.pad)
                        else:
                            self._not_tilling(n_index, c1_index,
                                              self.ho, self.hi,
                                              self.ho, self.hi,
                                              0, 0,
                                              0, None, shape, self.pad)

        else:
            # calculate how to tiling H to block nums
            div_list = self._get_core_divlist()
            block_num_inner, block_num_outer = 0, 0
            for i in div_list:
                if block_num >= i:
                    if self.ho >= CORE_NUM // i:
                        block_num_outer = i
                        block_num_inner = (block_num + i - 1) // i
                        break
            if block_num * self.ho < CORE_NUM:
                ho_outer = self.ho
                block_num_outer = block_num
                block_num_inner = 1
            else:
                if block_num_outer == 0:
                    ho_outer = CORE_NUM // block_num
                    block_num_outer = block_num
                    block_num_inner = 1
                else:
                    ho_outer = CORE_NUM // block_num_outer
            ho_inner = int(math.ceil(self.ho * 1.0 / ho_outer))
            # ho inner > 2, do tilling Ho
            ho_outer = int(math.ceil(self.ho * 1.0 / ho_inner))

            if self._if_block(ho_outer, ho_inner):
                self.tile_h_to_block = True
                with self.tik_instance.for_range(0, block_num_outer * ho_outer,
                                                 block_num=block_num_outer *
                                                           ho_outer) as block_outer_index:
                    with self.tik_instance.for_range(0, block_num_inner) as block_innner_index:
                        nc1_index = self.tik_instance.Scalar(dtype='int64', name='nc1_index')
                        nc1_index.set_as(
                            block_outer_index // ho_outer * block_num_inner + block_innner_index)
                        with self.tik_instance.if_scope(nc1_index < block_num):
                            n_index = self.tik_instance.Scalar(dtype='int64', name='n_index')
                            c1_index = self.tik_instance.Scalar(dtype='int64', name='c1_index')
                            ho_outer_index = self.tik_instance.Scalar(dtype='int64',
                                                                      name='ho_outer_index')
                            offset_gm_block = self.tik_instance.Scalar(dtype='int64',
                                                                       name='offset_gm_block')
                            n_index.set_as(nc1_index // self.c1)
                            c1_index.set_as(nc1_index % self.c1)
                            ho_outer_index.set_as(block_outer_index % ho_outer)

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
                            start_ho_index.set_as(ho_outer_index * ho_inner)
                            start_hi_index.set_as(start_ho_index * self.stride_h)
                            actual_start_ho_index.set_as(start_ho_index)
                            actual_start_hi_index.set_as(start_hi_index)
                            start_threshold.set_as(0)

                            with self.tik_instance.if_scope(start_hi_index <= self.pad_top):
                                offset_gm_block.set_as(
                                    (n_index * self.c1 + c1_index) * self.hi * self.wi * C0)
                                pad_top_block.set_as(self.pad_top - start_hi_index)
                                self.tik_instance.scalar_max(start_threshold, start_threshold,
                                                             pad_top_block)
                                actual_start_hi_index.set_as(0)
                            with self.tik_instance.else_scope():
                                offset_gm_block.set_as(((n_index * self.c1 + c1_index) * self.hi +
                                                        start_hi_index - self.pad_top) * self.wi * C0)
                                pad_top_block.set_as(0)
                                actual_start_hi_index.set_as(actual_start_hi_index - self.pad_top)

                            with self.tik_instance.if_scope(ho_outer_index != ho_outer - 1):
                                each_process_ho_block.set_as(ho_inner)
                            with self.tik_instance.else_scope():
                                each_process_ho_block.set_as(self.ho - ho_inner * (ho_outer - 1))
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

                            with self.tik_instance.if_scope(ho_outer_index == ho_outer - 1):
                                each_process_hi_block.set_as(self.hi + self.pad_top - start_hi_index)
                            with self.tik_instance.if_scope(
                                    start_hi_index + each_process_hi_block > self.hi + self.pad_top):
                                each_process_hi_block.set_as(self.hi + self.pad_top - start_hi_index)

                            pad = (self.pad_left, self.pad_right, pad_top_block, pad_bottom_block)
                            if self.kh > self.stride_h:
                                shape_ho = ho_inner + overlap_num
                                shape_hi = (ho_inner + overlap_num - 1) * self.stride_h + self.kh
                            else:
                                shape_ho = ho_inner
                                shape_hi = ho_inner * self.stride_h
                            if self.hi - ho_inner * self.stride_h * ho_outer > 0:
                                shape_hi += (self.hi - ho_inner * self.stride_h * ho_outer)
                            shape = (shape_ho, self.wo, shape_hi, self.wi)

                            need_cut_L1, need_cut_Ho, need_cut_Wo, \
                            each_process_ho, each_process_wo = \
                                self._tilling_factor((shape_hi, self.wi, C0), self.pad)

                            if need_cut_L1 and need_cut_Ho and need_cut_Wo:
                                self._tilling_l1_ho_wo(each_process_wo, n_index, c1_index,
                                                       each_process_ho_block, each_process_hi_block,
                                                       mov_len_ho, mov_len_hi,
                                                       actual_start_ho_index, actual_start_hi_index,
                                                       start_threshold,
                                                       offset_gm_block, shape, pad)
                            elif need_cut_L1 and need_cut_Ho:
                                self._tilling_l1_ho_only(each_process_ho, n_index, c1_index,
                                                         each_process_ho_block,
                                                         each_process_hi_block,
                                                         mov_len_ho, mov_len_hi,
                                                         actual_start_ho_index,
                                                         actual_start_hi_index,
                                                         start_threshold,
                                                         offset_gm_block, shape, pad)

                            elif need_cut_Ho:
                                self._tilling_ho_only(each_process_ho, n_index, c1_index,
                                                      each_process_ho_block, each_process_hi_block,
                                                      mov_len_ho, mov_len_hi,
                                                      actual_start_ho_index, actual_start_hi_index,
                                                      start_threshold,
                                                      offset_gm_block, shape, pad)
                            else:
                                self._not_tilling(n_index, c1_index,
                                                  each_process_ho_block, each_process_hi_block,
                                                  mov_len_ho, mov_len_hi,
                                                  actual_start_ho_index, actual_start_hi_index,
                                                  start_threshold,
                                                  offset_gm_block, shape, pad)

            else:
                nc1 = self.n * self.c1
                block = CORE_NUM
                while nc1 % block != 0:
                    block = block - 1
                nc1 = nc1 // block

                need_cut_L1, need_cut_Ho, need_cut_Wo, \
                each_process_ho, each_process_wo = \
                    self._tilling_factor((self.hi, self.wi, C0), self.pad)
                with self.tik_instance.for_range(0, block, block_num=block) as block_index:
                    with self.tik_instance.for_range(0, nc1, thread_num=1) as nc1_index:
                        self.offset_gm.set_as((block_index * nc1 + nc1_index) *
                                              self.hi * self.wi * C0)
                        n_index = (block_index * nc1 + nc1_index) // self.c1
                        c1_index = (block_index * nc1 + nc1_index) % self.c1

                        shape = (self.ho, self.wo, self.hi, self.wi)
                        if need_cut_L1 and need_cut_Ho and need_cut_Wo:
                            self._tilling_l1_ho_wo(each_process_wo, n_index, c1_index,
                                                   self.ho, self.hi,
                                                   self.ho, self.hi,
                                                   0, 0,
                                                   0,
                                                   None, shape, self.pad)
                        elif need_cut_L1 and need_cut_Ho:
                            self._tilling_l1_ho_only(each_process_ho, n_index, c1_index,
                                                     self.ho, self.hi,
                                                     self.ho, self.hi,
                                                     0, 0,
                                                     0,
                                                     None, shape, self.pad)

                        elif need_cut_Ho:
                            self._tilling_ho_only(each_process_ho, n_index, c1_index,
                                                  self.ho, self.hi,
                                                  self.ho, self.hi,
                                                  0, 0,
                                                  0,
                                                  None, shape, self.pad)
                        else:
                            self._not_tilling(n_index, c1_index,
                                              self.ho, self.hi,
                                              self.ho, self.hi,
                                              0, 0,
                                              0,
                                              None, shape, self.pad)

        self.tik_instance.BuildCCE(kernel_name=kernel_name, inputs=(
            self.ori_input_gm, self.ori_output_gm, self.grad_gm),
                                   outputs=(self.res_gm,), enable_l2=False)
        return self.tik_instance


class MaxpoolV3GradAtomic:
    """
        Function: use to store concat base parameters
    """
    def __init__(self, shape_list, params):
        # forward_in_shape, forward_ou_shape, grad_shape, ou_shape
        # list(ksize), list(strides), pads, dtype
        self.forward_in_shape = shape_list[0]
        self.forward_ou_shape = shape_list[1]
        self.grad_shape = shape_list[2]
        self.ou_shape = shape_list[3]
        self.core_ou_shape = []
        self.core_in_shape = []

        self.n = self.forward_in_shape[0]
        self.d = self.forward_in_shape[1]
        self.c1 = self.forward_in_shape[2]
        self.h = self.forward_in_shape[3]
        self.w = self.forward_in_shape[4]
        self.c0 = self.forward_in_shape[5]

        self.ksize = params[0]
        self.strides = params[1]
        self.dtype = params[3]
        self.kernel_name = params[4]
        self.pads = params[5].upper()

        self.kd = self.ksize[1]
        self.kh = self.ksize[2]
        self.kw = self.ksize[3]
        self.sd = self.strides[1]
        self.sh = self.strides[2]
        self.sw = self.strides[3]

        if self.pads in ["CALCULATED", ]:
            # Pytorch
            self.pads = "SAME"
            self.pad = [[params[2][0], params[2][1]],
                        [params[2][2], params[2][3]],
                        [params[2][4], params[2][5]]]
            self.do, self.ho, self.wo = self.grad_shape[1], self.grad_shape[3], \
                                        self.grad_shape[4]
        else:
            # TF
            self.do, self.ho, self.wo, self.pad = self._padding_mode()
        self.overlap_d = self._overlap_mode(self.sd, self.kd, self.do, self.d)
        self.overlap_h = self._overlap_mode(self.sh, self.kh, self.ho, self.h)
        self.overlap_w = self._overlap_mode(self.sw, self.kw, self.wo, self.w)
        self.di_invalid, \
        self.hi_invalid, self.wi_invalid = self._invalid_part()

        self.num_bit = 2
        self.num_bit_fp32 = 4
        self.mask_fp16 = 128
        self.mask_fp32 = 64
        self.ub_maxsize = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) // self.num_bit
        self.L1_maxsize = tbe_platform.get_soc_spec(tbe_platform.L1_SIZE) // self.num_bit
        self.orig_x_gm = None
        self.orig_y_gm = None
        self.grads_gm = None
        self.ou_y_gm = None

        self.check_load3d_support = tbe_platform.api_check_support("tik.load3dv1")

    @staticmethod
    def set_vector_dup(tik_instance, psm, dst, idx, number, dtype):
        """
        set_vector_dup
        """
        # idx is begin_index in dst,
        # must be 32B align
        if dtype == "float16":
            mask = 128
        else:
            mask = 64

        dup_psm = MAX_VECTOR_REPEATE_TIME * mask
        dup_repeat_merchant = psm // dup_psm
        dup_repeat_remainder = psm % dup_psm
        dst_blk_stride = 1
        dst_rep_stride = 8

        with tik_instance.for_range(0, dup_repeat_merchant) as i:
            tik_instance.vector_dup(mask,
                                    dst[idx + i * dup_psm],
                                    number,
                                    MAX_VECTOR_REPEATE_TIME,
                                    dst_blk_stride,
                                    dst_rep_stride)

        if dup_repeat_remainder != 0:
            repeats = dup_repeat_remainder // mask
            dup_remainder = dup_repeat_remainder % mask
            if repeats != 0:
                tik_instance.vector_dup(mask,
                                        dst[idx + dup_repeat_merchant * dup_psm],
                                        number,
                                        repeats,
                                        dst_blk_stride,
                                        dst_rep_stride)
            if dup_remainder != 0:
                tik_instance.vector_dup(dup_remainder,
                                        dst[idx + dup_repeat_merchant * dup_psm +
                                            repeats * mask],
                                        number,
                                        1,
                                        dst_blk_stride,
                                        dst_rep_stride)

    @staticmethod
    def norm_data_move(tik_instance, src_buf, dst_buf, in_list):
        """
        norm_data_move
        """
        src_idx, dst_idx = in_list[-2], in_list[-1]
        n_burst, burst_len = in_list[0], in_list[1]
        src_stride, dst_stride = in_list[2], in_list[3]

        tik_instance.data_move(dst_buf[dst_idx],
                               src_buf[src_idx],
                               0,
                               n_burst,
                               burst_len,
                               src_stride,
                               dst_stride)

    @staticmethod
    def grad(tik_instance, split_model,
             param, total_num, core_num, func):
        """
        grad
        """
        # just tiling do ho
        # support valid
        core_loop = tik_instance.Scalar("int64")
        sum_core = tik_instance.Scalar("int64")
        with tik_instance.for_range(0, core_num,
                                    block_num=core_num) as blk_idx:

            core_loop_uint64, sum_core_uint64 = cal_core(tik_instance, total_num,
                                                          blk_idx, core_num)
            core_loop.set_as(core_loop_uint64)
            sum_core.set_as(sum_core_uint64)

            func(tik_instance, core_loop, sum_core, split_model, param)

    @staticmethod
    def _overlap_mode(stride, size, xo, xi):
        # xo: direction of x can be slided by xo times
        # xi: the length of x
        if xo == 1:
            # If xo is 1,only xi >= stride, stride has work.
            # If xo is 1 and xi < stride, only kernel has work
            if xi >= stride:
                overlap = size - stride
            else:
                overlap = 0
        else:
            overlap = size - stride

        return overlap

    @staticmethod
    def _check_cut_model(cut_model, split_model, all_do, core_branch):
        # "not_tiling": 0
        # "tiling_do": 1
        # "tiling_do_ho": 2
        # "tiling_do_ho_wo": 3
        branch_list = ["not_tiling", "tiling_do",
                       "tiling_do_ho", "tiling_do_ho_wo"]

        if cut_model == [True, False, False]:
            if split_model[0] == all_do:
                model = 0
            else:
                model = 1
        elif cut_model == [True, True, False]:
            model = 2
        else:
            model = 3

        model = max(model, core_branch)
        return branch_list[model]

    @staticmethod
    def _ultimate_data_move(tik_instance, src_buf, dst_buf, in_list, num_bit):
        src_idx, dst_idx = in_list[-2], in_list[-1]
        n_burst, burst_len = in_list[0], in_list[1]
        src_stride, dst_stride = in_list[2], in_list[3]

        with tik_instance.for_range(0, n_burst) as i:
            src_idx += i * (src_stride + burst_len) * BLOCK_SIZE // num_bit
            dst_idx += i * (dst_stride + burst_len) * BLOCK_SIZE // num_bit

            tik_instance.data_move(dst_buf[dst_idx],
                                   src_buf[src_idx],
                                   0, 1, burst_len, 0, 0)

    @staticmethod
    def _vconv(tik_instance, src, src_start, dst,
               dst_start, ele_num, src_dtype):
        total_repeat_time = ele_num // VECTOR_FP32_SIZE
        remain_ele = ele_num % VECTOR_FP32_SIZE
        mask_value = VECTOR_FP32_SIZE

        repeat_max_time = total_repeat_time // MAX_VECTOR_REPEATE_TIME
        remain_repeat_time = total_repeat_time % MAX_VECTOR_REPEATE_TIME

        if src_dtype == 'float16':
            src_stride, dst_stride = 4, 8
            if repeat_max_time > 0:
                with tik_instance.for_range(0, repeat_max_time) as loop1:
                    tik_instance.vconv(
                        MASK64_VALUE, "",
                        dst[
                            dst_start + loop1 * MAX_VECTOR_REPEATE_TIME * mask_value],
                        src[
                            src_start + loop1 * MAX_VECTOR_REPEATE_TIME * mask_value],
                        MAX_VECTOR_REPEATE_TIME, 1, 1, dst_stride, src_stride)
            if remain_repeat_time > 0:
                tik_instance.vconv(
                    MASK64_VALUE, "",
                    dst[
                        dst_start + repeat_max_time * MAX_VECTOR_REPEATE_TIME * mask_value],
                    src[
                        src_start + repeat_max_time * MAX_VECTOR_REPEATE_TIME * mask_value],
                    remain_repeat_time, 1, 1, dst_stride, src_stride)
            if remain_ele > 0:
                tik_instance.vconv(
                    remain_ele, "",
                    dst[dst_start + repeat_max_time * MAX_VECTOR_REPEATE_TIME *
                        mask_value + remain_repeat_time * mask_value],
                    src[src_start + repeat_max_time * MAX_VECTOR_REPEATE_TIME *
                        mask_value + remain_repeat_time * mask_value],
                    1, 1, 1, dst_stride, src_stride)

        else:
            src_stride, dst_stride = 8, 4
            if repeat_max_time > 0:
                with tik_instance.for_range(0, repeat_max_time) as loop1:
                    tik_instance.vconv(
                        MASK64_VALUE, "",
                        dst[
                            dst_start + loop1 * MAX_VECTOR_REPEATE_TIME * mask_value],
                        src[
                            src_start + loop1 * MAX_VECTOR_REPEATE_TIME * mask_value],
                        MAX_VECTOR_REPEATE_TIME, 1, 1, dst_stride, src_stride)
            if remain_repeat_time > 0:
                tik_instance.vconv(
                    MASK64_VALUE, "",
                    dst[
                        dst_start + repeat_max_time * MAX_VECTOR_REPEATE_TIME * mask_value],
                    src[
                        src_start + repeat_max_time * MAX_VECTOR_REPEATE_TIME * mask_value],
                    remain_repeat_time, 1, 1, dst_stride, src_stride)
            if remain_ele > 0:
                tik_instance.vconv(
                    remain_ele, "",
                    dst[dst_start + repeat_max_time * MAX_VECTOR_REPEATE_TIME *
                        mask_value + remain_repeat_time * mask_value],
                    src[src_start + repeat_max_time * MAX_VECTOR_REPEATE_TIME *
                        mask_value + remain_repeat_time * mask_value],
                    1, 1, 1, dst_stride, src_stride)

    @staticmethod
    def _rewrite_fmap(tik_instance, operator,
                      src1, src2, dst, dtype, once_elem,
                      repeat_times, shape_map, shape_grad, config=None):
        # once_elem: amount of data processed at a time in the Wo direction.
        # shape_map: container size of src1[1:].
        # shape_grad: valid data size of src2.

        _, w, c0 = shape_map[0], shape_map[1], shape_map[2]
        _, wo = shape_grad[0], shape_grad[1]
        config = list(config)
        if dtype == "float16":
            max_mask = 128
            num_block = 8
            block_size = 16
        else:
            max_mask = 64
            num_block = 8
            block_size = 8

        # num_instr_loop_w: num of instructions on direct W
        # num_instr_loop_h: num of instructions on direct H
        num_instr_loop_w = math.ceil(once_elem/max_mask)
        remain_mask = once_elem % max_mask
        if remain_mask == 0 and once_elem != 0:
            remain_mask = max_mask
        num_instr_loop_h = math.ceil(repeat_times/MAX_VECTOR_REPEATE_TIME)
        remain_repeat = repeat_times % MAX_VECTOR_REPEATE_TIME
        if remain_repeat == 0 and repeat_times != 0:
            remain_repeat = MAX_VECTOR_REPEATE_TIME

        dst_offset = src1_offset = src2_offset = 0
        if operator == "vadd":
            for idx_h, _ in enumerate(range(num_instr_loop_h)):
                for idx_w, _ in enumerate(range(num_instr_loop_w)):
                    src1_offset = idx_w * num_block * config[1] * block_size + \
                                  idx_h * MAX_VECTOR_REPEATE_TIME * w * c0
                    src2_offset = idx_w * num_block * config[2] * block_size + \
                                  idx_h * MAX_VECTOR_REPEATE_TIME * wo * c0
                    dst_offset = idx_w * num_block * config[0] * block_size + \
                                 idx_h * MAX_VECTOR_REPEATE_TIME * w * c0

                    if idx_w < num_instr_loop_w - 1:
                        mask = max_mask
                    else:
                        mask = remain_mask
                    if idx_h < num_instr_loop_h - 1:
                        rep = MAX_VECTOR_REPEATE_TIME
                    else:
                        rep = remain_repeat
                    tik_instance.vadd(mask, dst[dst_offset],
                                      src1[src1_offset], src2[src2_offset],
                                      rep,
                                      config[0], config[1],
                                      config[2], config[3],
                                      config[4], config[5])

    @staticmethod
    def _division_nearest(number, base_num):
        # split number as n0 and n1,
        # return n1, base_num*n0 as new_number and core_num
        n1 = number
        new_base_num = base_num
        for n0 in range(1, number + 1):
            if number % n0 == 0:
                new_base_num = base_num * n0
                n1 = int(number / n0)
                if new_base_num >= CORE_NUM:
                    break
        return n1, new_base_num

    def set_tik_instance(self):
        """
        set tik_instance
        """
        tik_instance = tik.Tik()
        self.set_src_dst_tensor(tik_instance)

        return tik_instance

    def set_src_dst_tensor(self, tik_instance):
        """
        set input and output tensor
        """
        self.orig_x_gm = tik_instance.Tensor(self.dtype,
                                             self.forward_in_shape,
                                             name="orig_x_gm",
                                             scope=tik.scope_gm)

        self.orig_y_gm = tik_instance.Tensor(self.dtype,
                                             self.forward_ou_shape,
                                             name="orig_y_gm",
                                             scope=tik.scope_gm)

        self.grads_gm = tik_instance.Tensor(self.dtype,
                                            self.grad_shape,
                                            name="grads_gm",
                                            scope=tik.scope_gm)

        self.ou_y_gm = tik_instance.Tensor("float32",
                                           self.ou_shape,
                                           name="ou_y_gm",
                                           scope=tik.scope_gm,
                                           is_atomic_add=True)

    def not_tiling_main(self, tik_instance, core_loop, sum_core,
                        model, param):
        """
        not_tiling_main
        """

        do = model[0]
        ho = model[1]
        wo = model[2]
        di, hi, wi = self._infer_dim_return(do, ho, wo, True)
        c0 = self.c0
        c1 = self.c1

        buf_list = self._set_buf_tensor(tik_instance, param)
        l1_in_buf = buf_list[0]
        forward_ou_buf = buf_list[1]
        grad_buf = buf_list[2]
        col_in_buf = buf_list[3]
        mask_buf = buf_list[4]
        mask_or_buf = buf_list[5]
        mask_not_buf = buf_list[6]
        zero_buf = buf_list[7]
        grad_sel_fp16_buf = buf_list[8]
        grad_sel_fp32_buf = buf_list[9]
        f_map_fp32_buf = buf_list[10]
        self.set_vector_dup(tik_instance, param.zero_size,
                            zero_buf, 0, 0, self.dtype)

        with tik_instance.for_range(0, core_loop) as num_core_loop:
            # init
            self.set_vector_dup(tik_instance, param.f_map_fp32_size,
                                f_map_fp32_buf, 0, 0, "float32")
            merchant = (sum_core + num_core_loop) // c1
            remainder = (sum_core + num_core_loop) % c1

            # ----COPY_GM_2_L1_BUF----
            src_orig_x_gm = merchant * prod(self.forward_in_shape[1:]) + \
                            remainder * hi * wi * c0
            gm2l1_shape = [di, hi, wi, c0]
            self._copy_gm_to_l1(tik_instance, l1_in_buf,
                                src_orig_x_gm, 0, gm2l1_shape)

            # ----COPY_ORI_OUTPUT_2_FORWARD_OU_BUF----
            # ----COPY_GRAD_2_GRAD_BUF----
            src_orig_y_gm = merchant * prod(self.forward_ou_shape[1:]) + \
                            remainder * ho * wo * c0
            gm2ub_data_shape = [do, ho, wo, c0]
            self._copy_gm_to_ub(tik_instance, forward_ou_buf,
                                self.orig_y_gm, src_orig_y_gm, gm2ub_data_shape)

            src_grad_gm = src_orig_y_gm
            self._copy_gm_to_ub(tik_instance, grad_buf, self.grads_gm,
                                src_grad_gm, gm2ub_data_shape)

            # ---load3d l1 to col_in_buffer---
            repeat_times = ceil_div(ho*wo, 16)
            # window
            with tik_instance.for_range(0, do, thread_num=1) as idx_do:
                # number of hwc0 in window
                with tik_instance.for_range(0, self.kd, thread_num=1) as idx_d:
                    with tik_instance.for_range(0, self.kh, thread_num=1) as idx_h:
                        with tik_instance.for_range(0, self.kw, thread_num=1) as idx_w:
                            src_l1 = (idx_do * self.sd + idx_d) * hi * wi * c0
                            if self.check_load3d_support:
                                tik_instance.load3dv1(col_in_buf[0],
                                                      l1_in_buf[src_l1],
                                                      [0, 0, 0, 0],
                                                      hi, wi, 0,
                                                      idx_w, idx_h,
                                                      0, 0,
                                                      self.sw, self.sh,
                                                      self.kw, self.kh,
                                                      1, 1, 1, 1,
                                                      repeat_times, 0,
                                                      MIN_VALUE_FP16
                                                      )
                            else:
                                col_in_buf_shape = cal_shape_ele(col_in_buf.shape)
                                self.set_vector_dup(
                                    tik_instance, col_in_buf_shape, col_in_buf, 0, MIN_VALUE_FP16, "float16")
                                img2col(tik_instance, l1_in_buf, col_in_buf, src_l1, 0, idx_h, idx_w, 0, 0,
                                        hi, wi, self.kh, self.kw, self.sh, self.sw, repeat_times, 1, (0, 0, 0, 0))

                            # ---calculate mask---
                            with tik_instance.if_scope(tik.all(idx_d == 0,
                                                               idx_h == 0,
                                                               idx_w == 0)):
                                tik_instance.vcmpv_eq(mask_buf[0],
                                                      forward_ou_buf[idx_do*ho*wo*c0],
                                                      col_in_buf[0],
                                                      math.ceil(ho*wo*c0/VECTOR_FP16_SIZE),
                                                      1, 1, 8, 8)

                                tik_instance.data_move(mask_or_buf[0],
                                                       mask_buf[0], 0, 1,
                                                       param.mask_size//16, 0, 0)

                                tik_instance.vnot(self.mask_fp16,
                                                  mask_not_buf, mask_or_buf,
                                                  param.mask_size // VECTOR_FP16_SIZE,
                                                  1, 1, 8, 8)

                            with tik_instance.else_scope():
                                tik_instance.vcmpv_eq(mask_buf[0],
                                                      forward_ou_buf[idx_do*ho*wo*c0],
                                                      col_in_buf[0],
                                                      math.ceil(ho*wo*c0/VECTOR_FP16_SIZE),
                                                      1, 1, 8, 8)

                                tik_instance.vand(self.mask_fp16, mask_buf,
                                                  mask_not_buf, mask_buf,
                                                  param.mask_size // VECTOR_FP16_SIZE,
                                                  1, 1, 1, 8, 8, 8)

                                tik_instance.vor(self.mask_fp16, mask_or_buf,
                                                 mask_or_buf, mask_buf,
                                                 param.mask_size // VECTOR_FP16_SIZE,
                                                 1, 1, 1, 8, 8, 8)

                                tik_instance.vnot(self.mask_fp16, mask_not_buf,
                                                  mask_or_buf,
                                                  param.mask_size // VECTOR_FP16_SIZE,
                                                  1, 1, 8, 8)

                            # ---vsel(grad,zero,mask)---
                            repeat_times_sel = math.ceil(ho*wo*c0/VECTOR_FP16_SIZE)
                            with tik_instance.for_range(0, repeat_times_sel) as serial:
                                grad_sel_offset = serial * 128
                                grad_offset = serial * 128 + idx_do*ho*wo*c0
                                mask_offset = serial * 8
                                cmp_mask = tik_instance.mov_tensor_to_cmpmask(mask_buf[mask_offset])
                                tik_instance.vsel(self.mask_fp16, 0,
                                                  grad_sel_fp16_buf[grad_sel_offset],
                                                  cmp_mask,
                                                  grad_buf[grad_offset],
                                                  zero_buf,
                                                  1, 1, 1, 1, 8, 8, 0)

                            # ---vconv grad_sel_fp16 to fp32---
                            self._vconv(tik_instance, grad_sel_fp16_buf, 0,
                                        grad_sel_fp32_buf, 0,
                                        param.grad_sel_fp16_size, "float16")

                            # ---rewrite grad_sel_fp32 to f_map_fp32
                            config = (self.sw*2, self.sw*2, 2,
                                      self.sh*wi*2, self.sh*wi*2, wo*2)
                            if check_config(config):
                                with tik_instance.for_range(0, 1) as ho_idx:
                                    map_index = (idx_do*self.sd+idx_d)*hi*wi*c0 + \
                                                (idx_h*wi*c0+idx_w*c0)
                                    mask_index = wo * ho_idx * c0
                                    shape_map_hw = [hi, wi, c0]
                                    shape_grad = [ho, wo, c0]

                                    self._rewrite_fmap(tik_instance, "vadd",
                                                       f_map_fp32_buf[map_index:],
                                                       grad_sel_fp32_buf[mask_index:],
                                                       f_map_fp32_buf[map_index:],
                                                       "float32", wo*c0//2, ho,
                                                       shape_map_hw, shape_grad,
                                                       config=config)

                                    self._rewrite_fmap(tik_instance, "vadd",
                                                       f_map_fp32_buf[map_index+8:],
                                                       grad_sel_fp32_buf[mask_index+8:],
                                                       f_map_fp32_buf[map_index+8:],
                                                       "float32", wo*c0//2, ho,
                                                       shape_map_hw, shape_grad,
                                                       config=config)

                            else:
                                # map_index has three part: which hwc0 in
                                # which window, begin_index of kernel,
                                # begin_index of child kernel
                                with tik_instance.for_range(0, ho) as ho_idx:
                                    map_index = (idx_do*self.sd+idx_d)*hi*wi*c0 + \
                                                (ho_idx*self.sh*wi*c0) + \
                                                (idx_h*wi*c0+idx_w*c0)
                                    mask_index = wo * ho_idx * c0

                                    self._vector_op(tik_instance, "vadd",
                                                    f_map_fp32_buf[map_index:],
                                                    grad_sel_fp32_buf[mask_index:],
                                                    f_map_fp32_buf[map_index:],
                                                    "float32", wo*c0//2,
                                                    stride_config=(self.sw*2,
                                                                   self.sw*2, 2,
                                                                   self.sw*16,
                                                                   self.sw*16, 16))
                                    self._vector_op(tik_instance, "vadd",
                                                    f_map_fp32_buf[map_index+8:],
                                                    grad_sel_fp32_buf[mask_index+8:],
                                                    f_map_fp32_buf[map_index + 8:],
                                                    "float32", wo * c0 // 2,
                                                    stride_config=(self.sw*2,
                                                                   self.sw*2, 2,
                                                                   self.sw*16,
                                                                   self.sw*16, 16))

            # ---mov_out---
            dst_ou_gm = merchant * prod(self.forward_in_shape[1:]) + \
                        remainder * hi * wi * c0
            ub2gm_shape = [di, hi, wi, c0]
            self._copy_ub_to_gm(tik_instance, f_map_fp32_buf, 0,
                                self.ou_y_gm, dst_ou_gm,
                                ub2gm_shape)

    def tiling_do_main(self, tik_instance, core_loop,
                       sum_core, model, param):
        '''
        =========================
        Just only split do
        =========================
        '''
        do_batch = model[0]
        ho = model[1]
        wo = model[2]

        # batch + tail
        loop_do = self.do // do_batch
        di_batch, hi, wi = self._infer_dim_return(do_batch, ho, wo, True)
        c0 = self.c0
        c1 = self.c1
        if loop_do <= 0:
            error_manager_vector.raise_err_input_value_invalid("MaxPoolGRAD", "loop_do",
                                                               "more than or equal to 1",
                                                               str(loop_do))
        do_tail = self.do % do_batch
        di_tail, _, _ = self._infer_dim_return(do_tail, ho, wo, True)
        if do_tail == 0:
            di_tail = 0

        buf_list = self._set_buf_tensor(tik_instance, param)
        l1_in_buf = buf_list[0]
        forward_ou_buf = buf_list[1]
        grad_buf = buf_list[2]
        col_in_buf = buf_list[3]
        zero_buf = buf_list[7]
        grad_sel_fp16_buf = buf_list[8]
        grad_sel_fp32_buf = buf_list[9]
        f_map_fp32_buf = buf_list[10]
        self.set_vector_dup(tik_instance, param.zero_size,
                            zero_buf, 0, 0, self.dtype)
        self.set_vector_dup(tik_instance, param.f_map_fp32_size,
                            f_map_fp32_buf, 0, 0, "float32")

        # do is batch_do
        # can't fused loop_do in core_loop
        # due to overlap will init next loop_do,
        # if tail existed, fused loop will result in fail
        with tik_instance.for_range(0, core_loop) as num_core_loop:
            merchant = (sum_core + num_core_loop) // c1
            remainder = (sum_core + num_core_loop) % c1

            def _main(loop_idx, di, do):
                # ----Init_Begin_Idx----
                if self.kd >= self.sd:
                    di_coordinate = loop_idx * \
                                    (di_batch-self.overlap_d)
                else:
                    di_coordinate = loop_idx * di_batch
                do_coordinate = loop_idx * do_batch
                src_orig_x_gm = merchant * prod(self.forward_in_shape[1:]) + \
                                remainder * hi * wi * c0 + \
                                di_coordinate * c1 * hi * wi * c0
                src_orig_y_gm = merchant * prod(self.forward_ou_shape[1:]) + \
                                remainder * ho * wo * c0 + \
                                do_coordinate * c1 * ho * wo * c0
                src_grad_gm = src_orig_y_gm

                # ----COPY_GM_2_L1_BUF----
                # Prevent reading gm out of bounds
                # which only happened in kd<sd
                with tik_instance.if_scope(di_coordinate + di <= self.d):
                    self._copy_gm_to_l1(tik_instance, l1_in_buf,
                                        src_orig_x_gm, 0, [di, hi, wi, c0])
                with tik_instance.else_scope():
                    self._copy_gm_to_l1(tik_instance, l1_in_buf,
                                        src_orig_x_gm, 0,
                                        [di+self.overlap_d, hi, wi, c0])

                # ----COPY_ORI_OUTPUT_2_FORWARD_OU_BUF----
                self._copy_gm_to_ub(tik_instance, forward_ou_buf,
                                    self.orig_y_gm, src_orig_y_gm,
                                    [do, ho, wo, c0])

                # ----COPY_GRAD_2_GRAD_BUF----
                self._copy_gm_to_ub(tik_instance, grad_buf, self.grads_gm,
                                    src_grad_gm, [do, ho, wo, c0])

                # ---load3d l1 to col_in_buffer---
                repeat_times = ceil_div(ho*wo, 16)
                # which window
                with tik_instance.for_range(0, do, thread_num=1) as idx_do:
                    # which hwc0
                    with tik_instance.for_range(0, self.kd, thread_num=1) as idx_d:
                        with tik_instance.for_range(0, self.kh, thread_num=1) as idx_h:
                            with tik_instance.for_range(0, self.kw, thread_num=1) as idx_w:
                                src_l1 = (idx_do * self.sd + idx_d) * hi * wi * c0
                                if self.check_load3d_support:
                                    tik_instance.load3dv1(col_in_buf[0],
                                                          l1_in_buf[src_l1],
                                                          [0, 0, 0, 0],
                                                          hi, wi, 0,
                                                          idx_w, idx_h,
                                                          0, 0,
                                                          self.sw, self.sh,
                                                          self.kw, self.kh,
                                                          1, 1, 1, 1,
                                                          repeat_times, 0,
                                                          MIN_VALUE_FP16
                                                          )
                                else:
                                    col_in_buf_shape = cal_shape_ele(col_in_buf.shape)
                                    self.set_vector_dup(
                                        tik_instance, col_in_buf_shape, col_in_buf, 0, MIN_VALUE_FP16, "float16")
                                    img2col(tik_instance, l1_in_buf, col_in_buf, src_l1, 0, idx_h, idx_w, 0, 0,
                                            hi, wi, self.kh, self.kw, self.sh, self.sw, repeat_times, 1, (0, 0, 0, 0))

                                # ---calculate mask---
                                idx_list = [idx_do, idx_d, idx_h, idx_w]
                                const_list = [ho, wo, c0]
                                self._calc_mask(tik_instance, buf_list, param,
                                                idx_list, const_list)

                                # ---sel(grad,zero,mask)---
                                self._sel(tik_instance, buf_list,
                                          idx_list, const_list)

                                # ---vconv grad_sel_fp16 to fp32---
                                self._vconv(tik_instance, grad_sel_fp16_buf, 0,
                                            grad_sel_fp32_buf, 0,
                                            param.grad_sel_fp16_size, "float16")

                                # ---rewrite grad_sel_fp32 to f_map_fp32---
                                config = (self.sw*2, self.sw*2, 2,
                                          self.sh*wi*2, self.sh*wi*2, wo*2)
                                if check_config(config):
                                    with tik_instance.for_range(0, 1) as ho_idx:
                                        map_index = (idx_do*self.sd+idx_d)*hi*wi*c0 + \
                                                    (idx_h*wi*c0+idx_w*c0)
                                        mask_index = wo * ho_idx * c0
                                        shape_map_hw = [hi, wi, c0]
                                        shape_grad = [ho, wo, c0]

                                        self._rewrite_fmap(tik_instance, "vadd",
                                                           f_map_fp32_buf[map_index:],
                                                           grad_sel_fp32_buf[mask_index:],
                                                           f_map_fp32_buf[map_index:],
                                                           "float32", wo*c0//2, ho,
                                                           shape_map_hw, shape_grad,
                                                           config=config)

                                        self._rewrite_fmap(tik_instance, "vadd",
                                                           f_map_fp32_buf[map_index+8:],
                                                           grad_sel_fp32_buf[mask_index+8:],
                                                           f_map_fp32_buf[map_index+8:],
                                                           "float32", wo*c0//2, ho,
                                                           shape_map_hw, shape_grad,
                                                           config=config)
                                else:
                                    # map_index has three part: which hwc0 in
                                    # which window, begin_index of kernel,
                                    # begin_index of child kernel
                                    with tik_instance.for_range(0, ho) as ho_idx:
                                        map_index = (idx_do*self.sd+idx_d)*hi*wi*c0 + \
                                                    (ho_idx*self.sh*wi*c0) + \
                                                    (idx_h*wi*c0+idx_w*c0)
                                        mask_index = wo * ho_idx * c0

                                        self._vector_op(tik_instance, "vadd",
                                                        f_map_fp32_buf[map_index:],
                                                        grad_sel_fp32_buf[mask_index:],
                                                        f_map_fp32_buf[map_index:],
                                                        "float32", wo*c0//2,
                                                        stride_config=(self.sw*2,
                                                                       self.sw*2, 2,
                                                                       self.sw*16,
                                                                       self.sw*16, 16))
                                        self._vector_op(tik_instance, "vadd",
                                                        f_map_fp32_buf[map_index+8:],
                                                        grad_sel_fp32_buf[mask_index+8:],
                                                        f_map_fp32_buf[map_index + 8:],
                                                        "float32", wo * c0 // 2,
                                                        stride_config=(self.sw*2,
                                                                       self.sw*2, 2,
                                                                       self.sw*16,
                                                                       self.sw*16, 16))

                # ---mov_out---
                dst_ou_gm = merchant * prod(self.forward_in_shape[1:]) + \
                            remainder * hi * wi * c0 + \
                            di_coordinate * c1 * hi * wi * c0

                # effective boundary of d
                boundary_d = self.d - max(0, self.di_invalid)
                # di_coordinate + di < boundary_d means:
                # last effective kernel need SPECIAL TREATMENT
                with tik_instance.if_scope(di_coordinate + di < boundary_d):
                    if self.kd >= self.sd:
                        # move accumulated data to gm
                        ub2gm_shape = [di-self.overlap_d, hi, wi, c0]
                        self._copy_ub_to_gm(tik_instance, f_map_fp32_buf, 0,
                                            self.ou_y_gm, dst_ou_gm,
                                            ub2gm_shape)
                        # move overlap data to ub and vec_dup
                        ub2ub_shape = [di, hi, wi, c0]
                        num_overlap = prod(ub2ub_shape) // di * self.overlap_d
                        num_init_zero = prod(ub2ub_shape) - num_overlap
                        self._mov_init(tik_instance, f_map_fp32_buf, num_overlap,
                                       num_init_zero)
                    else:
                        # in case of sd > kd,
                        # di contains stride
                        ub2gm_shape = [di, hi, wi, c0]
                        self._copy_ub_to_gm(tik_instance, f_map_fp32_buf, 0,
                                            self.ou_y_gm, dst_ou_gm,
                                            ub2gm_shape)
                        self.set_vector_dup(tik_instance, param.f_map_fp32_size,
                                            f_map_fp32_buf, 0, 0, "float32")

                with tik_instance.else_scope():
                    if self.kd >= self.sd:
                        # the last kernel
                        ub2gm_shape = [di, hi, wi, c0]
                        self._copy_ub_to_gm(tik_instance, f_map_fp32_buf, 0,
                                            self.ou_y_gm, dst_ou_gm,
                                            ub2gm_shape)
                        self.set_vector_dup(tik_instance, param.f_map_fp32_size,
                                            f_map_fp32_buf, 0, 0, "float32")
                        if self.di_invalid != 0:
                            dst_ou_gm_new = dst_ou_gm + di * c1 * hi * wi * c0
                            ub2gm_shape = [self.di_invalid, hi, wi, c0]
                            self._copy_ub_to_gm(tik_instance, f_map_fp32_buf, 0,
                                                self.ou_y_gm, dst_ou_gm_new,
                                                ub2gm_shape)
                    else:
                        # useful data
                        if self.di_invalid <= 0:
                            # overlap_d make di exceed self.d
                            ub2gm_shape = [di+self.di_invalid, hi, wi, c0]
                            self._copy_ub_to_gm(tik_instance, f_map_fp32_buf, 0,
                                                self.ou_y_gm, dst_ou_gm,
                                                ub2gm_shape)
                            self.set_vector_dup(tik_instance, param.f_map_fp32_size,
                                                f_map_fp32_buf, 0, 0, "float32")
                        else:
                            ub2gm_shape = [di, hi, wi, c0]
                            self._copy_ub_to_gm(tik_instance, f_map_fp32_buf, 0,
                                                self.ou_y_gm, dst_ou_gm,
                                                ub2gm_shape)
                            dst_ou_gm_new = dst_ou_gm + di * c1 * hi * wi * c0
                            ub2gm_shape = [self.di_invalid, hi, wi, c0]
                            self.set_vector_dup(tik_instance, param.f_map_fp32_size,
                                                f_map_fp32_buf, 0, 0, "float32")
                            self._copy_ub_to_gm(tik_instance, f_map_fp32_buf, 0,
                                                self.ou_y_gm, dst_ou_gm_new,
                                                ub2gm_shape)

            with tik_instance.for_range(0, loop_do) as idx:
                # idx+1 represent kernel_d filter next position,
                # if self.overlap_d > 0, result of idx would be
                # used init idx+1(include tail)
                _main(idx, di_batch, do_batch)

            if do_tail != 0:
                _main(loop_do, di_tail, do_tail)

    def tiling_do_ho_main(self, tik_instance, core_loop,
                          sum_core, model, param):
        '''
        ============================
        Just only split do, ho
        ============================
        '''
        do_batch = model[0]
        ho_batch = model[1]
        wo = model[2]
        c0 = self.c0
        c1 = self.c1

        # batch + tail
        loop_do = self.do // do_batch
        loop_ho = self.ho // ho_batch
        ho_tail = self.ho % ho_batch

        di_batch, hi_batch, wi = self._infer_dim_return(do_batch, ho_batch,
                                                        wo, True)
        di_tail, hi_tail, _ = self._infer_dim_return(do_batch, ho_tail,
                                                           wo, True)
        if ho_tail == 0:
            hi_tail = 0

        buf_list = self._set_buf_tensor(tik_instance, param)
        l1_in_buf = buf_list[0]
        forward_ou_buf = buf_list[1]
        grad_buf = buf_list[2]
        col_in_buf = buf_list[3]
        zero_buf = buf_list[7]
        grad_sel_fp16_buf = buf_list[8]
        grad_sel_fp32_buf = buf_list[9]
        f_map_fp32_buf = buf_list[10]
        self.set_vector_dup(tik_instance, param.zero_size,
                            zero_buf, 0, 0, self.dtype)
        self.set_vector_dup(tik_instance, param.f_map_fp32_size,
                            f_map_fp32_buf, 0, 0, "float32")

        with tik_instance.for_range(0, core_loop) as num_core_loop:
            merchant = (sum_core + num_core_loop) // c1
            remainder = (sum_core + num_core_loop) % c1

            def _main(loop_do_idx, loop_ho_idx, di, do, hi, ho):
                # ==========================
                # Init_Begin_Idx
                # ==========================
                if self.kd >= self.sd:
                    di_coordinate = loop_do_idx * \
                                    (di_batch-self.overlap_d)
                else:
                    di_coordinate = loop_do_idx * di_batch

                if self.kh >= self.sh:
                    hi_coordinate = loop_ho_idx * \
                                    (hi_batch-self.overlap_h)
                else:
                    hi_coordinate = loop_ho_idx * hi_batch

                do_coordinate = loop_do_idx * do_batch
                ho_coordinate = loop_ho_idx * ho_batch

                src_orig_x_gm = merchant * prod(self.forward_in_shape[1:]) + \
                                remainder * self.h * wi * c0 + \
                                di_coordinate * c1 * self.h * wi * c0 + \
                                hi_coordinate * wi * c0
                src_orig_y_gm = merchant * prod(self.forward_ou_shape[1:]) + \
                                remainder * self.ho * wo * c0 + \
                                do_coordinate * c1 * self.ho * wo * c0 + \
                                ho_coordinate * wo * c0
                src_grad_gm = src_orig_y_gm
                # ================================
                # COPY_GM_2_L1_BUF
                # Prevent reading gm out of bounds
                # ================================
                with tik_instance.if_scope(di_coordinate + di <= self.d):
                    with tik_instance.if_scope(hi_coordinate + hi <= self.h):
                        self._gm2l1_tiling_do_ho(tik_instance, l1_in_buf,
                                                 src_orig_x_gm, 0,
                                                 [di, hi, wi, c0],
                                                 hi_batch)
                    with tik_instance.else_scope():
                        if self.overlap_h < 0:
                            self._gm2l1_tiling_do_ho(tik_instance, l1_in_buf,
                                                     src_orig_x_gm, 0,
                                                     [di, hi+self.overlap_h,
                                                      wi, c0],
                                                     hi_batch)

                with tik_instance.else_scope():
                    if self.overlap_d < 0:
                        with tik_instance.if_scope(hi_coordinate+hi <= self.h):
                            self._gm2l1_tiling_do_ho(tik_instance, l1_in_buf,
                                                     src_orig_x_gm, 0,
                                                     [di+self.overlap_d,
                                                      hi, wi, c0],
                                                     hi_batch)
                        with tik_instance.else_scope():
                            if self.overlap_h < 0:
                                self._gm2l1_tiling_do_ho(tik_instance,
                                                         l1_in_buf,
                                                         src_orig_x_gm, 0,
                                                         [di+self.overlap_d,
                                                          hi+self.overlap_h,
                                                          wi, c0],
                                                         hi_batch)

                # ================================
                # COPY_ORI_OUTPUT_2_FORWARD_OU_BUF
                # COPY_GRAD_2_GRAD_BUF
                # ================================
                self._copy_gm_to_ub(tik_instance, forward_ou_buf,
                                    self.orig_y_gm, src_orig_y_gm,
                                    [do, ho, wo, c0])

                self._copy_gm_to_ub(tik_instance, grad_buf, self.grads_gm,
                                    src_grad_gm, [do, ho, wo, c0])

                # ================================
                # load3d l1 to col_in_buffer
                # ================================
                repeat_times = ceil_div(ho*wo, 16)
                with tik_instance.for_range(0, do, thread_num=1) as idx_do:
                    with tik_instance.for_range(0, self.kd, thread_num=1) as idx_d:
                        with tik_instance.for_range(0, self.kh, thread_num=1) as idx_h:
                            with tik_instance.for_range(0, self.kw, thread_num=1) as idx_w:
                                src_l1 = (idx_do * self.sd + idx_d) * hi_batch * wi * c0
                                if self.check_load3d_support:
                                    tik_instance.load3dv1(col_in_buf[0],
                                                          l1_in_buf[src_l1],
                                                          [0, 0, 0, 0],
                                                          hi, wi, 0,
                                                          idx_w, idx_h,
                                                          0, 0,
                                                          self.sw, self.sh,
                                                          self.kw, self.kh,
                                                          1, 1, 1, 1,
                                                          repeat_times, 0,
                                                          MIN_VALUE_FP16
                                                          )
                                else:
                                    col_in_buf_shape = cal_shape_ele(col_in_buf.shape)
                                    self.set_vector_dup(
                                        tik_instance, col_in_buf_shape, col_in_buf, 0, MIN_VALUE_FP16, "float16")
                                    img2col(tik_instance, l1_in_buf, col_in_buf, src_l1, 0, idx_h, idx_w, 0, 0,
                                            hi, wi, self.kh, self.kw, self.sh, self.sw, repeat_times, 1, (0, 0, 0, 0))

                                # ---calculate mask---
                                idx_list = [idx_do, idx_d, idx_h, idx_w]
                                const_list = [ho, wo, c0]
                                self._calc_mask(tik_instance, buf_list, param,
                                                idx_list, const_list)

                                # ---sel(grad,zero,mask)---
                                self._sel(tik_instance, buf_list,
                                          idx_list, const_list)

                                # ---vconv grad_sel_fp16 to fp32---
                                self._vconv(tik_instance, grad_sel_fp16_buf, 0,
                                            grad_sel_fp32_buf, 0,
                                            param.grad_sel_fp16_size, "float16")

                                # ---rewrite grad_sel_fp32 to f_map_fp32
                                config = (self.sw*2, self.sw*2, 2,
                                          self.sh*wi*2, self.sh*wi*2, wo*2)
                                if check_config(config):
                                    with tik_instance.for_range(0, 1) as ho_idx:
                                        map_index = (idx_do*self.sd+idx_d)*hi_batch*wi*c0 + \
                                                    (idx_h*wi*c0+idx_w*c0)
                                        mask_index = wo * ho_idx * c0
                                        shape_map_hw = [hi_batch, wi, c0]
                                        shape_grad = [ho, wo, c0]

                                        self._rewrite_fmap(tik_instance, "vadd",
                                                           f_map_fp32_buf[map_index:],
                                                           grad_sel_fp32_buf[mask_index:],
                                                           f_map_fp32_buf[map_index:],
                                                           "float32", wo*c0//2, ho,
                                                           shape_map_hw, shape_grad,
                                                           config=config)

                                        self._rewrite_fmap(tik_instance, "vadd",
                                                           f_map_fp32_buf[map_index+8:],
                                                           grad_sel_fp32_buf[mask_index+8:],
                                                           f_map_fp32_buf[map_index+8:],
                                                           "float32", wo*c0//2, ho,
                                                           shape_map_hw, shape_grad,
                                                           config=config)
                                else:
                                    # map_index has three part: which hwc0 in
                                    # which window, begin_index of kernel,
                                    # begin_index of child kernel
                                    with tik_instance.for_range(0, ho) as ho_idx:
                                        map_index = (idx_do*self.sd+idx_d)*hi_batch*wi*c0 + \
                                                    (ho_idx*self.sh*wi*c0) + \
                                                    (idx_h*wi*c0+idx_w*c0)
                                        mask_index = wo * ho_idx * c0

                                        self._vector_op(tik_instance, "vadd",
                                                        f_map_fp32_buf[map_index:],
                                                        grad_sel_fp32_buf[mask_index:],
                                                        f_map_fp32_buf[map_index:],
                                                        "float32", wo*c0//2,
                                                        stride_config=(self.sw*2,
                                                                       self.sw*2, 2,
                                                                       self.sw*16,
                                                                       self.sw*16, 16))
                                        self._vector_op(tik_instance, "vadd",
                                                        f_map_fp32_buf[map_index+8:],
                                                        grad_sel_fp32_buf[mask_index+8:],
                                                        f_map_fp32_buf[map_index + 8:],
                                                        "float32", wo * c0 // 2,
                                                        stride_config=(self.sw*2,
                                                                       self.sw*2, 2,
                                                                       self.sw*16,
                                                                       self.sw*16, 16))

                # mov_out
                dst_ou_gm = merchant * prod(self.forward_in_shape[1:]) + \
                            remainder * self.h * wi * c0 + \
                            di_coordinate * c1 * self.h * wi * c0 + \
                            hi_coordinate * wi * c0

                def mov_atomic(num_d, dst, dst_idx, src_idx):
                    boundary_h = self.h - max(0, self.hi_invalid)
                    if self.kh > self.sh:
                        # ================================
                        # Split kernels as n-1 and the nth
                        # ================================
                        with tik_instance.if_scope(hi_coordinate+hi < boundary_h):
                            # ==============================
                            # move accumulated data to gm
                            # ==============================
                            in_shape = [num_d, hi-self.overlap_h, wi, c0]
                            self._ub2gm_split_do_ho(tik_instance,
                                                    f_map_fp32_buf,
                                                    src_idx, dst,
                                                    dst_idx, in_shape)

                            # ==============================
                            # mov to init and vec_dup
                            # ==============================
                            in_shape = [num_d, hi, wi, c0]
                            overlap = [num_d, self.overlap_h, wi, c0]
                            non_overlap = [num_d, hi-self.overlap_h, wi, c0]

                            n_burst = in_shape[0]
                            burst_len = prod(overlap[1:]) * \
                                        self.num_bit_fp32 // BLOCK_SIZE
                            src_stride = prod(non_overlap[1:]) * \
                                         self.num_bit_fp32 // BLOCK_SIZE
                            dst_stride = prod(non_overlap[1:]) * \
                                         self.num_bit_fp32 // BLOCK_SIZE
                            tik_instance.data_move(
                                f_map_fp32_buf[src_idx],
                                f_map_fp32_buf[src_idx + prod(non_overlap[1:])],
                                0,
                                n_burst,
                                burst_len,
                                src_stride,
                                dst_stride)

                            # vec_dup for next ho_idx
                            num_zero = prod(non_overlap[1:])
                            for i in range(in_shape[0]):
                                dst_vec_idx = src_idx + prod(overlap[1:]) + i * \
                                              prod(in_shape[1:])
                                self.set_vector_dup(tik_instance, num_zero,
                                                    f_map_fp32_buf, dst_vec_idx,
                                                    0, "float32")

                        with tik_instance.else_scope():
                            in_shape = [num_d, hi, wi, c0]
                            # if tail_h existed, ub2gm has different model
                            self._ub2gm_split_do_ho_2(tik_instance,
                                                      f_map_fp32_buf,
                                                      src_idx, dst,
                                                      dst_idx, in_shape,
                                                      hi_batch)

                            self.set_vector_dup(tik_instance,
                                                param.f_map_fp32_size,
                                                f_map_fp32_buf, 0, 0, "float32")

                    elif self.kh == self.sh:
                        in_shape = [num_d, hi, wi, c0]
                        self._ub2gm_split_do_ho_2(tik_instance,
                                                  f_map_fp32_buf,
                                                  src_idx, dst,
                                                  dst_idx, in_shape, hi_batch)

                        self.set_vector_dup(tik_instance, param.f_map_fp32_size,
                                            f_map_fp32_buf, 0, 0, "float32")

                    else:
                        if self.hi_invalid >= 0:
                            in_shape = [num_d, hi, wi, c0]
                            self._ub2gm_split_do_ho_2(tik_instance,
                                                      f_map_fp32_buf,
                                                      src_idx, dst,
                                                      dst_idx, in_shape,
                                                      hi_batch)

                            self.set_vector_dup(tik_instance,
                                                param.f_map_fp32_size,
                                                f_map_fp32_buf, 0, 0, "float32")
                        else:
                            with tik_instance.if_scope(hi_coordinate+hi < boundary_h):
                                in_shape = [num_d, hi, wi, c0]
                                self._ub2gm_split_do_ho_2(tik_instance,
                                                          f_map_fp32_buf,
                                                          src_idx, dst,
                                                          dst_idx, in_shape,
                                                          hi_batch)

                                self.set_vector_dup(tik_instance,
                                                    param.f_map_fp32_size,
                                                    f_map_fp32_buf, 0, 0,
                                                    "float32")
                            with tik_instance.else_scope():
                                in_shape = [num_d, hi+self.hi_invalid, wi, c0]
                                self._ub2gm_split_do_ho_2(tik_instance,
                                                          f_map_fp32_buf,
                                                          src_idx, dst,
                                                          dst_idx, in_shape,
                                                          hi_batch)
                                self.set_vector_dup(tik_instance,
                                                    param.f_map_fp32_size,
                                                    f_map_fp32_buf, 0, 0,
                                                    "float32")

                if self.kd >= self.sd:
                    tik_instance.set_atomic_add(1)
                    mov_atomic(di, self.ou_y_gm, dst_ou_gm, 0)
                    tik_instance.set_atomic_add(0)
                else:
                    # di_invalid can less than 0
                    tik_instance.set_atomic_add(1)
                    if self.di_invalid >= 0:
                        mov_atomic(di, self.ou_y_gm, dst_ou_gm, 0)
                    else:
                        with tik_instance.if_scope(di_coordinate+di <= self.d):
                            mov_atomic(di, self.ou_y_gm, dst_ou_gm, 0)
                        with tik_instance.else_scope():
                            mov_atomic(di+self.di_invalid,
                                       self.ou_y_gm, dst_ou_gm, 0)
                    tik_instance.set_atomic_add(0)

            if ho_tail != 0:
                with tik_instance.for_range(0, loop_do) as do_idx:
                    with tik_instance.for_range(0, loop_ho) as ho_idx:
                        _main(do_idx, ho_idx, di_batch, do_batch,
                              hi_batch, ho_batch)
                    _main(do_idx, loop_ho, di_tail, do_batch, hi_tail, ho_tail)
            else:
                with tik_instance.for_range(0, loop_do) as do_idx:
                    with tik_instance.for_range(0, loop_ho) as ho_idx:
                        _main(do_idx, ho_idx, di_batch, do_batch,
                              hi_batch, ho_batch)

    def tiling_do_ho_wo_main(self, tik_instance, core_loop,
                             sum_core, model, param):
        '''
        ============================
        Just split do, ho, wo
        ============================
        '''
        do_batch = model[0]
        ho_batch = model[1]
        wo_batch = model[2]
        c0 = self.c0
        c1 = self.c1

        if do_batch != ho_batch != 1:
            error_manager_vector.raise_err_input_value_invalid("MaxPoolGRAD", "do_batch and ho_batch",
                                                               "1", str(do_batch) + str(" and ") + str(ho_batch))

        loop_do = self.do // do_batch
        loop_ho = self.ho // ho_batch
        loop_wo = self.wo // wo_batch
        wo_tail = self.wo % wo_batch
        di_batch, hi_batch, wi_batch = self._infer_dim_return(do_batch,
                                                              ho_batch,
                                                              wo_batch, True)
        di_tail, hi_tail, wi_tail = self._infer_dim_return(do_batch,
                                                           ho_batch,
                                                           wo_tail, True)

        buf_list = self._set_buf_tensor(tik_instance, param)
        l1_in_buf = buf_list[0]
        forward_ou_buf = buf_list[1]
        grad_buf = buf_list[2]
        col_in_buf = buf_list[3]
        zero_buf = buf_list[7]
        grad_sel_fp16_buf = buf_list[8]
        grad_sel_fp32_buf = buf_list[9]
        f_map_fp32_buf = buf_list[10]
        self.set_vector_dup(tik_instance, param.zero_size,
                            zero_buf, 0, 0, self.dtype)
        self.set_vector_dup(tik_instance, param.f_map_fp32_size,
                            f_map_fp32_buf, 0, 0, "float32")

        with tik_instance.for_range(0, core_loop) as num_core_loop:
            merchant = (sum_core + num_core_loop) // c1
            remainder = (sum_core + num_core_loop) % c1

            def _main(loop_do_idx, loop_ho_idx, loop_wo_idx,
                      di, do, hi, ho, wi, wo):
                # ==========================
                # Init_Begin_Idx
                # ==========================
                if self.kd >= self.sd:
                    di_coordinate = loop_do_idx * \
                                    (di_batch-self.overlap_d)
                else:
                    di_coordinate = loop_do_idx * di_batch

                if self.kh >= self.sh:
                    hi_coordinate = loop_ho_idx * \
                                    (hi_batch-self.overlap_h)
                else:
                    hi_coordinate = loop_ho_idx * hi_batch

                if self.kw >= self.sw:
                    wi_coordinate = loop_wo_idx * \
                                    (wi_batch-self.overlap_w)
                else:
                    wi_coordinate = loop_wo_idx * wi_batch

                do_coordinate = loop_do_idx * do_batch
                ho_coordinate = loop_ho_idx * ho_batch
                wo_coordinate = loop_wo_idx * wo_batch

                src_orig_x_gm = merchant * prod(self.forward_in_shape[1:]) + \
                                remainder * self.h * self.w * c0 + \
                                di_coordinate * c1 * self.h * self.w * c0 + \
                                hi_coordinate * self.w * c0 + \
                                wi_coordinate * c0
                src_orig_y_gm = merchant * prod(self.forward_ou_shape[1:]) + \
                                remainder * self.ho * self.wo * c0 + \
                                do_coordinate * c1 * self.ho * self.wo * c0 + \
                                ho_coordinate * self.wo * c0 + \
                                wo_coordinate * c0
                src_grad_gm = src_orig_y_gm

                # ================================
                # COPY_GM_2_L1_BUF
                # Prevent reading gm out of bounds
                # ================================
                di_val = min(0, self.overlap_d) + di
                hi_val = min(0, self.overlap_h) + hi
                wi_val = min(0, self.overlap_w) + wi
                input0 = [di_val, hi_val, wi_val]
                input1 = [di_batch, hi_batch, wi_batch]
                self._gm2l1_tiling_do_ho_wo(tik_instance,
                                            l1_in_buf, src_orig_x_gm, 0,
                                            input0, input1)

                # ================================
                # COPY_ORI_OUTPUT_2_FORWARD_OU_BUF
                # COPY_GRAD_2_GRAD_BUF
                # in the branch, do and ho are 1.
                # ================================
                self._copy_gm_to_ub(tik_instance, forward_ou_buf,
                                    self.orig_y_gm, src_orig_y_gm,
                                    [do, ho, wo, c0])

                self._copy_gm_to_ub(tik_instance, grad_buf, self.grads_gm,
                                    src_grad_gm, [do, ho, wo, c0])

                # ================================
                # load3d l1 to col_in_buffer
                # ================================
                repeat_times = ceil_div(ho*wo, 16)
                with tik_instance.for_range(0, do, thread_num=1) as idx_do:
                    with tik_instance.for_range(0, self.kd, thread_num=1) as idx_d:
                        with tik_instance.for_range(0, self.kh, thread_num=1) as idx_h:
                            with tik_instance.for_range(0, self.kw, thread_num=1) as idx_w:
                                src_l1 = (idx_do * self.sd + idx_d) * hi_batch * wi_batch * c0
                                if self.check_load3d_support:
                                    tik_instance.load3dv1(col_in_buf[0],
                                                          l1_in_buf[src_l1],
                                                          [0, 0, 0, 0],
                                                          hi_val, wi_val, 0,
                                                          idx_w, idx_h,
                                                          0, 0,
                                                          self.sw, self.sh,
                                                          self.kw, self.kh,
                                                          1, 1, 1, 1,
                                                          repeat_times, 0,
                                                          MIN_VALUE_FP16
                                                          )
                                else:
                                    col_in_buf_shape = cal_shape_ele(col_in_buf.shape)
                                    self.set_vector_dup(
                                        tik_instance, col_in_buf_shape, col_in_buf, 0, MIN_VALUE_FP16, "float16")
                                    img2col(tik_instance, l1_in_buf, col_in_buf, src_l1, 0, idx_h, idx_w, 0, 0,
                                            hi_val, wi_val, self.kh, self.kw, self.sh, self.sw, repeat_times, 1,
                                            (0, 0, 0, 0))

                                # ---calculate mask---
                                idx_list = [idx_do, idx_d, idx_h, idx_w]
                                const_list = [ho, wo, c0]
                                self._calc_mask(tik_instance, buf_list, param,
                                                idx_list, const_list)

                                # ---sel(grad,zero,mask)---
                                self._sel(tik_instance, buf_list,
                                          idx_list, const_list)

                                # ---vconv grad_sel_fp16 to fp32---
                                self._vconv(tik_instance, grad_sel_fp16_buf, 0,
                                            grad_sel_fp32_buf, 0,
                                            param.grad_sel_fp16_size, "float16")

                                # ---rewrite grad_sel_fp32 to f_map_fp32
                                # `do = 1, ho = 1`
                                # map_index has two part: begin_index of kernel,
                                # begin_index of child kernel
                                # must use tik variable as index of grad_sel_fp32_buf,
                                # python variable is not work in grad_sel_fp32_buf[mask_index],
                                # `while x = grad_sel_fp32_buf[mask_index], y = x[n].`
                                with tik_instance.for_range(0, 1) as index_mask:
                                    map_index = idx_d * hi_batch * wi_batch * c0 + \
                                                idx_h * wi_batch * c0 + idx_w * c0
                                    mask_index = index_mask

                                    self._vector_op(tik_instance, "vadd",
                                                    f_map_fp32_buf[map_index:],
                                                    grad_sel_fp32_buf[mask_index:],
                                                    f_map_fp32_buf[map_index:],
                                                    "float32", wo*c0//2,
                                                    stride_config=(self.sw*2,
                                                                   self.sw*2, 2,
                                                                   self.sw*16,
                                                                   self.sw*16, 16))
                                    self._vector_op(tik_instance, "vadd",
                                                    f_map_fp32_buf[map_index+8:],
                                                    grad_sel_fp32_buf[mask_index+8:],
                                                    f_map_fp32_buf[map_index + 8:],
                                                    "float32", wo*c0//2,
                                                    stride_config=(self.sw*2,
                                                                   self.sw*2, 2,
                                                                   self.sw*16,
                                                                   self.sw*16, 16))

                # mov_out
                dst_ou_gm = merchant * prod(self.forward_in_shape[1:]) + \
                            remainder * self.h * self.w * c0 + \
                            di_coordinate * c1 * self.h * self.w * c0 + \
                            hi_coordinate * self.w * c0 + \
                            wi_coordinate * c0

                def mov_atomic(num_d, dst, dst_idx, src_idx):
                    num_h = hi + min(0, self.overlap_h)
                    num_w = wi + min(0, self.overlap_w)
                    in_shape = [num_d, num_h, num_w, c0]
                    self._ub2gm_split_do_ho_wo(tik_instance, f_map_fp32_buf,
                                               src_idx, dst, dst_idx,
                                               in_shape, hi_batch, wi_batch)

                    self.set_vector_dup(tik_instance, param.f_map_fp32_size,
                                        f_map_fp32_buf, 0, 0, "float32")

                tik_instance.set_atomic_add(1)
                mov_atomic(di + min(0, self.overlap_d),
                           self.ou_y_gm, dst_ou_gm, 0)
                tik_instance.set_atomic_add(0)

            if wo_tail != 0:
                with tik_instance.for_range(0, loop_do) as do_idx:
                    with tik_instance.for_range(0, loop_ho) as ho_idx:
                        with tik_instance.for_range(0, loop_wo) as wo_idx:
                            _main(do_idx, ho_idx, wo_idx,
                                  di_batch, do_batch,
                                  hi_batch, ho_batch,
                                  wi_batch, wo_batch)
                        _main(do_idx, ho_idx, loop_wo,
                              di_tail, do_batch,
                              hi_tail, ho_batch,
                              wi_tail, wo_tail)
            else:
                with tik_instance.for_range(0, loop_do) as do_idx:
                    with tik_instance.for_range(0, loop_ho) as ho_idx:
                        with tik_instance.for_range(0, loop_wo) as wo_idx:
                            _main(do_idx, ho_idx, wo_idx,
                                  di_batch, do_batch,
                                  hi_batch, ho_batch,
                                  wi_batch, wo_batch)

    def same_pure_atomic_tiling_do(self, tik_instance, core_loop,
                                   sum_core, model, param):
        '''
        ==============================================================
        In the case, [do,ho,wo] will be infer return
        [di_batch,hi_batch,wi_batch] and [map_di, map_hi, map_wi].
        xi_batch: size of input_data which restored in l1_in_buf.
        map_xi: size of feature_map which restored in f_map_fp32_buf.
        ==============================================================
        '''
        do_batch = model[0]
        ho_batch = model[1]
        wo_batch = model[2]
        c0 = self.c0
        c1 = self.c1

        # size of input_data
        loop_do = self.core_ou_shape[0] // do_batch
        di_batch, hi_batch, wi_batch = self._infer_dim_return(do_batch,
                                                              ho_batch,
                                                              wo_batch, True)
        do_tail = self.core_ou_shape[0] % do_batch
        di_tail, _, _ = self._infer_dim_return(do_tail,
                                                           ho_batch,
                                                           wo_batch, True)
        # feature_map's size
        _, map_hi, map_wi = self._infer_map_return(do_batch, ho_batch,
                                                        wo_batch)

        pad_d_top, pad_d_bottom = self.pad[0][0], self.pad[0][1]
        pad_hw_top, pad_hw_bottom = self.pad[1][0], self.pad[1][1]
        pad_hw_left, pad_hw_right = self.pad[2][0], self.pad[2][1]
        pad_hw_list = [pad_hw_left, pad_hw_right, pad_hw_top, pad_hw_bottom]

        buf_list = self._set_buf_tensor(tik_instance, param)
        l1_in_buf = buf_list[0]
        forward_ou_buf = buf_list[1]
        grad_buf = buf_list[2]
        col_in_buf = buf_list[3]
        zero_buf = buf_list[7]
        grad_sel_fp16_buf = buf_list[8]
        grad_sel_fp32_buf = buf_list[9]
        f_map_fp32_buf = buf_list[10]
        self.set_vector_dup(tik_instance, param.zero_size,
                            zero_buf, 0, 0, self.dtype)
        self.set_vector_dup(tik_instance, param.f_map_fp32_size,
                            f_map_fp32_buf, 0, 0, "float32")

        with tik_instance.for_range(0, core_loop) as num_core_loop:
            # ======================
            # Assume  ori_output_shape is [n,do,c1,ho,wo,c0],
            # split [n,do,c1] as core_num.
            # merchant: index of n-axis
            # merchant_c1: index of c1-axis
            # remainder_c1: index of do-axis
            # ======================
            core_do = self.core_ou_shape[0]
            core_do_times = self.do // core_do
            core_di = self.core_in_shape[0]

            merchant = (sum_core + num_core_loop) // (c1 * core_do_times)
            remainder = (sum_core + num_core_loop) % (c1 * core_do_times)
            merchant_c1 = remainder // core_do_times
            remainder_c1 = remainder % core_do_times

            def _main(loop_idx, di, do):
                # ============================================================
                # ----Init_Begin_Idx----
                # If pad_d_top exist, actual begin_idx of d_axis is -pad_d_top.
                # Meanwhile, don't move pad_d_x to l1_in_buf, but leave space
                # enough in l1_in_buf.
                # ============================================================
                if self.kd >= self.sd:
                    di_coordinate = loop_idx * (di_batch-self.overlap_d) + \
                                    remainder_c1 * (core_di-self.overlap_d) - \
                                    pad_d_top
                else:
                    di_coordinate = loop_idx * di_batch + \
                                    remainder_c1 * core_di - \
                                    pad_d_top

                do_coordinate = loop_idx * do_batch + remainder_c1 * core_do

                # if pad_d_top exist, the begin_index would be less than 0
                di_coord = init_coordinate(tik_instance, pad_d_top,
                                           di_coordinate)

                src_orig_x_gm = merchant * prod(self.forward_in_shape[1:]) + \
                                merchant_c1 * self.h * self.w * c0 + \
                                di_coord * c1 * self.h * self.w * c0
                src_orig_y_gm = merchant * prod(self.forward_ou_shape[1:]) + \
                                merchant_c1 * self.ho * self.wo * c0 + \
                                do_coordinate * c1 * self.ho * self.wo * c0
                src_grad_gm = src_orig_y_gm

                # ----COPY_GM_2_L1_BUF----
                # Prevent reading gm out of bounds.
                # Judge value of di_val according to do_coordinate.
                # di_val contains pad_d_top and pad_d_bottom.
                di_value = min(0, self.overlap_d) + di
                di_val = di_value
                l1_idx = 0
                d_top, d_bottom = calc_pad(tik_instance, pad_d_top, pad_d_bottom,
                                           di_coordinate, di_value, self.d)

                # gm2l1: filled regions don't move except d
                if pad_d_top != 0:
                    di_val -= d_top
                    l1_idx = d_top
                if pad_d_bottom != 0:
                    di_val -= d_bottom

                in_shape = [di_val, hi_batch, wi_batch, c0]
                self._copy_gm_to_l1(tik_instance, l1_in_buf,
                                    src_orig_x_gm, l1_idx*hi_batch*wi_batch*c0,
                                    in_shape)

                # ----COPY_ORI_OUTPUT_2_FORWARD_OU_BUF----
                self._copy_gm_to_ub(tik_instance, forward_ou_buf,
                                    self.orig_y_gm, src_orig_y_gm,
                                    [do, ho_batch, wo_batch, c0])

                # ----COPY_GRAD_2_GRAD_BUF----
                self._copy_gm_to_ub(tik_instance, grad_buf, self.grads_gm,
                                    src_grad_gm, [do, ho_batch, wo_batch, c0])

                # ---load3d l1 to col_in_buffer---
                load3d_mark = tik_instance.Scalar(dtype='int64', name='load3d_mark')
                repeat_times = ceil_div(ho_batch*wo_batch, 16)
                with tik_instance.for_range(0, do, thread_num=1) as idx_do:
                    with tik_instance.for_range(0, self.kd, thread_num=1) as idx_d:
                        # =====================================================
                        # if window in position of pad, not load3d, but vec_dup.
                        # =====================================================
                        self.filled_vec_dup(tik_instance, load3d_mark, di_value,
                                            pad_d_top, pad_d_bottom,
                                            idx_do, idx_d, d_top, d_bottom,
                                            param, col_in_buf)

                        with tik_instance.for_range(0, self.kh, thread_num=1) as idx_h:
                            with tik_instance.for_range(0, self.kw, thread_num=1) as idx_w:
                                src_l1 = (idx_do * self.sd + idx_d) * hi_batch * wi_batch * c0

                                with tik_instance.if_scope(load3d_mark != 1):
                                    if self.check_load3d_support:
                                        tik_instance.load3dv1(col_in_buf[0],
                                                              l1_in_buf[src_l1],
                                                              pad_hw_list,
                                                              hi_batch, wi_batch, 0,
                                                              idx_w, idx_h,
                                                              -pad_hw_left,
                                                              -pad_hw_top,
                                                              self.sw, self.sh,
                                                              self.kw, self.kh,
                                                              1, 1, 1, 1,
                                                              repeat_times, 0,
                                                              MIN_VALUE_FP16
                                                              )
                                    else:
                                        col_in_buf_shape = cal_shape_ele(col_in_buf.shape)
                                        self.set_vector_dup(
                                            tik_instance, col_in_buf_shape, col_in_buf, 0, MIN_VALUE_FP16, "float16")
                                        img2col(tik_instance, l1_in_buf, col_in_buf, src_l1, 0, idx_h, idx_w,
                                                -pad_hw_top, -pad_hw_left, hi_batch, wi_batch, self.kh, self.kw,
                                                self.sh, self.sw, repeat_times, 1, pad_hw_list)

                                # ---calculate mask---
                                idx_list = [idx_do, idx_d, idx_h, idx_w]
                                const_list = [ho_batch, wo_batch, c0]
                                self._calc_mask(tik_instance, buf_list, param,
                                                idx_list, const_list)

                                # ---sel(grad,zero,mask)---
                                self._sel(tik_instance, buf_list,
                                          idx_list, const_list)

                                # ---vconv grad_sel_fp16 to fp32---
                                self._vconv(tik_instance, grad_sel_fp16_buf, 0,
                                            grad_sel_fp32_buf, 0,
                                            param.grad_sel_fp16_size, "float16")

                                # ---rewrite grad_sel_fp32 to f_map_fp32
                                config = (self.sw*2, self.sw*2, 2,
                                          self.sh*map_wi*2, self.sh*map_wi*2, wo_batch*2)
                                if check_config(config):
                                    with tik_instance.for_range(0, 1) as ho_idx:
                                        map_index = (idx_do*self.sd+idx_d)*map_hi*map_wi*c0 + \
                                                    (idx_h*map_wi*c0+idx_w*c0)
                                        mask_index = wo_batch * ho_idx * c0
                                        shape_map_hw = [map_hi, map_wi, c0]
                                        shape_grad = [ho_batch, wo_batch, c0]

                                        self._rewrite_fmap(tik_instance, "vadd",
                                                           f_map_fp32_buf[map_index:],
                                                           grad_sel_fp32_buf[mask_index:],
                                                           f_map_fp32_buf[map_index:],
                                                           "float32", wo_batch*c0//2, ho_batch,
                                                           shape_map_hw, shape_grad,
                                                           config=config)

                                        self._rewrite_fmap(tik_instance, "vadd",
                                                           f_map_fp32_buf[map_index+8:],
                                                           grad_sel_fp32_buf[mask_index+8:],
                                                           f_map_fp32_buf[map_index+8:],
                                                           "float32", wo_batch*c0//2, ho_batch,
                                                           shape_map_hw, shape_grad,
                                                           config=config)
                                else:
                                    # map_index has three part: which hwc0 in
                                    # which window, begin_index of kernel,
                                    # begin_index of child kernel
                                    with tik_instance.for_range(0, ho_batch) as ho_idx:
                                        map_index = (idx_do*self.sd+idx_d)*map_hi*map_wi*c0 + \
                                                    (ho_idx*self.sh*map_wi*c0) + \
                                                    (idx_h*map_wi*c0+idx_w*c0)
                                        mask_index = wo_batch * ho_idx * c0

                                        self._vector_op(tik_instance, "vadd",
                                                        f_map_fp32_buf[map_index:],
                                                        grad_sel_fp32_buf[mask_index:],
                                                        f_map_fp32_buf[map_index:],
                                                        "float32", wo_batch*c0//2,
                                                        stride_config=(self.sw*2,
                                                                       self.sw*2, 2,
                                                                       self.sw*16,
                                                                       self.sw*16, 16))
                                        self._vector_op(tik_instance, "vadd",
                                                        f_map_fp32_buf[map_index+8:],
                                                        grad_sel_fp32_buf[mask_index+8:],
                                                        f_map_fp32_buf[map_index + 8:],
                                                        "float32", wo_batch*c0//2,
                                                        stride_config=(self.sw*2,
                                                                       self.sw*2, 2,
                                                                       self.sw*16,
                                                                       self.sw*16, 16))

                # ---mov_out---
                dst_ou_gm = merchant * prod(self.forward_in_shape[1:]) + \
                            merchant_c1 * self.h * self.w * c0 + \
                            di_coord * c1 * self.h * self.w * c0

                def mov_atomic(num_d, dst, dst_idx, src_idx):
                    ub2gm_shape = [num_d, hi_batch, wi_batch, c0]
                    src_idx += (pad_hw_top * map_wi + pad_hw_left) * c0

                    num_bit = self.num_bit_fp32
                    n_burst = ub2gm_shape[1]
                    burst_len = prod(ub2gm_shape[2:]) * num_bit // BLOCK_SIZE
                    # c0 * num_bit // BLOCK_SIZE is 2
                    src_stride = (pad_hw_left + pad_hw_right) * 2
                    dst_stride = 0

                    with tik_instance.for_range(0, ub2gm_shape[0]) as idx:
                        src_idx_new = src_idx + idx * map_hi*map_wi*c0
                        dst_idx_new = dst_idx + prod(self.forward_in_shape[3:]) * c1 * idx

                        in_list = [n_burst, burst_len, src_stride,
                                   dst_stride, src_idx_new, dst_idx_new]
                        if src_stride > MAX_STRIDE:
                            self._ultimate_data_move(tik_instance, f_map_fp32_buf,
                                                     dst, in_list, num_bit)
                        else:
                            self.norm_data_move(tik_instance, f_map_fp32_buf,
                                                dst, in_list)

                    self.set_vector_dup(tik_instance, param.f_map_fp32_size,
                                        f_map_fp32_buf, 0, 0, "float32")

                tik_instance.set_atomic_add(1)
                mov_atomic(di_val, self.ou_y_gm, dst_ou_gm,
                           l1_idx*map_hi*map_wi*c0)
                tik_instance.set_atomic_add(0)

            with tik_instance.for_range(0, loop_do) as idx:
                _main(idx, di_batch, do_batch)

            if do_tail != 0:
                _main(loop_do, di_tail, do_tail)

    def same_pure_atomic_tiling_do_ho(self, tik_instance, core_loop,
                                      sum_core, model, param):
        '''
        ===================================================
        In the case, hi will be split/tiling.Due to load3d
        has the ability to fill h*w, l1_in_buf will save
        factual data(h*w).
        ===================================================
        '''
        do_batch = model[0]
        ho_batch = model[1]
        wo_batch = model[2]
        c0 = self.c0
        c1 = self.c1

        # size of input_data
        loop_do = self.core_ou_shape[0] // do_batch
        loop_ho = self.core_ou_shape[1] // ho_batch
        ho_tail = self.core_ou_shape[1] % ho_batch

        di_batch, hi_batch, wi_batch = self._infer_dim_return(do_batch,
                                                              ho_batch,
                                                              wo_batch, True)
        di_tail, hi_tail, _ = self._infer_dim_return(do_batch,
                                                           ho_tail,
                                                           wo_batch, True)

        # size of feature map
        _, map_hi, map_wi = self._infer_map_return(do_batch, ho_batch,
                                                        wo_batch)

        pad_d_top, pad_d_bottom = self.pad[0][0], self.pad[0][1]
        pad_hw_top, pad_hw_bottom = self.pad[1][0], self.pad[1][1]
        pad_hw_left, pad_hw_right = self.pad[2][0], self.pad[2][1]
        pad_hw_list = [pad_hw_left, pad_hw_right, pad_hw_top, pad_hw_bottom]

        buf_list = self._set_buf_tensor(tik_instance, param)
        l1_in_buf = buf_list[0]
        forward_ou_buf = buf_list[1]
        grad_buf = buf_list[2]
        col_in_buf = buf_list[3]
        zero_buf = buf_list[7]
        grad_sel_fp16_buf = buf_list[8]
        grad_sel_fp32_buf = buf_list[9]
        f_map_fp32_buf = buf_list[10]
        self.set_vector_dup(tik_instance, param.zero_size,
                            zero_buf, 0, 0, self.dtype)
        self.set_vector_dup(tik_instance, param.f_map_fp32_size,
                            f_map_fp32_buf, 0, 0, "float32")

        with tik_instance.for_range(0, core_loop) as num_core_loop:
            # ==============================================
            # Assume  ori_output_shape is [n,do,c1,ho,wo,c0],
            # split [n,do,c1,ho] as core_num.
            # merchant: index of n-axis
            # merchant_c1: index of c1-axis
            # merchant_d: index of do-axis
            # remainder_d: index of ho-axis
            # ==============================================
            core_do = self.core_ou_shape[0]
            core_do_times = self.do // core_do
            core_di = self.core_in_shape[0]

            core_ho = self.core_ou_shape[1]
            core_ho_times = self.ho // core_ho
            core_hi = self.core_in_shape[1]

            merchant = (sum_core+num_core_loop) // (c1*core_do_times*core_ho_times)
            remainder = (sum_core+num_core_loop) % (c1*core_do_times*core_ho_times)

            merchant_c1 = remainder // (core_do_times*core_ho_times)
            remainder_c1 = remainder % (core_do_times*core_ho_times)

            merchant_d = remainder_c1 // core_ho_times
            remainder_d = remainder_c1 % core_ho_times

            def _main(loop_do_idx, loop_ho_idx, di, do, hi, ho):
                # ==========================
                # Init_Begin_Idx
                # ==========================
                if self.kd >= self.sd:
                    di_coordinate = loop_do_idx * (di_batch-self.overlap_d) + \
                                    merchant_d * (core_di-self.overlap_d) - \
                                    pad_d_top
                else:
                    di_coordinate = loop_do_idx * di_batch + \
                                    merchant_d * core_di - \
                                    pad_d_top

                if self.kh >= self.sh:
                    hi_coordinate = loop_ho_idx * (hi_batch-self.overlap_h) + \
                                    remainder_d * (core_hi-self.overlap_h) - \
                                    pad_hw_top
                else:
                    hi_coordinate = loop_ho_idx * hi_batch + \
                                    remainder_d * core_hi - \
                                    pad_hw_top

                do_coordinate = loop_do_idx * do_batch + merchant_d * core_do
                ho_coordinate = loop_ho_idx * ho_batch + remainder_d * core_ho

                # init begin coordinate of di,hi.
                di_coord = init_coordinate(tik_instance, pad_d_top,
                                           di_coordinate)
                hi_coord = init_coordinate(tik_instance, pad_hw_top,
                                           hi_coordinate)

                src_orig_x_gm = merchant * prod(self.forward_in_shape[1:]) + \
                                merchant_c1 * self.h * self.w * c0 + \
                                di_coord * c1 * self.h * self.w * c0 + \
                                hi_coord * self.w * c0
                src_orig_y_gm = merchant * prod(self.forward_ou_shape[1:]) + \
                                merchant_c1 * self.ho * self.wo * c0 + \
                                do_coordinate * c1 * self.ho * self.wo * c0 + \
                                ho_coordinate * self.wo * c0
                src_grad_gm = src_orig_y_gm

                # ================================
                # COPY_GM_2_L1_BUF
                # Prevent reading gm out of bounds.
                # ================================
                # use immediate number
                di_value = min(0, self.overlap_d) + di
                hi_value = min(0, self.overlap_h) + hi
                di_val = di_value
                hi_val = hi_value
                l1_idx = 0

                d_top, d_bottom = calc_pad(tik_instance, pad_d_top, pad_d_bottom,
                                           di_coordinate, di_value, self.d)
                h_top, h_bottom = calc_pad(tik_instance, pad_hw_top, pad_hw_bottom,
                                           hi_coordinate, hi_value, self.h)
                pad_hw_list[-1] = h_bottom
                pad_hw_list[-2] = h_top

                # gm2l1: filled regions don't move except d
                if pad_d_top != 0:
                    di_val -= d_top
                    l1_idx = d_top
                if pad_d_bottom != 0:
                    di_val -= d_bottom

                if pad_hw_top != 0:
                    hi_val -= h_top
                if pad_hw_bottom != 0:
                    hi_val -= h_bottom

                in_shape = [di_val, hi_val, wi_batch, c0]
                self._gm2l1_tiling_do_ho(tik_instance, l1_in_buf,
                                         src_orig_x_gm,
                                         l1_idx*hi_batch*wi_batch*c0,
                                         in_shape,
                                         hi_batch)

                # ================================
                # COPY_ORI_OUTPUT_2_FORWARD_OU_BUF
                # COPY_GRAD_2_GRAD_BUF
                # ================================
                self._copy_gm_to_ub(tik_instance, forward_ou_buf,
                                    self.orig_y_gm, src_orig_y_gm,
                                    [do, ho, wo_batch, c0])

                self._copy_gm_to_ub(tik_instance, grad_buf, self.grads_gm,
                                    src_grad_gm, [do, ho, wo_batch, c0])

                # ================================
                # load3d l1 to col_in_buffer
                # ================================
                load3d_mark = tik_instance.Scalar(dtype='int64', name='load3d_mark')
                repeat_times = ceil_div(ho*wo_batch, 16)
                with tik_instance.for_range(0, do, thread_num=1) as idx_do:
                    with tik_instance.for_range(0, self.kd, thread_num=1) as idx_d:
                        # =====================================================
                        # if window in position of pad, not load3d, but vec_dup.
                        # =====================================================
                        self.filled_vec_dup(tik_instance, load3d_mark, di_value,
                                            pad_d_top, pad_d_bottom,
                                            idx_do, idx_d, d_top, d_bottom,
                                            param, col_in_buf)

                        with tik_instance.for_range(0, self.kh, thread_num=1) as idx_h:
                            with tik_instance.for_range(0, self.kw, thread_num=1) as idx_w:
                                src_l1 = (idx_do * self.sd + idx_d) * hi_batch * wi_batch * c0

                                with tik_instance.if_scope(load3d_mark != 1):
                                    # in the case, l1_h must be hi_val to assure
                                    # correctness of result after filled.
                                    if self.check_load3d_support:
                                        tik_instance.load3dv1(col_in_buf[0],
                                                              l1_in_buf[src_l1],
                                                              pad_hw_list,
                                                              hi_val, wi_batch, 0,
                                                              idx_w, idx_h,
                                                              -pad_hw_left,
                                                              -h_top,
                                                              self.sw, self.sh,
                                                              self.kw, self.kh,
                                                              1, 1, 1, 1,
                                                              repeat_times, 0,
                                                              MIN_VALUE_FP16
                                                              )
                                    else:
                                        col_in_buf_shape = cal_shape_ele(col_in_buf.shape)
                                        self.set_vector_dup(
                                            tik_instance, col_in_buf_shape, col_in_buf, 0, MIN_VALUE_FP16, "float16")
                                        img2col(tik_instance, l1_in_buf, col_in_buf, src_l1, 0, idx_h, idx_w,
                                                -h_top, -pad_hw_left, hi_val, wi_batch, self.kh, self.kw,
                                                self.sh, self.sw, repeat_times, 1, pad_hw_list)

                                # ---calculate mask---
                                idx_list = [idx_do, idx_d, idx_h, idx_w]
                                const_list = [ho, wo_batch, c0]
                                self._calc_mask(tik_instance, buf_list, param,
                                                idx_list, const_list)

                                # ---sel(grad,zero,mask)---
                                self._sel(tik_instance, buf_list,
                                          idx_list, const_list)

                                # ---vconv grad_sel_fp16 to fp32---
                                self._vconv(tik_instance, grad_sel_fp16_buf, 0,
                                            grad_sel_fp32_buf, 0,
                                            param.grad_sel_fp16_size, "float16")

                                # ---rewrite grad_sel_fp32 to f_map_fp32
                                config = (self.sw*2, self.sw*2, 2,
                                          self.sh*map_wi*2, self.sh*map_wi*2, wo_batch*2)
                                if check_config(config):
                                    with tik_instance.for_range(0, 1) as ho_idx:
                                        map_index = (idx_do*self.sd+idx_d)*map_hi*map_wi*c0 + \
                                                    (idx_h*map_wi*c0+idx_w*c0)
                                        mask_index = wo_batch * ho_idx * c0
                                        shape_map_hw = [map_hi, map_wi, c0]
                                        shape_grad = [ho, wo_batch, c0]

                                        self._rewrite_fmap(tik_instance, "vadd",
                                                           f_map_fp32_buf[map_index:],
                                                           grad_sel_fp32_buf[mask_index:],
                                                           f_map_fp32_buf[map_index:],
                                                           "float32", wo_batch*c0//2, ho,
                                                           shape_map_hw, shape_grad,
                                                           config=config)

                                        self._rewrite_fmap(tik_instance, "vadd",
                                                           f_map_fp32_buf[map_index+8:],
                                                           grad_sel_fp32_buf[mask_index+8:],
                                                           f_map_fp32_buf[map_index+8:],
                                                           "float32", wo_batch*c0//2, ho,
                                                           shape_map_hw, shape_grad,
                                                           config=config)
                                else:
                                    # map_index has three part: which hwc0 in
                                    # which window, begin_index of kernel,
                                    # begin_index of child kernel
                                    with tik_instance.for_range(0, ho) as ho_idx:
                                        map_index = (idx_do*self.sd+idx_d)*map_hi*map_wi*c0 + \
                                                    (ho_idx*self.sh*map_wi*c0) + \
                                                    (idx_h*map_wi*c0+idx_w*c0)
                                        mask_index = wo_batch * ho_idx * c0

                                        self._vector_op(tik_instance, "vadd",
                                                        f_map_fp32_buf[map_index:],
                                                        grad_sel_fp32_buf[mask_index:],
                                                        f_map_fp32_buf[map_index:],
                                                        "float32", wo_batch*c0//2,
                                                        stride_config=(self.sw*2,
                                                                       self.sw*2, 2,
                                                                       self.sw*16,
                                                                       self.sw*16, 16))
                                        self._vector_op(tik_instance, "vadd",
                                                        f_map_fp32_buf[map_index+8:],
                                                        grad_sel_fp32_buf[mask_index+8:],
                                                        f_map_fp32_buf[map_index + 8:],
                                                        "float32", wo_batch*c0//2,
                                                        stride_config=(self.sw*2,
                                                                       self.sw*2, 2,
                                                                       self.sw*16,
                                                                       self.sw*16, 16))

                # mov_out
                dst_ou_gm = merchant * prod(self.forward_in_shape[1:]) + \
                            merchant_c1 * self.h * self.w * c0 + \
                            di_coord * c1 * self.h * self.w * c0 + \
                            hi_coord * self.w * c0

                def mov_atomic(num_d, num_h, dst, dst_idx, src_idx):
                    ub2gm_shape = [num_d, num_h, wi_batch, c0]
                    src_idx += (h_top * map_wi + pad_hw_left) * c0

                    num_bit = self.num_bit_fp32
                    n_burst = ub2gm_shape[1]
                    burst_len = prod(ub2gm_shape[2:]) * num_bit // BLOCK_SIZE
                    # c0 * num_bit // BLOCK_SIZE is 2
                    src_stride = (pad_hw_left + pad_hw_right) * 2
                    dst_stride = 0

                    with tik_instance.for_range(0, ub2gm_shape[0]) as idx:
                        src_idx_new = src_idx + idx * map_hi*map_wi*c0
                        dst_idx_new = dst_idx + prod(self.forward_in_shape[3:]) * c1 * idx

                        in_list = [n_burst, burst_len, src_stride,
                                   dst_stride, src_idx_new, dst_idx_new]
                        if src_stride > MAX_STRIDE:
                            self._ultimate_data_move(tik_instance, f_map_fp32_buf,
                                                     dst, in_list, num_bit)
                        else:
                            self.norm_data_move(tik_instance, f_map_fp32_buf,
                                                dst, in_list)

                    # vec_dup
                    self.set_vector_dup(tik_instance, param.f_map_fp32_size,
                                        f_map_fp32_buf, 0, 0, "float32")

                tik_instance.set_atomic_add(1)
                mov_atomic(di_val, hi_val, self.ou_y_gm, dst_ou_gm,
                           l1_idx*map_hi*map_wi*c0)
                tik_instance.set_atomic_add(0)

            if ho_tail != 0:
                with tik_instance.for_range(0, loop_do) as do_idx:
                    with tik_instance.for_range(0, loop_ho) as ho_idx:
                        _main(do_idx, ho_idx, di_batch, do_batch,
                              hi_batch, ho_batch)
                    _main(do_idx, loop_ho, di_tail, do_batch, hi_tail, ho_tail)
            else:
                with tik_instance.for_range(0, loop_do) as do_idx:
                    with tik_instance.for_range(0, loop_ho) as ho_idx:
                        _main(do_idx, ho_idx, di_batch, do_batch,
                              hi_batch, ho_batch)

    def same_pure_atomic_tiling_do_ho_wo(self, tik_instance, core_loop,
                                         sum_core, model, param):
        '''
        ===================================================
        In the case, do,ho,wo will be split/tiling.So,need
        to assure pad_value of different axis.
        ===================================================
        '''
        do_batch = model[0]
        ho_batch = model[1]
        wo_batch = model[2]
        c0 = self.c0
        c1 = self.c1

        if do_batch != ho_batch != 1:
            error_manager_vector.raise_err_input_value_invalid("MaxPoolGRAD",
                                                               "In the branch of 'tiling_do_ho' do_batch and ho_batch",
                                                               "1", str(do_batch) + str(" and ") + str(ho_batch))

        # size of input_data
        loop_do = self.core_ou_shape[0] // do_batch
        loop_ho = self.core_ou_shape[1] // ho_batch
        loop_wo = self.core_ou_shape[2] // wo_batch
        wo_tail = self.core_ou_shape[2] % wo_batch
        di_batch, hi_batch, wi_batch = self._infer_dim_return(do_batch,
                                                              ho_batch,
                                                              wo_batch, True)
        di_tail, hi_tail, wi_tail = self._infer_dim_return(do_batch,
                                                           ho_batch,
                                                           wo_tail, True)

        # size of feature_map
        _, map_hi, map_wi = self._infer_map_return(do_batch, ho_batch,
                                                        wo_batch)
        pad_d_top, pad_d_bottom = self.pad[0][0], self.pad[0][1]
        pad_hw_top, pad_hw_bottom = self.pad[1][0], self.pad[1][1]
        pad_hw_left, pad_hw_right = self.pad[2][0], self.pad[2][1]
        pad_hw_list = [pad_hw_left, pad_hw_right, pad_hw_top, pad_hw_bottom]

        buf_list = self._set_buf_tensor(tik_instance, param)
        l1_in_buf = buf_list[0]
        forward_ou_buf = buf_list[1]
        grad_buf = buf_list[2]
        col_in_buf = buf_list[3]
        zero_buf = buf_list[7]
        grad_sel_fp16_buf = buf_list[8]
        grad_sel_fp32_buf = buf_list[9]
        f_map_fp32_buf = buf_list[10]
        self.set_vector_dup(tik_instance, param.zero_size,
                            zero_buf, 0, 0, self.dtype)
        self.set_vector_dup(tik_instance, param.f_map_fp32_size,
                            f_map_fp32_buf, 0, 0, "float32")

        with tik_instance.for_range(0, core_loop) as num_core_loop:
            # ==============================================
            # Assume  ori_output_shape is [n,do,c1,ho,wo,c0],
            # split [n,do,c1,ho] as core_num.
            # merchant: index of n-axis
            # merchant_c1: index of c1-axis
            # merchant_d: index of do-axis
            # merchant_h: index of ho-axis
            # remainder_h: index of wo-axis
            # ==============================================
            core_do = self.core_ou_shape[0]
            core_do_times = self.do // core_do
            core_di = self.core_in_shape[0]

            core_ho = self.core_ou_shape[1]
            core_ho_times = self.ho // core_ho
            core_hi = self.core_in_shape[1]

            core_wo = self.core_ou_shape[2]
            core_wo_times = self.wo // core_wo
            core_wi = self.core_in_shape[2]

            merchant = (sum_core+num_core_loop) // \
                       (c1*core_do_times*core_ho_times*core_wo_times)
            remainder = (sum_core+num_core_loop) % \
                        (c1*core_do_times*core_ho_times*core_wo_times)

            merchant_c1 = remainder // (core_do_times*core_ho_times*core_wo_times)
            remainder_c1 = remainder % (core_do_times*core_ho_times*core_wo_times)

            merchant_d = remainder_c1 // (core_ho_times*core_wo_times)
            remainder_d = remainder_c1 % (core_ho_times*core_wo_times)

            merchant_h = remainder_d // core_wo_times
            remainder_h = remainder_d % core_wo_times

            def _main(loop_do_idx, loop_ho_idx, loop_wo_idx,
                      di, do, hi, ho, wi, wo):
                # ==========================
                # Init_Begin_Idx
                # ==========================
                if self.kd >= self.sd:
                    di_coordinate = loop_do_idx * (di_batch-self.overlap_d) + \
                                    merchant_d * (core_di-self.overlap_d) - \
                                    pad_d_top
                else:
                    di_coordinate = loop_do_idx * di_batch + \
                                    merchant_d * core_di - \
                                    pad_d_top

                if self.kh >= self.sh:
                    hi_coordinate = loop_ho_idx * (hi_batch-self.overlap_h) + \
                                    merchant_h * (core_hi-self.overlap_h) - \
                                    pad_hw_top
                else:
                    hi_coordinate = loop_ho_idx * hi_batch + \
                                    merchant_h * core_hi - \
                                    pad_hw_top

                if self.kw >= self.sw:
                    wi_coordinate = loop_wo_idx * (wi_batch-self.overlap_w) + \
                                    remainder_h * (core_wi-self.overlap_w) - \
                                    pad_hw_left
                else:
                    wi_coordinate = loop_wo_idx * wi_batch + \
                                    remainder_h * core_wi - \
                                    pad_hw_left

                do_coordinate = loop_do_idx * do_batch + merchant_d * core_do
                ho_coordinate = loop_ho_idx * ho_batch + merchant_h * core_ho
                wo_coordinate = loop_wo_idx * wo_batch + remainder_h * core_wo

                # init begin coordinate of di,hi,wi
                di_coord = init_coordinate(tik_instance, pad_d_top,
                                           di_coordinate)
                hi_coord = init_coordinate(tik_instance, pad_hw_top,
                                           hi_coordinate)
                wi_coord = init_coordinate(tik_instance, pad_hw_left,
                                           wi_coordinate)

                src_orig_x_gm = merchant * prod(self.forward_in_shape[1:]) + \
                                merchant_c1 * self.h * self.w * c0 + \
                                di_coord * c1 * self.h * self.w * c0 + \
                                hi_coord * self.w * c0 + \
                                wi_coord * c0
                src_orig_y_gm = merchant * prod(self.forward_ou_shape[1:]) + \
                                merchant_c1 * self.ho * self.wo * c0 + \
                                do_coordinate * c1 * self.ho * self.wo * c0 + \
                                ho_coordinate * self.wo * c0 + \
                                wo_coordinate * c0
                src_grad_gm = src_orig_y_gm

                # ================================
                # COPY_GM_2_L1_BUF
                # Prevent reading gm out of bounds
                # ================================
                di_value = min(0, self.overlap_d) + di
                hi_value = min(0, self.overlap_h) + hi
                wi_value = min(0, self.overlap_w) + wi
                di_val = di_value
                hi_val = hi_value
                wi_val = wi_value

                d_top, d_bottom = calc_pad(tik_instance, pad_d_top, pad_d_bottom,
                                           di_coordinate, di_value, self.d)
                h_top, h_bottom = calc_pad(tik_instance, pad_hw_top, pad_hw_bottom,
                                           hi_coordinate, hi_value, self.h)
                w_top, w_bottom = calc_pad(tik_instance, pad_hw_left, pad_hw_right,
                                           wi_coordinate, wi_value, self.w)
                pad_hw_list[-1], pad_hw_list[-2] = h_bottom, h_top
                pad_hw_list[-3], pad_hw_list[-4] = w_bottom, w_top

                # gm2l1: filled regions don't move except d
                di_val = di_val - d_top - d_bottom
                hi_val = hi_val - h_top - h_bottom
                wi_val = wi_val - w_top - w_bottom
                l1_idx = d_top

                input0 = [di_val, hi_val, wi_val]
                input1 = [di_batch, hi_batch, wi_batch]
                self._gm2l1_tiling_do_ho_wo(tik_instance,
                                            l1_in_buf, src_orig_x_gm,
                                            l1_idx*hi_batch*wi_batch*c0,
                                            input0, input1)

                # ================================
                # COPY_ORI_OUTPUT_2_FORWARD_OU_BUF
                # COPY_GRAD_2_GRAD_BUF
                # in the branch, do and ho are 1.
                # ================================
                self._copy_gm_to_ub(tik_instance, forward_ou_buf,
                                    self.orig_y_gm, src_orig_y_gm,
                                    [do, ho, wo, c0])

                self._copy_gm_to_ub(tik_instance, grad_buf, self.grads_gm,
                                    src_grad_gm, [do, ho, wo, c0])

                # ================================
                # load3d l1 to col_in_buffer
                # ================================
                load3d_mark = tik_instance.Scalar(dtype='int64', name='load3d_mark')
                repeat_times = ceil_div(ho*wo, 16)
                with tik_instance.for_range(0, do, thread_num=1) as idx_do:
                    with tik_instance.for_range(0, self.kd, thread_num=1) as idx_d:
                        # =====================================================
                        # if window in position of pad, not load3d, but vec_dup.
                        # =====================================================
                        self.filled_vec_dup(tik_instance, load3d_mark, di_value,
                                            pad_d_top, pad_d_bottom,
                                            idx_do, idx_d, d_top, d_bottom,
                                            param, col_in_buf)

                        with tik_instance.for_range(0, self.kh, thread_num=1) as idx_h:
                            with tik_instance.for_range(0, self.kw, thread_num=1) as idx_w:
                                src_l1 = (idx_do * self.sd + idx_d) * hi_batch * wi_batch * c0

                                with tik_instance.if_scope(load3d_mark != 1):
                                    if self.check_load3d_support:
                                        tik_instance.load3dv1(col_in_buf[0],
                                                              l1_in_buf[src_l1],
                                                              pad_hw_list,
                                                              hi_val, wi_val, 0,
                                                              idx_w, idx_h,
                                                              -w_top, -h_top,
                                                              self.sw, self.sh,
                                                              self.kw, self.kh,
                                                              1, 1, 1, 1,
                                                              repeat_times, 0,
                                                              MIN_VALUE_FP16
                                                              )
                                    else:
                                        col_in_buf_shape = cal_shape_ele(col_in_buf.shape)
                                        self.set_vector_dup(
                                            tik_instance, col_in_buf_shape, col_in_buf, 0, MIN_VALUE_FP16, "float16")
                                        img2col(tik_instance, l1_in_buf, col_in_buf, src_l1, 0, idx_h, idx_w,
                                                -h_top, -w_top, hi_val, wi_val, self.kh, self.kw,
                                                self.sh, self.sw, repeat_times, 1, pad_hw_list)

                                # ---calculate mask---
                                idx_list = [idx_do, idx_d, idx_h, idx_w]
                                const_list = [ho, wo, c0]
                                self._calc_mask(tik_instance, buf_list, param,
                                                idx_list, const_list)

                                # ---sel(grad,zero,mask)---
                                self._sel(tik_instance, buf_list,
                                          idx_list, const_list)

                                # ---vconv grad_sel_fp16 to fp32---
                                self._vconv(tik_instance, grad_sel_fp16_buf, 0,
                                            grad_sel_fp32_buf, 0,
                                            param.grad_sel_fp16_size, "float16")

                                # ---rewrite grad_sel_fp32 to f_map_fp32
                                # `do = 1, ho = 1`
                                # map_index has two part: begin_index of kernel,
                                # begin_index of child kernel
                                # must use tik variable as index of grad_sel_fp32_buf,
                                # python variable is not work in grad_sel_fp32_buf[mask_index],
                                # `while x = grad_sel_fp32_buf[mask_index], y = x[n].`
                                with tik_instance.for_range(0, 1) as index_mask:
                                    map_index = idx_d * map_hi * map_wi * c0 + \
                                                idx_h * map_wi * c0 + idx_w * c0
                                    mask_index = index_mask

                                    self._vector_op(tik_instance, "vadd",
                                                    f_map_fp32_buf[map_index:],
                                                    grad_sel_fp32_buf[mask_index:],
                                                    f_map_fp32_buf[map_index:],
                                                    "float32", wo*c0//2,
                                                    stride_config=(self.sw*2,
                                                                   self.sw*2, 2,
                                                                   self.sw*16,
                                                                   self.sw*16, 16))
                                    self._vector_op(tik_instance, "vadd",
                                                    f_map_fp32_buf[map_index+8:],
                                                    grad_sel_fp32_buf[mask_index+8:],
                                                    f_map_fp32_buf[map_index + 8:],
                                                    "float32", wo*c0//2,
                                                    stride_config=(self.sw*2,
                                                                   self.sw*2, 2,
                                                                   self.sw*16,
                                                                   self.sw*16, 16))

                # mov_out
                dst_ou_gm = merchant * prod(self.forward_in_shape[1:]) + \
                            merchant_c1 * self.h * self.w * c0 + \
                            di_coord * c1 * self.h * self.w * c0 + \
                            hi_coord * self.w * c0 + \
                            wi_coord * c0

                def mov_atomic(num_d, num_h, num_w, dst, dst_idx, src_idx):
                    ub2gm_shape = [num_d, num_h, num_w, c0]
                    src_idx += (h_top * map_wi + w_top) * c0

                    num_bit = self.num_bit_fp32
                    n_burst = ub2gm_shape[1]
                    burst_len = prod(ub2gm_shape[2:]) * num_bit // BLOCK_SIZE
                    # c0 * num_bit // BLOCK_SIZE is 2
                    src_stride = (map_wi - num_w) * 2
                    dst_stride = (self.w - num_w) * 2

                    with tik_instance.for_range(0, ub2gm_shape[0]) as idx:
                        src_idx_new = src_idx + idx * map_hi*map_wi*c0
                        dst_idx_new = dst_idx + prod(self.forward_in_shape[3:]) * c1 * idx

                        in_list = [n_burst, burst_len, src_stride,
                                   dst_stride, src_idx_new, dst_idx_new]
                        check = isinstance(src_stride, int)

                        with tik_instance.if_scope(
                                tik.any(src_stride > MAX_STRIDE,
                                        dst_stride > MAX_STRIDE)):
                            self._ultimate_data_move(tik_instance, f_map_fp32_buf,
                                                     dst, in_list, num_bit)

                        with tik_instance.else_scope():
                            if check:
                                if src_stride <= MAX_STRIDE and dst_stride <= MAX_STRIDE:
                                    self.norm_data_move(tik_instance, f_map_fp32_buf,
                                                        dst, in_list)
                            else:
                                self.norm_data_move(tik_instance, f_map_fp32_buf,
                                                    dst, in_list)

                    self.set_vector_dup(tik_instance, param.f_map_fp32_size,
                                        f_map_fp32_buf, 0, 0, "float32")

                tik_instance.set_atomic_add(1)
                mov_atomic(di_val, hi_val, wi_val,
                           self.ou_y_gm, dst_ou_gm,
                           l1_idx*map_hi*map_wi*c0)
                tik_instance.set_atomic_add(0)

            if wo_tail != 0:
                with tik_instance.for_range(0, loop_do) as do_idx:
                    with tik_instance.for_range(0, loop_ho) as ho_idx:
                        with tik_instance.for_range(0, loop_wo) as wo_idx:
                            _main(do_idx, ho_idx, wo_idx,
                                  di_batch, do_batch,
                                  hi_batch, ho_batch,
                                  wi_batch, wo_batch)
                        _main(do_idx, ho_idx, loop_wo,
                              di_tail, do_batch,
                              hi_tail, ho_batch,
                              wi_tail, wo_tail)
            else:
                with tik_instance.for_range(0, loop_do) as do_idx:
                    with tik_instance.for_range(0, loop_ho) as ho_idx:
                        with tik_instance.for_range(0, loop_wo) as wo_idx:
                            _main(do_idx, ho_idx, wo_idx,
                                  di_batch, do_batch,
                                  hi_batch, ho_batch,
                                  wi_batch, wo_batch)

    def filled_vec_dup(self, tik_instance, mark, di_value, pad_d_top,
                       pad_d_bottom, idx_do, idx_d, d_top, d_bottom,
                       param, dst_buf):
        """
        filled_vec_dup
        """
        # make filled region in l1_buf, not move to
        # col_in_buf by load3d, but vec_dup in col_in_buf.
        mark.set_as(0)
        win_idx = idx_do * self.sd + idx_d
        if pad_d_top != 0:
            with tik_instance.if_scope(win_idx < d_top):
                self.set_vector_dup(tik_instance, param.col_in_size,
                                    dst_buf, 0, MIN_VALUE_FP16, "float16")
                mark.set_as(1)

        if pad_d_bottom != 0:
            with tik_instance.if_scope(win_idx > di_value-d_bottom-1):
                self.set_vector_dup(tik_instance, param.col_in_size,
                                    dst_buf, 0, MIN_VALUE_FP16, "float16")
                mark.set_as(1)

    def get_tik_instance(self):
        """
        obtain tik instance
        """
        tik_instance = self._compute()
        tik_instance.BuildCCE(kernel_name=self.kernel_name,
                              inputs=[self.orig_x_gm,
                                      self.orig_y_gm,
                                      self.grads_gm],
                              outputs=[self.ou_y_gm])

        return tik_instance

    def _padding_mode(self,):
        # NDC1HWC0
        _, map_d, _, map_h, map_w, _ = self.forward_in_shape
        _, kernel_d, kernel_h, kernel_w, _ = self.ksize
        if self.pads.upper() == 'VALID':
            do = int(math.ceil((map_d - kernel_d + 1) * 1.0 / self.sd))
            ho = int(math.ceil((map_h - kernel_h + 1) * 1.0 / self.sh))
            wo = int(math.ceil((map_w - kernel_w + 1) * 1.0 / self.sw))
            pad_d_top = pad_d_bottom = \
                pad_hw_top = pad_hw_left = pad_hw_bottom = pad_hw_right = 0

        else:
            do = (map_d + self.sd - 1) // self.sd
            ho = (map_h + self.sh - 1) // self.sh
            wo = (map_w + self.sw - 1) // self.sw

            pad_h = max((ho - 1) * self.sh + kernel_h - map_h, 0)
            pad_hw_top = pad_h // 2
            pad_hw_bottom = pad_h - pad_hw_top
            pad_w = max((wo - 1) * self.sw + kernel_w - map_w, 0)
            pad_hw_left = pad_w // 2
            pad_hw_right = pad_w - pad_hw_left

            pad_d = max((do - 1) * self.sd + kernel_d - map_d, 0)
            pad_d_top = pad_d // 2
            pad_d_bottom = pad_d - pad_d_top

        pad_model = [[pad_d_top, pad_d_bottom],
                     [pad_hw_top, pad_hw_bottom],
                     [pad_hw_left, pad_hw_right]]

        return [do, ho, wo, pad_model]

    def _infer_dim_return(self, do, ho, wo, model):

        if self.kd >= self.sd:
            di = self.kd + (do-1) * self.sd
        else:
            di = do * self.sd

        if self.kh >= self.sh:
            hi = self.kh + (ho-1) * self.sh
        else:
            hi = ho * self.sh

        if self.kw > self.sw:
            wi = self.kw + (wo-1) * self.sw
        else:
            wi = wo * self.sw

        # model: True, work for real split
        # model: False, calc used part for _invalid_part()
        # if not split do,ho,wo, all dim would
        # be return.
        if model:
            if self.do == do:
                # in "SAME", return the filled di
                di = self.d + self.pad[0][0] + self.pad[0][1]
            if self.ho == ho:
                hi = self.h
            if self.wo == wo:
                wi = self.w

        return di, hi, wi

    def _infer_map_return(self, do, ho, wo):
        # Only work in "SAME", return size of feature_map.
        # Because in "VALID", feature_map's size is as same as l1_in_buf.

        if self.kd >= self.sd:
            di = self.kd + (do-1) * self.sd
        else:
            di = do * self.sd

        if self.kh >= self.sh:
            hi = self.kh + (ho-1) * self.sh
        else:
            hi = ho * self.sh

        if self.kw > self.sw:
            wi = self.kw + (wo-1) * self.sw
        else:
            wi = wo * self.sw

        if self.do == do:
            di = self.d + self.pad[0][0] + self.pad[0][1]
        if self.ho == ho:
            hi = self.h + self.pad[1][0] + self.pad[1][1]
        if self.wo == wo:
            wi = self.w + self.pad[2][0] + self.pad[2][1]

        return di, hi, wi

    def _invalid_part(self):
        # return area of kernel doesn't slides
        di, hi, wi = self._infer_dim_return(self.do, self.ho, self.wo, False)
        invalid_d = self.d - di
        invalid_h = self.h - hi
        invalid_w = self.w - wi

        return invalid_d, invalid_h, invalid_w

    def _check_process_space(self, do, ho, wo):
        """
        =====================================
        UB_space: compute virtual space in UB
        =====================================
        col_in_shape:ho wo c0 (512B) ---> for load3d
        forward_ou_shape: do ho wo c0(last do have 256B) ---> for vcmp_eq
        mask_shape: ho wo (uint16,)
        mask_or_shape: same
        mask_not_shape: same
        grad_shape: do ho wo c0(as same as forward_ou_shape)
        zero_shape: 256B
        grad_vsel_fp16_shape: ho wo c0(256B)
        grad_vsel_fp32_shape: ho wo c0(256B)
        f_map_fp32_shape: di hi wi c0
        """
        # If consider padding, L1_size must be less
        # than computation of _infer_dim_return.
        # So,actual of data move in L1 may less than space of malloc
        # If data of L1 col2img to UB, l1_in_data would be released.
        # Then, L1 will be used to save overlap which may include
        # overlap_d and overlap_h.
        # if l1_split = True, UB also can't process do ho wo.

        # due to valid, self.pads is [[0,0],[0,0],[0,0]]
        # l1_in_shape is most
        infer_di, infer_hi, infer_wi = self._infer_dim_return(do, ho, wo, True)
        l1_in_shape = [infer_di, infer_hi, infer_wi, self.c0]
        l1_in_size = prod(l1_in_shape)

        col_in_shape = [ho, wo, self.c0]
        col_in_size = ceil_div(prod(col_in_shape), 256) * 256

        forward_ou_shape_last_do = [1, ho, wo, self.c0]
        forward_ou_shape_except_last = [do-1, ho, wo, self.c0]
        forward_ou_size = ceil_div(prod(forward_ou_shape_last_do), 128) * 128
        forward_ou_size += prod(forward_ou_shape_except_last)

        mask_shape = [ho, wo]
        mask_size = ceil_div(prod(mask_shape), 128) * 128
        grad_size = forward_ou_size
        zero_size = 128

        grad_sel_fp16_shape = [ho, wo, self.c0]
        grad_sel_fp16_size = ceil_div(prod(grad_sel_fp16_shape), 128) * 128
        grad_sel_fp32_size = grad_sel_fp16_size

        if self.pads.upper() == "VALID":
            f_map_fp32_shape = [infer_di, infer_hi, infer_wi, self.c0]
        else:
            map_di, map_hi, map_wi = self._infer_map_return(do, ho, wo)
            f_map_fp32_shape = [map_di, map_hi, map_wi, self.c0]
        f_map_fp32_size = prod(f_map_fp32_shape)

        used_ub_byte = (col_in_size + forward_ou_size + mask_size * 3 +
                        grad_size + zero_size +
                        grad_sel_fp16_size) * self.num_bit + \
                       (grad_sel_fp32_size + f_map_fp32_size) * 4

        ub_split = used_ub_byte > self.ub_maxsize * self.num_bit

        param = Params(ub_split, col_in_size, forward_ou_size,
                       mask_size, grad_size, zero_size, grad_sel_fp16_size,
                       grad_sel_fp32_size, f_map_fp32_size, l1_in_size)
        return param

    def _pattern(self, core_ou_shape, core_branch):
        # valid
        # D H W C0 -> Do Ho Wo C0
        all_wo = core_ou_shape[-2]
        all_ho = core_ou_shape[-3]
        all_do = core_ou_shape[-4]

        wo = all_wo
        ho = all_ho
        do = all_do

        split_do = False
        split_ho = False
        split_wo = False

        for k in range(all_do):
            do = all_do - k
            param = self._check_process_space(do, ho, wo)
            if not param.ub_split:
                split_do = True
                break

        if not split_do:
            do = 1
            for k in range(all_ho):
                ho = all_ho - k
                param = self._check_process_space(do, ho, wo)
                if not param.ub_split:
                    split_do = True
                    split_ho = True
                    break

        if not split_do and not split_ho:
            do = ho = 1
            for k in range(all_wo):
                wo = all_wo - k
                param = self._check_process_space(do, ho, wo)
                if not param.ub_split:
                    split_do = True
                    split_ho = True
                    split_wo = True
                    break

        cut_model = [split_do, split_ho, split_wo]
        split_model = [do, ho, wo]
        if cut_model == [False, False, False]:
            error_manager_vector.raise_err_specific_reson("MaxPoolGRAD", "kernel is too larger")

        # avoid hardware bugs that load3dv1 can't
        # support wo=1 and ho != 1 in cloud_v100
        if split_model[-1] == 1 and split_model[-2] != 1:
            param = self._check_process_space(1, 1, 1)
            cut_model = [True, True, True]
            split_model = [1, 1, 1]

        branch = self._check_cut_model(cut_model, split_model,
                                       all_do, core_branch)

        return branch, split_model, param

    def _copy_gm_to_l1(self, tik_instance, l1_buf, src_idx, dst_idx, in_shape):
        n_burst = in_shape[0]
        burst_len = prod(in_shape[1:]) * self.num_bit // BLOCK_SIZE
        src_stride = (prod(self.forward_in_shape[3:]) * (self.c1-1) +
                      prod(self.forward_in_shape[4:]) *
                      (self.h-in_shape[1])) * self.num_bit // BLOCK_SIZE
        dst_stride = 0

        in_list = [n_burst, burst_len, src_stride, dst_stride, src_idx, dst_idx]
        check = isinstance(src_stride, int)
        with tik_instance.if_scope(src_stride > MAX_STRIDE):
            self._ultimate_data_move(tik_instance, self.orig_x_gm,
                                     l1_buf, in_list, self.num_bit)

        with tik_instance.else_scope():
            if check:
                if src_stride <= MAX_STRIDE:
                    self.norm_data_move(tik_instance, self.orig_x_gm,
                                        l1_buf, in_list)
            else:
                self.norm_data_move(tik_instance, self.orig_x_gm,
                                    l1_buf, in_list)

    def _gm2l1_tiling_do_ho(self, tik_instance, l1_buf, src_idx, dst_idx,
                            in_shape, hi_batch):
        n_burst = in_shape[0]
        burst_len = prod(in_shape[1:]) * self.num_bit // BLOCK_SIZE
        src_stride = (prod(self.forward_in_shape[3:]) * (self.c1-1) +
                      prod(self.forward_in_shape[4:]) *
                      (self.h-in_shape[1])) * self.num_bit // BLOCK_SIZE
        dst_stride = (hi_batch - in_shape[1]) * self.w * self.c0 * \
                     self.num_bit // BLOCK_SIZE

        in_list = [n_burst, burst_len, src_stride,
                   dst_stride, src_idx, dst_idx]
        # dst_stride and src_stride must be same type
        check = isinstance(src_stride, int)

        with tik_instance.if_scope(
                tik.any(src_stride > MAX_STRIDE,
                        dst_stride > MAX_STRIDE)):
            self._ultimate_data_move(tik_instance, self.orig_x_gm,
                                     l1_buf, in_list, self.num_bit)
        with tik_instance.else_scope():
            if check:
                if src_stride <= MAX_STRIDE and dst_stride <= MAX_STRIDE:
                    self.norm_data_move(tik_instance, self.orig_x_gm,
                                        l1_buf, in_list)
            else:
                self.norm_data_move(tik_instance, self.orig_x_gm,
                                    l1_buf, in_list)

    def _gm2l1_tiling_do_ho_wo(self, tik_instance,
                               l1_buf, src_idx, dst_idx,
                               input0, input1):

        di_val, hi_val, wi_val = input0[0], input0[1], input0[2]
        _, hi_batch, wi_batch = input1[0], input1[1], input1[2]

        # ==================================
        # copy gm to l1
        # ==================================
        c1 = self.c1
        c0 = self.c0
        in_shape = [hi_val, wi_val, c0]
        n_burst = in_shape[0]
        burst_len = prod(in_shape[1:]) * self.num_bit // BLOCK_SIZE
        src_stride = (self.w - wi_val) * c0 * self.num_bit // BLOCK_SIZE
        dst_stride = 0

        with tik_instance.for_range(0, di_val) as idx:
            src_idx_new = src_idx + prod(self.forward_in_shape[3:]) * c1 * idx
            dst_idx_new = dst_idx + hi_batch * wi_batch * c0 * idx

            in_list = [n_burst, burst_len, src_stride,
                       dst_stride, src_idx_new, dst_idx_new]
            check = isinstance(src_stride, int)

            with tik_instance.if_scope(src_stride > MAX_STRIDE):
                self._ultimate_data_move(tik_instance, self.orig_x_gm,
                                         l1_buf, in_list, self.num_bit)

            with tik_instance.else_scope():
                if check:
                    if src_stride <= MAX_STRIDE:
                        self.norm_data_move(tik_instance, self.orig_x_gm,
                                            l1_buf, in_list)
                else:
                    self.norm_data_move(tik_instance, self.orig_x_gm,
                                        l1_buf, in_list)

    def _copy_ub_to_gm(self, tik_instance, src_buf, src_idx,
                       dst_buf, dst_idx, in_shape):
        # "float32"
        n_burst = in_shape[0]
        burst_len = prod(in_shape[1:]) * self.num_bit_fp32 // BLOCK_SIZE
        src_stride = 0
        dst_stride = (prod(self.forward_in_shape[3:]) * (self.c1-1) +
                      prod(self.forward_in_shape[4:]) *
                      (self.h-in_shape[1])) * self.num_bit_fp32 // BLOCK_SIZE

        in_list = [n_burst, burst_len, src_stride,
                   dst_stride, src_idx, dst_idx]
        check = isinstance(dst_stride, int)

        with tik_instance.if_scope(dst_stride > MAX_STRIDE):
            self._ultimate_data_move(tik_instance, src_buf,
                                     dst_buf, in_list, self.num_bit_fp32)
        with tik_instance.else_scope():
            if check:
                if dst_stride <= MAX_STRIDE:
                    self.norm_data_move(tik_instance, src_buf, dst_buf, in_list)
            else:
                self.norm_data_move(tik_instance, src_buf, dst_buf, in_list)

    def _ub2gm_split_do_ho_2(self, tik_instance, src,
                             src_idx, dst, dst_idx, in_shape, hi_batch):
        n_burst = in_shape[0]
        burst_len = prod(in_shape[1:]) * \
                    self.num_bit_fp32 // BLOCK_SIZE
        src_stride = (hi_batch - in_shape[1]) * self.w * self.c0 * \
                     self.num_bit_fp32 // BLOCK_SIZE
        dst_stride = (prod(self.forward_in_shape[3:]) * (self.c1-1) +
                      prod(self.forward_in_shape[4:]) *
                      (self.h-in_shape[1])) * \
                     self.num_bit_fp32 // BLOCK_SIZE

        in_list = [n_burst, burst_len, src_stride,
                   dst_stride, src_idx, dst_idx]
        # src_stride and dst_stride must be same type
        check = isinstance(src_stride, int)

        with tik_instance.if_scope(
                tik.any(src_stride > MAX_STRIDE,
                        dst_stride > MAX_STRIDE)):
            self._ultimate_data_move(tik_instance, src, dst,
                                     in_list, self.num_bit_fp32)

        with tik_instance.else_scope():
            if check:
                if src_stride <= MAX_STRIDE and dst_stride <= MAX_STRIDE:
                    self.norm_data_move(tik_instance, src, dst, in_list)
            else:
                self.norm_data_move(tik_instance, src, dst, in_list)

    def _ub2gm_split_do_ho(self, tik_instance, src_buf, src_idx,
                           dst_buf, dst_idx, in_shape):
        n_burst = in_shape[0]
        burst_len = prod(in_shape[1:]) * \
                    self.num_bit_fp32 // BLOCK_SIZE
        src_stride = self.overlap_h * self.w * self.c0 * \
                     self.num_bit_fp32 // BLOCK_SIZE
        dst_stride = (prod(self.forward_in_shape[3:]) * (self.c1-1) +
                      prod(self.forward_in_shape[4:]) *
                      (self.h-in_shape[1])) * \
                     self.num_bit_fp32 // BLOCK_SIZE

        in_list = [n_burst, burst_len, src_stride,
                   dst_stride, src_idx, dst_idx]
        # src_stride is int,
        check = isinstance(dst_stride, int)

        with tik_instance.if_scope(
                tik.any(src_stride > MAX_STRIDE,
                        dst_stride > MAX_STRIDE)):
            self._ultimate_data_move(tik_instance, src_buf,
                                     dst_buf, in_list, self.num_bit_fp32)

        with tik_instance.else_scope():
            if check:
                if src_stride <= MAX_STRIDE and dst_stride <= MAX_STRIDE:
                    self.norm_data_move(tik_instance, src_buf, dst_buf, in_list)
            else:
                if src_stride <= MAX_STRIDE:
                    self.norm_data_move(tik_instance, src_buf, dst_buf, in_list)

    def _ub2gm_split_do_ho_wo(self, tik_instance, src, src_idx, dst, dst_idx,
                              in_shape, hi_batch, wi_batch):

        c0 = in_shape[-1]
        c1 = self.c1
        num_bit = self.num_bit_fp32

        n_burst = in_shape[1]
        burst_len = prod(in_shape[2:]) * num_bit // BLOCK_SIZE
        src_stride = (wi_batch - in_shape[2]) * c0 * num_bit // BLOCK_SIZE
        dst_stride = (self.w - in_shape[2]) * c0 * num_bit // BLOCK_SIZE

        for idx in range(in_shape[0]):
            dst_idx_new = dst_idx + prod(self.forward_in_shape[3:]) * c1 * idx
            src_idx_new = src_idx + hi_batch * wi_batch * c0 * idx

            in_list = [n_burst, burst_len, src_stride,
                       dst_stride, src_idx_new, dst_idx_new]
            # type of src_stride is as same as dst_stride
            check = isinstance(src_stride, int)

            with tik_instance.if_scope(
                    tik.any(src_stride > MAX_STRIDE,
                            dst_stride > MAX_STRIDE)):
                self._ultimate_data_move(tik_instance, src,
                                         dst, in_list, num_bit)
            with tik_instance.else_scope():
                if check:
                    if src_stride <= MAX_STRIDE and dst_stride <= MAX_STRIDE:
                        self.norm_data_move(tik_instance, src, dst, in_list)
                else:
                    self.norm_data_move(tik_instance, src, dst, in_list)

    def _mov_init(self, tik_instance, ubuf,
                  num_overlap, num_init_zero):
        # mov  float32
        all_num = num_overlap + num_init_zero
        n_burst = 1
        burst_len = num_overlap * 4 // BLOCK_SIZE
        src_stride = 0
        dst_stride = 0
        if num_overlap > 0:
            tik_instance.data_move(ubuf[0],
                                   ubuf[all_num-num_overlap],
                                   0,
                                   n_burst,
                                   burst_len,
                                   src_stride,
                                   dst_stride)
        # vec_dup
        self.set_vector_dup(tik_instance, num_init_zero,
                            ubuf, num_overlap, 0, "float32")

    def _copy_gm_to_ub(self, tik_instance, dst_buf, src_buf, src_idx, in_shape):
        # Only split do, self.ho is equal to in_shape[1], self.wo is equal
        # to in_shape[2].
        # Only split do and ho, self.wo is equal to in_shape[2], and do is 1.
        # Only split do, ho, wo, do and ho is 1.
        n_burst = in_shape[0]
        burst_len = prod(in_shape[1:]) * self.num_bit // BLOCK_SIZE
        src_stride = (prod(self.forward_ou_shape[3:]) * (self.c1-1) +
                      prod(self.forward_ou_shape[4:]) * (self.ho-in_shape[1]) +
                      self.c0 * (self.wo-in_shape[2])) * \
                     self.num_bit // BLOCK_SIZE
        dst_stride = 0

        if src_stride > MAX_STRIDE or dst_stride > MAX_STRIDE:
            in_list = [n_burst, burst_len, src_stride,
                       dst_stride, src_idx, 0]
            self._ultimate_data_move(tik_instance, src_buf,
                                     dst_buf, in_list, self.num_bit)
        else:
            tik_instance.data_move(dst_buf[0],
                                   src_buf[src_idx],
                                   0,
                                   n_burst,
                                   burst_len,
                                   src_stride,
                                   dst_stride)

    def _vector_op(self, tik_instance, operator,
                   src1, src2, dst, dtype, ele_num,
                   stride_config=None):
        stride_config = list(stride_config)
        if dtype == "float16":
            repeat_times = ele_num // VECTOR_FP16_SIZE
            remain_ele = ele_num % VECTOR_FP16_SIZE
            mask = VECTOR_FP16_SIZE
        else:
            repeat_times = ele_num // VECTOR_FP32_SIZE
            remain_ele = ele_num % VECTOR_FP32_SIZE
            mask = VECTOR_FP32_SIZE

        repeat_max_loop = repeat_times // MAX_VECTOR_REPEATE_TIME
        remain_max_loop = repeat_times % MAX_VECTOR_REPEATE_TIME

        if operator == "vadd":
            if stride_config is None:
                stride_config = 1, 1, 1, 8, 8, 8
            dst_offset = 0
            src1_offset = 0
            src2_offset = 0
            if repeat_max_loop > 0:
                tik_instance.vadd(mask, dst[dst_offset], src1[src1_offset],
                                  src2[src2_offset],
                                  MAX_VECTOR_REPEATE_TIME,
                                  stride_config[0], stride_config[1],
                                  stride_config[2], stride_config[3],
                                  stride_config[4], stride_config[5])
                dst_offset += BLOCK_SIZE // (tbe_platform.get_bit_len(
                    dst.dtype.lower()) // 8) * stride_config[3] * 255
                src1_offset += BLOCK_SIZE // (tbe_platform.get_bit_len(
                    src1.dtype.lower()) // 8) * stride_config[4] * 255
                src2_offset += BLOCK_SIZE // (tbe_platform.get_bit_len(
                    src2.dtype.lower()) // 8) * stride_config[5] * 255
            if remain_max_loop > 0:
                # rep_stride maybe more than 255 while repeat_times=1.
                if remain_max_loop == 1:
                    tik_instance.vadd(mask, dst[dst_offset], src1[src1_offset],
                                      src2[src2_offset],
                                      remain_max_loop,
                                      stride_config[0], stride_config[1],
                                      stride_config[2], 0, 0, 0)
                else:
                    if self.sw >= 16:
                        error_manager_vector.raise_err_specific_reson(
                            "maxpoolgrad", "dst_rep_stride exceed limit")
                    tik_instance.vadd(mask, dst[dst_offset], src1[src1_offset],
                                      src2[src2_offset],
                                      remain_max_loop,
                                      stride_config[0], stride_config[1],
                                      stride_config[2], stride_config[3],
                                      stride_config[4], stride_config[5])
                dst_offset += BLOCK_SIZE // (tbe_platform.get_bit_len(
                    dst.dtype.lower()) // 8) * stride_config[3] * remain_max_loop
                src1_offset += BLOCK_SIZE // (tbe_platform.get_bit_len(
                    src1.dtype.lower()) // 8) * stride_config[4] * remain_max_loop
                src2_offset += BLOCK_SIZE // (tbe_platform.get_bit_len(
                    src2.dtype.lower()) // 8) * stride_config[5] * remain_max_loop
            if remain_ele > 0:
                stride_config[3] = stride_config[4] = stride_config[5] = 0
                tik_instance.vadd(remain_ele, dst[dst_offset], src1[src1_offset],
                                  src2[src2_offset], 1,
                                  stride_config[0], stride_config[1],
                                  stride_config[2], stride_config[3],
                                  stride_config[4], stride_config[5])

    def _set_buf_tensor(self, tik_instance, param):

        l1_in_buf = tik_instance.Tensor(self.dtype,
                                        [param.l1_in_size, ],
                                        name="l1_in_buf",
                                        scope=tik.scope_cbuf)
        forward_ou_buf = tik_instance.Tensor(self.dtype,
                                             [param.forward_ou_size, ],
                                             name="forward_ou_buf",
                                             scope=tik.scope_ubuf)
        grad_buf = tik_instance.Tensor(self.dtype,
                                       [param.grad_size, ],
                                       name="grad_buf",
                                       scope=tik.scope_ubuf)
        col_in_buf = tik_instance.Tensor(self.dtype,
                                         [param.col_in_size, ],
                                         name="col_in_buf",
                                         scope=tik.scope_ubuf)
        mask_buf = tik_instance.Tensor("uint16",
                                       [param.mask_size, ],
                                       name='mask_buf',
                                       scope=tik.scope_ubuf)
        mask_or_buf = tik_instance.Tensor("uint16",
                                          [param.mask_size, ],
                                          name='mask_or_buf',
                                          scope=tik.scope_ubuf)
        mask_not_buf = tik_instance.Tensor("uint16",
                                           [param.mask_size, ],
                                           name='mask_not_buf',
                                           scope=tik.scope_ubuf)
        zero_buf = tik_instance.Tensor(self.dtype,
                                       [param.zero_size, ],
                                       name='zero_buf',
                                       scope=tik.scope_ubuf)

        grad_sel_fp16_buf = tik_instance.Tensor(self.dtype,
                                                [param.grad_sel_fp16_size, ],
                                                name='grad_sel_fp16_buf',
                                                scope=tik.scope_ubuf)
        grad_sel_fp32_buf = tik_instance.Tensor("float32",
                                                [param.grad_sel_fp32_size, ],
                                                name='grad_sel_fp32_buf',
                                                scope=tik.scope_ubuf)
        f_map_fp32_buf = tik_instance.Tensor("float32",
                                             [param.f_map_fp32_size, ],
                                             name='f_map_fp32_buf',
                                             scope=tik.scope_ubuf)

        buf_list = [l1_in_buf, forward_ou_buf, grad_buf, col_in_buf,
                    mask_buf, mask_or_buf, mask_not_buf, zero_buf,
                    grad_sel_fp16_buf, grad_sel_fp32_buf,
                    f_map_fp32_buf]

        return buf_list

    def _calc_mask(self, tik_instance, buf_list, param,
                   idx_list, const_list):
        # ---calculate mask---
        forward_ou_buf = buf_list[1]
        col_in_buf = buf_list[3]
        mask_buf = buf_list[4]
        mask_or_buf = buf_list[5]
        mask_not_buf = buf_list[6]

        idx_do = idx_list[0]
        idx_d = idx_list[1]
        idx_h = idx_list[2]
        idx_w = idx_list[3]
        ho, wo, c0 = const_list

        with tik_instance.if_scope(tik.all(idx_d == 0, idx_h == 0, idx_w == 0)):
            tik_instance.vcmpv_eq(mask_buf[0],
                                  forward_ou_buf[idx_do*ho*wo*c0],
                                  col_in_buf[0],
                                  math.ceil(ho*wo*c0/VECTOR_FP16_SIZE),
                                  1, 1, 8, 8)

            tik_instance.data_move(mask_or_buf[0],
                                   mask_buf[0], 0, 1,
                                   param.mask_size//16, 0, 0)

            tik_instance.vnot(self.mask_fp16, mask_not_buf, mask_or_buf,
                              param.mask_size // VECTOR_FP16_SIZE,
                              1, 1, 8, 8)

        with tik_instance.else_scope():
            tik_instance.vcmpv_eq(mask_buf[0],
                                  forward_ou_buf[idx_do*ho*wo*c0],
                                  col_in_buf[0],
                                  math.ceil(ho*wo*c0/VECTOR_FP16_SIZE),
                                  1, 1, 8, 8)

            tik_instance.vand(self.mask_fp16, mask_buf, mask_not_buf, mask_buf,
                              param.mask_size // VECTOR_FP16_SIZE,
                              1, 1, 1, 8, 8, 8)

            tik_instance.vor(self.mask_fp16, mask_or_buf, mask_or_buf, mask_buf,
                             param.mask_size // VECTOR_FP16_SIZE,
                             1, 1, 1, 8, 8, 8)

            tik_instance.vnot(self.mask_fp16, mask_not_buf, mask_or_buf,
                              param.mask_size // VECTOR_FP16_SIZE,
                              1, 1, 8, 8)

    def _sel(self, tik_instance, buf_list, idx_list, const_list):
        mask_buf = buf_list[4]
        zero_buf = buf_list[7]
        grad_buf = buf_list[2]
        grad_sel_fp16_buf = buf_list[8]

        ho, wo, c0 = const_list
        idx_do = idx_list[0]

        repeat_times_sel = math.ceil(ho*wo*c0/VECTOR_FP16_SIZE)
        with tik_instance.for_range(0, repeat_times_sel) as serial:
            grad_sel_offset = serial * 128
            grad_offset = serial * 128 + idx_do*ho*wo*c0
            mask_offset = serial * 8
            cmp_mask = tik_instance.mov_tensor_to_cmpmask(mask_buf[mask_offset])
            tik_instance.vsel(self.mask_fp16, 0,
                              grad_sel_fp16_buf[grad_sel_offset],
                              cmp_mask,
                              grad_buf[grad_offset],
                              zero_buf,
                              1, 1, 1, 1, 8, 8, 0)

    def _split_core(self):
        # ============================
        # SPLIT Do,Ho,Wo for core_num
        # core_branch:
        # 0: "not_split"
        # 1: "split_do"
        # 2: "split_do_ho"
        # 3: "split_do_ho_wo"
        # =============================
        n, do, c1, ho, wo, c0 = self.n, self.do, self.c1, self.ho, self.wo, \
                                self.c0
        core_ou_shape = [do, ho, wo, c0]
        base_num = n * c1


        total_num = base_num
        core_num = CORE_NUM
        core_branch = 0

        do, ho, wo = core_ou_shape[0], core_ou_shape[1], core_ou_shape[2]
        di, hi, wi = self._infer_dim_return(do, ho, wo, True)
        core_in_shape = [di, hi, wi, c0]

        return [total_num, core_num, core_ou_shape, core_in_shape, core_branch]

    def _compute(self):
        """
        the overall data move process
        """
        tik_instance = self.set_tik_instance()
        total_num, core_num, core_ou_shape, \
        core_in_shape, core_branch = self._split_core()
        self.core_ou_shape = core_ou_shape
        self.core_in_shape = core_in_shape
        branch, split_model, param = self._pattern(core_ou_shape, core_branch)
        if self.pads.upper() == 'VALID':
            # =====================
            # case0: n*c1 as core
            # =====================
            if branch == "not_tiling":
                self.grad(tik_instance, split_model, param,
                            total_num, core_num,
                            self.not_tiling_main)

            elif branch == "tiling_do":
                self.grad(tik_instance, split_model, param,
                            total_num, core_num,
                            self.tiling_do_main)

            elif branch == "tiling_do_ho":
                self.grad(tik_instance, split_model, param,
                            total_num, core_num,
                            self.tiling_do_ho_main)

            else:
                # "tiling_do_ho_wo"
                self.grad(tik_instance, split_model, param,
                            total_num, core_num,
                            self.tiling_do_ho_wo_main)

        else:
            if branch in ["tiling_do", "not_tiling"]:
                self.grad(tik_instance, split_model, param,
                          total_num, core_num,
                          self.same_pure_atomic_tiling_do)

            elif branch == "tiling_do_ho":
                self.grad(tik_instance, split_model, param,
                          total_num, core_num,
                          self.same_pure_atomic_tiling_do_ho)
            else:
                # "tiling_do_ho_wo"
                self.grad(tik_instance, split_model, param,
                          total_num, core_num,
                          self.same_pure_atomic_tiling_do_ho_wo)

        return tik_instance


class Params:
    """
        Function: use to store concat base parameters
    """
    def __init__(self, ub_split, col_in_size,
                 forward_ou_size, mask_size, grad_size,
                 zero_size, grad_sel_fp16_size, grad_sel_fp32_size,
                 f_map_fp32_size, l1_in_size):
        self.ub_split = ub_split
        self.l1_in_size = l1_in_size
        self.col_in_size = col_in_size
        self.forward_ou_size = forward_ou_size
        self.mask_size = mask_size
        self.grad_size = grad_size
        self.zero_size = zero_size
        self.grad_sel_fp16_size = grad_sel_fp16_size
        self.grad_sel_fp32_size = grad_sel_fp32_size
        self.f_map_fp32_size = f_map_fp32_size


# 'pylint: disable=too-many-arguments,unused-argument,invalid-name,huawei-too-many-arguments
def check_supported(ori_input,
                     ori_output,
                     grad,
                     y,
                     ksize,
                     strides,
                     padding="CALCULATED",
                     pads=[0, 0, 0, 0],
                     data_format="NCHW",
                     global_pooling=False,
                     ceil_mode=False,
                     kernel_name="max_pool_v3_grad"):
    """
    check whether ai_core is supported
    """
    return True, ""


# 'pylint: disable=dangerous-default-value
# 'pylint: disable=too-few-public-methods,too-many-statements,too-many-branches,no-self-use,huawei-too-many-arguments
# 'pylint: disable=too-many-instance-attributes,unused-argument,too-many-lines
# 'pylint: disable=too-many-lines,too-many-locals,too-many-statements,unused-variable,too-many-arguments
def op_select_format(orig_x,
                     orig_y,
                     grads,
                     y,
                     ksize,
                     strides,
                     padding="CALCULATED",
                     pads=[0, 0, 0, 0],
                     data_format="NCHW",
                     global_pooling=False,
                     ceil_mode=False,
                     kernel_name="max_pool_v3_grad"):
    x1_format,  x2_format,  grad_format,  y_format = \
        ["NC1HWC0", ], ["NC1HWC0", ], ["NC1HWC0", ], ["NC1HWC0", ]
    x1_dy_format, x2_dy_format, grad_dy_format, y_dy_format = \
        ["NC1HWC0", ], ["NC1HWC0", ], ["NC1HWC0", ], ["NC1HWC0", ]
    x1_dtype, x2_dtype, grad_dtype, y_dtype = \
        ["float16", ], ["float16", ], ["float16", ], ["float16", ]

    input0 = gen_param(classify="input0", name="orig_input",
                       datatype=",".join(x1_dtype),
                       format=",".join(x1_format),
                       unknownshape_format=",".join(x1_dy_format))
    input1 = gen_param(classify="input1", name="orig_output",
                       datatype=",".join(x2_dtype),
                       format=",".join(x2_format),
                       unknownshape_format=",".join(x2_dy_format))
    input2 = gen_param(classify="input2", name="grad",
                       datatype=",".join(grad_dtype),
                       format=",".join(grad_format),
                       unknownshape_format=",".join(grad_dy_format))
    output0 = gen_param(classify="output0", name="out_grad",
                        datatype=",".join(y_dtype),
                        format=",".join(y_format),
                        unknownshape_format=",".join(y_dy_format))

    param_list = [input0, input1, input2, output0]
    param_dynamic_in_json = get_dynamic_param_in_json(param_list)
    return param_dynamic_in_json
