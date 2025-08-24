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
max_pool_grad_with_argmax_resnet50
"""

from te import tik
from tbe.common import platform as tbe_platform
from impl import common_util
from impl import constant_util as constant
from tbe.common.platform import ASCEND_910
from tbe.common.platform import ASCEND_310
from tbe.common.platform.platform_info import get_soc_spec
from tbe.common.platform import SHORT_SOC_VERSION

# size of vector calc one repeat
ONE_REPEAT = 256
# max repeat of vector calc
V_MAX_REPEAT = 255
# max num of fp16 in one repeat
FP16_MAX = 128
# max num of fp32 in one repeat
FP32_MAX = 64
# max num of fp16 mask handle one time
MASK_MAX = 8


# 'pylint: disable=locally-disabled,too-few-public-methods,
# 'pylint: disable=too-many-instance-attributes
class MaxpoolGradResnet50():
    """
    parameter for max_pool_grad_with_pool
    """

    # 'pylint: disable=locally-disabled,too-many-locals,too-many-arguments
    def __init__(self, grad, argmax, input_x, ksize, strides, padding):
        """
        init compare and bit pack base parameters
        Parameters
        ----------
        input_x: input of maxpool, useless for maxpool gard
        grad: input of maxpoolgard or output of maxpool
        argmax:output of maxpool mask or index
        strides: stride , minimum length is 4,
                 just like [1, poolingStrideH, poolingStrideW, 1]
        padding: pad mode, just support "SANME" or "VALID"
        Returns
        -------
        None
        """
        self.blocknum = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.ub_size = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)

        self.input_gard_shape = grad.get("shape")
        self.argmax_shape = argmax.get("shape")
        self.y_shape = input_x.get("shape")
        self.dtype = grad.get("dtype").lower()
        self.dtype_size = common_util.get_data_size(self.dtype)
        self.nc1 = 1
        self.block = self.input_gard_shape[0] * self.input_gard_shape[1]
        self.tik_instance = tik.Tik()
        self.ksize = ksize
        self.strides = strides
        self.padding = padding
        dyh, dyw = self.input_gard_shape[2:4]
        dxh, dxw = self.y_shape[2:4]
        strideh, stridew = self.strides[1:3]
        windowh, windoww = self.ksize[1:3]
        pad_top = 0
        pad_bottom = 0
        pad_right = 0
        pad_left = 0
        if padding == "SAME":
            padh = (dyh - 1) * strideh + windowh - dxh
            padw = (dyw - 1) * stridew + windoww - dxw
            padh = max([padh, 0])
            padw = max([padw, 0])
            pad_top = padh // 2
            pad_bottom = padh - pad_top
            pad_left = padw // 2
            pad_right = padw - pad_left
        self.pad = (pad_top, pad_bottom, pad_left, pad_right)

        self.hoverlap = 0
        if windowh > strideh:
            self.hoverlap = windowh - strideh
        self.woverlap = 0
        if windoww > stridew:
            self.woverlap = windoww - stridew

    def tik_instance_cut_nc1_cut_h_v2(self, kernel_name):
        """
        function for max_pool_grad_with_pool calc for normal shape
        """
        batch_num = self.input_gard_shape[0]
        c0_dim = constant.C0_SIZE
        c1_dim = self.input_gard_shape[1]
        block_num = batch_num * c1_dim
        input_h, input_w = self.input_gard_shape[2:4]
        output_h, output_w = self.y_shape[2:4]
        strideh, stridew = self.strides[1:3]
        windowh, windoww = self.ksize[1:3]
        input_h_once = 8  # tiling strategy
        output_h_once = (input_h_once - 1) * strideh + windowh  # 17
        output_w_once = (input_w - 1) * stridew + windoww  # 113
        filter_size = windowh * windoww
        input_block_size = input_h_once * input_w * c0_dim
        dtype = self.dtype

        # gm
        data_input = self.tik_instance.Tensor(dtype, (block_num, input_h, input_w, c0_dim),
                                              name="data_input", scope=tik.scope_gm)
        mask_one_window = ((input_h * input_w + 15) // 16 + 1) * 16
        data_mask = self.tik_instance.Tensor("uint16", (block_num, filter_size, mask_one_window),
                                             name="data_mask", scope=tik.scope_gm)
        data_output = self.tik_instance.Tensor(dtype, (block_num, output_h, output_w, c0_dim),
                                               name="data_output", scope=tik.scope_gm)
        data_input_origin = self.tik_instance.Tensor(dtype, (block_num, output_h, output_w, c0_dim),
                                                     name="data_input_origin", scope=tik.scope_gm)

        with self.tik_instance.for_range(0, block_num, block_num=block_num) as block_idx:
            input_ub_tensor0 = self.tik_instance.Tensor(dtype, (input_h_once, input_w, c0_dim),
                                                        name="input_ub0", scope=tik.scope_ubuf)
            input_ub_tensor1 = self.tik_instance.Tensor(dtype, (input_h_once, input_w, c0_dim),
                                                        name="input_ub1", scope=tik.scope_ubuf)
            mask_ub_tensor0 = self.tik_instance.Tensor("uint16", (filter_size, input_h_once, input_w),
                                                       name="mask_ub0", scope=tik.scope_ubuf)
            mask_ub_tensor1 = self.tik_instance.Tensor("uint16", (filter_size, input_h_once, input_w),
                                                       name="mask_ub1", scope=tik.scope_ubuf)
            select_ub_tensor = self.tik_instance.Tensor(dtype, (input_h_once, input_w, c0_dim),
                                                        name="select_ub0", scope=tik.scope_ubuf)
            output_ub_tensor0 = self.tik_instance.Tensor(dtype, (output_h_once, output_w_once, c0_dim),
                                                         name="output_ub0", scope=tik.scope_ubuf)
            output_ub_tensor1 = self.tik_instance.Tensor(dtype, (output_h_once, output_w_once, c0_dim),
                                                         name="output_ub1", scope=tik.scope_ubuf)

            if get_soc_spec(SHORT_SOC_VERSION) in (ASCEND_910, ASCEND_310):
                data_vsel_ub_zero = self.tik_instance.Tensor(dtype, (128,),
                                                             name="data_vsel_ub_zero",
                                                             scope=tik.scope_ubuf)
                self.tik_instance.vector_dup(constant.MASK128,
                                             data_vsel_ub_zero,
                                             0,
                                             1,
                                             1, 8)
            with self.tik_instance.for_range(0, (input_h // input_h_once + 1) // 2) as loop_idx:
                # ping
                # vector dup output
                with self.tik_instance.if_scope(loop_idx > 0):
                    self.tik_instance.vadds(constant.MASK128,
                                            output_ub_tensor0,
                                            output_ub_tensor1[output_h_once - 1, 0, 0],
                                            0,
                                            output_w * c0_dim // 128,
                                            1, 1, 8, 8)
                    self.tik_instance.vector_dup(constant.MASK128,
                                                 output_ub_tensor0[1, 0, 0],
                                                 0,
                                                 (output_h_once - 1) * output_w_once * c0_dim // 128,
                                                 1, 8)
                with self.tik_instance.else_scope():
                    self.tik_instance.vector_dup(constant.MASK128,
                                                 output_ub_tensor0,
                                                 0,
                                                 output_h_once * output_w_once * c0_dim // 128,
                                                 1, 8)
                # move input to ub
                self.tik_instance.data_move(input_ub_tensor0,
                                            data_input[block_idx, loop_idx * 2 * input_h_once, 0, 0],
                                            0,
                                            1,
                                            input_h_once * input_w,
                                            0, 0)
                # move mask to ub
                self.tik_instance.data_move(mask_ub_tensor0,
                                            data_mask[block_idx, 0, loop_idx * 2 * input_h_once * input_w],
                                            0,
                                            filter_size,
                                            input_h_once * input_w // 16,
                                            (mask_one_window - input_h_once * input_w) // 16, 0)

                with self.tik_instance.for_range(0, filter_size) as flt_idx:
                    # calc select ub
                    if get_soc_spec(SHORT_SOC_VERSION) in (ASCEND_910, ASCEND_310):
                        with self.tik_instance.for_range(0, input_block_size // 128) as cycle:
                            cmpmask = self.tik_instance.mov_tensor_to_cmpmask(
                                mask_ub_tensor0[flt_idx, cycle * 8 // input_w, cycle * 8 % input_w])
                            self.tik_instance.vsel(constant.MASK128, 0,
                                                   select_ub_tensor[cycle * 8 // input_w, cycle * 8 % input_w, 0],
                                                   cmpmask,
                                                   input_ub_tensor0[cycle * 8 // input_w, cycle * 8 % input_w, 0],
                                                   data_vsel_ub_zero,
                                                   1,
                                                   1, 1, 1, 8, 8, 8)
                    else:
                        self.tik_instance.vsel(constant.MASK128, 1,
                                               select_ub_tensor,
                                               mask_ub_tensor0[flt_idx, 0, 0],
                                               input_ub_tensor0,
                                               0,
                                               input_block_size // 128,
                                               1, 1, 0, 8, 8, 0)

                    # col2img
                    with self.tik_instance.for_range(0, input_h_once) as looph_idx:
                        self.tik_instance.vadd(constant.MASK128,
                                               output_ub_tensor0[looph_idx * 2 + flt_idx // windoww,
                                                                 flt_idx % windoww, 0],
                                               output_ub_tensor0[looph_idx * 2 + flt_idx // windoww,
                                                                 flt_idx % windoww, 0],
                                               select_ub_tensor[looph_idx, 0, 0],
                                               input_w * c0_dim // 128,
                                               stridew,
                                               stridew,
                                               1,
                                               8 * stridew,
                                               8 * stridew,
                                               8)

                # move output to gm
                self.tik_instance.data_move(data_output[block_idx, loop_idx * 2 * strideh * input_h_once, 0, 0],
                                            output_ub_tensor0,
                                            0,
                                            strideh * input_h_once,
                                            output_w,
                                            output_w_once - output_w,
                                            0)

                # pang
                with self.tik_instance.if_scope(loop_idx < (input_h // input_h_once) // 2):
                    # vector dup output
                    self.tik_instance.vadds(constant.MASK128,
                                            output_ub_tensor1,
                                            output_ub_tensor0[output_h_once - 1, 0, 0],
                                            0,
                                            output_w * c0_dim // 128,
                                            1, 1, 8, 8)
                    self.tik_instance.vector_dup(constant.MASK128,
                                                 output_ub_tensor1[1, 0, 0],
                                                 0,
                                                 (output_h_once - 1) * output_w_once * c0_dim // 128,
                                                 1, 8)
                    # move input to ub
                    self.tik_instance.data_move(input_ub_tensor1,
                                                data_input[block_idx, (loop_idx * 2 + 1) * input_h_once, 0, 0],
                                                0,
                                                1,
                                                input_h_once * input_w,
                                                0, 0)
                    # move mask to ub
                    self.tik_instance.data_move(mask_ub_tensor1,
                                                data_mask[block_idx, 0, (loop_idx * 2 + 1) * input_h_once * input_w],
                                                0,
                                                filter_size,
                                                input_h_once * input_w // 16,
                                                (mask_one_window - input_h_once * input_w) // 16, 0)

                    with self.tik_instance.for_range(0, filter_size) as flt_idx:
                        # calc select ub
                        if get_soc_spec(SHORT_SOC_VERSION) in (ASCEND_910, ASCEND_310):
                            with self.tik_instance.for_range(0, input_block_size // 128) as cycle:
                                cmpmask = self.tik_instance.mov_tensor_to_cmpmask(
                                    mask_ub_tensor1[flt_idx, cycle * 8 // input_w, cycle * 8 % input_w])
                                self.tik_instance.vsel(constant.MASK128, 0,
                                                       select_ub_tensor[cycle * 8 // input_w, cycle * 8 % input_w, 0],
                                                       cmpmask,
                                                       input_ub_tensor1[cycle * 8 // input_w, cycle * 8 % input_w, 0],
                                                       data_vsel_ub_zero,
                                                       1,
                                                       1, 1, 1, 8, 8, 0)
                        else:
                            self.tik_instance.vsel(constant.MASK128, 1,
                                                   select_ub_tensor,
                                                   mask_ub_tensor1[flt_idx, 0, 0],
                                                   input_ub_tensor1,
                                                   0,
                                                   input_block_size // 128,
                                                   1, 1, 0, 8, 8, 8)

                        # col2img
                        with self.tik_instance.for_range(0, input_h_once) as looph_idx:
                            self.tik_instance.vadd(constant.MASK128,
                                                   output_ub_tensor1[looph_idx * 2 + flt_idx // windoww,
                                                                     flt_idx % windoww, 0],
                                                   output_ub_tensor1[looph_idx * 2 + flt_idx // windoww,
                                                                     flt_idx % windoww, 0],
                                                   select_ub_tensor[looph_idx, 0, 0],
                                                   input_w * c0_dim // 128,
                                                   stridew,
                                                   stridew,
                                                   1,
                                                   8 * stridew,
                                                   8 * stridew,
                                                   8)

                    # move output to gm
                    self.tik_instance.data_move(data_output[block_idx, (loop_idx * 2 + 1) * strideh * input_h_once,
                                                            0, 0],
                                                output_ub_tensor1,
                                                0,
                                                strideh * input_h_once,
                                                output_w,
                                                output_w_once - output_w,
                                                0)

        self.tik_instance.BuildCCE(kernel_name=kernel_name,
                                   inputs=[data_input_origin, data_input,
                                           data_mask],
                                   outputs=[data_output], enable_l2=False)
        return self.tik_instance


# 'pylint: disable=invalid-name, too-many-arguments
def is_max_pool_grad_with_argmax_param(grad, argmax, x, ksize, strides,
                                       padding):
    """
    test if the param suitable for this module to treat
    :param grad: dict of shape and dtype of the input grad
    :param argmax: dict of shape and dtype of the input argmax
    :param x: dict of shape and dtype of the input x
    :param ksize: value of ksize
    :param strides: value of strides
    :param padding: value of padding
    :return: Bool, if the param suitable for this module to treat return True,
             if not return False
    """
    resnet50_grad = {"shape": (32, 4, 56, 56, 16), "dtype": "float16"}
    resnet50_argmax = {"shape": (32, 4, 9, 197, 16), "dtype": "uint16"}
    resnet50_x = {"shape": (32, 4, 112, 112, 16), "dtype": "float16"}
    resnet50_ksize = [1, 3, 3, 1]
    resnet50_strides = [1, 2, 2, 1]
    resnet50_padding = "SAME"

    def is_valid_shape(resnet50shape, shape):
        if shape.get("dtype") != resnet50shape.get("dtype"):
            return False

        if len(shape.get("shape")) != len(resnet50shape.get("shape")):
            return False

        resnet50_last3dims = resnet50shape.get("shape")[2:]
        last3dims = shape.get("shape")[2:]

        return list(resnet50_last3dims) == list(last3dims)

    ksize = list(ksize)
    strides = list(strides)

    if resnet50_padding != padding:
        return False

    if (resnet50_ksize == ksize and resnet50_strides == strides and
            is_valid_shape(resnet50_grad, grad) and
            is_valid_shape(resnet50_argmax, argmax) and
            is_valid_shape(resnet50_x, x)):
        return True

    return False


# 'pylint: disable=invalid-name, too-many-arguments
def max_pool_grad_with_argmax(grad, argmax, x, ksize, strides, padding,
                              kernel_name):
    """
    implementation of max_pool_with_argmax and return the tik instance
    :param grad: dict of shape and dtype of the input grad
    :param argmax: dict of shape and dtype of the input argmax
    :param x: dict of shape and dtype of the input x
    :param ksize: value of ksize
    :param strides: value of strides
    :param padding: value of padding
    :param kernel_name: kernel's name
    :return: tik instance
    """
    max_pool_grad = MaxpoolGradResnet50(grad, argmax, x, ksize, strides,
                                        padding)
    return max_pool_grad.tik_instance_cut_nc1_cut_h_v2(kernel_name)
