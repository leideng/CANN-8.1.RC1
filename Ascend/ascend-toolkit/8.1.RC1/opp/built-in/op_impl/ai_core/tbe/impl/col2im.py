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
# Col2im

from impl.util.platform_adapter import tik
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import para_check
from impl import constant_util as constant


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument,invalid-name
# 'pylint: disable=too-many-locals
@register_operator_compute("col2im", op_mode="static", support_fusion=True)
def col2im_compute(
        x_gm, output_size_gm, y_gm, kernel_size, dilation, padding, stride, tik_instance
    ):
    """
    do Col2im compute
    Parameters:
    ----------------
    x : input tensor x
    output_size : input tensor output_size
    y : output tensor  y
    kernel_size : value of kernel_size, data type int[2]
    dilation : value of dilation, data type int[2]
    padding : value of padding, data type int[2]
    stride : value of stride, data type int[2]
    kernel_name : cce kernel name, default value is "Col2im"
    ----------------
    """
    output_batch, output_c1, output_h, output_w, output_c0 = y_gm.shape
    _, _, _, _, input_c0 = x_gm.shape

    input_dtype = x_gm.dtype
    output_dtype = y_gm.dtype
    if input_dtype == "float32":
        dtype_byte_num = constant.DATA_SIZE_FOUR
    else:
        dtype_byte_num = constant.DATA_SIZE_TWO
    kernel_h, kernel_w = kernel_size
    kernel_num = kernel_h * kernel_w
    stride_h, stride_w = stride
    padding_h, padding_w = padding
    dilation_h, dilation_w = dilation
    ho = (output_h + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
    wo = (output_w + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1

    with tik_instance.for_range(
            0, output_batch * output_c1, block_num=output_batch * output_c1) as nc:
        n = nc // output_c1
        ci = nc % output_c1
        output_ub = tik_instance.Tensor(
            output_dtype, (constant.VECTOR_BYTE_SIZE//dtype_byte_num,), tik.scope_ubuf, "output_ub"
        )

        input_ub = tik_instance.Tensor(
            input_dtype, (constant.VECTOR_BYTE_SIZE//dtype_byte_num,), tik.scope_ubuf, "input_ub"
        )
        with tik_instance.for_range(0, kernel_num) as mask_id:
            width = mask_id % kernel_w
            height = mask_id // kernel_w
            with tik_instance.for_range(0, ho) as h:
                output_offset_h = height * dilation_h + h * stride_h - padding_h
                with tik_instance.for_range(0, wo) as w:
                    output_offset_w = width * dilation_w + w * stride_w - padding_w
                    with  tik_instance.if_scope(tik.all(
                        output_offset_h >= 0, output_offset_h < output_h,
                        output_offset_w >= 0, output_offset_w < output_w)):
                        tik_instance.data_move(
                            input_ub, x_gm[n, ci, mask_id, h*wo + w, 0],
                            constant.SID, constant.DEFAULT_NBURST,
                            (input_c0 * dtype_byte_num) // constant.BLOCK_SIZE,
                            constant.STRIDE_ZERO, constant.STRIDE_ZERO
                        )
                        tik_instance.data_move(
                            output_ub, y_gm[n, ci, output_offset_h, output_offset_w, 0],
                            constant.SID, constant.DEFAULT_NBURST,
                            (output_c0 * dtype_byte_num) // constant.BLOCK_SIZE,
                            constant.STRIDE_ZERO, constant.STRIDE_ZERO
                        )
                        tik_instance.vadd(
                            output_c0, output_ub, output_ub, input_ub,
                            constant.DEFAULT_REPEAT_TIME, constant.BLOCK_STRIDE_ONE, constant.BLOCK_STRIDE_ONE,
                            constant.BLOCK_STRIDE_ONE, constant.REPEAT_STRIDE_EIGHT,
                            constant.REPEAT_STRIDE_EIGHT, constant.REPEAT_STRIDE_EIGHT
                        )
                        tik_instance.data_move(
                            y_gm[n, ci, output_offset_h, output_offset_w, 0], output_ub,
                            constant.SID, constant.DEFAULT_NBURST,
                            (output_c0 * dtype_byte_num) // constant.BLOCK_SIZE,
                            constant.STRIDE_ZERO, constant.STRIDE_ZERO
                        )


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument,invalid-name
@para_check.check_op_params(
    para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
    para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_LIST_INT,
    para_check.REQUIRED_ATTR_LIST_INT, para_check.REQUIRED_ATTR_LIST_INT,
    para_check.REQUIRED_ATTR_LIST_INT, para_check.KERNEL_NAME
)
def col2im(
        x, output_size, y, kernel_size, dilation, padding, stride, kernel_name="Col2im"
    ):
    """
    do col2im operation on x, result is y, and y's height/width is value of output_size
    Parameters:
    ----------
    x : dict of x, include shape and dtype, dtype support float32
    output_size : dict of output_size, include shape and dtype, dtype support int32
    y : dict of y, include shape and dtype, dtype support float32
    kernel_size : value of kernel_size, data type int[2]
    dilation : value of dilation, data type int[2]
    padding : value of padding, data type int[2]
    stride : value of stride, data type int[2]
    kernel_name : cce kernel name, default value is "Col2im"
    -------
    """
    tik_instance = tik.Tik()
    y_gm = tik_instance.Tensor(y["dtype"], y["shape"], tik.scope_gm, "y_gm", is_atomic_add=True)
    x_gm = tik_instance.Tensor(x["dtype"], x["shape"], tik.scope_gm, "x_gm")
    output_size_gm = tik_instance.Tensor(output_size["dtype"], output_size["shape"], tik.scope_gm, "output_size_gm")
    col2im_compute(x_gm, output_size_gm, y_gm, kernel_size, dilation, padding, stride, tik_instance)
    tik_instance.BuildCCE(
        kernel_name=kernel_name,
        inputs=[x_gm, output_size_gm],
        outputs=[y_gm]
    )
    return tik_instance
