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
aipp_resize_padding
"""
# 'pylint: disable=invalid-name,too-many-locals,too-many-branches,too-many-statements,too-many-return-values
# 'pylint: disable=too-many-arguments,too-many-boolean-expressions,import-error,too-many-lines
import math

from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from tbe.dsl.instrinsic.cce_util import get_const

from impl import aipp_comm


def set_padding_ub(ib, dtype, padding_ub_buf, vadds_src_ub_buf, src_block_stride, num):
    """
    set padding_ub
    """
    if dtype == "float16":
        aipp_comm.vadds_zero(ib, dtype, padding_ub_buf, vadds_src_ub_buf, 0, num)
    else:
        with ib.new_scope():
            tmp_padding_ub = ib.allocate("float16", (num,), "tmp_padding_ub", scope=tbe_platform.scope_ubuf)
            tmp_padding_ub_buf = tvm.decl_buffer((num,), "float16", "tmp_padding_ub_buf",
                                                 scope=tbe_platform.scope_ubuf, data=tmp_padding_ub)
            aipp_comm.vadds_zero(ib, "float16", tmp_padding_ub_buf, vadds_src_ub_buf, src_block_stride, num)
            aipp_comm.conv(ib, dtype, "float16", padding_ub_buf, tmp_padding_ub_buf, num)


def padding_size_tiling(padding_size, w, buffer_upper_limit, dtype, output_format):
    """
    calculate tiling info of padding size
    """
    tiling_w, w_loop = aipp_comm.get_tiling_w(w, buffer_upper_limit, 1)
    tiling_h = buffer_upper_limit // tiling_w
    tail_w = w % tiling_w
    tail_h = padding_size % tiling_h

    if output_format == "NC1HWC0_C04" and tail_w > 0:
        if dtype == "float16":
            if tail_w < 4:
                if 4 // w_loop > 0 and 4 % w_loop == 0:
                    tiling_w = tiling_w - (4 // w_loop)
                elif 4 // w_loop > 0 and 4 % w_loop != 0:
                    tiling_w = tiling_w - (4 // w_loop) - 1
                else:
                    tiling_w = tiling_w - 1
                tiling_h = buffer_upper_limit // tiling_w
                tail_w = w - w_loop * tiling_w
                tail_h = padding_size % tiling_h
        else:
            if tail_w < 8:
                if 8 // w_loop > 0 and 8 % w_loop == 0:
                    tiling_w = tiling_w - (8 // w_loop)
                elif 8 // w_loop > 0 and 8 % w_loop != 0:
                    tiling_w = tiling_w - (8 // w_loop) - 1
                else:
                    tiling_w = tiling_w - 1
                tiling_h = buffer_upper_limit // tiling_w
                tail_w = w - w_loop * tiling_w
                tail_h = padding_size % tiling_h

    h_loop = padding_size // tiling_h

    return tiling_w, w_loop, tail_w, tiling_h, h_loop, tail_h


def move_padding_ub_to_gm(ib, padding_ub_buf, output_buf, params):
    """
    move padding_ub to gm
    """
    gm_offset, elems, dtype, dsize, output_format = params

    if output_format == "NC1HWC0_C04" and elems * dsize % 32 != 0 and elems * dsize > 32:
        tail_ub = ib.allocate(dtype, (32 // dsize,), "tail_ub", scope=tbe_platform.scope_ubuf)
        tail_ub_buf = tvm.decl_buffer((32 // dsize,), dtype, "tail_ub_buf", scope=tbe_platform.scope_ubuf,
                                      data=tail_ub)
        aipp_comm.copy_ubuf_to_gm_tail(ib, dtype, output_buf, padding_ub_buf, tail_ub_buf, elems, gm_offset, 0)

    ib.emit(tvm.call_extern(dtype, 'copy_ubuf_to_gm',
                            output_buf.access_ptr("w", ptr_type=dtype, offset=gm_offset),
                            padding_ub_buf.access_ptr("rw", ptr_type=dtype, offset=0),
                            0, 1, elems*dsize // 32, 0, 0))


def process_padding_value_impl(ib, output_buf, vadds_src_ub_buf, output_format, params):
    """
    process_padding_value implement
    params: tuple, (dtype, dsize, elems, src_block_stride, gm_offset)
    """
    dtype, dsize, elems, src_block_stride, gm_offset = params
    num = (((elems * dsize + 255) // 256) * 256) // dsize
    padding_ub = ib.allocate(dtype, (num,), "padding_ub", scope=tbe_platform.scope_ubuf)
    padding_ub_buf = tvm.decl_buffer((num,), dtype, "padding_ub_buf", scope=tbe_platform.scope_ubuf,
                                     data=padding_ub)
    set_padding_ub(ib, dtype, padding_ub_buf, vadds_src_ub_buf, src_block_stride, num)

    move_padding_ub_to_gm(ib, padding_ub_buf, output_buf, (gm_offset, elems, dtype, dsize, output_format))


def process_padding_value(ib, input_data, output_buf, input_format, output_format="NC1HWC0"):
    """
    process top or bottom padding value
    """
    dtype, w, c0, dsize, padding_size, offset, padding_value = input_data
    ub_size = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) - 2048

    fp16 = "float16"
    dsize_fp16 = 2
    src_vadds_ub_size = 32
    src_block_stride = 0
    if dtype == fp16:
        buffer_upper_limit = (ub_size // c0) // dsize
    else:
        if output_format == "NC1HWC0":
            src_vadds_ub_size = 256
            src_block_stride = 1
        buffer_upper_limit = (ub_size // 3 // c0) // dsize

    step = 4
    loop = 1
    channel_valid = 3
    if input_format in ("YUV400_U8",):
        channel_valid = 1
    if input_format in ("YUV420SP_U8", "YUYV_U8", "YUV422SP_U8", "RGB888_U8", "XRGB8888_U8", "RGB16"):
        channel_valid = 3
    if input_format in ("AYUV444_U8", "ARGB8888_U8"):
        channel_valid = 4
    if output_format == "NC1HWC0_C04":
        loop = 32 // step
    if src_vadds_ub_size == 256:
        step = 32
        loop = 128 // step

    with ib.new_scope():
        # set src_ub of vadds
        vadds_src_ub = ib.allocate(fp16, (src_vadds_ub_size // dsize_fp16,), "vadds_src_ub",
                                   scope=tbe_platform.scope_ubuf)
        vadds_src_ub_buf = tvm.decl_buffer((src_vadds_ub_size // dsize_fp16,), fp16, "vadds_src_ub_buf",
                                           scope=tbe_platform.scope_ubuf, data=vadds_src_ub)
        if src_vadds_ub_size == 32:
            for i in range(src_vadds_ub_size // dsize_fp16):
                ib.emit(tvm.call_extern(fp16, "reg_mov", vadds_src_ub_buf.access_ptr("w", offset=i),
                                        tvm.call_extern(fp16, "reg", 0)))
        else:
            ib.emit(tvm.call_extern(fp16, "vector_dup", vadds_src_ub_buf.access_ptr("w", offset=0),
                                    tvm.const(0, dtype=fp16), 1, 1, 1, 8, 8))
        for i in range(loop):
            for j in range(channel_valid):
                ib.emit(tvm.call_extern(fp16, "reg_mov", vadds_src_ub_buf.access_ptr("w", offset=step*i + j),
                                        tvm.call_extern(fp16, "reg", padding_value)))

        if buffer_upper_limit >= padding_size * w:
            with ib.new_scope():
                elems = padding_size * w * c0
                process_padding_value_impl(ib, output_buf, vadds_src_ub_buf, output_format,
                                           (dtype, dsize, elems, src_block_stride, offset))
        else:
            tiling_w, w_loop, tail_w, tiling_h, h_loop, tail_h = \
                padding_size_tiling(padding_size, w, buffer_upper_limit, dtype, output_format)
            zero_const = tvm.const(0, dtype="uint64")
            h_loop_const = tvm.const(h_loop, dtype="uint64")
            w_loop_const = tvm.const(w_loop, dtype="uint64")

            with ib.for_range(zero_const, h_loop_const, name="h1", dtype="uint64") as h1:
                with ib.for_range(zero_const, w_loop_const, name="w1", dtype="uint64") as w1:
                    with ib.new_scope():
                        elems = tiling_h * tiling_w * c0
                        gm_offset = offset + h1*tiling_h*w*c0 + w1*tiling_w*tiling_h*c0
                        process_padding_value_impl(ib, output_buf, vadds_src_ub_buf, output_format,
                                                   (dtype, dsize, elems, src_block_stride, gm_offset))

                    if tail_w != 0:
                        with ib.new_scope():
                            elems = tiling_h * tail_w * c0
                            gm_offset = offset + h1*tiling_h*w*c0 + w_loop*tiling_w*tiling_h*c0
                            process_padding_value_impl(ib, output_buf, vadds_src_ub_buf, output_format,
                                                       (dtype, dsize, elems, src_block_stride, gm_offset))

            if tail_h != 0:
                with ib.for_range(zero_const, w_loop_const, name="w1", dtype="uint64") as w1:
                    with ib.new_scope():
                        elems = tail_h * tiling_w * c0
                        gm_offset = offset + h_loop*tiling_h*w*c0 + w1*tiling_w*tail_h*c0
                        process_padding_value_impl(ib, output_buf, vadds_src_ub_buf, output_format,
                                                   (dtype, dsize, elems, src_block_stride, gm_offset))
                if tail_w != 0:
                    with ib.new_scope():
                        elems = tail_h * tail_w * c0
                        gm_offset = offset + h_loop*tiling_h*w*c0 + w_loop*tiling_w*tail_h*c0
                        process_padding_value_impl(ib, output_buf, vadds_src_ub_buf, output_format,
                                                   (dtype, dsize, elems, src_block_stride, gm_offset))


def process_padding(ib, input_data, output_buf, output_format="NC1HWC0"):
    """
    :param ib:
    :param input_data:
    :param output_buf:
    :return:
    """

    dtype = input_data[0]
    w = input_data[1]
    c0 = input_data[2]
    size = input_data[3]
    padding_size = input_data[4]
    offset = input_data[5]
    padding_value = input_data[6]

    ub_size = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
    if output_format == "NC1HWC0_C04":
        c0 = 4
        ub_size = ub_size - 64
    buffer_upper_limit = ub_size//2//2//c0

    if buffer_upper_limit >= padding_size*w:
        with ib.new_scope():
            num = (((padding_size*w*c0*size + 31) // 32)*32) // size
            padding_ub = ib.allocate(dtype, (num,), "padding_ub",
                                     scope=tbe_platform.scope_ubuf)
            padding_ub_buf = tvm.decl_buffer((num,), dtype,
                                             "padding_ub", scope=tbe_platform.scope_ubuf,
                                             data=padding_ub)
            if dtype == "float16":
                aipp_comm.vector_dup(ib, "float16", padding_ub_buf, num, padding_value)
            else:
                with ib.new_scope():
                    tmp_padding_ub = ib.allocate("float16",
                                                 (num,),
                                                 "tmp_padding_ub",
                                                 scope=tbe_platform.scope_ubuf)
                    tmp_padding_ub_buf = tvm.decl_buffer((num,),
                                                         "float16",
                                                         "tmp_padding_ub_buf",
                                                         scope=tbe_platform.scope_ubuf,
                                                         data=tmp_padding_ub)

                    aipp_comm.vector_dup(ib, "float16", tmp_padding_ub_buf, num, padding_value)
                    aipp_comm.conv(ib, dtype, "float16", padding_ub_buf,
                                   tmp_padding_ub_buf, num)

            if output_format == "NC1HWC0_C04" and \
                    padding_size*w*c0*size % 32 != 0 and \
                    padding_size*w*c0*size > 32:

                tail_ub = ib.allocate(dtype, (32 // size,), "tail_ub", scope=tbe_platform.scope_ubuf)
                tail_ub_buf = tvm.decl_buffer((32 // size,), dtype, "tail_ub_buf",
                                              scope=tbe_platform.scope_ubuf, data=tail_ub)
                aipp_comm.copy_ubuf_to_gm_tail(
                    ib, dtype, output_buf, padding_ub_buf,
                    tail_ub_buf,
                    padding_size*w*c0, offset,
                    0)

            ib.emit(tvm.call_extern(
                dtype, 'copy_ubuf_to_gm',
                output_buf.access_ptr("w", ptr_type=dtype, offset=offset),
                padding_ub_buf.access_ptr("rw", ptr_type=dtype, offset=0),
                0, 1, padding_size*w*c0*size//32, 0, 0))
    else:
        tiling_w, w_loop = aipp_comm.get_tiling_w(w, buffer_upper_limit, 1)
        tiling_h = buffer_upper_limit // tiling_w
        tail_w = w % tiling_w
        tail_h = padding_size % tiling_h
        if output_format == "NC1HWC0_C04" and tail_w > 0:
            if dtype == "float16":
                if tail_w < 4:
                    if 4 // w_loop > 0 and 4 % w_loop == 0:
                        tiling_w = tiling_w - (4 // w_loop)
                    elif 4 // w_loop > 0 and 4 % w_loop != 0:
                        tiling_w = tiling_w - (4 // w_loop) - 1
                    else:
                        tiling_w = tiling_w - 1
                    tiling_h = buffer_upper_limit // tiling_w
                    tail_w = w - w_loop * tiling_w
                    tail_h = padding_size % tiling_h
            else:
                if tail_w < 8:
                    if 8 // w_loop > 0 and 8 % w_loop == 0:
                        tiling_w = tiling_w - (8 // w_loop)
                    elif 8 // w_loop > 0 and 8 % w_loop != 0:
                        tiling_w = tiling_w - (8 // w_loop) - 1
                    else:
                        tiling_w = tiling_w - 1
                    tiling_h = buffer_upper_limit // tiling_w
                    tail_w = w - w_loop * tiling_w
                    tail_h = padding_size % tiling_h

        h_loop = padding_size // tiling_h
        zero_const = tvm.const(0, dtype="uint64")
        h_loop_const = tvm.const(h_loop, dtype="uint64")
        w_loop_const = tvm.const(w_loop, dtype="uint64")

        with ib.for_range(zero_const, h_loop_const, name="h1",
                          dtype="uint64") as h1:
            with ib.for_range(zero_const, w_loop_const,
                              name="w1", dtype="uint64") as w1:
                with ib.new_scope():
                    num = (((tiling_h*tiling_w*c0*size+31)//32)*32) // size
                    padding_ub = ib.allocate(dtype, (num,), "padding_ub",
                                             scope=tbe_platform.scope_ubuf)
                    padding_ub_buf = tvm.decl_buffer((num,), dtype,
                                                     "padding_ub",
                                                     scope=tbe_platform.scope_ubuf,
                                                     data=padding_ub)

                    if dtype == "float16":
                        aipp_comm.vector_dup(ib, "float16", padding_ub_buf, num, padding_value)
                    else:
                        with ib.new_scope():
                            tmp_padding_ub = ib.allocate("float16",
                                                         (num,),
                                                         "tmp_padding_ub",
                                                         scope=tbe_platform.scope_ubuf)
                            tmp_padding_ub_buf = tvm.decl_buffer((num,),
                                                                 "float16",
                                                                 "tmp_padding_ub_buf",
                                                                 scope=tbe_platform.scope_ubuf,
                                                                 data=tmp_padding_ub)

                            aipp_comm.vector_dup(ib, "float16", tmp_padding_ub_buf, num, padding_value)
                            aipp_comm.conv(ib, dtype, "float16", padding_ub_buf,
                                           tmp_padding_ub_buf, num)

                    if output_format == "NC1HWC0_C04" and \
                            tiling_h*tiling_w*c0*size % 32 != 0 and \
                            tiling_h*tiling_w*c0*size > 32:
                        tail_ub = ib.allocate(dtype, (32 // size,), "tail_ub", scope=tbe_platform.scope_ubuf)
                        tail_ub_buf = tvm.decl_buffer((32 // size,), dtype, "tail_ub_buf",
                                                      scope=tbe_platform.scope_ubuf, data=tail_ub)
                        aipp_comm.copy_ubuf_to_gm_tail(
                            ib, dtype, output_buf, padding_ub_buf,
                            tail_ub_buf,
                            tiling_h*tiling_w*c0,
                            offset + h1*tiling_h*w*c0 + w1*tiling_w*tiling_h*c0,
                            0)

                    ib.emit(tvm.call_extern(
                        dtype, 'copy_ubuf_to_gm',
                        output_buf.access_ptr("w", ptr_type=dtype,
                                              offset=offset + h1*tiling_h*w*c0 + \
                                                     w1*tiling_w*tiling_h*c0),
                        padding_ub_buf.access_ptr("rw", ptr_type=dtype, offset=0),
                        0, 1, tiling_h*tiling_w*c0*size//32, 0, 0))

                if tail_w != 0:
                    with ib.new_scope():
                        num = (((tiling_h*tail_w*c0*size+31)//32)*32) // size
                        padding_ub = ib.allocate(dtype, (num,), "padding_ub",
                                                 scope=tbe_platform.scope_ubuf)
                        padding_ub_buf = tvm.decl_buffer((num,), dtype,
                                                         "padding_ub",
                                                         scope=tbe_platform.scope_ubuf,
                                                         data=padding_ub)

                        if dtype == "float16":
                            aipp_comm.vector_dup(ib, "float16", padding_ub_buf, num, padding_value)
                        else:
                            with ib.new_scope():
                                tmp_padding_ub = ib.allocate("float16",
                                                             (num,),
                                                             "tmp_padding_ub",
                                                             scope=tbe_platform.scope_ubuf)
                                tmp_padding_ub_buf = tvm.decl_buffer((num,),
                                                                     "float16",
                                                                     "tmp_padding_ub_buf",
                                                                     scope=tbe_platform.scope_ubuf,
                                                                     data=tmp_padding_ub)

                                aipp_comm.vector_dup(ib, "float16", tmp_padding_ub_buf, num, padding_value)
                                aipp_comm.conv(ib, dtype, "float16", padding_ub_buf,
                                               tmp_padding_ub_buf, num)

                        if output_format == "NC1HWC0_C04" and \
                                tiling_h*tail_w*c0*size % 32 != 0 and \
                                tiling_h*tail_w*c0*size > 32:

                            tail_ub = ib.allocate(dtype, (32 // size,), "tail_ub", scope=tbe_platform.scope_ubuf)
                            tail_ub_buf = tvm.decl_buffer((32 // size,), dtype, "tail_ub_buf",
                                                          scope=tbe_platform.scope_ubuf, data=tail_ub)
                            aipp_comm.copy_ubuf_to_gm_tail(
                                ib, dtype, output_buf, padding_ub_buf,
                                tail_ub_buf,
                                tiling_h*tail_w*c0,
                                offset + h1*tiling_h*tiling_w*c0 + w_loop*tiling_w*tiling_h*c0,
                                0)

                        ib.emit(tvm.call_extern(
                            dtype, 'copy_ubuf_to_gm',
                            output_buf.access_ptr("w", ptr_type=dtype,
                                                  offset=offset +
                                                  h1*tiling_h*tiling_w*c0 +
                                                  w_loop*tiling_w*tiling_h*c0),
                            padding_ub_buf.access_ptr("rw", ptr_type=dtype, offset=0),
                            0, 1, tiling_h*tail_w*c0*size//32, 0, 0))

        if tail_h != 0:
            with ib.for_range(zero_const,
                              w_loop_const,
                              name="w1", dtype="uint64") as w1:
                with ib.new_scope():
                    num = (((tail_h*tiling_w*c0*size+31)//32)*32) // size
                    padding_ub = ib.allocate(dtype, (num,),
                                             "padding_ub", scope=tbe_platform.scope_ubuf)
                    padding_ub_buf = tvm.decl_buffer((num,), dtype,
                                                     "padding_ub",
                                                     scope=tbe_platform.scope_ubuf,
                                                     data=padding_ub)

                    if dtype == "float16":
                        aipp_comm.vector_dup(ib, "float16", padding_ub_buf, num, padding_value)
                    else:
                        with ib.new_scope():
                            tmp_padding_ub = ib.allocate("float16", (num,),
                                                         "tmp_padding_ub",
                                                         scope=tbe_platform.scope_ubuf)
                            tmp_padding_ub_buf = tvm.decl_buffer((num,),
                                                                 "float16",
                                                                 "tmp_padding_ub_buf",
                                                                 scope=tbe_platform.scope_ubuf,
                                                                 data=tmp_padding_ub)

                            aipp_comm.vector_dup(ib, "float16", tmp_padding_ub_buf, num, padding_value)
                            aipp_comm.conv(ib, dtype, "float16", padding_ub_buf,
                                           tmp_padding_ub_buf, num)

                    if output_format == "NC1HWC0_C04" and \
                            tail_h*tiling_w*c0*size % 32 != 0 and \
                            tail_h*tiling_w*c0*size > 32:

                        tail_ub = ib.allocate(dtype, (32 // size,), "tail_ub", scope=tbe_platform.scope_ubuf)
                        tail_ub_buf = tvm.decl_buffer((32 // size,), dtype, "tail_ub_buf",
                                                      scope=tbe_platform.scope_ubuf, data=tail_ub)
                        aipp_comm.copy_ubuf_to_gm_tail(
                            ib, dtype, output_buf, padding_ub_buf,
                            tail_ub_buf,
                            tail_h*tiling_w*c0,
                            offset + h_loop*tiling_h*w*c0 + w1*tiling_w*tail_h*c0,
                            0)

                    ib.emit(tvm.call_extern(
                        dtype, 'copy_ubuf_to_gm',
                        output_buf.access_ptr("w", ptr_type=dtype,
                                              offset=offset + h_loop*tiling_h*w*c0 + \
                                                     w1*tiling_w*tail_h*c0),
                        padding_ub_buf.access_ptr("rw", ptr_type=dtype, offset=0),
                        0, 1, tail_h*tiling_w*c0*size//32, 0, 0))
            if tail_w != 0:
                with ib.new_scope():
                    num = (((tail_h*tail_w*c0*size+31)//32)*32) // size
                    padding_ub = ib.allocate(dtype, (num,), "padding_ub",
                                             scope=tbe_platform.scope_ubuf)
                    padding_ub_buf = tvm.decl_buffer((num,), dtype,
                                                     "padding_ub",
                                                     scope=tbe_platform.scope_ubuf,
                                                     data=padding_ub)

                    if dtype == "float16":
                        aipp_comm.vector_dup(ib, "float16", padding_ub_buf, num, padding_value)
                    else:
                        with ib.new_scope():
                            tmp_padding_ub = ib.allocate("float16",
                                                         (((tail_h*tail_w*c0+31)//32)*32,),
                                                         "tmp_padding_ub",
                                                         scope=tbe_platform.scope_ubuf)
                            tmp_padding_ub_buf = tvm.decl_buffer((((tail_h*tail_w*c0+31)//32)*32,),
                                                                 "float16",
                                                                 "tmp_padding_ub_buf",
                                                                 scope=tbe_platform.scope_ubuf,
                                                                 data=tmp_padding_ub)

                            aipp_comm.vector_dup(ib, "float16", tmp_padding_ub_buf, num, padding_value)
                            aipp_comm.conv(ib, dtype, "float16", padding_ub_buf,
                                           tmp_padding_ub_buf, tail_h*tail_w*c0)

                    if output_format == "NC1HWC0_C04" and \
                            tail_h*tail_w*c0*size % 32 != 0 and \
                            tail_h*tail_w*c0*size > 32:
                        with ib.new_scope():
                            tail_ub = ib.allocate(dtype, (32 // size,), "tail_ub", scope=tbe_platform.scope_ubuf)
                            tail_ub_buf = tvm.decl_buffer((32 // size,), dtype, "tail_ub_buf",
                                                          scope=tbe_platform.scope_ubuf, data=tail_ub)
                            aipp_comm.copy_ubuf_to_gm_tail(
                                ib, dtype, output_buf, padding_ub_buf,
                                tail_ub_buf,
                                tail_h*tail_w*c0,
                                offset + h_loop*tiling_h*w*c0 + w_loop*tiling_w*tail_h*c0,
                                0)

                    ib.emit(tvm.call_extern(
                        dtype, 'copy_ubuf_to_gm',
                        output_buf.access_ptr("w", ptr_type=dtype,
                                              offset=offset + h_loop*tiling_h*w*c0 + \
                                                     w_loop*tiling_w*tail_h*c0),
                        padding_ub_buf.access_ptr("rw", ptr_type=dtype, offset=0),
                        0, 1, tail_h*tail_w*c0*size//32, 0, 0))


def move_data_from_l1_to_gm(ib, totol_num, dtype, output_cb_buf, output_buf, gm_output_offset,
                            output_format="NC1HWC0"):
    """
    :param ib:
    :param totol_num:
    :param dtype:
    :param output_cb_buf:
    :param output_buf:
    :param gm_output_offset:
    :return:
    """

    ub_size = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)

    if output_format == "NC1HWC0_C04":
        ub_size = ub_size - 64

    if dtype == "float16":
        size = 2
    else:
        size = 1

    if totol_num*size < (ub_size//2):
        with ib.new_scope():
            num = (((totol_num*size+31)//32)*32) // size
            output_ub = ib.allocate(dtype, (num,),
                                    "output_ub",
                                    scope=tbe_platform.scope_ubuf)
            output_ub_buf = tvm.decl_buffer(
                (num,), dtype, "output_ub_buf",
                scope=tbe_platform.scope_ubuf, data=output_ub)

            len_burst, n_burst = \
                aipp_comm.get_lenburst_and_nburst(
                    (totol_num*size + 31)//32, 1)

            ib.emit(tvm.call_extern(
                dtype, 'copy_cbuf_to_ubuf',
                output_ub_buf.access_ptr("w",
                                         ptr_type=dtype,
                                         offset=0),
                output_cb_buf.access_ptr("rw",
                                         ptr_type=dtype,
                                         offset=0),
                0, n_burst, len_burst, 0, 0))

            if output_format == "NC1HWC0_C04" and \
                    totol_num*size % 32 != 0 and \
                    totol_num*size > 32:
                len_burst, n_burst = \
                    aipp_comm.get_lenburst_and_nburst(
                        (totol_num*size)//32, 1)

                tail_ub = ib.allocate(dtype, (32 // size,), "tail_ub", scope=tbe_platform.scope_ubuf)
                tail_ub_buf = tvm.decl_buffer((32 // size,), dtype, "tail_ub_buf",
                                              scope=tbe_platform.scope_ubuf, data=tail_ub)
                aipp_comm.copy_ubuf_to_gm_tail(
                    ib, dtype, output_buf, output_ub_buf,
                    tail_ub_buf,
                    totol_num, gm_output_offset,
                    0)

            ib.emit(tvm.call_extern(
                dtype, 'copy_ubuf_to_gm',
                output_buf.access_ptr(
                    "w", ptr_type=dtype,
                    offset=gm_output_offset),
                output_ub_buf.access_ptr("rw", ptr_type=dtype,
                                         offset=0),
                0, n_burst, len_burst, 0, 0))
    else:
        cycle_num = (ub_size//2)//size
        move_num = totol_num // cycle_num
        move_tail = totol_num - move_num * cycle_num
        if output_format == "NC1HWC0_C04" and move_tail*size < 32:
            cycle_num = (cycle_num*size - 32) // size
            move_num = totol_num // cycle_num
            move_tail = totol_num - move_num*cycle_num

        with ib.for_range(tvm.const(0, dtype="uint64"),
                          tvm.const(move_num,
                                    dtype="uint64"),
                          name="move_index",
                          dtype="uint64") as move_index:
            with ib.new_scope():
                output_ub = ib.allocate(dtype, (ub_size//2,),
                                        "output_ub",
                                        scope=tbe_platform.scope_ubuf)
                output_ub_buf = tvm.decl_buffer(
                    (ub_size // 2,), dtype, "output_ub_buf",
                    scope=tbe_platform.scope_ubuf, data=output_ub)

                len_burst, n_burst = \
                    aipp_comm.get_lenburst_and_nburst(
                        (cycle_num*size) // 32, 1)

                ib.emit(tvm.call_extern(
                    dtype, 'copy_cbuf_to_ubuf',
                    output_ub_buf.access_ptr("w", ptr_type=dtype,
                                             offset=0),
                    output_cb_buf.access_ptr("rw", ptr_type=dtype,
                                             offset=move_index*cycle_num),
                    0, n_burst, len_burst, 0, 0))

                ib.emit(tvm.call_extern(
                    dtype, 'copy_ubuf_to_gm',
                    output_buf.access_ptr("w", ptr_type=dtype,
                                          offset=gm_output_offset +
                                          move_index*cycle_num),
                    output_ub_buf.access_ptr("rw", ptr_type=dtype,
                                             offset=0),
                    0, n_burst, len_burst, 0, 0))
        if move_tail != 0:
            with ib.new_scope():
                output_ub = ib.allocate(dtype, (ub_size // 2,),
                                        "output_ub",
                                        scope=tbe_platform.scope_ubuf)
                output_ub_buf = tvm.decl_buffer(
                    (ub_size // 2,), dtype, "output_ub_buf",
                    scope=tbe_platform.scope_ubuf, data=output_ub)

                len_burst, n_burst = \
                    aipp_comm.get_lenburst_and_nburst(
                        (move_tail*size + 31)//32, 1)

                ib.emit(tvm.call_extern(
                    dtype, 'copy_cbuf_to_ubuf',
                    output_ub_buf.access_ptr("w", ptr_type=dtype,
                                             offset=0),
                    output_cb_buf.access_ptr("rw", ptr_type=dtype,
                                             offset=move_num*cycle_num),
                    0, n_burst, len_burst, 0, 0))

                if output_format == "NC1HWC0_C04" and \
                        (move_tail*size) % 32 != 0 and \
                        move_tail*size > 32:
                    len_burst, n_burst = \
                        aipp_comm.get_lenburst_and_nburst(
                            (move_tail*size)//32, 1)

                    tail_ub = ib.allocate(dtype, (32 // size,), "tail_ub", scope=tbe_platform.scope_ubuf)
                    tail_ub_buf = tvm.decl_buffer((32 // size,), dtype, "tail_ub_buf",
                                                  scope=tbe_platform.scope_ubuf, data=tail_ub)
                    aipp_comm.copy_ubuf_to_gm_tail(
                        ib, dtype, output_buf, output_ub_buf,
                        tail_ub_buf,
                        move_tail,
                        gm_output_offset + move_num*cycle_num,
                        0)

                ib.emit(tvm.call_extern(
                    dtype, 'copy_ubuf_to_gm',
                    output_buf.access_ptr("w", ptr_type=dtype,
                                          offset=gm_output_offset +
                                          move_num*cycle_num),
                    output_ub_buf.access_ptr("rw", ptr_type=dtype,
                                             offset=0),
                    0, n_burst, len_burst, 0, 0))


def aipp_compute(input_tensor, input_shape, input_format,
                 output_data, aipp_config):
    """
    :param input_tensor:
    :param param_tensor:
    :param input_shape:
    :param input_format:
    :param output_data:
    :param aipp_config:
    :return:
    """

    if input_format == "NHWC":
        n, h, w, c = input_shape
    else:
        n, c, h, w = input_shape

    output_shape = output_data.get('shape')
    n, c1, h, w, c0 = output_shape

    src_image_size_h = aipp_config.get("src_image_size_h")
    src_image_size_w = aipp_config.get("src_image_size_w")
    load_image_h = src_image_size_h
    load_image_w = src_image_size_w
    load_start_pos_h = 0
    load_start_pos_w = 0

    output_format = output_data.get('format')

    if "crop" in aipp_config and aipp_config.get("crop") == 1:
        src_image_size_h, src_image_size_w, load_start_pos_h, load_start_pos_w, \
        load_image_h, load_image_w = aipp_comm.get_crop_info(aipp_config,
                                                             src_image_size_h,
                                                             src_image_size_w)

    dtype = output_data.get('dtype')
    if dtype == "float16":
        size = 2
        c0 = 16
    else:
        size = 1
        c0 = 32

    if output_format == "NC1HWC0_C04":
        c0 = 4

    c1 = (c + c0 - 1) // c0

    actual_col_size = h * w
    if "crop" in aipp_config and aipp_config.get("crop") == 1 or \
       "resize" in aipp_config and aipp_config.get("resize") == 1 or \
       "padding" in aipp_config and aipp_config.get("padding") == 1:
        actual_col_size = aipp_comm.get_actual_col_size(aipp_config, h, w)

    l1_image_buf_max = \
        aipp_comm.get_l1_image_buf_max(actual_col_size, dtype, False, output_format)

    def aipp_ir(input_buf, output_buf):
        ib = tvm.tir.ir_builder.create()

        cur_cce_product = tbe_platform.get_soc_spec("SHORT_SOC_VERSION")

        if cur_cce_product not in aipp_comm.Const.STC_AIPP_SUPPORT_SOC_VERSION_SET:
            cause_dec = "Only support " + ", ".join(aipp_comm.Const.STC_AIPP_SUPPORT_SOC_VERSION_SET) + \
                        ", cur_cce_product is %s" % cur_cce_product
            aipp_comm.raise_runtime_error(cause_dec)

        device_core_num = \
            tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        batch_num = n
        batch_factor = 1

        if batch_num % device_core_num == 0:
            batch_factor = batch_num // device_core_num

        block_index = tvm.thread_axis("blockIdx.x")
        ib.scope_attr(block_index, "thread_extent", batch_num // batch_factor)

        if aipp_config.get('input_format') == "YUV420SP_U8":
            input_offset = batch_factor*((c*src_image_size_h*src_image_size_w) // 2)
        elif aipp_config.get('input_format') in \
                ["XRGB8888_U8", "RGB888_U8", "ARGB8888_U8",
                 "AYUV444_U8", "YUV400_U8", "RGB16"]:
            input_offset = batch_factor*((c*src_image_size_h*src_image_size_w))
        elif aipp_config.get('input_format') in ["YUYV_U8", "YUV422SP_U8"]:
            input_offset = batch_factor*((2*src_image_size_h*src_image_size_w))

        offset = batch_factor*c1*h*w*c0

        def _aipp_intrin():
            # config SPR2~SPR9
            aipp_comm.set_spr2_spr9(ib, aipp_config, dtype, cur_cce_product, output_format)

            aipp_xt = (src_image_size_w - 1)

            left_padding_size, right_padding_size, \
            top_padding_size, bottom_padding_size =\
                aipp_comm.get_padding_size(aipp_config)

            with ib.for_range(tvm.const(0, dtype="uint64"),
                              tvm.const(batch_factor, dtype="uint64"),
                              name="n1", dtype="uint64") as n1:
                if aipp_config.get('input_format') == "YUV420SP_U8":
                    spr0 = get_const(
                        input_buf.access_ptr(
                            'r',
                            offset=block_index*input_offset +
                            n1*((c*src_image_size_h*src_image_size_w) // 2)))
                elif aipp_config.get('input_format') in \
                        ["XRGB8888_U8", "RGB888_U8",
                         "ARGB8888_U8", "AYUV444_U8", "YUV400_U8", "RGB16"]:
                    spr0 = get_const(
                        input_buf.access_ptr(
                            'r',
                            offset=block_index*input_offset +
                            n1*((c*src_image_size_h*src_image_size_w))))
                elif aipp_config.get('input_format') in ["YUYV_U8", "YUV422SP_U8"]:
                    spr0 = get_const(
                        input_buf.access_ptr(
                            'r',
                            offset=block_index*input_offset +
                            n1*((2*src_image_size_h*src_image_size_w))))

                spr1 = tvm.const(0, dtype="uint64")
                if ('csc_switch' in aipp_config) and \
                        (aipp_config.get('csc_switch') == 1):
                    spr1 = tvm.const(1 << 63, dtype="uint64")
                else:
                    if cur_cce_product in ["Hi3796CV300ES", "Hi3796CV300CS", "SD3403"]:
                        if aipp_config.get('input_format') in \
                                ["YUV420SP_U8", "YUYV_U8",
                                 "YUV422SP_U8", "AYUV444_U8"]:
                            spr1 = tvm.const(1 << 63, dtype="uint64")

                if aipp_config.get('input_format') == "YUV420SP_U8":
                    spr1 = get_const(
                        input_buf.access_ptr(
                            'r',
                            offset=block_index*input_offset +
                            n1*((c*src_image_size_h*src_image_size_w) // 2) +
                            src_image_size_h*src_image_size_w)) | \
                           spr1
                elif aipp_config.get('input_format') == "YUV422SP_U8":
                    spr1 = get_const(
                        input_buf.access_ptr(
                            'r',
                            offset=block_index*input_offset +
                            n1*((2*src_image_size_h*src_image_size_w)) +
                            src_image_size_h*src_image_size_w)) | \
                           spr1

                ib.emit(tvm.call_extern(dtype, "set_aipp_spr_0", spr0))
                ib.emit(tvm.call_extern(dtype, "set_aipp_spr_1", spr1))

                if "padding" in aipp_config and \
                        aipp_config.get("padding") == 1:
                    padding_value = aipp_config.get("padding_value", 0)

                    if "top_padding_size" in aipp_config and \
                            aipp_config.get("top_padding_size") > 0:
                        top_offset = block_index*offset + n1*c1*h*w*c0
                        if padding_value != 0 and not (c0 == 4 and aipp_config.get("input_format") in \
                                                       ("AYUV444_U8", "ARGB8888_U8")):
                            with ib.new_scope():
                                process_padding_value(ib, (dtype, w, c0, size, top_padding_size,
                                                           top_offset, padding_value),
                                                      output_buf, aipp_config.get("input_format"), output_format)
                        else:
                            with ib.new_scope():
                                process_padding(
                                    ib,
                                    (dtype, w, c0, size,
                                     top_padding_size, top_offset, padding_value),
                                    output_buf, output_format)

                    if "bottom_padding_size" in aipp_config and \
                            aipp_config.get("bottom_padding_size") > 0:
                        bottom_offset = block_index*offset + n1*c1*h*w*c0 + (h - bottom_padding_size)*w*c0
                        if padding_value != 0 and not (c0 == 4 and aipp_config.get("input_format") in \
                                                       ("AYUV444_U8", "ARGB8888_U8")):
                            with ib.new_scope():
                                process_padding_value(ib, (dtype, w, c0, size, bottom_padding_size,
                                                           bottom_offset, padding_value),
                                                      output_buf, aipp_config.get("input_format"), output_format)
                        else:
                            with ib.new_scope():
                                process_padding(
                                    ib,
                                    (dtype, w, c0, size,
                                     bottom_padding_size, bottom_offset, padding_value),
                                    output_buf, output_format)
                scf_inc_vscl = 0
                scf_inc_hscl = 0
                if cur_cce_product in ["Hi3796CV300ES", "Hi3796CV300CS", "SD3403"]:
                    spr13 = 0
                    spr16 = 0
                    if "resize" in aipp_config and \
                            aipp_config.get("resize") == 1:
                        resize_model = 0
                        if "resize_model" in aipp_config:
                            resize_model = aipp_config.get("resize_model")
                        if resize_model == 1:
                            spr13 = 1 << 8 | 1 << 9 | 1 << 10 | 1 << 11
                        else:
                            spr13 = 0

                        if aipp_config.get("resize_input_h") != \
                                aipp_config.get("resize_output_h"):
                            spr13 = spr13 | 1

                            scf_inc_vscl = \
                                math.floor(
                                    (aipp_config.get("resize_input_h") - 1) * 262144 /
                                    (aipp_config.get("resize_output_h") - 1)) & \
                                0xFFFFFC
                            spr16 = spr16 | scf_inc_vscl

                        if aipp_config.get("resize_input_w") != \
                                aipp_config.get("resize_output_w"):
                            spr13 = spr13 | 1 << 2

                            scf_inc_hscl = \
                                math.floor(
                                    (aipp_config.get("resize_input_w") - 1) * 262144 /
                                    (aipp_config.get("resize_output_w") - 1)) & \
                                0xFFFFFC
                            spr16 = spr16 | scf_inc_hscl << 32

                        if aipp_config.get("resize_output_w") > \
                                aipp_config.get("resize_input_w"):
                            spr13 = spr13 | 1 << 7

                if l1_image_buf_max >= actual_col_size:
                    with ib.new_scope():
                        output_cb_buf, output_ub_buf = \
                            aipp_comm.new_alloc(ib, dtype,
                                                c1 * l1_image_buf_max * c0)

                        if cur_cce_product in ["Hi3796CV300ES", "Hi3796CV300CS", "SD3403"]:
                            spr12 = 0
                            spr15 = 0
                            if ("resize") not in aipp_config or \
                                    aipp_config.get("resize") == 0:
                                spr12 = ((load_image_h - 1)) | \
                                        ((load_image_w - 1) << 16)
                            else:
                                spr12 = \
                                    (aipp_config.get("resize_output_h") - 1) | \
                                    ((aipp_config.get("resize_output_w") - 1) << 16)
                            ib.emit(tvm.call_extern(dtype, "set_aipp_spr_12",
                                                    tvm.const(spr12,
                                                              dtype="uint64")))
                            ib.emit(tvm.call_extern(dtype, "set_aipp_spr_13",
                                                    tvm.const(spr13,
                                                              dtype="uint64")))
                            ib.emit(tvm.call_extern(dtype, "set_aipp_spr_15",
                                                    tvm.const(spr15,
                                                              dtype="uint64")))
                            ib.emit(tvm.call_extern(dtype, "set_aipp_spr_16",
                                                    tvm.const(spr16,
                                                              dtype="uint64")))

                        aipp_xt = aipp_xt | (left_padding_size & 0xff) << 32 | \
                                 (right_padding_size & 0xff) << 45

                        aipp_xs = tvm.const(
                            (load_image_w - 1) |
                            (load_image_h - 1) << 16 |
                            (load_start_pos_w) << 32 |
                            (load_start_pos_h) << 48, dtype="uint64")
                        ib.emit(tvm.call_extern(dtype, "load_image_to_cbuf",
                                                output_cb_buf.access_ptr(
                                                    "rw",
                                                    ptr_type=dtype,
                                                    offset=0),
                                                aipp_xs,
                                                tvm.const(aipp_xt,
                                                          dtype="uint64")))

                        output_offset = n1*c1*h*w*c0

                        ib.emit(tvm.call_extern(
                            dtype, 'copy_cbuf_to_ubuf',
                            output_ub_buf.access_ptr("w",
                                                     ptr_type=dtype,
                                                     offset=0),
                            output_cb_buf.access_ptr("rw",
                                                     ptr_type=dtype,
                                                     offset=0),
                            0, 1,
                            (c1*(h - top_padding_size - bottom_padding_size)*w*c0*size + 31) // 32,
                            0, 0))

                        num = c1*(h - top_padding_size - bottom_padding_size)*w*c0
                        if output_format == "NC1HWC0_C04" and \
                                num*size > 32 and num*size % 32 != 0:

                            tail_ub = ib.allocate(dtype, (32 // size,), "tail_ub", scope=tbe_platform.scope_ubuf)
                            tail_ub_buf = tvm.decl_buffer((32 // size,), dtype, "tail_ub_buf",
                                                          scope=tbe_platform.scope_ubuf, data=tail_ub)
                            aipp_comm.copy_ubuf_to_gm_tail(
                                ib, dtype, output_buf, output_ub_buf,
                                tail_ub_buf,
                                num,
                                block_index*offset +
                                top_padding_size*w*c0 +
                                output_offset,
                                0)
                        ib.emit(tvm.call_extern(
                            dtype, 'copy_ubuf_to_gm',
                            output_buf.access_ptr("w",
                                                  ptr_type=dtype,
                                                  offset=block_index*offset +
                                                  top_padding_size*w*c0 +
                                                  output_offset),
                            output_ub_buf.access_ptr("rw",
                                                     ptr_type=dtype, offset=0),
                            0, 1,
                            c1*(h - top_padding_size - bottom_padding_size)*w*c0*size // 32,
                            0, 0))

                else:
                    buffer_upper_limit = l1_image_buf_max
                    l1_size = tbe_platform.get_soc_spec(tbe_platform.L1_SIZE)
                    if output_format == "NC1HWC0_C04":
                        l1_size = l1_size - 64

                    if aipp_config.get('input_format') == "YUV420SP_U8":
                        if 2*w > l1_image_buf_max:
                            buffer_upper_limit = l1_size // size // c0
                    else:
                        if w > l1_image_buf_max:
                            buffer_upper_limit = l1_size // size // c0

                    tiling_h = buffer_upper_limit // w

                    if aipp_config.get('input_format') == "YUV420SP_U8":
                        #tiling_h must be even
                        if tiling_h % 2 != 0:
                            if tiling_h > 1:
                                tiling_h = tiling_h - 1

                    if "resize" in aipp_config and aipp_config.get("resize") == 1:
                        h_loop = aipp_config.get("resize_output_h")//tiling_h
                    else:
                        h_loop = load_image_h // tiling_h

                    h_start_pos = tvm.const(48, dtype="uint64")
                    w_start_pos = tvm.const(32, dtype="uint64")
                    load_h_pos = tvm.const(16, dtype="uint64")
                    tiling_h_const = tvm.const(tiling_h, dtype="uint64")
                    load_w = tvm.const(load_image_w - 1, dtype="uint64")
                    load_h = tvm.const(tiling_h - 1, dtype="uint64")
                    zero_const = tvm.const(0, dtype="uint64")
                    h_loop_const = tvm.const(h_loop, dtype="uint64")

                    with ib.for_range(zero_const, h_loop_const, name="h1",
                                      dtype="uint64") as h1:
                        with ib.new_scope():
                            num = (((c1*buffer_upper_limit*c0*size+31)//32)*32) // size
                            output_cb = ib.allocate(dtype,
                                                    (num,),
                                                    "output_cb",
                                                    scope=tbe_platform.scope_cbuf)
                            output_cb_buf = tvm.decl_buffer(
                                (num,), dtype,
                                "output_cb_buf", scope=tbe_platform.scope_cbuf,
                                data=output_cb)

                            aipp_xt = aipp_xt | \
                                     (left_padding_size & 0xff) << 32 | \
                                     (right_padding_size & 0xff) << 45

                            output_w = w
                            output_h = tiling_h
                            output_offset = n1*c1*h*w*c0 + \
                                            c1*(h1*tiling_h)*output_w*c0
                            resize_input_h_stat_pos = 0
                            if cur_cce_product in ["Hi3796CV300ES", "Hi3796CV300CS", "SD3403"]:
                                spr12 = 0
                                spr15 = 0
                                if "resize" not in aipp_config or \
                                        aipp_config.get("resize") == 0:
                                    spr12 = (tiling_h - 1) | \
                                            (load_image_w - 1) << 16
                                else:
                                    spr12 = \
                                        (tiling_h - 1) | \
                                        ((aipp_config.get("resize_output_w") - 1) << 16)

                                    if aipp_config.get("resize_input_h") != \
                                            aipp_config.get("resize_output_h"):
                                        resize_output_h_start_pos = h1*tiling_h
                                        resize_output_h_end_pos = \
                                            ((h1 + 1)*tiling_h - 1)

                                        resize_input_h_stat_pos = \
                                            (scf_inc_vscl*resize_output_h_start_pos) >> 18
                                        resize_input_h_end_pos = \
                                            ((scf_inc_vscl*resize_output_h_end_pos) +
                                             (1 << 18) - 1) >> 18
                                        if aipp_config.get("input_format") == "YUV420SP_U8":
                                            resize_input_h_stat_pos = \
                                                resize_input_h_stat_pos & \
                                                0xfffffffffffffffe
                                            resize_input_h_end_pos += \
                                                (resize_input_h_end_pos -
                                                 resize_input_h_stat_pos + 1) & \
                                                0x1

                                        acc_vscl = \
                                            (scf_inc_vscl*resize_output_h_start_pos) -\
                                            (resize_input_h_stat_pos << 18)
                                        spr15 = acc_vscl

                                        load_h = \
                                            get_const(
                                                (resize_input_h_end_pos -
                                                 resize_input_h_stat_pos))
                                ib.emit(
                                    tvm.call_extern(
                                        dtype, "set_aipp_spr_12",
                                        tvm.const(spr12, dtype="uint64")))
                                ib.emit(
                                    tvm.call_extern(
                                        dtype, "set_aipp_spr_13",
                                        tvm.const(spr13, dtype="uint64")))
                                ib.emit(
                                    tvm.call_extern(dtype, "set_aipp_spr_15",
                                                    get_const(spr15)))
                                ib.emit(
                                    tvm.call_extern(dtype, "set_aipp_spr_16",
                                                    tvm.const(spr16,
                                                              dtype="uint64")))

                            if "resize" in aipp_config and \
                                    aipp_config.get("resize") == 1 and \
                                    aipp_config.get("resize_input_h") != \
                                    aipp_config.get("resize_output_h"):
                                aipp_xs = get_const(
                                    load_w | load_h << load_h_pos |
                                    (tvm.const(load_start_pos_w, dtype="uint64")) << w_start_pos |
                                    (load_start_pos_h + resize_input_h_stat_pos) << h_start_pos)
                            else:
                                aipp_xs = get_const(
                                    load_w | load_h << load_h_pos |
                                    (tvm.const(load_start_pos_w, dtype="uint64")) << w_start_pos |
                                    (load_start_pos_h + h1*tiling_h_const) << h_start_pos)
                            ib.emit(tvm.call_extern(dtype, "load_image_to_cbuf",
                                                    output_cb_buf.access_ptr(
                                                        "rw", ptr_type=dtype,
                                                        offset=0),
                                                    aipp_xs,
                                                    tvm.const(aipp_xt,
                                                              dtype="uint64")))

                            move_data_from_l1_to_gm(ib, c1*output_h*w*c0, dtype,
                                                    output_cb_buf, output_buf,
                                                    block_index*offset+top_padding_size*w*c0+output_offset,
                                                    output_format)

                    if "resize" in aipp_config and \
                            aipp_config.get("resize") == 1:
                        tail_h = aipp_config.get("resize_output_h") % tiling_h
                    else:
                        tail_h = load_image_h % tiling_h
                    if tail_h != 0:
                        tail_h_postion = tvm.const(
                            load_start_pos_h + h_loop*tiling_h, dtype="uint64")
                        load_tail_h = tvm.const(tail_h - 1, dtype="uint64")

                        with ib.new_scope():
                            num = (((c1*buffer_upper_limit*c0*size + 31) // 32)*32) // size
                            output_cb = ib.allocate(dtype,
                                                    (num,),
                                                    "output_cb",
                                                    scope=tbe_platform.scope_cbuf)
                            output_cb_buf = tvm.decl_buffer(
                                (num,), dtype,
                                "output_cb_buf", scope=tbe_platform.scope_cbuf,
                                data=output_cb)

                            output_w = w
                            aipp_xt = aipp_xt | \
                                     (left_padding_size & 0xff) << 32 | \
                                     (right_padding_size & 0xff) << 45
                            output_h = tail_h
                            output_offset = \
                                n1 * c1 * h * w * c0 + c1 * (h_loop_const * tiling_h) * output_w * c0

                            if cur_cce_product in ["Hi3796CV300ES", "Hi3796CV300CS", "SD3403"]:
                                spr12 = 0
                                spr15 = 0
                                if ("resize") not in aipp_config or \
                                        aipp_config.get("resize") == 0:
                                    spr12 = (tail_h - 1) | \
                                            (load_image_w - 1) << 16
                                else:
                                    spr12 = \
                                        (tail_h - 1) | \
                                        ((aipp_config.get("resize_output_w") - 1) << 16)


                                    if aipp_config.get("resize_input_h") != \
                                            aipp_config.get("resize_output_h"):
                                        resize_output_h_start_pos = h_loop*tiling_h
                                        resize_output_h_end_pos = \
                                            aipp_config.get("resize_output_h") - 1

                                        resize_input_h_stat_pos = \
                                            (scf_inc_vscl*resize_output_h_start_pos) >> 18
                                        resize_input_h_end_pos = \
                                            ((scf_inc_vscl*resize_output_h_end_pos) +
                                             (1 << 18) - 1) >> 18
                                        if aipp_config.get("input_format") == "YUV420SP_U8":
                                            resize_input_h_stat_pos = \
                                                resize_input_h_stat_pos & \
                                                0xfffffffffffffffe
                                            resize_input_h_end_pos += \
                                                (resize_input_h_end_pos -
                                                 resize_input_h_stat_pos + 1) & \
                                                0x1

                                        acc_vscl = \
                                            (scf_inc_vscl*resize_output_h_start_pos) - \
                                            (resize_input_h_stat_pos << 18)
                                        spr15 = acc_vscl

                                        load_tail_h = tvm.const(
                                            (resize_input_h_end_pos -
                                             resize_input_h_stat_pos),
                                            dtype="uint64")
                                        tail_h_postion = tvm.const(
                                            (load_start_pos_h +
                                             resize_input_h_stat_pos),
                                            dtype="uint64")
                                ib.emit(
                                    tvm.call_extern(
                                        dtype, "set_aipp_spr_12",
                                        tvm.const(spr12, dtype="uint64")))
                                ib.emit(
                                    tvm.call_extern(
                                        dtype, "set_aipp_spr_13",
                                        tvm.const(spr13, dtype="uint64")))
                                ib.emit(
                                    tvm.call_extern(
                                        dtype, "set_aipp_spr_15",
                                        tvm.const(spr15, dtype="uint64")))
                                ib.emit(
                                    tvm.call_extern(
                                        dtype, "set_aipp_spr_16",
                                        tvm.const(spr16, dtype="uint64")))

                            aipp_xs = get_const(
                                load_w | load_tail_h << load_h_pos |
                                tail_h_postion << h_start_pos |
                                (tvm.const(load_start_pos_w, dtype="uint64")) << w_start_pos)

                            ib.emit(tvm.call_extern(dtype, "load_image_to_cbuf",
                                                    output_cb_buf.access_ptr(
                                                        "rw", ptr_type=dtype,
                                                        offset=0),
                                                    tvm.const(aipp_xs,
                                                              dtype="uint64"),
                                                    tvm.const(aipp_xt,
                                                              dtype="uint64")))
                            move_data_from_l1_to_gm(ib, c1*output_h*w*c0, dtype,
                                                    output_cb_buf, output_buf,
                                                    block_index*offset+top_padding_size*w*c0+output_offset,
                                                    output_format)

        _aipp_intrin()
        return ib.get()

    return tvm.extern([(n, c1, h, w, c0)], [input_tensor],
                      lambda ins, outs: aipp_ir(ins[0], outs[0]),
                      dtype=[dtype], name="aipp")
