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
target_crop_and_resize
"""
# 'pylint: disable=invalid-name,unused-argument,unused-variable,too-many-statements
# 'pylint: disable=too-many-arguments,too-many-locals
from tbe import tvm
import te.platform as tbe_platform
from te.utils import para_check
from impl.util.platform_adapter import build_config
from impl.util.platform_adapter import get_const

from impl import aipp_comm


def check_input_format(input_format):
    """
    check input format
    """
    if input_format not in ["YUV420SP_U8", "RGB888_U8", "XRGB8888_U8", "ARGB8888_U8", "YUYV_U8",
                            "YUV422SP_U8", "AYUV444_U8", "YUV400_U8"]:
        cause_desc = "Hi3796CV300CS and SD3403 only support YUV420SP_U8, RGB888_U8, XRGB8888_U8, XRGB8888_U8, " \
                     "YUYV_U8, YUV422SP_U8, AYUV444_U8, YUV400_U8, current input format is %s" % input_format
        aipp_comm.raise_runtime_error(cause_desc)


def check_src_image_size_w(src_image_size_w, input_format):
    """
    check source image width
    """
    if input_format in ["YUV420SP_U8", "YUV400_U8", "YUV422SP_U8", "RAW10", "RAW12", "RAW16", "uint16"]:
        if src_image_size_w % 16 != 0:
            cause_desc = "src_image_size_w[%d] must be multiples of 16 for %s" % \
                         (src_image_size_w, input_format)
            aipp_comm.raise_runtime_error(cause_desc)
    elif input_format in ["RGB888_U8"]:
        if (src_image_size_w*3) % 16 != 0:
            cause_desc = "src_image_size_w[%d]*3 must be multiples of 16 for %s" % \
                         (src_image_size_w, input_format)
            aipp_comm.raise_runtime_error(cause_desc)
    elif input_format in ["XRGB8888_U8", "ARGB8888_U8", "AYUV444_U8"]:
        if (src_image_size_w*4) % 16 != 0:
            cause_desc = "src_image_size_w[%d]*4 must be multiples of 16 for %s" % \
                         (src_image_size_w, input_format)
            aipp_comm.raise_runtime_error(cause_desc)
    elif input_format in ["YUYV_U8"]:
        if (src_image_size_w*2) % 16 != 0:
            cause_desc = "src_image_size_w[%d]*2 must be multiples of 16 for %s" % \
                         (src_image_size_w, input_format)
            aipp_comm.raise_runtime_error(cause_desc)
    else:
        pass


def set_spr2_spr9(ib, input_format, dtype):
    """
    set spr 2~9
    """
    spr2 = 0
    spr3 = 0
    spr4 = 0
    if input_format in ["YUV420SP_U8", "YUYV_U8", "YUV422SP_U8", "AYUV444_U8"]:
        # spr_2->bits.csc_matrix_r0_c2 = (uint16_t)(1<<10)
        spr2 = spr2 | (1 << 10 & 0xffff) << 32

        spr3 = 0
        # spr_3->bits.csc_matrix_r1_c1
        spr3 = spr3 | (1 << 10 & 0xffff)
        # spr_3->bits.csc_matrix_r2_c0
        spr3 = spr3 | (1 << 10 & 0xffff) << 32

        spr4 = 0

    spr9 = 0
    if input_format == "YUV420SP_U8":
        spr9 = spr9 | 0 << 19
    elif input_format == "XRGB8888_U8":
        spr9 = spr9 | 1 << 19
    elif input_format == "RGB888_U8":
        spr9 = spr9 | 4 << 19
    elif input_format == "ARGB8888_U8":
        spr9 = spr9 | 5 << 19
    elif input_format == "YUYV_U8":
        spr9 = spr9 | 6 << 19
    elif input_format == "YUV422SP_U8":
        spr9 = spr9 | 7 << 19
    elif input_format == "AYUV444_U8":
        spr9 = spr9 | 8 << 19
    elif input_format == "YUV400_U8":
        spr9 = spr9 | 9 << 19
    elif input_format == "RAW10":
        spr9 = spr9 | 10 << 19
    elif input_format == "RAW12":
        spr9 = spr9 | 11 << 19
    elif input_format == "RAW16":
        spr9 = spr9 | 12 << 19

    spr9 = spr9 | (1 << 40)

    ib.emit(tvm.call_extern(dtype, "set_aipp_spr_2", tvm.const(spr2, dtype="uint64")))
    ib.emit(tvm.call_extern(dtype, "set_aipp_spr_3", tvm.const(spr3, dtype="uint64")))
    ib.emit(tvm.call_extern(dtype, "set_aipp_spr_4", tvm.const(spr4, dtype="uint64")))
    ib.emit(tvm.call_extern(dtype, "set_aipp_spr_9", tvm.const(spr9, dtype="uint64")))


def set_spr0_spr1(ib, x_buf, x_dic, input_format, box_index_reg):
    """
    set spr 0 and 1
    """
    x_shape = x_dic.get('shape')
    x_format = x_dic.get('format')

    if x_format == "NHWC":
        src_h = x_shape[1]
        src_w = x_shape[2]
        src_c = x_shape[3]
    elif x_format == "NCHW":
        src_c = x_shape[1]
        src_h = x_shape[2]
        src_w = x_shape[3]

    spr0 = tvm.const(0, dtype="uint64")
    spr1 = tvm.const(0, dtype="uint64")

    if input_format == "YUV420SP_U8":
        spr0 = get_const(
            x_buf.access_ptr('r', offset=box_index_reg[0]*(src_c*src_h*src_w) // 2))
    elif input_format in ["XRGB8888_U8", "RGB888_U8", "ARGB8888_U8", "AYUV444_U8", "YUV400_U8",
                          "RAW10", "RAW12", "RAW16"]:
        spr0 = get_const(
            x_buf.access_ptr('r', offset=box_index_reg[0]*(src_c*src_h*src_w)))
    elif input_format in ["YUYV_U8", "YUV422SP_U8"]:
        spr0 = get_const(
            x_buf.access_ptr('r', offset=box_index_reg[0]*(2*src_h*src_w)))

    cur_cce_product = tbe_platform.get_soc_spec("SHORT_SOC_VERSION")
    if cur_cce_product in ["Hi3796CV300ES", "Hi3796CV300CS", "SD3403"]:
        if input_format in ["YUV420SP_U8", "YUYV_U8", "YUV422SP_U8", "AYUV444_U8"]:
            spr1 = tvm.const(1 << 63, dtype="uint64")

    if input_format == "YUV420SP_U8":
        spr1 = get_const(
            x_buf.access_ptr('r', offset=box_index_reg[0]*(src_c*src_h*src_w) // 2 + src_h*src_w)) | spr1
    elif input_format == "YUV422SP_U8":
        spr1 = get_const(
            x_buf.access_ptr('r', offset=box_index_reg[0]*(2*src_h*src_w) + src_h*src_w)) | spr1

    ib.emit(tvm.call_extern("uint8", "set_aipp_spr_0", spr0))
    ib.emit(tvm.call_extern("uint8", "set_aipp_spr_1", spr1))


def move_data_from_l1_to_gm(ib, total_num, dtype, output_cb_buf, output_buf,
                            gm_output_offset, output_format="NC1HWC0_C04"):
    """
    move data from l1 to gm
    """
    ub_size = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
    if output_format == "NC1HWC0_C04":
        ub_size = ub_size - 64
    ub_size -= (1024 - 1) // 1024 * 1024

    if dtype == "float16":
        size = 2
    else:
        size = 1

    if total_num*size < (ub_size // 2):
        with ib.new_scope():
            num = (((total_num*size + 31) // 32)*32) // size
            output_ub = ib.allocate(dtype, (num,), "output_ub", scope=tbe_platform.scope_ubuf)
            output_ub_buf = tvm.decl_buffer((num,), dtype, "output_ub_buf",
                                            scope=tbe_platform.scope_ubuf, data=output_ub)

            burst_len, n_burst = aipp_comm.get_lenburst_and_nburst((total_num * size + 31) // 32, 1)

            ib.emit(tvm.call_extern(
                dtype, 'copy_cbuf_to_ubuf',
                output_ub_buf.access_ptr("w", ptr_type=dtype, offset=0),
                output_cb_buf.access_ptr("rw", ptr_type=dtype, offset=0),
                0, n_burst, burst_len, 0, 0))

            if output_format == "NC1HWC0_C04" and total_num * size % 32 != 0 and total_num * size > 32:
                burst_len, n_burst = aipp_comm.get_lenburst_and_nburst((total_num * size) // 32, 1)

                tail_ub = ib.allocate(dtype, (32 // size,), "tail_ub", scope=tbe_platform.scope_ubuf)
                tail_ub_buf = tvm.decl_buffer((32 // size,), dtype, "tail_ub_buf",
                                              scope=tbe_platform.scope_ubuf, data=tail_ub)
                aipp_comm.copy_ubuf_to_gm_tail(ib, dtype, output_buf, output_ub_buf, tail_ub_buf,
                                               total_num, gm_output_offset, 0)

            ib.emit(tvm.call_extern(
                dtype, 'copy_ubuf_to_gm',
                output_buf.access_ptr("w", ptr_type=dtype, offset=gm_output_offset),
                output_ub_buf.access_ptr("rw", ptr_type=dtype, offset=0),
                0, n_burst, burst_len, 0, 0))
    else:
        cycle_num = (ub_size // 2) // size
        move_num = total_num // cycle_num
        move_tail = total_num - move_num*cycle_num
        if output_format == "NC1HWC0_C04" and move_tail*size < 32:
            cycle_num = (cycle_num*size - 32) // size
            move_num = total_num // cycle_num
            move_tail = total_num - move_num*cycle_num

        with ib.for_range(tvm.const(0, dtype="uint64"), tvm.const(move_num, dtype="uint64"),
                          name="move_index", dtype="uint64") as move_index:
            with ib.new_scope():
                output_ub = ib.allocate(dtype, (ub_size // 2,), "output_ub", scope=tbe_platform.scope_ubuf)
                output_ub_buf = tvm.decl_buffer((ub_size // 2,), dtype, "output_ub_buf",
                                                scope=tbe_platform.scope_ubuf, data=output_ub)

                burst_len, n_burst = aipp_comm.get_lenburst_and_nburst((cycle_num * size) // 32, 1)

                ib.emit(tvm.call_extern(
                    dtype, 'copy_cbuf_to_ubuf',
                    output_ub_buf.access_ptr("w", ptr_type=dtype, offset=0),
                    output_cb_buf.access_ptr("rw", ptr_type=dtype, offset=move_index*cycle_num),
                    0, n_burst, burst_len, 0, 0))

                ib.emit(tvm.call_extern(
                    dtype, 'copy_ubuf_to_gm',
                    output_buf.access_ptr("w", ptr_type=dtype, offset=gm_output_offset + move_index*cycle_num),
                    output_ub_buf.access_ptr("rw", ptr_type=dtype, offset=0),
                    0, n_burst, burst_len, 0, 0))
        if move_tail != 0:
            with ib.new_scope():
                output_ub = ib.allocate(dtype, (ub_size // 2,), "output_ub", scope=tbe_platform.scope_ubuf)
                output_ub_buf = tvm.decl_buffer((ub_size // 2,), dtype, "output_ub_buf",
                                                scope=tbe_platform.scope_ubuf, data=output_ub)

                burst_len, n_burst = aipp_comm.get_lenburst_and_nburst((move_tail * size + 31) // 32, 1)

                ib.emit(tvm.call_extern(
                    dtype, 'copy_cbuf_to_ubuf',
                    output_ub_buf.access_ptr("w", ptr_type=dtype, offset=0),
                    output_cb_buf.access_ptr("rw", ptr_type=dtype, offset=move_num*cycle_num),
                    0, n_burst, burst_len, 0, 0))

                if output_format == "NC1HWC0_C04" and (move_tail * size) % 32 != 0 and move_tail * size > 32:
                    burst_len, n_burst = aipp_comm.get_lenburst_and_nburst((move_tail * size) // 32, 1)

                    tail_ub = ib.allocate(dtype, (32 // size,), "tail_ub", scope=tbe_platform.scope_ubuf)
                    tail_ub_buf = tvm.decl_buffer((32 // size,), dtype, "tail_ub_buf",
                                                  scope=tbe_platform.scope_ubuf, data=tail_ub)
                    aipp_comm.copy_ubuf_to_gm_tail(
                        ib, dtype, output_buf, output_ub_buf, tail_ub_buf, move_tail,
                        gm_output_offset + move_num*cycle_num, 0)

                ib.emit(tvm.call_extern(
                    dtype, 'copy_ubuf_to_gm',
                    output_buf.access_ptr("w", ptr_type=dtype, offset=gm_output_offset + move_num*cycle_num),
                    output_ub_buf.access_ptr("rw", ptr_type=dtype, offset=0),
                    0, n_burst, burst_len, 0, 0))


def target_crop_and_resize_compute(x, boxes, box_index, x_dic, y_dic, input_format):
    """
    compute of target_crop_and_resize op
    """
    x_format = x_dic.get('format')
    x_shape = x_dic.get('shape')
    output_dtype = y_dic.get('dtype')
    output_shape = y_dic.get('shape')

    if output_dtype == "float16":
        size = 2
    else:
        size = 1

    if x_format == "NHWC":
        src_w = x_shape[2]
    elif x_format == "NCHW":
        src_w = x_shape[3]

    batch, channel_1, height, width, channel_0 = output_shape

    actual_col_size = height * width

    l1_image_buf_max = aipp_comm.get_l1_image_buf_max(actual_col_size, output_dtype,
                                                      True, "NC1HWC0_C04")

    def target_crop_and_resize_ir(x_buf, boxes_buf, box_index_buf, output_buf):
        """
        ir of target_crop_and_resize op
        """
        ib = tvm.tir.ir_builder.create()
        set_spr2_spr9(ib, input_format, output_dtype)

        device_core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        batch_num = batch
        batch_factor = 1

        if batch_num % device_core_num == 0:
            batch_factor = batch_num // device_core_num

        block_index = tvm.thread_axis("blockIdx.x")
        ib.scope_attr(block_index, "thread_extent", batch_num // batch_factor)

        offset = batch_factor * channel_1 * height * width * channel_0
        zero_const = tvm.const(0, dtype="uint64")

        boxes_ub = ib.allocate("int32", [8], "boxes_ub", scope=tbe_platform.scope_ubuf)
        boxes_ub_buf = tvm.decl_buffer([8], "int32", "boxes_ub_buf",
                                       scope=tbe_platform.scope_ubuf, data=boxes_ub)
        boxes_reg = ib.allocate("uint64", [4], name="boxes_reg", scope=tbe_platform.scope_reg)
        tmp = ib.allocate("uint64", [1], name="tmp", scope=tbe_platform.scope_reg)

        box_index_ub = ib.allocate("int32", [8], "box_index_ub", scope=tbe_platform.scope_ubuf)
        box_index_ub_buf = tvm.decl_buffer([8], "int32", "box_index_ub_buf",
                                           scope=tbe_platform.scope_ubuf, data=box_index_ub)
        box_index_reg = ib.allocate("uint64", [1], name="box_index_reg", scope=tbe_platform.scope_reg)

        scf_inc_vscl = ib.allocate("uint64", [1], name="scfIncVscl", scope=tbe_platform.scope_reg)
        scf_inc_hscl = ib.allocate("uint64", [1], name="scfIncHscl", scope=tbe_platform.scope_reg)

        spr13 = ib.allocate("uint64", [1], name="spr13", scope=tbe_platform.scope_reg)
        spr15 = ib.allocate("uint64", [1], name="spr15", scope=tbe_platform.scope_reg)
        spr16 = ib.allocate("uint64", [1], name="spr16", scope=tbe_platform.scope_reg)

        with ib.for_range(zero_const, tvm.const(batch_factor, dtype="uint64"),
                          name="n1", dtype="uint64") as n1:
            spr13[0] = tvm.const(0, dtype="uint64")
            spr15[0] = tvm.const(0, dtype="uint64")
            spr16[0] = tvm.const(0, dtype="uint64")

            batch_id = batch_factor * block_index + n1

            ib.emit(tvm.call_extern("int32", 'copy_gm_to_ubuf',
                                    box_index_ub_buf.access_ptr("w", ptr_type="int32", offset=0),
                                    box_index_buf.access_ptr("rw", ptr_type="int32", offset=batch_id),
                                    0, 1, 1, 0, 0))
            ib.emit(tvm.call_extern("int32", "reg_mov", tvm.call_extern("uint64", "reg", box_index_reg[0]),
                                    box_index_ub_buf.access_ptr('r', offset=0)))

            with ib.if_scope(box_index_reg[0] >= 0):
                ib.emit(tvm.call_extern("int32", 'copy_gm_to_ubuf',
                                        boxes_ub_buf.access_ptr("w", ptr_type="int32", offset=0),
                                        boxes_buf.access_ptr("rw", ptr_type="int32", offset=4*batch_id),
                                        0, 1, 1, 0, 0))

                # get load_start_pos_h
                ib.emit(tvm.call_extern("int32", "reg_mov", tvm.call_extern("uint64", "reg", tmp[0]),
                                        boxes_ub_buf.access_ptr('r', offset=0)))
                boxes_reg[0] = tmp[0]

                # get load_start_pos_w
                ib.emit(tvm.call_extern("int32", "reg_mov", tvm.call_extern("uint64", "reg", tmp[0]),
                                        boxes_ub_buf.access_ptr('r', offset=1)))
                boxes_reg[1] = tmp[0]

                # get crop_size_h
                ib.emit(tvm.call_extern("int32", "reg_mov", tvm.call_extern("uint64", "reg", tmp[0]),
                                        boxes_ub_buf.access_ptr('r', offset=2)))
                boxes_reg[2] = tmp[0]

                # get crop_size_w
                ib.emit(tvm.call_extern("int32", "reg_mov", tvm.call_extern("uint64", "reg", tmp[0]),
                                        boxes_ub_buf.access_ptr('r', offset=3)))
                boxes_reg[3] = tmp[0]

                aipp_xt = src_w - 1

                set_spr0_spr1(ib, x_buf, x_dic, input_format, box_index_reg)

                with ib.if_scope(boxes_reg[2] != height):
                    spr13[0] = spr13[0] | tvm.const(1, dtype="uint64")
                    scf_inc_vscl[0] = ((boxes_reg[2] - tvm.const(1, dtype="uint64"))*tvm.const(262144, dtype="uint64")
                                       / (tvm.const(height - 1, dtype="uint64"))) & 0xFFFFFC
                    spr16[0] = spr16[0] | scf_inc_vscl[0]

                with ib.if_scope(boxes_reg[3] != width):
                    spr13[0] = spr13[0] | tvm.const(1, dtype="uint64") << 2

                    scf_inc_hscl[0] = ((boxes_reg[3] - tvm.const(1, dtype="uint64"))*tvm.const(262144, dtype="uint64")
                                       / (tvm.const(width - 1, dtype="uint64"))) & 0xFFFFFC
                    spr16[0] = spr16[0] | scf_inc_hscl[0] << 32

                with ib.if_scope(width > boxes_reg[3]):
                    spr13[0] = spr13[0] | tvm.const(1, dtype="uint64") << 7

                ib.emit(tvm.call_extern(output_dtype, "set_aipp_spr_13", spr13[0]))
                ib.emit(tvm.call_extern(output_dtype, "set_aipp_spr_16", spr16[0]))

                if l1_image_buf_max >= actual_col_size:
                    ib.emit(tvm.call_extern(output_dtype, "set_aipp_spr_15", spr15[0]))

                    spr12 = (height - 1) | ((width - 1) << 16)
                    ib.emit(tvm.call_extern(output_dtype, "set_aipp_spr_12", tvm.const(spr12, dtype="uint64")))

                    with ib.new_scope():
                        output_cb_buf, output_ub_buf = \
                            aipp_comm.new_alloc(ib, output_dtype, l1_image_buf_max * channel_0)

                        aipp_xs = get_const(
                            (boxes_reg[3] - 1) |
                            (boxes_reg[2] - 1) << 16 |
                            (boxes_reg[1]) << 32 |
                            (boxes_reg[0]) << 48)

                        ib.emit(tvm.call_extern(output_dtype, "load_image_to_cbuf",
                                                output_cb_buf.access_ptr("rw", ptr_type=output_dtype, offset=0),
                                                aipp_xs, get_const(aipp_xt)))

                        output_offset = n1 * channel_1 * height * width * channel_0

                        ib.emit(tvm.call_extern(
                            output_dtype, 'copy_cbuf_to_ubuf',
                            output_ub_buf.access_ptr("w", ptr_type=output_dtype, offset=0),
                            output_cb_buf.access_ptr("rw", ptr_type=output_dtype, offset=0),
                            0, 1, channel_1 * height * width * channel_0 * size // 32, 0, 0))

                        if (channel_1 * height * width * channel_0 * size) % 32 != 0:
                            with ib.new_scope():
                                tail_ub = ib.allocate(output_dtype, (32 // size,), "tail_ub",
                                                      scope=tbe_platform.scope_ubuf)
                                tail_ub_buf = tvm.decl_buffer((32 // size,), output_dtype, "tail_ub_buf",
                                                              scope=tbe_platform.scope_ubuf, data=tail_ub)
                                aipp_comm.copy_ubuf_to_gm_tail(
                                    ib, output_dtype, output_buf, output_ub_buf, tail_ub_buf,
                                    channel_1*height*width*channel_0, block_index*offset + output_offset, 0)

                        ib.emit(tvm.call_extern(
                            output_dtype, 'copy_ubuf_to_gm',
                            output_buf.access_ptr("w", ptr_type=output_dtype,
                                                  offset=block_index*offset + output_offset),
                            output_ub_buf.access_ptr("rw", ptr_type=output_dtype, offset=0),
                            0, 1, channel_1*height*width*channel_0*size // 32, 0, 0))
                else:
                    buffer_upper_limit = l1_image_buf_max
                    l1_size = tbe_platform.get_soc_spec(tbe_platform.L1_SIZE) - 64

                    if 2*width > l1_image_buf_max:
                        buffer_upper_limit = l1_size // size // channel_0

                    tiling_h = buffer_upper_limit // width

                    if input_format == "YUV420SP_U8":
                        # tiling_h must be even
                        if tiling_h % 2 != 0:
                            if tiling_h > 1:
                                tiling_h = tiling_h - 1

                    h_loop = height // tiling_h

                    h_start_pos = tvm.const(48, dtype="uint64")
                    w_start_pos = tvm.const(32, dtype="uint64")
                    load_h_pos = tvm.const(16, dtype="uint64")
                    tiling_h_const = tvm.const(tiling_h, dtype="uint64")

                    resize_input_h_stat_pos = ib.allocate("uint64", [1], name="resize_input_h_stat_pos",
                                                          scope=tbe_platform.scope_reg)
                    resize_input_h_end_pos = ib.allocate("uint64", [1], name="resize_input_h_end_pos",
                                                         scope=tbe_platform.scope_reg)
                    load_h = ib.allocate("uint64", [1], name="load_h", scope=tbe_platform.scope_reg)
                    load_h[0] = tvm.const(tiling_h - 1, dtype="uint64")

                    xs = ib.allocate("uint64", [1], name="Xs", scope=tbe_platform.scope_reg)
                    xs[0] = tvm.const(0, dtype="uint64")

                    with ib.for_range(zero_const, h_loop, name="h1", dtype="uint64") as h1:
                        with ib.new_scope():
                            num = (((channel_1 * buffer_upper_limit * channel_0 * size + 31) // 32) * 32) // size
                            output_cb = ib.allocate(output_dtype, (num,), "output_cb",
                                                    scope=tbe_platform.scope_cbuf)
                            output_cb_buf = tvm.decl_buffer((num,), output_dtype, "output_cb_buf",
                                                            scope=tbe_platform.scope_cbuf, data=output_cb)

                            output_w = width
                            output_h = tiling_h
                            output_offset = n1 * channel_1 * height * width * channel_0 + \
                                            channel_1 * (h1 * tiling_h) * output_w * channel_0

                            spr12 = (tvm.const(tiling_h - 1, dtype="uint64")) | \
                                    (tvm.const(width - 1, dtype="uint64")) << 16
                            ib.emit(tvm.call_extern(output_dtype, "set_aipp_spr_12", spr12))

                            with ib.if_scope(boxes_reg[2] != height):
                                resize_input_h_stat_pos[0] = tvm.const(0, dtype="uint64")
                                resize_input_h_end_pos[0] = tvm.const(0, dtype="uint64")

                                resize_output_h_start_pos = h1 * tiling_h
                                resize_output_h_end_pos = (h1 + 1) * tiling_h - 1

                                resize_input_h_stat_pos[0] = \
                                    (scf_inc_vscl[0] * get_const(resize_output_h_start_pos)) >> 18
                                resize_input_h_end_pos[0] = \
                                    ((scf_inc_vscl[0] * get_const(resize_output_h_end_pos)) +
                                     tvm.const((1 << 18) - 1, dtype="uint64")) >> 18

                                if input_format == "YUV420SP_U8":
                                    resize_input_h_stat_pos[0] = resize_input_h_stat_pos[0] & 0xfffffffffffffffe
                                    resize_input_h_end_pos[0] += \
                                        (resize_input_h_end_pos[0] -
                                         resize_input_h_stat_pos[0] + tvm.const(1, dtype="uint64")) & 0x1

                                acc_vscl = \
                                    (scf_inc_vscl[0]*get_const(resize_output_h_start_pos)) - \
                                    (resize_input_h_stat_pos[0] << 18)
                                spr15[0] = acc_vscl

                                ib.emit(tvm.call_extern(output_dtype, "set_aipp_spr_15",
                                                        get_const(spr15[0])))

                                load_h[0] = (resize_input_h_end_pos[0] - resize_input_h_stat_pos[0])
                            with ib.else_scope():
                                ib.emit(tvm.call_extern(output_dtype, "set_aipp_spr_15", tvm.const(0, dtype="uint64")))

                            with ib.if_scope(boxes_reg[2] != height):
                                xs[0] = get_const(
                                    (boxes_reg[3] - 1) | load_h[0] << load_h_pos |
                                    (get_const(boxes_reg[1])) << w_start_pos |
                                    (boxes_reg[0] + resize_input_h_stat_pos[0]) << h_start_pos)
                            with ib.else_scope():
                                xs[0] = get_const(
                                    (boxes_reg[3] - 1) | load_h[0] << load_h_pos |
                                    (get_const(boxes_reg[1])) << w_start_pos |
                                    (boxes_reg[0] + h1*tiling_h_const) << h_start_pos)

                            ib.emit(tvm.call_extern(output_dtype, "load_image_to_cbuf",
                                                    output_cb_buf.access_ptr("rw", ptr_type=output_dtype, offset=0),
                                                    xs[0], get_const(aipp_xt)))

                            move_data_from_l1_to_gm(ib, channel_1*output_h*width*channel_0, output_dtype,
                                                    output_cb_buf, output_buf, block_index*offset+output_offset,
                                                    "NC1HWC0_C04")

                    tail_h = height % tiling_h
                    if tail_h != 0:
                        tail_h_postion = ib.allocate("uint64", [1], name="tail_h_postion", scope=tbe_platform.scope_reg)
                        tail_h_postion[0] = boxes_reg[0] + tvm.const(h_loop*tiling_h, dtype="uint64")
                        load_tail_h = ib.allocate("uint64", [1], name="load_tail_h", scope=tbe_platform.scope_reg)
                        load_tail_h[0] = tvm.const(tail_h - 1, dtype="uint64")

                        with ib.new_scope():
                            num = (((channel_1*buffer_upper_limit*channel_0*size + 31) // 32)*32) // size
                            output_cb = ib.allocate(output_dtype, (num,), "output_cb", scope=tbe_platform.scope_cbuf)
                            output_cb_buf = tvm.decl_buffer((num,), output_dtype, "output_cb_buf",
                                                            scope=tbe_platform.scope_cbuf, data=output_cb)

                            output_w = width
                            output_h = tail_h
                            output_offset = \
                                n1*channel_1*height*width*channel_0 + channel_1*(h_loop*tiling_h)*output_w*channel_0

                            spr12 = (tvm.const(tail_h - 1, dtype="uint64")) | \
                                    (tvm.const(width - 1, dtype="uint64")) << 16
                            ib.emit(tvm.call_extern(output_dtype, "set_aipp_spr_12", spr12))

                            with ib.if_scope(boxes_reg[2] != height):
                                resize_input_h_stat_pos[0] = tvm.const(0, dtype="uint64")
                                resize_input_h_end_pos[0] = tvm.const(0, dtype="uint64")

                                resize_output_h_start_pos = tvm.const(h_loop*tiling_h, dtype="uint64")
                                resize_output_h_end_pos = tvm.const(height - 1, dtype="uint64")

                                resize_input_h_stat_pos[0] = (scf_inc_vscl[0]*resize_output_h_start_pos) >> 18
                                resize_input_h_end_pos[0] = ((scf_inc_vscl[0]*resize_output_h_end_pos) +
                                                             tvm.const((1 << 18) - 1, dtype="uint64")) >> 18

                                if input_format == "YUV420SP_U8":
                                    resize_input_h_stat_pos[0] = resize_input_h_stat_pos[0] & 0xfffffffffffffffe
                                    resize_input_h_end_pos[0] += (resize_input_h_end_pos[0] -
                                                                  resize_input_h_stat_pos[0] +
                                                                  tvm.const(1, dtype="uint64")) & 0x1

                                acc_vscl = (scf_inc_vscl[0]*resize_output_h_start_pos) - \
                                           (resize_input_h_stat_pos[0] << 18)
                                spr15[0] = acc_vscl

                                ib.emit(tvm.call_extern(output_dtype, "set_aipp_spr_15",
                                                        get_const(spr15[0])))

                                load_tail_h[0] = resize_input_h_end_pos[0] - resize_input_h_stat_pos[0]
                                tail_h_postion[0] = boxes_reg[0] + resize_input_h_stat_pos[0]
                            with ib.else_scope():
                                ib.emit(tvm.call_extern(output_dtype, "set_aipp_spr_15", tvm.const(0, dtype="uint64")))

                            aipp_xs = (boxes_reg[3] - 1) | load_tail_h[0] << load_h_pos | \
                                      tail_h_postion[0] << h_start_pos | (boxes_reg[1]) << w_start_pos

                            ib.emit(tvm.call_extern(output_dtype, "load_image_to_cbuf",
                                                    output_cb_buf.access_ptr("rw", ptr_type=output_dtype, offset=0),
                                                    get_const(aipp_xs),
                                                    get_const(aipp_xt)))

                            move_data_from_l1_to_gm(ib, channel_1*output_h*width*channel_0, output_dtype,
                                                    output_cb_buf, output_buf, block_index*offset+output_offset,
                                                    "NC1HWC0_C04")

        return ib.get()

    return tvm.extern([(batch, channel_1, height, width, channel_0)], [x, boxes, box_index],
                      lambda ins, outs: target_crop_and_resize_ir(ins[0], ins[1], ins[2], outs[0]),
                      dtype=[output_dtype], name="target_crop_and_resize")


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_INT, para_check.REQUIRED_ATTR_INT,
                            para_check.REQUIRED_ATTR_STR, para_check.KERNEL_NAME)
def target_crop_and_resize(x_dic, boxes_dic, box_index_dic, y_dic,
                           output_h, output_w, input_format,
                           kernel_name="target_crop_and_resize"):
    """
    Operation for target_crop_and_resize, only support Hi3796CV300CS.

    Parameters
    ----------
    x_dic: dict of input x, include shape and dtype, dtype support uint8
    boxes_dic: dict of input boxes, include shape and dtype, dtype support int32
    box_index_dic: dict of input box_index, include shape and dtype, dtype support int32
    y_dic: dict of output y, include shape and dtype, dtype support uint8, NC1HWC0_C04
    output_h : output height
    output_w : output width
    input_format: input format
    kernel_name : cce kernel name, default value is "target_crop_and_resize"

    Returns
    -------
    None
    """

    cur_cce_product = tbe_platform.get_soc_spec("SHORT_SOC_VERSION")
    if cur_cce_product not in ["Hi3796CV300CS", "SD3403"]:
        cause_desc = "TargetCropAndResize only support Hi3796CV300CS and SD3403"
        aipp_comm.raise_runtime_error(cause_desc)

    x_shape = x_dic.get('shape')
    x_dtype = x_dic.get('dtype')
    x_format = x_dic.get('format')
    output_dtype = y_dic.get('dtype')
    output_shape = y_dic.get('shape')
    output_format = y_dic.get('format')

    para_check.check_shape(x_shape, param_name="x")

    src_w = x_shape[2]
    if x_format == "NHWC":
        src_w = x_shape[2]
    elif x_format == "NCHW":
        src_w = x_shape[3]

    check_input_format(input_format)
    check_src_image_size_w(src_w, input_format)

    check_list = ["uint8"]
    para_check.check_dtype(x_dtype.lower(), check_list, param_name="x")
    para_check.check_dtype(output_dtype.lower(), check_list, param_name="y")

    format_list = ["NCHW", "NHWC"]
    para_check.check_format(x_format, format_list, param_name="x")

    format_list = ["NC1HWC0_C04"]
    para_check.check_format(output_format, format_list, param_name="y")

    batch, channel_1, height, width, channel_0 = output_shape

    if height != output_h:
        cause_desc = "height of output y's shape must be equal to output_h"
        aipp_comm.raise_runtime_error(cause_desc)

    if width != output_w:
        cause_desc = "width of output y's shape must be equal to output_w"
        aipp_comm.raise_runtime_error(cause_desc)

    boxes_shape = boxes_dic.get('shape')
    boxes_dtype = boxes_dic.get('dtype')

    box_index_shape = box_index_dic.get('shape')
    box_index_dtype = box_index_dic.get('dtype')

    check_list = ["int32"]
    para_check.check_dtype(boxes_dtype.lower(), check_list, param_name="boxes")
    para_check.check_dtype(box_index_dtype.lower(), check_list, param_name="box_index")

    if boxes_shape[0] != box_index_shape[0]:
        cause_desc = "boxes_shape[0] must be equal to box_index_shape[0]"
        aipp_comm.raise_runtime_error(cause_desc)

    if boxes_shape[0] != batch:
        cause_desc = "batch of output y's shape must be equal to boxes_shape[0]"
        aipp_comm.raise_runtime_error(cause_desc)

    if boxes_shape[1] != 4:
        cause_desc = "boxes_shape[1] must be equal to 4"
        aipp_comm.raise_runtime_error(cause_desc)

    x = tvm.placeholder(x_shape, name='x', dtype=x_dtype.lower())
    boxes = tvm.placeholder(boxes_shape, name='boxes', dtype=boxes_dtype)
    box_index = tvm.placeholder(box_index_shape, name='box_index', dtype=box_index_dtype)

    output = target_crop_and_resize_compute(x, boxes, box_index, x_dic, y_dic, input_format)

    # Schedule
    s = tvm.create_schedule([output.op])
    with build_config():
        tvm.build(s, [x, boxes, box_index, output], "cce", name=kernel_name)
