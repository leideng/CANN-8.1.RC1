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
aipp_stc_dyn
"""
# 'pylint: disable=invalid-name,too-many-locals,too-many-branches,too-many-statements
# 'pylint: disable=too-many-arguments,too-many-boolean-expressions,import-error,too-many-lines
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from tbe.dsl.instrinsic.cce_util import get_const
from impl.util.platform_adapter import tbe_build
from impl import aipp_comm


def _get_spr0(aipp_config, input_buf, param_list):
    spr0 = tvm.const(0, dtype="uint64")
    n1, offset, c_ori, src_image_size_h, src_image_size_w = param_list
    if aipp_config.get("input_format") == "YUV420SP_U8":
        spr0 = get_const(
            input_buf.access_ptr(
                "r",
                offset=offset + n1 * ((c_ori * src_image_size_h * src_image_size_w) // 2),
            )
        )
    elif aipp_config.get("input_format") in [
        "XRGB8888_U8",
        "RGB888_U8",
        "ARGB8888_U8",
        "AYUV444_U8",
        "YUV400_U8",
        "RGB16",
    ]:
        spr0 = get_const(
            input_buf.access_ptr("r", offset=offset + n1 * ((c_ori * src_image_size_h * src_image_size_w)))
        )
    elif aipp_config.get("input_format") in ["YUYV_U8", "YUV422SP_U8"]:
        spr0 = get_const(
            input_buf.access_ptr("r", offset=offset + n1 * ((2 * src_image_size_h * src_image_size_w)))
        )

    return spr0


def _get_spr1(aipp_config, input_buf, param_list):
    n1, offset, c_ori, src_image_size_h, src_image_size_w = param_list
    spr1 = tvm.const(0, dtype="uint64")
    if ("csc_switch" in aipp_config) and (aipp_config.get("csc_switch") == 1):
        spr1 = tvm.const(1 << 63, dtype="uint64")

    if aipp_config.get("input_format") == "YUV420SP_U8":
        spr1 = (
            get_const(
                input_buf.access_ptr(
                    "r",
                    offset=offset
                    + n1 * ((c_ori * src_image_size_h * src_image_size_w) // 2)
                    + src_image_size_h * src_image_size_w,
                )
            )
            | spr1
        )
    elif aipp_config.get("input_format") == "YUV422SP_U8":
        spr1 = (
            get_const(
                input_buf.access_ptr(
                    "r",
                    offset=offset
                    + n1 * ((2 * src_image_size_h * src_image_size_w))
                    + src_image_size_h * src_image_size_w,
                )
            )
            | spr1
        )

    return spr1


def _get_dyn_spr0_spr1(input_buf, param_list):
    (
        ib,
        input_format,
        batch_factor,
        n1,
        src_image_size,
        block_index,
        csc_switch,
    ) = param_list
    spr = ib.allocate("uint64", [2], name="spr", scope=tbe_platform.scope_reg)
    spr[0] = tvm.const(0, dtype="uint64")
    spr[1] = tvm.const(0, dtype="uint64")
    # YUV420SP_U8
    with ib.if_scope(input_format == 1):
        input_offset = batch_factor * ((3 * src_image_size[0] * src_image_size[1]) // 2)
        spr[0] = spr[0] | get_const(
            input_buf.access_ptr(
                "r",
                offset=block_index * input_offset + n1 * ((3 * src_image_size[0] * src_image_size[1]) // 2),
            )
        )
        uv_addr = get_const(
            input_buf.access_ptr(
                "r",
                offset=block_index * input_offset
                + n1 * ((3 * src_image_size[0] * src_image_size[1]) // 2)
                + src_image_size[0] * src_image_size[1],
            )
        )
        spr[1] = (csc_switch & 0x1) << 63 | uv_addr
    # XRGB8888_U8
    with ib.if_scope(input_format == 2):
        input_offset = batch_factor * ((4 * src_image_size[0] * src_image_size[1]))
        spr[0] = spr[0] | get_const(
            input_buf.access_ptr(
                "r",
                offset=block_index * input_offset + n1 * ((4 * src_image_size[0] * src_image_size[1])),
            )
        )
        spr[1] = (csc_switch & 0x1) << 63
    # RGB888_U8
    with ib.if_scope(input_format == 5):
        input_offset = batch_factor * ((3 * src_image_size[0] * src_image_size[1]))
        spr[0] = spr[0] | get_const(
            input_buf.access_ptr(
                "r",
                offset=block_index * input_offset + n1 * ((3 * src_image_size[0] * src_image_size[1])),
            )
        )
        spr[1] = (csc_switch & 0x1) << 63
    # ARGB8888
    with ib.if_scope(input_format == 6):
        input_offset = batch_factor * ((4 * src_image_size[0] * src_image_size[1]))
        spr[0] = spr[0] | get_const(
            input_buf.access_ptr(
                "r",
                offset=block_index * input_offset + n1 * ((4 * src_image_size[0] * src_image_size[1])),
            )
        )
        spr[1] = (csc_switch & 0x1) << 63
    # YUYV_U8
    with ib.if_scope(input_format == 7):
        input_offset = batch_factor * ((2 * src_image_size[0] * src_image_size[1]))
        spr[0] = spr[0] | get_const(
            input_buf.access_ptr(
                "r",
                offset=block_index * input_offset + n1 * ((2 * src_image_size[0] * src_image_size[1])),
            )
        )
        spr[1] = (csc_switch & 0x1) << 63
    # YUV422SP_U8
    with ib.if_scope(input_format == 8):
        input_offset = batch_factor * ((2 * src_image_size[0] * src_image_size[1]))
        spr[0] = spr[0] | get_const(
            input_buf.access_ptr(
                "r",
                offset=block_index * input_offset + n1 * ((2 * src_image_size[0] * src_image_size[1])),
            )
        )
        uv_addr = get_const(
            input_buf.access_ptr(
                "r",
                offset=block_index * input_offset
                + n1 * ((2 * src_image_size[0] * src_image_size[1]))
                + src_image_size[0] * src_image_size[1],
            )
        )
        spr[1] = (csc_switch & 0x1) << 63 | uv_addr
    # AYUV444_U8
    with ib.if_scope(input_format == 9):
        input_offset = batch_factor * ((2 * src_image_size[0] * src_image_size[1]))
        spr[0] = spr[0] | get_const(
            input_buf.access_ptr(
                "r",
                offset=block_index * input_offset + n1 * ((2 * src_image_size[0] * src_image_size[1])),
            )
        )
        spr[1] = (csc_switch & 0x1) << 63
    # YUV400_U8
    with ib.if_scope(input_format == 10):
        input_offset = batch_factor * ((src_image_size[0] * src_image_size[1]))
        spr[0] = spr[0] | get_const(
            input_buf.access_ptr(
                "r",
                offset=block_index * input_offset + n1 * ((src_image_size[0] * src_image_size[1])),
            )
        )
        spr[1] = tvm.const(0, dtype="uint64") << 63

    return spr


def _platform_instruction(len_burst):
    cur_platform = tbe_platform.get_soc_spec("SHORT_SOC_VERSION")
    instruction = "copy_cbuf_to_gm_align" if cur_platform in ["Ascend610Lite", "BS9SX2A", "MC61AM21A"] \
        else "copy_cbuf_to_gm"
    burst_len = 32 * len_burst if cur_platform in ["Ascend610Lite", "BS9SX2A", "MC61AM21A"] else len_burst
    return instruction, burst_len


def _padding_compute(ib, input_data, output_buf, l1_image_buf_max):
    """
    process top or bottom padding value
    """
    (
        dtype,
        out_w,
        c0,
        _,
        padding_size,
        offset,
        load_image_w,
        load_image_h,
        load_start_pos_w,
        load_start_pos_h,
        src_image_size_w,
    ) = input_data

    new_left_pad = ib.allocate("uint64", [1], name="new_left_pad", scope=tbe_platform.scope_reg)
    new_left_pad[0] = tvm.const(0, dtype="uint64")
    pad_limit = tvm.const(4095, dtype="uint64")
    w_const = tvm.const(out_w, dtype="uint64")
    with ib.if_scope(padding_size * w_const <= pad_limit):
        new_left_pad[0] = padding_size * w_const
    with ib.else_scope():
        new_left_pad[0] = pad_limit

    loop_times = padding_size * w_const // new_left_pad[0]
    tail = padding_size * w_const % new_left_pad[0]
    zero_const = tvm.const(0, dtype="uint64")

    with ib.new_scope():
        output_cb = ib.allocate(dtype, (l1_image_buf_max * c0,), "output_cb", scope=tbe_platform.scope_cbuf)
        output_cb_buf = tvm.decl_buffer(
            (l1_image_buf_max * c0,),
            dtype,
            "output_cb_buf",
            scope=tbe_platform.scope_cbuf,
            data=output_cb,
        )
        if isinstance(src_image_size_w, (int,)):
            src_image_size_w = tvm.const(src_image_size_w, dtype="uint64")

        aipp_xt = (src_image_size_w - 1) | (new_left_pad[0] & 0xFFF) << 32
        aipp_xm = (load_image_w - 1) | (load_image_h - 1) << 16 | load_start_pos_w << 32 | load_start_pos_h << 48

        if isinstance(aipp_xm, (int,)):
            aipp_xm = tvm.const(aipp_xm, dtype="uint64")

        ib.emit(
            tvm.call_extern(
                dtype,
                "load_image_to_cbuf",
                output_cb_buf.access_ptr("rw", ptr_type=dtype, offset=0),
                aipp_xm,
                aipp_xt,
            )
        )

        n_burst = 1
        instruction, len_burst = _platform_instruction(new_left_pad[0])
        with ib.for_range(zero_const, loop_times, name="loop_times", dtype="uint64") as time:
            output_offset = offset + time * new_left_pad[0] * c0
            ib.emit(
                tvm.call_extern(
                    dtype,
                    instruction,
                    output_buf.access_ptr("w", ptr_type=dtype, offset=output_offset),
                    output_cb_buf.access_ptr("rw", ptr_type=dtype, offset=0),
                    0,
                    n_burst,
                    len_burst,
                    0,
                    0,
                )
            )

        with ib.if_scope(tail > 0):
            output_offset = offset + loop_times * new_left_pad[0] * c0
            instruction, len_burst = _platform_instruction(tail)
            ib.emit(
                tvm.call_extern(
                    dtype,
                    instruction,
                    output_buf.access_ptr("w", ptr_type=dtype, offset=output_offset),
                    output_cb_buf.access_ptr("rw", ptr_type=dtype, offset=0),
                    0,
                    n_burst,
                    len_burst,
                    0,
                    0,
                )
            )


def _dynamic_aipp_compute(input_tensor, param_tensor, output_data, cur_cce_product):
    """
    :param input_tensor: tensor for input image
    :param param_tensor: tensor for dynamic aipp config params
    :param output_data: dict of output, include shape, format and dtype
    :param cur_cce_product: current soc version
    :return: tensor for aipp output image
    """

    output_shape = output_data.get("shape")
    out_n, c1, out_h, out_w, c0 = output_shape

    dtype = output_data.get("dtype")
    if dtype == "float16":
        size = 2
        c0 = 16
    else:
        size = 1
        c0 = 32

    if c1 != 1:
        cause_desc = "network output c1 should be equal to 1"
        aipp_comm.raise_runtime_error(cause_desc)

    l1_size = tbe_platform.get_soc_spec(tbe_platform.L1_SIZE)
    l1_image_buf_max = l1_size // size // c0

    def aipp_ir(input_buf, param_buf, output_buf):
        """
        :param input_buf: allocated buffer for aipp input image
        :param param_buf: allocated buffer for dynamic aipp configs
        :param output_buf: allocatedd buffer for aipp output image
        :return: result ir
        """
        ib = tvm.tir.ir_builder.create()

        device_core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        batch_num = out_n
        batch_factor = 1

        if batch_num % device_core_num == 0:
            batch_factor = batch_num // device_core_num

        block_index = tvm.thread_axis("blockIdx.x")
        ib.scope_attr(block_index, "thread_extent", batch_num // batch_factor)
        offset = batch_factor * c1 * out_h * out_w * c0
        zero_const = tvm.const(0, dtype="uint64")

        def _padding_top_bottom(spr, src_image_size, load_image_info, padding_info, top_offset, bottom_offset):
            # set single line mode
            spr9 = spr[9] | (1 & 0x1) << 24
            ib.emit(tvm.call_extern(dtype, "set_aipp_spr_9", spr9))
            # top_padding_size
            with ib.if_scope(padding_info[0] > 0):
                with ib.new_scope():
                    _padding_compute(
                        ib,
                        (
                            dtype,
                            out_w,
                            c0,
                            size,
                            padding_info[0],
                            top_offset,
                            load_image_info[3],
                            load_image_info[2],
                            load_image_info[1],
                            load_image_info[0],
                            src_image_size[1],
                        ),
                        output_buf,
                        l1_image_buf_max,
                    )
            # bottom_padding_size
            with ib.if_scope(padding_info[1] > 0):
                with ib.new_scope():
                    _padding_compute(
                        ib,
                        (
                            dtype,
                            out_w,
                            c0,
                            size,
                            padding_info[0],
                            bottom_offset,
                            load_image_info[3],
                            load_image_info[2],
                            load_image_info[1],
                            load_image_info[0],
                            src_image_size[1],
                        ),
                        output_buf,
                        l1_image_buf_max,
                    )

            ib.emit(tvm.call_extern(dtype, "set_aipp_spr_9", spr[9]))

        def _aipp_single_process(aipp_xt, aipp_xm, output_offset, len_burst):
            instruction, len_burst = _platform_instruction(len_burst)
            with ib.new_scope():
                output_cb = ib.allocate(
                    dtype,
                    (l1_image_buf_max * c0,),
                    "output_cb",
                    scope=tbe_platform.scope_cbuf,
                )
                output_cb_buf = tvm.decl_buffer(
                    (l1_image_buf_max * c0,),
                    dtype,
                    "output_cb_buf",
                    scope=tbe_platform.scope_cbuf,
                    data=output_cb,
                )
                ib.emit(
                    tvm.call_extern(
                        dtype,
                        "load_image_to_cbuf",
                        output_cb_buf.access_ptr("rw", ptr_type=dtype, offset=0),
                        aipp_xm,
                        aipp_xt,
                    )
                )
                ib.emit(
                    tvm.call_extern(
                        dtype,
                        instruction,
                        output_buf.access_ptr("w", ptr_type=dtype, offset=output_offset),
                        output_cb_buf.access_ptr("rw", ptr_type=dtype, offset=0),
                        0,
                        1,
                        len_burst,
                        0,
                        0,
                    )
                )

        def _aipp_intrin():
            # config SPR2~SPR9
            spr, tmp = aipp_comm.set_spr_dync_from_gm(ib, param_buf, dtype)

            src_image_size = ib.allocate("uint64", [2], name="src_image_size", scope=tbe_platform.scope_reg)
            src_image_size[0] = tvm.const(out_h, dtype="uint64")
            src_image_size[1] = tvm.const(out_w, dtype="uint64")
            aipp_comm.get_dync_src_image_size(ib, param_buf, tmp, src_image_size)

            load_image_info = ib.allocate("uint64", [4], name="load_image_info", scope=tbe_platform.scope_reg)

            crop = ib.allocate("uint64", [1], name="crop", scope=tbe_platform.scope_reg)
            crop[0] = tvm.const(0, dtype="uint64")

            padding_info = ib.allocate("uint64", [4], name="padding_info", scope=tbe_platform.scope_reg)

            padding = ib.allocate("uint64", [1], name="padding", scope=tbe_platform.scope_reg)
            padding[0] = tvm.const(0, dtype="uint64")

            input_format = ib.allocate("uint64", [1], name="input_format", scope=tbe_platform.scope_reg)
            input_format[0] = tvm.const(0, dtype="uint64")

            h_loop = ib.allocate("uint64", [1], name="h_loop", scope=tbe_platform.scope_reg)
            h_loop[0] = tvm.const(1, dtype="uint64")

            load_h = ib.allocate("uint64", [1], name="load_h", scope=tbe_platform.scope_reg)
            load_h[0] = tvm.const(0, dtype="uint64")

            tail_h = ib.allocate("uint64", [1], name="tail_h", scope=tbe_platform.scope_reg)
            tail_h[0] = tvm.const(0, dtype="uint64")

            tail_h_postion = ib.allocate("uint64", [1], name="tail_h_postion", scope=tbe_platform.scope_reg)
            tail_h_postion[0] = tvm.const(0, dtype="uint64")

            load_tail_h = ib.allocate("uint64", [1], name="load_tail_h", scope=tbe_platform.scope_reg)
            load_tail_h[0] = tvm.const(0, dtype="uint64")

            with ib.for_range(
                zero_const,
                tvm.const(batch_factor, dtype="uint64"),
                name="n1",
                dtype="uint64",
            ) as n1:
                batch_id = batch_factor * block_index + n1
                param_offset = aipp_comm.Const.DYNC_PARAM_HEAD_STRUCT_SIZE + \
                               aipp_comm.Const.DYNC_PARAM_BATCH_STRUCT_SIZE * batch_id

                # load_start_pos_h
                load_image_info[0] = tvm.const(0, dtype="uint64")
                # load_start_pos_w
                load_image_info[1] = tvm.const(0, dtype="uint64")
                # load_image_h
                load_image_info[2] = src_image_size[0]
                # load_image_w
                load_image_info[3] = src_image_size[1]

                # top_padding_size
                padding_info[0] = tvm.const(0, dtype="uint64")
                # bottom_padding_size
                padding_info[1] = tvm.const(0, dtype="uint64")
                # left_padding_size
                padding_info[2] = tvm.const(0, dtype="uint64")
                # right_padding_size
                padding_info[3] = tvm.const(0, dtype="uint64")

                ib.emit(
                    tvm.call_extern(
                        "int8",
                        "reg_mov",
                        tvm.call_extern("uint64", "reg", tmp[0]),
                        param_buf.access_ptr(
                            "r",
                            offset=param_offset + aipp_comm.Const.BATCH_OFFSET_CROP_SWITCH,
                        ),
                    )
                )
                crop[0] = tmp[0]
                # crop enable
                with ib.if_scope(crop[0] > 0):
                    aipp_comm.get_dync_crop_info(ib, param_buf, tmp, load_image_info, param_offset)

                ib.emit(
                    tvm.call_extern(
                        "uint8",
                        "reg_mov",
                        tvm.call_extern("uint64", "reg", tmp[0]),
                        param_buf.access_ptr("r", offset=aipp_comm.Const.HEAD_OFFSET_INPUT_FORMAT),
                    )
                )

                input_format[0] = tmp[0]
                ib.emit(
                    tvm.call_extern(
                        "int8",
                        "reg_mov",
                        tvm.call_extern("uint64", "reg", tmp[0]),
                        param_buf.access_ptr("r", offset=aipp_comm.Const.HEAD_OFFSET_CSC_SWITCH),
                    )
                )
                param_list = (
                    ib,
                    input_format[0],
                    batch_factor,
                    n1,
                    src_image_size,
                    block_index,
                    tmp[0],
                )
                spr0_1 = _get_dyn_spr0_spr1(input_buf, param_list)
                ib.emit(tvm.call_extern(dtype, "set_aipp_spr_0", spr0_1[0]))
                ib.emit(tvm.call_extern(dtype, "set_aipp_spr_1", spr0_1[1]))

                if cur_cce_product in aipp_comm.Const.V300_SOC_VERSION_LIST:
                    aipp_comm.set_spr_dync_in_batch_v300(ib, dtype, param_buf, spr, tmp, offset=param_offset)
                else:
                    aipp_comm.set_spr_dync_in_batch(ib, dtype, param_buf, spr, tmp, offset=param_offset)
                aipp_comm.get_dync_padding_size(ib, param_buf, tmp, padding_info, param_offset)

                support_vertical_padding = cur_cce_product in aipp_comm.Const.V300_SOC_VERSION_LIST
                actual_col_size_reg = ib.allocate("uint64", [1], name="actual_col_size_reg",
                    scope=tbe_platform.scope_reg)
                aipp_comm.get_dync_actual_col_size(out_h, out_w, padding_info, actual_col_size_reg,
                    support_vertical_padding)

                with ib.if_scope(l1_image_buf_max >= actual_col_size_reg[0]):
                    # +----+---------+---------------------------------+
                    # | Xt | [57:45] | right padding size              |
                    # | Xt | [44:32] | left padding size               |
                    # | Xt | [29:24] | bottom padding size             |
                    # | Xt | [21:16] | top padding size                |
                    # | Xt | [15:0]  | horizontal size of source image |
                    # +----+---------+---------------------------------+
                    # | Xm | [60:48] | start positon h of matted image |
                    # | Xm | [44:32] | start positon w of matted image |
                    # | Xm | [28:16] | height of matted image          |
                    # | Xm | [12:0]  | width of matted image           |
                    # +----+---------+---------------------------------+
                    if support_vertical_padding:
                        aipp_xt = get_const(
                            (src_image_size[1] - 1) |
                            (padding_info[0] & 0x1F) << 16 |
                            (padding_info[1] & 0x1F) << 24 |
                            (padding_info[2] & 0xFFF) << 32 |
                            (padding_info[3] & 0xFFF) << 45,
                        )

                        aipp_xm = get_const(
                            (load_image_info[3] - 1) |
                            (load_image_info[2] - 1) << 16 |
                            (load_image_info[1]) << 32 |
                            (load_image_info[0]) << 48
                        )

                        output_offset = block_index * offset + n1 * c1 * out_h * out_w * c0
                        len_burst = c1 * out_h * out_w
                        _aipp_single_process(aipp_xt, aipp_xm, output_offset, len_burst)
                    else:
                        top_offset = block_index * offset + n1 * c1 * out_h * out_w * c0
                        bottom_offset = (
                            block_index * offset
                            + n1 * c1 * out_h * out_w * c0
                            + (out_h - padding_info[1]) * out_w * c0
                        )
                        _padding_top_bottom(spr, src_image_size, load_image_info, padding_info, top_offset,
                                            bottom_offset)

                        aipp_xt = get_const(
                            (src_image_size[1] - 1) |
                            (padding_info[2] & 0xFFF) << 32 |
                            (padding_info[3] & 0xFFF) << 45
                        )

                        aipp_xm = get_const(
                            (load_image_info[3] - 1) |
                            (load_image_info[2] - 1) << 16 |
                            (load_image_info[1]) << 32 |
                            (load_image_info[0]) << 48
                        )

                        output_offset = \
                            block_index * offset + n1 * c1 * out_h * out_w * c0 + padding_info[0] * out_w * c0
                        len_burst = c1 * (out_h - padding_info[0] - padding_info[1]) * out_w
                        _aipp_single_process(aipp_xt, aipp_xm, output_offset, len_burst)

                with ib.else_scope():
                    tiling_h = l1_image_buf_max // out_w

                    h_loop[0] = tvm.const(1, dtype="uint64")
                    h_loop[0] = tvm.div(load_image_info[2], tvm.const(tiling_h, dtype="uint64"))

                    h_start_pos = tvm.const(48, dtype="uint64")
                    w_start_pos = tvm.const(32, dtype="uint64")
                    load_h_pos = tvm.const(16, dtype="uint64")
                    tiling_h_const = tvm.const(tiling_h, dtype="uint64")
                    load_w = get_const(load_image_info[3] - 1)
                    load_h[0] = tvm.const(tiling_h - 1, dtype="uint64")

                    with ib.for_range(zero_const, h_loop[0], name="h1", dtype="uint64") as h1:
                        aipp_xt = get_const(
                            (src_image_size[1] - 1) |
                            (padding_info[2] & 0xFFF) << 32 |
                            (padding_info[3] & 0xFFF) << 45
                        )

                        aipp_xm = get_const(
                            load_w |
                            load_h[0] << load_h_pos |
                            load_image_info[1] << w_start_pos |
                            (load_image_info[0] + h1 * tiling_h_const) << h_start_pos
                        )

                        output_offset = (
                            block_index * offset
                            + padding_info[0] * out_w * c0
                            + n1 * c1 * out_h * out_w * c0
                            + c1 * (h1 * tiling_h) * out_w * c0
                        )

                        len_burst = tiling_h * out_w
                        _aipp_single_process(aipp_xt, aipp_xm, output_offset, len_burst)

                    tail_h[0] = load_image_info[2] % tvm.const(tiling_h, dtype="uint64")
                    with ib.if_scope(tail_h[0] != 0):
                        tail_h_postion[0] = load_image_info[0] + h_loop[0] * tvm.const(tiling_h, dtype="uint64")
                        load_tail_h[0] = tail_h[0] - tvm.const(1, dtype="uint64")

                        aipp_xt = get_const(
                            (src_image_size[1] - 1) |
                            (padding_info[2] & 0xFFF) << 32 |
                            (padding_info[3] & 0xFFF) << 45
                        )

                        aipp_xm = get_const(
                            load_w |
                            load_tail_h[0] << load_h_pos |
                            tail_h_postion[0] << h_start_pos |
                            (load_image_info[1]) << w_start_pos
                        )

                        output_offset = (
                            block_index * offset
                            + padding_info[0] * out_w * c0
                            + n1 * c1 * out_h * out_w * c0
                            + c1 * (h_loop[0] * tiling_h) * out_w * c0
                        )

                        len_burst = tail_h[0] * out_w
                        _aipp_single_process(aipp_xt, aipp_xm, output_offset, len_burst)

        _aipp_intrin()
        return ib.get()

    return tvm.extern(
        [(out_n, c1, out_h, out_w, c0)],
        [input_tensor, param_tensor],
        lambda ins, outs: aipp_ir(ins[0], ins[1], outs[0]),
        dtype=[dtype],
        name="aipp",
    )


def _static_aipp_compute(data, input_shape, input_format, output_data, aipp_config, cur_cce_product):
    """
    :param data: input tensor of image data
    :param input_shape: shape of input tensor
    :param input_format: format of input tensor
    :param output_data: dict of output, include shape, format and dtype
    :param aipp_config: config dict of aipp
    :param cur_cce_product: current soc version
    :return: output tensor
    """
    para_check.check_format(input_format, ("NHWC", "NCHW"), param_name="input")

    c_ori = input_shape[input_format.index("C")]
    h_ori = input_shape[input_format.index("H")]
    w_ori = input_shape[input_format.index("W")]
    output_shape = output_data.get("shape")
    out_n, c1, out_h, out_w, c0 = output_shape

    src_image_size_h = h_ori
    src_image_size_w = w_ori
    load_image_h = h_ori
    load_image_w = w_ori
    load_start_pos_h = 0
    load_start_pos_w = 0

    output_format = output_data.get("format")

    (
        src_image_size_h,
        src_image_size_w,
        load_start_pos_h,
        load_start_pos_w,
        load_image_h,
        load_image_w,
    ) = aipp_comm.get_crop_info(aipp_config, h_ori, w_ori)

    dtype = output_data.get("dtype")
    if dtype == "float16":
        size = 2
        c0 = 16
    else:
        size = 1
        c0 = 32

    if c1 != 1:
        cause_desc = "network output c1 should be equal to 1"
        aipp_comm.raise_runtime_error(cause_desc)

    def aipp_ir(input_buf, output_buf):
        """
        :param input_buf: allocated buffer for input image
        :param output_buf: allocated buffer for aipp output image
        :return: result ir
        """
        ib = tvm.tir.ir_builder.create()

        device_core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        batch_num = out_n
        batch_factor = 1

        support_vertical_padding = cur_cce_product in aipp_comm.Const.V300_SOC_VERSION_LIST
        actual_col_size = aipp_comm.get_actual_col_size(aipp_config, out_h, out_w, support_vertical_padding)
        l1_size = tbe_platform.get_soc_spec(tbe_platform.L1_SIZE)
        l1_image_buf_max = l1_size // size // c0

        if batch_num % device_core_num == 0:
            batch_factor = batch_num // device_core_num

        block_index = tvm.thread_axis("blockIdx.x")
        ib.scope_attr(block_index, "thread_extent", batch_num // batch_factor)

        if aipp_config.get("input_format") == "YUV420SP_U8":
            input_offset = batch_factor * ((c_ori * src_image_size_h * src_image_size_w) // 2)
        elif aipp_config.get("input_format") in [
            "XRGB8888_U8",
            "RGB888_U8",
            "ARGB8888_U8",
            "AYUV444_U8",
            "YUV400_U8",
            "RGB16",
        ]:
            input_offset = batch_factor * ((c_ori * src_image_size_h * src_image_size_w))
        elif aipp_config.get("input_format") in ["YUYV_U8", "YUV422SP_U8"]:
            input_offset = batch_factor * ((2 * src_image_size_h * src_image_size_w))

        offset = batch_factor * c1 * out_h * out_w * c0

        def _padding_top_bottom(top_padding_size, bottom_padding_size, top_offset, bottom_offset):
            spr9 = aipp_comm.get_spr9(aipp_config, dtype, output_format)
            # set single line mode
            spr9 = spr9 | (1 & 0x1) << 24
            ib.emit(tvm.call_extern(dtype, "set_aipp_spr_9", tvm.const(spr9, dtype="uint64")))
            if top_padding_size > 0:
                with ib.new_scope():
                    top_padding_size = tvm.const(top_padding_size, dtype="uint64")
                    _padding_compute(
                        ib,
                        (
                            dtype,
                            out_w,
                            c0,
                            size,
                            top_padding_size,
                            top_offset,
                            load_image_w,
                            load_image_h,
                            load_start_pos_w,
                            load_start_pos_h,
                            src_image_size_w,
                        ),
                        output_buf,
                        l1_image_buf_max,
                    )
            if bottom_padding_size > 0:
                with ib.new_scope():
                    bottom_padding_size = tvm.const(bottom_padding_size, dtype="uint64")
                    _padding_compute(
                        ib,
                        (
                            dtype,
                            out_w,
                            c0,
                            size,
                            bottom_padding_size,
                            bottom_offset,
                            load_image_w,
                            load_image_h,
                            load_start_pos_w,
                            load_start_pos_h,
                            src_image_size_w,
                        ),
                        output_buf,
                        l1_image_buf_max,
                    )

            spr9 = aipp_comm.get_spr9(aipp_config, dtype, output_format)
            ib.emit(tvm.call_extern(dtype, "set_aipp_spr_9", tvm.const(spr9, dtype="uint64")))

        def _aipp_single_process(aipp_xt, aipp_xm, output_offset, len_burst):
            instruction, len_burst = _platform_instruction(len_burst)
            with ib.new_scope():
                output_cb = ib.allocate(
                    dtype,
                    (l1_image_buf_max * c0,),
                    "output_cb",
                    scope=tbe_platform.scope_cbuf,
                )
                output_cb_buf = tvm.decl_buffer(
                    (l1_image_buf_max * c0,),
                    dtype,
                    "output_cb_buf",
                    scope=tbe_platform.scope_cbuf,
                    data=output_cb,
                )
                ib.emit(
                    tvm.call_extern(
                        dtype,
                        "load_image_to_cbuf",
                        output_cb_buf.access_ptr("rw", ptr_type=dtype, offset=0),
                        aipp_xm,
                        aipp_xt,
                    )
                )
                ib.emit(
                    tvm.call_extern(
                        dtype,
                        instruction,
                        output_buf.access_ptr("w", ptr_type=dtype, offset=output_offset),
                        output_cb_buf.access_ptr("rw", ptr_type=dtype, offset=0),
                        0,
                        1,
                        len_burst,
                        0,
                        0,
                    )
                )

        def _aipp_intrin():
            # set spr2~spr9
            aipp_comm.set_spr2_spr9(ib, aipp_config, dtype, cur_cce_product, output_format)
            # set spr18~spr21
            if cur_cce_product in aipp_comm.Const.V300_SOC_VERSION_LIST:
                aipp_comm.set_spr18_spr21(ib, aipp_config, dtype, output_format)

            horizontal_pad_mask = 0XFFF
            verical_pad_mask = 0x1F

            with ib.for_range(
                tvm.const(0, dtype="uint64"), tvm.const(batch_factor, dtype="uint64"), name="n1", dtype="uint64"
            ) as n1:
                param_list = (
                    n1,
                    block_index * input_offset,
                    c_ori,
                    src_image_size_h,
                    src_image_size_w,
                )
                spr0 = _get_spr0(aipp_config, input_buf, param_list)

                param_list = (
                    n1,
                    block_index * input_offset,
                    c_ori,
                    src_image_size_h,
                    src_image_size_w,
                )
                spr1 = _get_spr1(aipp_config, input_buf, param_list)

                ib.emit(tvm.call_extern(dtype, "set_aipp_spr_0", spr0))
                ib.emit(tvm.call_extern(dtype, "set_aipp_spr_1", spr1))

                (
                    left_padding_size,
                    right_padding_size,
                    top_padding_size,
                    bottom_padding_size,
                ) = aipp_comm.get_padding_size(aipp_config)

                if l1_image_buf_max >= actual_col_size:
                    # +----+---------+---------------------------------+
                    # | Xt | [57:45] | right padding size              |
                    # | Xt | [44:32] | left padding size               |
                    # | Xt | [29:24] | bottom padding size             |
                    # | Xt | [21:16] | top padding size                |
                    # | Xt | [15:0]  | horizontal size of source image |
                    # +----+---------+---------------------------------+
                    # | Xm | [60:48] | start positon h of matted image |
                    # | Xm | [44:32] | start positon w of matted image |
                    # | Xm | [28:16] | height of matted image          |
                    # | Xm | [12:0]  | width of matted image           |
                    # +----+---------+---------------------------------+
                    if support_vertical_padding:
                        aipp_xt = tvm.const(
                            (src_image_size_w - 1) |
                            (left_padding_size & horizontal_pad_mask) << 32 |
                            (right_padding_size & horizontal_pad_mask) << 45 |
                            (top_padding_size & verical_pad_mask) << 16 |
                            (bottom_padding_size & verical_pad_mask) << 24,
                            dtype="uint64"
                        )

                        aipp_xm = tvm.const(
                            (load_image_w - 1) |
                            (load_image_h - 1) << 16 |
                            (load_start_pos_w) << 32 |
                            (load_start_pos_h) << 48,
                            dtype="uint64"
                        )

                        output_offset = block_index * offset + n1 * c1 * out_h * out_w * c0
                        len_burst = c1 * out_h * out_w
                        _aipp_single_process(aipp_xt, aipp_xm, output_offset, len_burst)
                    else:
                        top_offset = block_index * offset + n1 * c1 * out_h * out_w * c0
                        bottom_offset = (
                            block_index * offset
                            + n1 * c1 * out_h * out_w * c0
                            + (out_h - bottom_padding_size) * out_w * c0
                        )
                        _padding_top_bottom(top_padding_size, bottom_padding_size, top_offset, bottom_offset)

                        aipp_xt = tvm.const(
                            (src_image_size_w - 1) |
                            (left_padding_size & horizontal_pad_mask) << 32 |
                            (right_padding_size & horizontal_pad_mask) << 45,
                            dtype="uint64"
                        )

                        aipp_xm = tvm.const(
                            (load_image_w - 1) |
                            (load_image_h - 1) << 16 |
                            (load_start_pos_w) << 32 |
                            (load_start_pos_h) << 48,
                            dtype="uint64"
                        )

                        output_offset = block_index * offset + n1 * c1 * out_h * out_w * c0 \
                            + top_padding_size * out_w * c0
                        len_burst = c1 * (out_h - top_padding_size - bottom_padding_size) * out_w
                        _aipp_single_process(aipp_xt, aipp_xm, output_offset, len_burst)
                else:
                    top_offset = block_index * offset + n1 * c1 * out_h * out_w * c0
                    bottom_offset = (
                        block_index * offset
                        + n1 * c1 * out_h * out_w * c0
                        + (out_h - bottom_padding_size) * out_w * c0
                    )
                    _padding_top_bottom(top_padding_size, bottom_padding_size, top_offset, bottom_offset)

                    tiling_h = l1_image_buf_max // out_w
                    h_loop = load_image_h // tiling_h

                    h_start_pos = tvm.const(48, dtype="uint64")
                    w_start_pos = tvm.const(32, dtype="uint64")
                    load_h_pos = tvm.const(16, dtype="uint64")
                    tiling_h_const = tvm.const(tiling_h, dtype="uint64")
                    load_w = tvm.const(load_image_w - 1, dtype="uint64")
                    load_h = tvm.const(tiling_h - 1, dtype="uint64")
                    zero_const = tvm.const(0, dtype="uint64")
                    h_loop_const = tvm.const(h_loop, dtype="uint64")

                    with ib.for_range(zero_const, h_loop_const, name="h1", dtype="uint64") as h1:
                        aipp_xt = tvm.const(
                            (src_image_size_w - 1) |
                            (left_padding_size & horizontal_pad_mask) << 32 |
                            (right_padding_size & horizontal_pad_mask) << 45,
                            dtype="uint64"
                        )

                        aipp_xm = get_const(
                            load_w |
                            load_h << load_h_pos |
                            tvm.const(load_start_pos_w, dtype="uint64") << w_start_pos |
                            (load_start_pos_h + h1 * tiling_h_const) << h_start_pos
                        )

                        output_offset = (
                            block_index * offset
                            + n1 * c1 * out_h * out_w * c0
                            + c1 * (h1 * tiling_h) * out_w * c0
                            + top_padding_size * out_w * c0
                        )

                        len_burst = tiling_h * out_w
                        _aipp_single_process(aipp_xt, aipp_xm, output_offset, len_burst)

                    tail_h = load_image_h % tiling_h
                    if tail_h != 0:
                        tail_h_postion = tvm.const(load_start_pos_h + h_loop * tiling_h, dtype="uint64")
                        load_tail_h = tvm.const(tail_h - 1, dtype="uint64")

                        aipp_xt = tvm.const(
                            (src_image_size_w - 1) |
                            (left_padding_size & horizontal_pad_mask) << 32 |
                            (right_padding_size & horizontal_pad_mask) << 45,
                            dtype="uint64"
                        )

                        aipp_xm = get_const(
                            load_w |
                            load_tail_h << load_h_pos |
                            tail_h_postion << h_start_pos |
                            tvm.const(load_start_pos_w, dtype="uint64") << w_start_pos
                        )
                        aipp_xm = tvm.const(aipp_xm, dtype="uint64")

                        output_offset = (
                            block_index * offset
                            + top_padding_size * out_w * c0
                            + n1 * c1 * out_h * out_w * c0
                            + c1 * (h_loop_const * tiling_h) * out_w * c0
                        )

                        len_burst = tail_h * out_w
                        _aipp_single_process(aipp_xt, aipp_xm, output_offset, len_burst)

        _aipp_intrin()
        return ib.get()

    return tvm.extern(
        [(out_n, c1, out_h, out_w, c0)], [data], lambda ins, outs: aipp_ir(ins[0], outs[0]), dtype=[dtype], name="aipp"
    )


def new_aipp_compute(input_data, input_dync_param, output_data, aipp_config, cur_cce_product, kernel_name="aipp"):
    """
    Parameters
    ----------
    input_data: dict of input, include shape and dtype, dtype support uint8
    input_dync_param: dict of dynamic parameter, include shape and dtype, dtype support uint8
    output_data:  dict of output, include shape and dtype, dtype support uint8, int8 and float16
    aipp_config : dict of aipp config
    cur_cce_product: current soc version
    kernel_name : cce kernel name, default value is "aipp"
    Returns
    -------
        None
    """
    input_shape = input_data.get("shape")
    input_dtype = input_data.get("dtype")
    input_format = input_data.get("format")
    output_dtype = output_data.get("dtype")
    output_format = output_data.get("format")
    if output_format != "NC1HWC0":
        cause_desc = "aipp single op only support NC1HWC0 output_format"
        aipp_comm.raise_runtime_error(cause_desc)

    aipp_mode = aipp_config.get("aipp_mode")
    if aipp_mode == "dynamic":
        input_dync_param_shape = input_dync_param.get("shape")
        input_dync_param_dtype = input_dync_param.get("dtype")
        para_check.check_shape(input_dync_param_shape, param_name="input_dync_param")
        para_check.check_dtype(input_dync_param_dtype, ["uint8"], param_name="input_dync_param_dtype")

        # Compute
        data = tvm.placeholder(input_shape, name="input", dtype=input_dtype.lower())
        param = tvm.placeholder(input_dync_param_shape, name="param", dtype=input_dync_param_dtype)
        output = _dynamic_aipp_compute(data, param, output_data, cur_cce_product)
        tensor_list = [data, param, output]
    else:
        aipp_comm.set_aipp_default_params(aipp_config)
        aipp_comm.check_aipp_static_config(input_shape, input_format, output_data, aipp_config, cur_cce_product)
        aipp_comm.check_aipp_dtype(aipp_config, input_dtype, output_dtype)

        # Compute
        data = tvm.placeholder(input_shape, name="input", dtype=input_dtype.lower())
        output = _static_aipp_compute(data, input_shape, input_format, \
            output_data, aipp_config, cur_cce_product)
        tensor_list = [data, output]

    # Schedule
    s = tvm.create_schedule([output.op])
    with tbe_build.build_config():
        tvm.build(s, tensor_list, "cce", name=kernel_name)
