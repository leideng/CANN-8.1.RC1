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
trans_data_2d
"""
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import platform_info
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check


def _apply_mem(tik_instance, dtype, shape, name, scope=tbe_platform.scope_ubuf):
    """apply mem fuc

    Parameters
    ----------
    tik_instance: tik_instance
        tik_instance
    dtype: str
        ub dtype
    shape: list
        ub shape
    name: str
        ub name
    scope: scope
        scope_ubuf or scope_gm
    Returns
    -------
    Tensor: Tensor
    """
    return tik_instance.Tensor(dtype, shape, name=name, scope=scope)


def _get_ceil_int(int1, int2):
    _result = int1 // int2
    if int1 % int2 == 0:
        return _result
    return _result + 1


# 'pylint: disable=locally-disabled,too-many-instance-attributes
# 'pylint: disable=locally-disabled,too-few-public-methods
class TransData2D():
    """
       Function: use to finish Iou main functions
    """

    def __init__(self, src, dst, src_format, dst_format):
        """
        init MaxPoolWithargmax parameters

        Parameters
        ----------
        bboxes : TVM tensor
            the placeholder of bboxes
        gtboxes : TVM tensor
            the placeholder of gtboxes
        overlap : dict
            shape and dtype of overlap
            result shape is [m, n]
        mode :  str
            ('iou','iof')
            iou : the output is gtbox and bbox iou
            iof :

        Returns
        -------
        None
        """
        self.src_shape = src.get("shape")
        self.src_dtype = src.get("dtype").lower() if src.get("dtype") != "bfloat16" else "float16"
        self.src_format = src_format
        self.dst_shape = dst.get("shape")
        self.dst_dtype = dst.get("dtype").lower() if dst.get("dtype") != "bfloat16" else "float16"
        if self.dst_dtype == "bool":
            self.dst_dtype = "int8"
        self.dst_format = dst_format
        self.data_size = para_check.check_tensor_shape_size(list(self.dst_shape))
        # get dtype size, float16 size = 2 byte   / float32 size = 4 byte
        self.dtype_size = \
            tbe_platform.get_bit_len(self.src_dtype) // 8
        # get one block data size, block align len, 1 block = 16 fp16 and = 8 fp32
        self.data_len_one_bloack = 32 // self.dtype_size
        self.data_len_one_vector = self.data_len_one_bloack * 8

        self.ub_availble = \
            tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) - 8*1024
        self.ub_max_data = self.ub_availble // self.dtype_size

        self.tik_instance = tik.Tik()
        self.core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)

        # input and output tensor in gm
        self.src_gm = self.tik_instance.Tensor(
            self.src_dtype,
            self.src_shape,
            name="src_gm",
            scope=tbe_platform.scope_gm)
        self.dst_gm = self.tik_instance.Tensor(
            self.dst_dtype,
            self.dst_shape,
            name="dst_gm",
            scope=tbe_platform.scope_gm)
        self.data_ub = None

    def _transdata_2d_2_5d(self):
        # calcu for get core scedule
        dim_c_2d = self.src_shape[0]
        dim_c_2d = self.src_shape[1]
        pad_num = dim_c_2d % platform_info.C0_SIZE
        if pad_num == 0:
            # just copy in and copy out
            self._nopad_transdata()

    def _transdata_5d_2_2d(self):
        # calcu for get core scedule
        dim_c_2d = self.dst_shape[0]
        dim_c_2d = self.dst_shape[1]
        pad_num = dim_c_2d % platform_info.C0_SIZE
        if pad_num == 0:
            # just copy in and copy out
            self._nopad_transdata()

    def _nopad_transdata(self):
        # core scedule
        core_data = self.data_size // self.core_num
        if core_data == 0:
            core_data = 1
        core_data = \
            _get_ceil_int(core_data, self.data_len_one_bloack) * \
            self.data_len_one_bloack
        core_used = _get_ceil_int(self.data_size, core_data)
        core_last = self.data_size - (core_data * (core_used - 1))
        # calcu copy max segment
        copy_segment = self.ub_max_data // 2
        copy_segment = \
            (_get_ceil_int(copy_segment, self.data_len_one_bloack) - 1) * \
            self.data_len_one_bloack
        # core process
        with self.tik_instance.for_range(
                0, core_used, block_num=core_used) as core_index:
            with self.tik_instance.if_scope(core_index < (core_used - 1)):
                copy_loop = core_data // copy_segment
                copy_tail = core_data % copy_segment
                thread_num = 2
                if copy_loop < 2:
                    thread_num = 1
                with self.tik_instance.for_range(
                        0, copy_loop, thread_num=thread_num) as loop_index:
                    gm_in_offset = core_index*core_data + \
                                   loop_index*copy_segment
                    gm_out_offset = core_index*core_data + \
                                    loop_index*copy_segment
                    self._copy_in_and_out(copy_segment,
                                          gm_in_offset,
                                          gm_out_offset)
                if copy_tail != 0:
                    gm_in_offset = core_index*core_data + \
                                   copy_loop*copy_segment
                    gm_out_offset = core_index*core_data + \
                                    copy_loop*copy_segment
                    self._copy_in_and_out(copy_tail,
                                          gm_in_offset,
                                          gm_out_offset)
            with self.tik_instance.else_scope():
                copy_loop = core_last // copy_segment
                copy_tail = core_last % copy_segment
                thread_num = 2
                if copy_loop < 2:
                    thread_num = 1
                with self.tik_instance.for_range(
                        0, copy_loop, thread_num=thread_num) as loop_index:
                    gm_in_offset = core_index*core_data + \
                                   loop_index*copy_segment
                    gm_out_offset = core_index*core_data +\
                                    loop_index*copy_segment
                    self._copy_in_and_out(copy_segment,
                                          gm_in_offset, gm_out_offset)
                if copy_tail != 0:
                    gm_in_offset = core_index*core_data + \
                                   copy_loop*copy_segment
                    gm_out_offset = core_index*core_data + \
                                    copy_loop*copy_segment
                    self._copy_in_and_out(copy_tail,
                                          gm_in_offset, gm_out_offset)

    def _copy_in_and_out(self, copy_len, copy_in_offset, copy_out_offset):
        nbust = _get_ceil_int(copy_len, self.data_len_one_bloack)
        self.data_ub = _apply_mem(self.tik_instance, self.dst_dtype,
                                  [nbust*self.data_len_one_bloack],
                                  "data_ub")
        self.tik_instance.data_move(self.data_ub[0],
                                    self.src_gm[copy_in_offset],
                                    0, 1, nbust, 0, 0)
        self.tik_instance.data_move(self.dst_gm[copy_out_offset],
                                    self.data_ub[0],
                                    0, 1, nbust, 0, 0)

    def run_tik(self, kernel_name, mode):
        """
        cal tik_instance according to mode
        """
        if mode == "2to5":
            self._transdata_2d_2_5d()
        elif mode == "5to2":
            self._transdata_5d_2_2d()
        self.tik_instance.BuildCCE(
            kernel_name=kernel_name,
            inputs=[self.src_gm],
            outputs=[self.dst_gm])
        return self.tik_instance


@register_operator_compute("trans_data_2d", op_mode="static", support_fusion=True)
def trans_data_2d_compute(src, dst, src_format, dst_format,
                          kernel_name):
    """
    algorithm: format_transfer
    just 2d to 5d, and 5d to 2d

    Parameters
    ----------
    src : dict
        shape and dtype of input
    dst: dict
        shape and dtype of output, should be same shape and type as input
    src_format: str
        source data format, can be NHWC, NCHW, NC1HWC0 etc.
    dst_format: str
        target data format, can be NC1HWC0, NHWC, NCHW etc.
    kernel_name: str
        kernel name, default value is "format_transfer"

    Returns
    -------
    None
    """
    mode = ""
    if src_format == "NHWC":
        mode = "2to5"
        src["shape"] = [src["shape"][0], src["shape"][3]]
    elif src_format == "NCHW":
        mode = "2to5"
        src["shape"] = [src["shape"][0], src["shape"][1]]
    elif dst_format == "NHWC":
        mode = "5to2"
        dst["shape"] = [dst["shape"][0], dst["shape"][3]]
    elif dst_format == "NCHW":
        mode = "5to2"
        dst["shape"] = [dst["shape"][0], dst["shape"][1]]
    iou_res = TransData2D(src, dst, src_format, dst_format)

    return iou_res.run_tik(kernel_name, mode)


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_STR,
                            para_check.REQUIRED_ATTR_STR, para_check.KERNEL_NAME)
def trans_data_2d(src, dst, src_format, dst_format,
                  kernel_name):
    """
    algorithm: format_transfer
    just 2d to 5d, and 5d to 2d

    Parameters
    ----------
    src : dict
        shape and dtype of input
    dst: dict
        shape and dtype of output, should be same shape and type as input
    src_format: str
        source data format, can be NHWC, NCHW, FRACTAL_Zn etc.
    dst_format: str
        target data format, can be NC1HWC0, NCHW, FRACTAL_Zn etc.
    kernel_name: str
        kernel name, default value is "format_transfer"

    Returns
    -------
    None
    """
    res = trans_data_2d_compute(src, dst, src_format, dst_format, kernel_name)

    return res
