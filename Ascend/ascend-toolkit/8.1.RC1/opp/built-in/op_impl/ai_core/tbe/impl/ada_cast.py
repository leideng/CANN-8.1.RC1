#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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
ada_cast
"""

import functools
from te import tik
from te.utils import para_check

from tbe.common.utils import errormgr


def check_supported(x, y, kernel_name="ada_cast"):
    """
    x : shape: [1, 512, 512, 3], dtype: uint16, 
    y : shape: [1, 512, 512, 3], dtype: float16,
    """
    # HDRnet网络定制算子，仅支持shape [1, 512, 512, 3] tensor
    x_shape = x.get("shape")
    need_support_shape_list = [(1, 512, 512, 3)]
    if x_shape not in need_support_shape_list:
        reason = "Ascend610Lite HDRnet network custom operator, only supports shape [1, 512, 512, 3] tensor"
        return False, reason
    return True, ""


class AdaCast:
    
    def __init__(self, x, y, pixel):
        """
        Init AdaCast parameters

        Returns
        -------
        None
        """
        self.x_shape = x.get("shape")
        self.x_type = x.get("dtype")
        self.x_format = x.get("format")
        self.x_elements_num = functools.reduce(lambda x, y: x * y, self.x_shape)
        self.pixel = pixel
        self.pixel_quotient = 1 / self.pixel

        self.y_shape = y.get("shape")
        self.y_type = y.get("dtype")

        self.input_x_gm = None
        self.output_y_gm = None
        self.x_block_num = None
        self.input_x_ub = None
        self.output_y_ub = None 
        self.src_v_uint16 = None
        self.dst_v_float16_even_tmp = None
        self.dst_v_float16_even_dintly = None
        self.dst_v_float16_even_dintly_tmp = None
        self.dst_v_float16 = None
        self.tik_instance = tik.Tik()   

    def init_gm_mem(self):
        """
        init tik gm mem
        """
        # init gm input
        self.input_x_gm = self.tik_instance.Tensor(self.x_type, self.x_shape, tik.scope_gm, "input_x_gm")
        # init gm output
        self.output_y_gm = self.tik_instance.Tensor(self.y_type, self.y_shape, tik.scope_gm, "output_y_gm")   

    def data_move_in(self, w, j):
        """
        data_move_in
        """
        # load 128个uit16数
        self.src_v_uint16 = self.tik_instance.Vector("uint16")
        self.tik_instance.vector_load(self.src_v_uint16, self.input_x_ub[w * 24576 + j * 128])

    def computer_even(self):
        """
        computer_even
        """
        # 对奇数位进行处理
        dst_v_uint32_even = self.tik_instance.Vector("uint32")
        dst_v_uint32_even_ub = self.tik_instance.Tensor("uint32", \
            (64,), tik.scope_ubuf, "dst_v_uint32_even_ub")
        dst_v_int32_even = self.tik_instance.Vector("int32")
        dst_v_float32_even = self.tik_instance.Vector("float32")
        dst_v_float32_even_mul = self.tik_instance.Vector("float32")
        dst_v_float16_even = self.tik_instance.Vector("float16")
        self.dst_v_float16_even_tmp = self.tik_instance.Vector("float16")
        self.dst_v_float16_even_dintly = self.tik_instance.Vector("float16")
        self.dst_v_float16_even_dintly_tmp = self.tik_instance.Vector("float16")
        # u16-u32-s32
        self.tik_instance.vector_cast(None, dst_v_uint32_even, self.src_v_uint16, \
            saturate_flag=False, part_indicator="PART_EVEN", round_mode="")
        self.tik_instance.vector_store(dst_v_uint32_even_ub, dst_v_uint32_even)
        dst_v_int32_even_ub = dst_v_uint32_even_ub.reinterpret_cast_to("int32")
        self.tik_instance.vector_load(dst_v_int32_even, dst_v_int32_even_ub)
        # s32-fp32
        self.tik_instance.vector_cast(None, dst_v_float32_even, dst_v_int32_even, \
            saturate_flag=False, part_indicator=None, round_mode="round")
        # muls 
        self.tik_instance.vector_vmul(None, dst_v_float32_even_mul, \
            dst_v_float32_even, self.pixel_quotient)
        # f32-fp16
        self.tik_instance.vector_cast(None, dst_v_float16_even, dst_v_float32_even_mul, \
            saturate_flag=True, part_indicator="PART_EVEN", round_mode="round")
        # dintly
        self.tik_instance.vector_vdintlv(self.dst_v_float16_even_dintly, \
            self.dst_v_float16_even_dintly_tmp, dst_v_float16_even, self.dst_v_float16_even_tmp)

    def computer_odd(self):
        """
        computer_odd
        """
        # 对偶数位进行处理
        dst_v_uint32_odd = self.tik_instance.Vector("uint32")
        dst_v_uint32_odd_ub = self.tik_instance.Tensor("uint32", \
            (64,), tik.scope_ubuf, "dst_v_uint32_odd_ub")
        dst_v_int32_odd = self.tik_instance.Vector("int32")
        dst_v_float32_odd = self.tik_instance.Vector("float32")
        dst_v_float32_odd_mul = self.tik_instance.Vector("float32")
        dst_v_float16_odd = self.tik_instance.Vector("float16")
        dst_v_float16_odd_dintly = self.tik_instance.Vector("float16")
        # u16-u32-s32
        self.tik_instance.vector_cast(None, dst_v_uint32_odd, self.src_v_uint16, \
            saturate_flag=False, part_indicator="PART_ODD", round_mode="")
        self.tik_instance.vector_store(dst_v_uint32_odd_ub, dst_v_uint32_odd)
        dst_v_int32_odd_ub = dst_v_uint32_odd_ub.reinterpret_cast_to("int32")
        self.tik_instance.vector_load(dst_v_int32_odd, dst_v_int32_odd_ub)
        # s32-fp32
        self.tik_instance.vector_cast(None, dst_v_float32_odd, dst_v_int32_odd, \
            saturate_flag=False, part_indicator=None, round_mode="round")
        # muls
        self.tik_instance.vector_vmul(None, dst_v_float32_odd_mul, \
            dst_v_float32_odd, self.pixel_quotient)
        # fp32-fp16
        self.tik_instance.vector_cast(None, dst_v_float16_odd, dst_v_float32_odd_mul, \
            saturate_flag=True, part_indicator="PART_EVEN", round_mode="round")
        # dintly
        self.tik_instance.vector_vdintlv(dst_v_float16_odd_dintly, \
            self.dst_v_float16_even_dintly_tmp, dst_v_float16_odd, self.dst_v_float16_even_tmp)

        # intlv
        self.dst_v_float16 = self.tik_instance.Vector("float16")
        self.tik_instance.vector_vintlv(self.dst_v_float16, self.dst_v_float16_even_tmp, \
            self.dst_v_float16_even_dintly, dst_v_float16_odd_dintly)
    
    def data_move_out(self, blockidx, i, w):
        """
        data_move_out
        """
        # store
        self.tik_instance.data_move(self.output_y_gm[blockidx * 98304 + i * 4 * 98304 + w * 24576], \
            self.output_y_ub, 0, 1, 1536, 0, 0)

    def ada_cast_computer(self):
        """
        cast: U16-->U32-->S32-->FP32-->FP16
        """
        # 计算每次分核分块处理数据量
        for_num = 2
        self.x_block_num = int(self.x_elements_num / (for_num * 4))

        # 分核处理
        with self.tik_instance.for_range(0, 4, name="blockidx", block_num=4) as blockidx:
            self.input_x_ub = self.tik_instance.Tensor(self.x_type, (self.x_block_num,), tik.scope_ubuf, "input_x_ub")
            with self.tik_instance.for_range(0, for_num, name="i") as i:
                self.tik_instance.data_move(self.input_x_ub, \
                    self.input_x_gm[blockidx * 98304 + i * 4 * 98304], 0, 1, 6144, 0, 0)
                with self.tik_instance.for_range(0, 4, name="w") as w:
                    self.output_y_ub = self.tik_instance.Tensor(self.y_type, (24576,), tik.scope_ubuf, "output_y_ub")
                    with self.tik_instance.for_range(0, 192, name="j", thread_num=2) as j:
                        self.data_move_in(w, j)
                        self.computer_even()
                        self.computer_odd()
                        self.tik_instance.vector_store(self.output_y_ub[j * 128], self.dst_v_float16)
                    self.data_move_out(blockidx, i, w)                    

    def build_tik_instance(self, kernel_name_value):
        """
        build tik instance
        """
        self.tik_instance.BuildCCE(kernel_name=kernel_name_value,
                                inputs=[self.input_x_gm],
                                outputs=[self.output_y_gm],
                                output_files_path=None,
                                enable_l2=False)

        return self.tik_instance                


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_INT, para_check.KERNEL_NAME)
def ada_cast(x, y, pixel, kernel_name="ada_cast"):
    """
    do ada_cast

    Parameters:
    ----------
    x : dict.
        dict info of x value, must include the keys(shape and dtype).
        and shape will be NHWC
    kernel_name : str.
        cce kernel name, default value is "ada_cast"

    Returns
    -------
    tik_instance
    """
    if pixel <= 0:
        errormgr.raise_err_input_value_invalid(op_name="AdaCast", \
            param_name="pixel", excepted_value="positive integer", real_value=pixel)
    ada_cast_obj = AdaCast(x, y, pixel)
    ada_cast_obj.init_gm_mem()
    ada_cast_obj.ada_cast_computer()
    tik_instance = ada_cast_obj.build_tik_instance(kernel_name)

    return tik_instance