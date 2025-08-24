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
trans_argb
"""
import functools
from te import tik
from te.utils import para_check


class TransArgb:

    def __init__(self, x, y):
        """
        Init TransArgb parameters

        Returns
        -------
        None
        """
        self.x_shape = x.get("shape")
        self.x_type = x.get("dtype")
        self.x_format = x.get("format")
        self.x_elements_num = functools.reduce(lambda x, y: x * y, self.x_shape)

        self.y_shape = y.get("shape")
        self.y_type = y.get("dtype")

        self.input_x_gm = None
        self.output_y_gm = None
        self.input_x_one_row_shape = None
        self.input_x_row_ub = None
        self.output_y_row_ub = None

        self.tik_instance = tik.Tik()
    
    def get_tik_instance(self):
        """get_tik_instance
        """
        return self.tik_instance
    
    def init_gm_mem(self):
        """init tik gm mem
        """
        # init gm input
        self.input_x_gm = self.tik_instance.Tensor(self.x_type, self.x_shape, tik.scope_gm, "input_x_gm")

        # init gm output
        self.output_y_gm = self.tik_instance.Tensor(self.y_type, self.y_shape, tik.scope_gm, "output_y_gm")

    def data_move_in(self):
        """
        data_move_in
        """
        self.input_x_one_row_shape = self.x_shape[1] * self.x_shape[2] * self.x_shape[4]
        self.input_x_row_ub = self.tik_instance.Tensor(self.x_type, \
            (16, self.input_x_one_row_shape), tik.scope_ubuf, "input_x_row_ub")
        # 循环搬入
        with self.tik_instance.for_range(0, 16, thread_num=0) as i:
            self.tik_instance.data_move(self.input_x_row_ub[1920 * i], self.input_x_gm[16 * i], 0, 120, 1, 15, 0)
        
    def computer_transpose(self):
        """
        transpose (vnchwconv, ub_to_ub)
        """
        # 1、开启新的作用域，进行tranpose
        with self.tik_instance.new_stmt_scope(disable_sync=False):
            input_x_col_ub = self.tik_instance.Tensor(self.x_type, \
                (self.input_x_one_row_shape, 16), tik.scope_ubuf, "input_x_col_ub")
            input_x_col_trans_ub = self.tik_instance.Tensor(self.x_type, \
                (self.input_x_one_row_shape, 16), tik.scope_ubuf, "input_x_col_trans_ub")
            # 1.1、首次转置-vnchwconv
            first_dst_high_half = False
            first_src_high_half = False
            first_dst_list = [input_x_col_ub[16 * j] for j in range(16)]
            first_src_list = [self.input_x_row_ub[1920 * j] for j in range(16)]
            first_repeat_time = 120
            # 源操作数相邻两次迭代直接间隔16个操作数，源操作数和目的操作数第一次迭代与第二次迭代直接间隔512B
            first_dst_rep_stride = 16
            first_src_rep_stride = 1
            self.tik_instance.vnchwconv(first_dst_high_half, first_src_high_half, first_dst_list, \
                first_src_list, first_repeat_time, first_dst_rep_stride, first_src_rep_stride)
            
            # 1.2、ub to ub
            with self.tik_instance.for_range(0, 16) as p:
                with self.tik_instance.for_range(0, 10) as q:
                    self.tik_instance.data_move(input_x_col_trans_ub[((p * 120) + (q * 12)) * 16], \
                        input_x_col_ub[(p + (q * 16)) * 16], 0, 12, 1, 159, 0)
            
            # 1.3、第二次转置-vnchwconv
            second_dst_high_half = False
            second_src_high_half = False
            second_dst_list = [self.input_x_row_ub[1920 * j] for j in range(16)]
            second_src_list = [input_x_col_trans_ub[16 * j] for j in range(16)]
            second_repeat_time = 120
            second_dst_rep_stride = 1
            second_src_rep_stride = 16
            self.tik_instance.vnchwconv(second_dst_high_half, second_src_high_half, \
                second_dst_list, second_src_list, second_repeat_time, second_dst_rep_stride, second_src_rep_stride)

    def computer_cast(self):
        """
        cast
        """
        self.output_y_row_ub = self.tik_instance.Tensor(self.y_type, \
            (16, self.input_x_one_row_shape), tik.scope_ubuf, "output_y_row_ub")
        self.tik_instance.vconv(128, "round", self.output_y_row_ub, self.input_x_row_ub, 240, 1, 1, 8, 8)

    def data_move_out(self):
        """
        data_move_out
        """
        self.tik_instance.data_move(self.output_y_gm, self.output_y_row_ub, 0, 1, 1920, 0, 0)

    def trans_argb_computer(self):
        """
        transpose (vnchwconv, ub_to_ub)+ cast
        """
        self.data_move_in()
        self.computer_transpose()
        self.computer_cast()
        self.data_move_out()
        
    def build_tik_instance(self, kernel_name_value):
        """build tik instance
        """
        self.tik_instance.BuildCCE(kernel_name=kernel_name_value,
                                   inputs=[self.input_x_gm],
                                   outputs=[self.output_y_gm],
                                   output_files_path=None,
                                   enable_l2=False)

        return self.tik_instance
    

def check_supported(x, y, kernel_name="trans_argb"):
    """
    x : shape: [1, 12, 10, 16, 16], dtype: float16, 
    y : shape: [1, 16, 16, 10, 12], dtype: int16,
    """
    # HDRnet网络定制算子，仅支持shape [1, 12, 10, 16, 16] tensor
    x_shape = x.get("shape")
    need_support_shape_list = [(1, 12, 10, 16, 16)]
    if x_shape not in need_support_shape_list:
        reason = "Ascend610Lite HDRnet network custom operator, only supports shape [1, 12, 10, 16, 16] tensor"
        return False, reason
    return True, ""


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def trans_argb(x, y, kernel_name="trans_argb"):
    """
    do trans_argb

    Parameters:
    ----------
    x : dict.
        dict info of x value, must include the keys(shape and dtype).
        and shape will be ND
    kernel_name : str.
        cce kernel name, default value is "trans_argb"

    Returns
    -------
    tik_instance
    """
    # init object for TransArgb
    trans_argb_obj = TransArgb(x, y)
    # init gm ub
    trans_argb_obj.init_gm_mem()
    trans_argb_obj.trans_argb_computer()
    tik_instance = trans_argb_obj.build_tik_instance(kernel_name)

    return tik_instance