# -*- coding: UTF-8 -*-
# Copyright 2021 Huawei Technologies Co., Ltd
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
ascend_tensor_operator_param.py
"""

import numpy as np

from . import _check as a_check
from . import _internal_lib as a_lib
from . import ascend_container as a_container


class TensorOperatorParam(a_lib.ObjWithConst):
    """
    the params of tensor Operator
    """
    # 'pylint: disable=too-many-arguments
    def __init__(self, tensor, proc_num, offset, debug=False,
                 num_per_cmd=None, num_stride_blk=None, num_stride_cmd=None):
        """
        tensor api
        :param tensor: tensor
        :param proc_num: process num
        :param offset: offset
        :param debug: debug
        :param num_per_cmd: number per cmd
        :param num_stride_blk: number stride
        :param num_stride_cmd: number stride
        """
        a_check.check_param_type(debug, bool, "error debug type")
        self.debug = debug
        a_check.set_tik_debug_flag(self.debug)
        self._check_proc_num(proc_num, a_container.AContainer.get_instance())
        self._check_offset(offset, a_container.AContainer.get_instance())

        self.const_tensor = tensor
        self._proc_num = None
        self._offset = None
        self.set_proc_info(proc_num, offset)

        dtype = self.const_tensor.dtype
        self.const_max_num_per_cmd = a_container.AContainer.get_instance().get_vec_proc_num_per_cmd(dtype)
        self.const_num_per_blk = a_container.AContainer.get_instance().get_vec_proc_num_per_cmd_blk(dtype)
        self.set_proc_num_per_cmd(num_per_cmd, False)
        self.set_num_stride(num_stride_blk, num_stride_cmd, True)

    def set_proc_info(self, proc_num, offset, b_check=False):
        """
        set info
        """
        tik = a_container.AContainer.get_instance().tik
        if (a_check.is_tik_dynamic(proc_num, tik) or
            a_check.is_tik_dynamic(offset, tik)):
            self._set_proc_info_tik(proc_num, offset)
        else:
            self._set_proc_info_python(proc_num, offset)
        self._check_num_infos(b_check)

    def _set_proc_info_python(self, proc_num, offset):
        """
        set python info
        """
        total_num = np.product(self.const_tensor.shape)
        if total_num >= (proc_num + offset):
            self._proc_num = proc_num
            self._offset = offset
        else:
            raise RuntimeError("invalid prof info:", proc_num, offset)

    def _set_proc_info_tik(self, proc_num, offset):
        """
        set tik info
        """
        self._proc_num = proc_num
        self._offset = offset
        if self.debug:
            total_num = np.product(self.const_tensor.shape)
            proc_end = proc_num + offset
            tinst = a_container.AContainer.get_instance().tinst
            with tinst.if_scope(proc_end > total_num):
                tinst.tik_return()
            with tinst.else_scope():
                pass

    def set_proc_num_per_cmd(self, num_per_cmd, b_check=True):
        """
        set number
        """
        if num_per_cmd is not None:
            self._num_per_cmd = num_per_cmd
        else:
            self._num_per_cmd = self.const_max_num_per_cmd

        self._check_num_infos(b_check)

    def set_num_stride(self, num_stride_blk, num_stride_cmd, b_check=True):
        """
        set number
        """
        if num_stride_blk is None:
            self._num_stride_blk = self.const_num_per_blk
        else:
            self._num_stride_blk = num_stride_blk

        if num_stride_cmd is None:
            self._num_stride_cmd = self._num_per_cmd
        else:
            self._num_stride_cmd = num_stride_cmd

        self._check_num_infos(b_check)

    def calc_stride_for_cmd(self):
        """
        calculate stride
        """
        blk_stride = a_container.AContainer.get_instance().calc_block_num(
            self.const_tensor.dtype, self._num_stride_blk)
        rpt_stride = a_container.AContainer.get_instance().calc_block_num(
            self.const_tensor.dtype, self._num_stride_cmd)
        return blk_stride, rpt_stride

    def calc_proced_num_shift(self, exec_times):
        """
        calculate shift
        """
        return exec_times * self._num_stride_cmd

    def calc_loop_info_for_cmd(self):
        """
        calculate loop info
        """
        max_cmd_rpt = a_container.AContainer.get_instance().const_vector_proc_max_rpt
        max_proc_num = max_cmd_rpt * self._num_per_cmd
        loop = self._proc_num // max_proc_num
        loop_proc = loop * max_proc_num
        repeat = (self._proc_num - loop_proc) // self._num_per_cmd
        left = self._proc_num - loop_proc - repeat * self._num_per_cmd
        return a_lib.VecLoopInfo(self._num_per_cmd, max_cmd_rpt, loop, repeat, left)

    def is_proc_num_tik_dynamic(self):
        """
        judge dynamic
        """
        return a_check.is_tik_dynamic(self._proc_num, a_container.AContainer.get_instance().tik)

    def get_buf_addr(self, shift):
        """
        get buffer address
        """
        tik = a_container.AContainer.get_instance().tik
        tinst = a_container.AContainer.get_instance().tinst
        if a_check.is_tik_dynamic(shift, tik):
            err_str = '"get_buf_addr invalid shift:"+str(param)'
            a_check.check_tik_param_dtype(shift, ("int32",), tik)
            a_check.check_tik_param_low(shift, tik, tinst, 0, err_str)
            a_check.check_tik_param_high(shift, tik, tinst,
                                         self._proc_num, err_str)
            a_check.check_tik_param_not_equal(shift, tik, tinst,
                                              self._proc_num, err_str)
        else:
            a_check.check_param_type(shift, int, "error shift type")
            a_check.check_param_low(shift, 0, "invalid shift")
            a_check.check_param_high(shift, self._proc_num, "invalid shift")
            a_check.check_param_not_equal(shift, self._proc_num,
                                          "invalid shift {}".format(shift))
        return self.const_tensor[self._offset + shift]

    @staticmethod
    def _check_proc_num(proc_num, container):
        """
        check process number
        """
        tik = container.tik
        tinst = container.tinst
        if a_check.is_tik_dynamic(proc_num, tik):
            err_str = '"_check_proc_num invalid proc_num:"+str(param)'
            a_check.check_tik_param_dtype(proc_num, ("int32",), tik)
            a_check.check_tik_param_low(proc_num, tik, tinst, 0, err_str)
            a_check.check_tik_param_not_equal(proc_num, tik, tinst, 0, err_str)
        else:
            a_check.check_param_type(proc_num, int, "error proc_num")
            a_check.check_param_low(proc_num, 0, "invalid proc_num")
            a_check.check_param_not_equal(proc_num, 0, "invalid proc_num")

    @staticmethod
    def _check_offset(offset, container):
        """
        check offset
        """
        tik = container.tik
        tinst = container.tinst
        if a_check.is_tik_dynamic(offset, tik):
            err_str = '"_check_offset invalid offset:"+str(param)'
            a_check.check_tik_param_dtype(offset, ("int32",), tik)
            a_check.check_tik_param_low(offset, tik, tinst, 0, err_str)
        else:
            a_check.check_param_type(offset, int, "error offset")
            a_check.check_param_low(offset, 0, "invalid offset")

    def _check_num_infos(self, b_check):
        """
        check number
        """
        if not b_check:
            return

        a_check.check_param_type(self._num_per_cmd, int,
                                  "error num_per_cmd type")
        a_check.check_param_low(self._num_per_cmd, 0, "error num_per_cmd low")
        a_check.check_param_not_equal(self._num_per_cmd, 0, "err num_per_cmd")
        a_check.check_param_high(self._num_per_cmd, self.const_max_num_per_cmd,
                                 "error num_per_cmd")

        a_check.check_param_type(self._num_stride_blk, int,
                                  "error num_stride_blk type")
        a_check.check_param_low(self._num_stride_blk, 0, "err num_stride_blk")
        a_check.check_param_mod(self._num_stride_blk, self.const_num_per_blk,
                                "error num_stride_blk mod num_per_blk")

        a_check.check_param_type(self._num_stride_cmd, int,
                                 "error num_stride_cmd type")
        a_check.check_param_low(self._num_stride_cmd, 0, "err num_stride_cmd")
        a_check.check_param_mod(self._num_stride_cmd, self.const_num_per_blk,
                                "error num_stride_cmd % num_per_cmd")

        if self._num_stride_blk != 0:
            a_check.check_param_mod(self._num_stride_cmd, self._num_stride_blk,
                                    "error num_stride_cmd % num_stride_blk")
