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
ascend_vec_executor.py
"""
from . import _check as a_check
from . import _internal_lib as a_lib
from . import ascend_container as a_container
from . import ascend_tensor_operator_param as a_param
from . import ascend_vec_cmd as a_cmd


# 'pylint: disable=too-many-arguments, too-many-instance-attributes, too-few-public-methods
class VecExecutor(a_lib.ObjWithConst):
    """
    executor
    """
    @staticmethod
    def exec_vec_cmd(bufs, cmds, drive_buf_name):
        """
        功能描述：在bufs上执行一系列的Vector指令（gather模式）
        输入 bufs:               dict type, {str: a_param.TensorOperatorParam}
        输入 cmds:               list type, [a_cmd.VecCmd]
        输入 drive_buf_name:     str
        """
        a_check.check_dict(bufs, str, a_param.TensorOperatorParam, "error bufs")
        a_check.check_list_tuple(cmds, a_cmd.VecCmd, "error cmds")
        drive_buf = bufs.get(drive_buf_name)
        if drive_buf is None:
            raise ValueError("error drive_buf_name {}".format(drive_buf_name))
        for cmd in cmds:
            if cmd.dst_name not in bufs:
                raise ValueError("error dst_name {}".format(cmd.dst_name))

        if drive_buf.is_proc_num_tik_dynamic():
            VecExecutor._exec_cmd_tik(bufs, cmds, drive_buf)
        else:
            VecExecutor._exec_cmd_python(bufs, cmds, drive_buf)

    @staticmethod
    def _exec_cmd_python(bufs, cmds, drive_buf):
        tinst = a_container.AContainer.get_instance().tinst
        loop_info = drive_buf.calc_loop_info_for_cmd()
        num_per_cmd = loop_info.num_per_cmd
        max_cmd_rpt = loop_info.max_cmd_rpt
        loop = loop_info.loop
        repeat = loop_info.repeat
        left = loop_info.left
        if loop > 0:
            with tinst.for_range(0, loop) as loop_i:
                executed_cmd_cnt = loop_i * max_cmd_rpt
                VecExecutor._build_cmd(bufs, cmds, num_per_cmd,
                                       max_cmd_rpt, False, executed_cmd_cnt)
        if repeat > 0:
            executed_cmd_cnt = loop * max_cmd_rpt
            VecExecutor._build_cmd(bufs, cmds, num_per_cmd,
                                   repeat, False, executed_cmd_cnt)

        if left > 0:
            executed_cmd_cnt = loop * max_cmd_rpt + repeat
            VecExecutor._build_cmd(bufs, cmds, left,
                                   1, True, executed_cmd_cnt)

    @staticmethod
    def _exec_cmd_tik(bufs, cmds, drive_buf):
        tinst = a_container.AContainer.get_instance().tinst
        loop_info = drive_buf.calc_loop_info_for_cmd()
        num_per_cmd = loop_info.num_per_cmd
        max_cmd_rpt = loop_info.max_cmd_rpt
        loop = loop_info.loop
        repeat = loop_info.repeat
        left = loop_info.left
        sca_max_rpt = tinst.Scalar("int32", name="max_rpt", init_value=max_cmd_rpt)
        with tinst.if_scope(loop > 0):
            with tinst.for_range(0, loop) as loop_i:
                executed_cmd_cnt = loop_i * max_cmd_rpt
                VecExecutor._build_cmd(bufs, cmds, num_per_cmd,
                                       sca_max_rpt, False, executed_cmd_cnt)
        with tinst.else_scope():
            pass

        with tinst.if_scope(repeat > 0):
            executed_cmd_cnt = loop * max_cmd_rpt
            VecExecutor._build_cmd(bufs, cmds, num_per_cmd,
                                   repeat, False, executed_cmd_cnt)
        with tinst.else_scope():
            pass

        with tinst.if_scope(left > 0):
            executed_cmd_cnt = loop * max_cmd_rpt + repeat
            VecExecutor._build_cmd(bufs, cmds, left,
                                   1, True, executed_cmd_cnt)
        with tinst.else_scope():
            pass

    @staticmethod
    def _build_cmd(bufs, cmds, mask, repeat, b_run_left,
                   executed_cmd_cnt):
        for cmd in cmds:
            dst_info = VecExecutor._get_buf_info(bufs.get(cmd.dst_name),
                                                 executed_cmd_cnt, b_run_left)
            src0_info = VecExecutor._get_buf_info(bufs.get(cmd.src0_name),
                                                  executed_cmd_cnt, b_run_left)
            src1_info = VecExecutor._get_buf_info(bufs.get(cmd.src1_name),
                                                  executed_cmd_cnt, b_run_left)
            cmd.run(mask, repeat, dst_info, src0_info, src1_info)

    @staticmethod
    def _get_buf_info(buf, executed_cmd_cnt, b_run_left):
        if buf is None:
            return a_lib.VecBufInfo(None, None, None)

        shift = buf.calc_proced_num_shift(executed_cmd_cnt)
        buf_addr = buf.get_buf_addr(shift)
        blk_stride, rpt_stride = buf.calc_stride_for_cmd()
        if b_run_left:
            rpt_stride = 0
        return a_lib.VecBufInfo(buf_addr, blk_stride, rpt_stride)
