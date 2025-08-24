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
ascend_vec_cmd.py
"""

from .ascend_container import AContainer
from ._internal_types import VecGatherCmdType as VGCT
from ._internal_lib import ObjWithConst
from ._internal_lib import VecBufInfo
from . import _check as G_CHECK


# 'pylint: disable=too-many-arguments, too-many-instance-attributes, too-few-public-methods
class _BaseVecCmdRunner(ObjWithConst):
    def __init__(self, cmd_name, mask, repeat,
                 dst_info, src0_info, src1_info):
        self.cmd_name = cmd_name
        self.mask = mask
        self.repeat = repeat
        self.dst_info = dst_info
        self.src0_info = src0_info
        self.src1_info = src1_info

    def run(self, kwargs):
        """
        pass
        """
        pass


class _SingleRunner(_BaseVecCmdRunner):
    def run(self, kwargs):
        """
        single run
        """
        cmd_func = AContainer.get_instance().get_vec_cmd_func(self.cmd_name)
        return cmd_func(self.mask, self.dst_info.addr, self.src0_info.addr, self.repeat,
                        self.dst_info.blk_stride, self.src0_info.blk_stride,
                        self.dst_info.rpt_stride, self.src0_info.rpt_stride)


class _MultiRunner(_BaseVecCmdRunner):
    def run(self, kwargs):
        """
        multi run
        """
        cmd_func = AContainer.get_instance().get_vec_cmd_func(self.cmd_name)
        return cmd_func(self.mask, self.dst_info.addr, self.src0_info.addr, self.src1_info.addr, self.repeat,
                        self.dst_info.blk_stride, self.src0_info.blk_stride, self.src1_info.blk_stride,
                        self.dst_info.rpt_stride, self.src0_info.rpt_stride, self.src1_info.rpt_stride)


class _ScalarMultiRunner(_BaseVecCmdRunner):
    def run(self, kwargs):
        """
        scalar run
        """
        cmd_func = AContainer.get_instance().get_vec_cmd_func(self.cmd_name)
        return cmd_func(self.mask, self.dst_info.addr, self.src0_info.addr, kwargs.get("scalar"),
                        self.repeat, self.dst_info.blk_stride, self.src0_info.blk_stride,
                        self.dst_info.rpt_stride, self.src0_info.rpt_stride)


class _ConvRunner(_BaseVecCmdRunner):
    def run(self, kwargs):
        """
        conv run
        """
        cmd_func = AContainer.get_instance().get_vec_cmd_func(self.cmd_name)
        return cmd_func(self.mask, kwargs.get("round_mode"), self.dst_info.addr, self.src0_info.addr,
                        self.repeat, self.dst_info.blk_stride, self.src0_info.blk_stride,
                        self.dst_info.rpt_stride, self.src0_info.rpt_stride, kwargs.get("deqscale"))


class _InitRunner(_BaseVecCmdRunner):
    def run(self, kwargs):
        """
        init runner
        """
        cmd_func = AContainer.get_instance().get_vec_cmd_func(self.cmd_name)
        return cmd_func(self.mask, self.dst_info.addr, kwargs.get("scalar"), self.repeat,
                        self.dst_info.blk_stride, self.dst_info.rpt_stride)


class _RunnerFactory(ObjWithConst):
    _runner_map = {VGCT.SINGLE: _SingleRunner,
                   VGCT.DBL_TRI: _MultiRunner,
                   VGCT.SCA_DBL_TRI: _ScalarMultiRunner,
                   VGCT.CONV: _ConvRunner,
                   VGCT.INIT: _InitRunner}

    @classmethod
    def get_runner(cls, cmd_name, mask, repeat, dst_info,
                   src0_info, src1_info):
        """
        get runner
        """
        cmd_type = AContainer.get_instance().get_vec_cmd_type(cmd_name)
        runner_cls = cls._runner_map.get(cmd_type)
        runner = runner_cls(cmd_name, mask, repeat, dst_info, src0_info, src1_info)
        return runner


class VecCmd(ObjWithConst):
    """
    vector cmd
    """
    def __init__(self, cmd_name, dst_name, src0_name="", src1_name="",
                 dbg_print=None, **kwargs):
        """
        vector cmd
        """
        self._check_in_params(cmd_name, dst_name, src0_name, src1_name,
                              dbg_print)
        self.cmd_name = cmd_name
        self.dst_name = dst_name
        self.src0_name = src0_name
        self.src1_name = src1_name
        self.dbg_print = dbg_print
        self.kwargs = kwargs

    def run(self, mask, repeat, dst_info, src0_info, src1_info):
        """
        run vector cmd
        """
        self._call_dbg_print(True, mask, repeat, dst_info, src0_info, src1_info)
        runner = _RunnerFactory.get_runner(self.cmd_name, mask, repeat, dst_info, src0_info, src1_info)
        runner.run(self.kwargs)
        self._call_dbg_print(False, mask, repeat, dst_info, src0_info, src1_info)

    def _call_dbg_print(self, b_pre_run, mask, repeat,
                        dst_info, src0_info, src1_info):
        """
        print debug
        """
        if self.dbg_print is not None:
            tinst = AContainer.get_instance().tinst
            self.dbg_print(tinst, b_pre_run, dst_info.addr, src0_info.addr, src1_info.addr,
                           self.cmd_name, self.dst_name, repeat, mask, self.kwargs)

    @staticmethod
    def _check_in_params(cmd_name, dst_name, src0_name, src1_name, dbg_print):
        """
        check params
        """
        G_CHECK.check_param_type(cmd_name, str, "error cmd_name")
        G_CHECK.check_param_type(dst_name, str, "error dst_name")
        G_CHECK.check_param_type(src0_name, str, "error src0_name")
        G_CHECK.check_param_type(src1_name, str, "error src1_name")
        if dbg_print is not None:
            G_CHECK.check_func(dbg_print, "dbg_print is not a function")

    def __str__(self):
        msg = "cmd:\t{}(dst={}".format(self.cmd_name, self.dst_name)
        if self.src0_name != "":
            msg += ", src0={}".format(self.src0_name)
        if self.src1_name != "":
            msg += ", src1={}".format(self.src1_name)
        if self.kwargs.get("scalar") is not None:
            msg += ", scalar={}".format(self.kwargs.get("scalar"))
        if self.kwargs.get("round_mode") is not None:
            msg += ", round_mode={}".format(self.kwargs.get("round_mode"))
        msg += ")"
        return msg
