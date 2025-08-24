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
ascend_container.py
"""

from __future__ import annotations
import importlib

from . import _check as a_check
from . import _internal_lib as a_lib
from . import _internal_types as a_types


class AContainer(a_lib.ObjWithConst):
    """
    container object
    """
    _instance = None

    @classmethod
    def get_instance(cls):
        """
        get instance
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self._init_tik_module()
        self._init_const_value()
        self._init_vector_cmds()

    def get_vec_proc_num_per_cmd(self, dtype):
        """
        get vector process number
        """

        elm_byte = self.const_dtype_byte.get(dtype)
        if elm_byte is not None:
            return self.const_vector_proc_byte // elm_byte
        raise ValueError("not supported dtype:{}".format(dtype))

    def get_vec_proc_num_per_cmd_blk(self, dtype):
        """
        get vector process number per block
        """
        elm_byte = self.const_dtype_byte.get(dtype)
        if elm_byte is not None:
            return self.const_block_byte // elm_byte
        raise ValueError("not supported dtype:{}".format(dtype))

    def calc_block_num(self, dtype, proc_num):
        """
        calculate block number
        """
        a_check.check_param_type(dtype, str, "error dtype type")
        if a_check.is_tik_dynamic(proc_num, self.tik):
            return self._calc_block_num_tik(dtype, proc_num)
        return self._calc_block_num_python(dtype, proc_num)

    def _calc_block_num_python(self, dtype, proc_num):
        a_check.check_param_type(proc_num, int, "error proc_num type")
        a_check.check_param_low(proc_num, 0, "error proc_num")
        elm_byte = self.const_dtype_byte.get(dtype)
        if elm_byte is not None:
            proc_byte = proc_num * elm_byte
            a_check.check_param_mod(proc_byte, self.const_block_byte,
                                    "error num {} calc blk".format(proc_num))
            return proc_byte // self.const_block_byte
        raise ValueError("not supported dtype:{}".format(dtype))

    def _calc_block_num_tik(self, dtype, proc_num):
        a_check.check_tik_param_dtype(proc_num, ("int32",), self.tik)
        a_check.check_tik_param_low(proc_num, self.tik, self.tinst,
                                    0, '"error proc_num"+str(param)')
        elm_byte = self.const_dtype_byte.get(dtype)
        if elm_byte is not None:
            proc_byte = proc_num * elm_byte
            a_check.check_tik_param_mod(proc_byte, self.tik, self.tinst,
                                        self.const_block_byte,
                                        '"error proc_byte:"+str(param)')
            return proc_byte // self.const_block_byte
        raise ValueError("not supported dtype:{}".format(dtype))

    def calc_blk_align_num(self, dtype, cur_num):
        """
        calculate block number
        """
        a_check.check_param_type(dtype, str, "error dtype type")
        if a_check.is_tik_dynamic(cur_num, self.tik):
            a_check.check_tik_param_dtype(cur_num, ("int32",), self.tik)
            a_check.check_tik_param_low(cur_num, self.tik, self.tinst,
                                        0, '"error cur_num"+str(param)')
        else:
            a_check.check_param_type(cur_num, int, "error cur_num type")
            a_check.check_param_low(cur_num, 0, "error cur_num")
        elm_byte = self.const_dtype_byte.get(dtype)
        if elm_byte is not None:
            align_unit = self.const_block_byte // elm_byte
            return ((cur_num + align_unit - 1) // align_unit) * align_unit
        raise ValueError("not supported dtype:{}".format(dtype))

    # 'pylint: disable=no-self-use
    def get_c0_num(self, dtype):
        """
        get c0
        """
        if dtype in ("float32", "int32", "uint32", "int16",
                     "uint16", "float16"):
            c0_num = 16
        elif dtype in ("uint8", "int8"):
            c0_num = 32
        else:
            raise RuntimeError("not supported dtype:{}".format(dtype))
        return c0_num

    def get_tensor_type(self):
        """
        get tensor type
        """
        if hasattr(self.tik, "ir_builder_lib"):
            return self.tik.ir_builder_lib.ib_Tensor
        if hasattr(self.tik, "api"):
            return self.tik.api.tik_Tensor
        raise RuntimeError("not supported tik version")

    def get_vec_cmd_type(self, cmd_name):
        """
        get type of vector cmd
        """
        if cmd_name in self.const_vec_cmds.keys():
            return self.const_vec_cmds.get(cmd_name)[1]
        raise ValueError("not supported cmd_name:{}".format(cmd_name))

    def get_vec_cmd_func(self, cmd_name):
        """
        get function of vector cmd
        """
        if cmd_name in self.const_vec_cmds.keys():
            return self.const_vec_cmds.get(cmd_name)[0]
        raise ValueError("not supported cmd_name:{}".format(cmd_name))

    def _init_tik_module(self):
        self.tik = importlib.import_module("tbe.tik")
        self.plat = importlib.import_module("te.platform")
        self.tinst = self.tik.Tik(self.tik.Dprofile())

        get_f = self.plat.get_soc_spec
        # 8kb for scalar in ub
        self.const_ub_max_byte = get_f(self.plat.cce_conf.UB_SIZE) - 8192
        self.const_l1_max_byte = get_f(self.plat.cce_conf.L1_SIZE)
        self.const_l0a_max_byte = get_f(self.plat.cce_conf.L0A_SIZE)
        self.const_l0b_max_byte = get_f(self.plat.cce_conf.L0B_SIZE)
        self.const_l0c_max_byte = get_f(self.plat.cce_conf.L0C_SIZE)
        self.const_aicore_num = get_f(self.plat.cce_conf.CORE_NUM)

    def _init_const_value(self):
        self.const_dtype_byte = {"float16": 2, "float32": 4, "int32": 4,
                                 "uint8": 1, "uint16": 2, "int64": 8,
                                 "int8": 1, "int16": 2}
        self.const_block_byte = 32
        self.const_vector_proc_max_rpt = 255
        self.const_vector_proc_byte = 256
        self.const_proposal_data_num = 8
        self.const_proposal_repeat_num = 16

    def _init_vector_base_cmds(self):
        single_cmds = {"vrec": (self.tinst.vrec, a_types.VecGatherCmdType.SINGLE),
                       "vexp": (self.tinst.vexp, a_types.VecGatherCmdType.SINGLE),
                       "vabs": (self.tinst.vabs, a_types.VecGatherCmdType.SINGLE),
                       "vsqrt": (self.tinst.vsqrt, a_types.VecGatherCmdType.SINGLE)}

        dbl_tri_cmds = {"vadd": (self.tinst.vadd, a_types.VecGatherCmdType.DBL_TRI),
                        "vsub": (self.tinst.vsub, a_types.VecGatherCmdType.DBL_TRI),
                        "vmul": (self.tinst.vmul, a_types.VecGatherCmdType.DBL_TRI),
                        "vmin": (self.tinst.vmin, a_types.VecGatherCmdType.DBL_TRI),
                        "vmax": (self.tinst.vmax, a_types.VecGatherCmdType.DBL_TRI),
                        "vmla": (self.tinst.vmla, a_types.VecGatherCmdType.DBL_TRI),
                        "vmadd": (self.tinst.vmadd, a_types.VecGatherCmdType.DBL_TRI)}

        sca_dbl_tri_cmds = {"vadds": (self.tinst.vadds, a_types.VecGatherCmdType.SCA_DBL_TRI),
                            "vmuls": (self.tinst.vmuls, a_types.VecGatherCmdType.SCA_DBL_TRI),
                            "vaxpy": (self.tinst.vaxpy, a_types.VecGatherCmdType.SCA_DBL_TRI),
                            "vmins": (self.tinst.vmins, a_types.VecGatherCmdType.SCA_DBL_TRI)}

        init_cmds = {"vector_dup": (self.tinst.vector_dup, a_types.VecGatherCmdType.INIT)}
        conv_cmds = {"vconv": (self.tinst.vconv, a_types.VecGatherCmdType.CONV)}
        res = single_cmds
        res.update(dbl_tri_cmds)
        res.update(sca_dbl_tri_cmds)
        res.update(conv_cmds)
        res.update(init_cmds)
        return res

    def _init_vector_cmds(self):
        cmds = self._init_vector_base_cmds()

        dbl_tri_cmds = {"vdiv": (self.tinst.vdiv, a_types.VecGatherCmdType.DBL_TRI)}
        cmds.update(dbl_tri_cmds)
        self.const_vec_cmds = cmds

    @classmethod
    def reset_instance(cls):
        """
        reset
        """
        cls._instance = cls()
