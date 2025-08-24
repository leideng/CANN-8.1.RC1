#!/usr/bin/env python
# coding=utf-8

"""
Function:
This file mainly involves class for IR operator info.
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved © 2020
"""

from op_gen.interface import utils
from op_gen.interface.op_info_ir import IROpInfo
from op_gen.interface.const_manager import ConstManager


class MSIROpInfo(IROpInfo):
    """
    CLass for IR row for Mindspore.
    """

    @staticmethod
    def _mapping_input_output_type(ir_type: str, ir_name: str) -> any:
        file_type = ConstManager.INPUT_FILE_XLSX
        return utils.CheckFromConfig().trans_ms_io_dtype(ir_type, ir_name,
                                                         file_type)

    def get_op_path(self: any) -> str:
        """
        get op path
        """
        return self.op_path

    def get_gen_flag(self: any) -> str:
        """
        get gen flag
        """
        return self.gen_flag
