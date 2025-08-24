#!/usr/bin/env python
# coding=utf-8

"""
Function:
This file mainly involves class for parsing operator info.
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved © 2020
"""

from op_gen.interface.arg_parser import ArgParser
from op_gen.interface.op_info_ir import IROpInfo
from op_gen.interface.op_info_tf import TFOpInfo
from op_gen.interface.op_info_ir_mindspore import MSIROpInfo
from op_gen.interface.op_info_tf_mindspore import MSTFOpInfo
from op_gen.interface.op_info_ir_json import JsonIROpInfo
from op_gen.interface.op_info_ir_json_mindspore import JsonMSIROpInfo
from op_gen.interface.const_manager import ConstManager
from op_gen.interface import utils


class OpInfoParser:
    """
    CLass for parsing operator info
    """

    def __init__(self: any, argument: ArgParser) -> None:
        self.op_info = self._create_op_info(argument)
        self.op_info.parse()

    @staticmethod
    def get_gen_flag() -> str:
        """
        get gen flag
        """
        return ""

    @staticmethod
    def _create_op_info(argument: ArgParser) -> any:
        if argument.input_path.endswith(ConstManager.INPUT_FILE_EXCEL):
            utils.print_warn_log("Excel cannot be used as inputs in future "
                                 "versions. It is recommended that json "
                                 "files be used as inputs.")
            if argument.gen_flag and argument.framework in ConstManager.FMK_MS:
                return MSIROpInfo(argument)
            return IROpInfo(argument)
        if argument.input_path.endswith(ConstManager.INPUT_FILE_JSON):
            if argument.gen_flag and argument.framework in ConstManager.FMK_MS:
                return JsonMSIROpInfo(argument)
            return JsonIROpInfo(argument)
        if argument.gen_flag and argument.framework in ConstManager.FMK_MS:
            return MSTFOpInfo(argument)
        return TFOpInfo(argument)

    def get_op_info(self: any) -> any:
        """
        get op info
        """
        return self.op_info
