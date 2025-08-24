#!/usr/bin/env python
# coding=utf-8

"""
Function:
This file mainly involves class for operator info.
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved © 2020
"""
import collections


class OpInfo:
    """
    OpInfo store the op informat for generate the op files,
    parsed_input_info and parsed_output_info is dicts,eg:
    {name:
    {
    ir_type_list:[],
    param_type:""required,
    format_list:[]
    }
    }
    """

    def __init__(self: any) -> None:
        self.op_type = ""
        self.fix_op_type = ""
        self.parsed_input_info = collections.OrderedDict()
        self.parsed_output_info = collections.OrderedDict()
        self.parsed_attr_info = []

    def get_op_type(self: any) -> str:
        """
        get op type
        """
        return self.op_type

    def get_fix_op_type(self: any) -> str:
        """
        get fix op type
        """
        return self.fix_op_type
