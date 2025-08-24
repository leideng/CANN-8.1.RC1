#!/usr/bin/python
# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
from mskpp._C import arch
from .trace import core_type_list
from .common import checker


class Core:
    core_type = None

    def __init__(self, core_type_name):
        self.param_check(core_type_name)
        Core.set_core_type(core_type_name)
        if core_type_name not in core_type_list:
            core_type_list.append(core_type_name)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.clear_core_type()

    @staticmethod
    def set_core_type(core_type_input):
        Core.core_type = core_type_input

    @staticmethod
    def clear_core_type():
        Core.core_type = None

    @staticmethod
    def get_instr_pipe_name(instr_type=None, src=None, dst=None):
        '''
        :param instr_type: only mmad valid
        :param src: mov instr src tensor
        :param dst: mov instr dst tensor
        :return: pipe name
        '''
        base_pipe_name = "PIPE-V"
        if instr_type == "mmad":
            base_pipe_name = "PIPE-M"
        if src is not None and dst is not None:
            base_pipe_name = arch.get_pipe_by_io(src, dst)
        return base_pipe_name if Core.core_type is None else (Core.core_type + "-" + base_pipe_name)

    @staticmethod
    def param_check(core_type_name):
        checker.check_name_valid(core_type_name, "core_type_name")