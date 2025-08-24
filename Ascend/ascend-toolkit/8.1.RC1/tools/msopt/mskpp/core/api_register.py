#!/usr/bin/python
# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2025. All rights reserved.

import os
from .common.registries import BaseRegistry


class InstrApiRegister(BaseRegistry):
    """指令注册器，继承自基础注册器"""
    pass


def import_all_apis():
    import importlib
    # 动态导入intrisic_api下的各个模块
    for module_name in ['infer_invoke', 'instr_register', 'instr_strategy']:
        importlib.import_module(f'.intrisic_api.{module_name}', package='mskpp')


import_all_apis()