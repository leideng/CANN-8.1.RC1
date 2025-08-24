#!/usr/bin/python
# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2025. All rights reserved.

import os
from abc import ABC, abstractmethod
from .common.registries import BaseRegistry


class PrefModel(ABC):
    """
    通过性能数据建模，获得指令在特定输入输出下的性能数据
    """
    def __init__(self, name, inputs, outputs):
        self.name = name
        self.inputs = inputs
        self.outputs = outputs

    @abstractmethod
    def size(self):
        raise Exception("this api need to impl.")

    @abstractmethod
    def time(self):
        raise Exception("this api need to impl.")


class ProfDataRegister(BaseRegistry):
    """性能数据注册器，继承自基础注册器"""
    pass


def import_all_pref_datas():
    import importlib
    module_names = []
    api_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prof_data")
    for fname in os.listdir(api_dir):
        if os.path.isfile(os.path.join(api_dir, fname)) \
                and fname.endswith('prof.py') and not fname.startswith("ascend"):
            module_names.append(fname.split('.')[0])
    for module_name in module_names:
        importlib.import_module(f'.prof_data.{module_name}', package='mskpp')


import_all_pref_datas()