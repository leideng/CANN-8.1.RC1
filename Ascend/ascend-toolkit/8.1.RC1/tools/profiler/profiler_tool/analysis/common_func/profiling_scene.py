#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.

import os
import logging
from enum import Enum

from common_func.constant import Constant
from common_func.file_manager import check_so_valid
from common_func.platform.chip_manager import ChipManager
from common_func.singleton import singleton
from common_func.utils import Utils
from common_func.db_name_constant import DBNameConstant


class ExportMode(Enum):
    """
    数据导出模式
    """
    ALL_EXPORT = 0
    STEP_EXPORT = 1
    GRAPH_EXPORT = 2


@singleton
class ProfilingScene:
    """
    Differentiate the scene for the current profiling data
    """

    def __init__(self: any) -> None:
        self.project_path = None
        self._scene = None
        self._mode = ExportMode.ALL_EXPORT

    @staticmethod
    def is_cpp_parse_enable() -> bool:
        """
        check whether cpp_parse
        :return: cpp_parse
        """
        so_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "lib64", "msprof_analysis.so")
        if not check_so_valid(so_path):
            return False
        return True

    def set_mode(self: any, mode: ExportMode) -> None:
        self._mode = mode

    def get_mode(self: any) -> ExportMode:
        return self._mode

    def init(self: any, project_path: str) -> None:
        """
        init the class profiling scene and load the data from the current data
        :param project_path: project path
        :return: NA
        """
        self.project_path = project_path
        self._scene = None

    def get_scene(self: any) -> any:
        """
        get scene of the current data
        :return: scene
        """
        if self._scene is None:
            self._scene = Utils.get_scene(self.project_path)
            logging.info("Current scene of data is based on %s", self._scene)
        return self._scene

    def is_all_export(self: any) -> bool:
        """
        check whether all export
        通过用户输入确定模式
        :return: True or False
        """
        return self._mode == ExportMode.ALL_EXPORT

    def is_step_export(self: any) -> bool:
        """
        check whether step export
        通过用户输入确定模式
        :return: True or False
        """
        # 按step导数据必须要支持全导
        return self._mode == ExportMode.STEP_EXPORT

    def is_graph_export(self: any) -> bool:
        """
        check whether graph export
        通过用户输入确定模式
        :return: True or False
        """
        return self._mode == ExportMode.GRAPH_EXPORT

    def get_step_table_name(self: any) -> str:
        """
        根据不同场景, 返回step_trace_data或者StepTime表名
        """
        if self.is_step_export():
            return DBNameConstant.TABLE_STEP_TIME
        else:
            return DBNameConstant.TABLE_STEP_TRACE_DATA

    def is_operator(self: any) -> bool:
        """
        check whether operator
        通过step trace数据确定是否单算子
        :return: True or False
        """
        return self.get_scene() == Constant.SINGLE_OP

    def is_mix_operator_and_graph(self: any) -> bool:
        """
        check whether operator
        通过step trace数据确定是否图和单算子混合
        :return: True or False
        """
        return self.get_scene() == Constant.MIX_OP_AND_GRAPH
