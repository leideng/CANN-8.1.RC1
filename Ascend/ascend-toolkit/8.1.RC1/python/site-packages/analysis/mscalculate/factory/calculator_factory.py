#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.

from common_func.config_mgr import ConfigMgr
from common_func.ms_constant.str_constant import StrConstant
from framework.config_data_parsers import ConfigDataParsers
from framework.iprof_factory import IProfFactory
from msconfig.config_manager import ConfigManager


class CalculatorFactory(IProfFactory):
    """
    factory for data calculate
    """
    def __init__(self: any, file_list: dict, sample_config: dict, chip_model: str) -> None:
        self._sample_config = sample_config
        self._chip_model = chip_model
        self._file_list = file_list
        self._project_path = sample_config.get(StrConstant.SAMPLE_CONFIG_PROJECT_PATH)

    def generate(self: any, chip_model: str) -> dict:
        """
        get data parser
        :param chip_model:
        :return:
        """
        task_flag = ConfigMgr.is_ai_core_task_based(self._project_path)
        return ConfigDataParsers.get_parsers(ConfigManager.DATA_CALCULATOR, chip_model, task_flag)

    def run(self: any) -> None:
        """
        run for parsers
        :return:
        """
        self._launch_parser_list(self._sample_config, self._file_list, self.generate(self._chip_model))
