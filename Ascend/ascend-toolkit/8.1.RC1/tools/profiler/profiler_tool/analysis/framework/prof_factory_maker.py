#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.

import logging
import os

from common_func.ms_constant.str_constant import StrConstant
from common_func.platform.chip_manager import ChipManager
from mscalculate.factory.calculator_factory import CalculatorFactory
from msparser.factory.parser_factory import ParserFactory


class ProfFactoryMaker:
    """
    factory maker for creating factories
    """

    def __init__(self: any, sample_config: dict) -> None:
        self._sample_config = sample_config
        self._project_path = sample_config.get(StrConstant.SAMPLE_CONFIG_PROJECT_PATH)
        self._file_list = {}

    @staticmethod
    def _get_chip_model() -> any:
        return ChipManager().get_chip_id()

    def create_parser_factory(self: any, file_list: dict) -> None:
        """
        entry for creating data parser factory.
        :param file_list: file list
        :return: NA
        """
        if not file_list:
            logging.warning("No profiling data detected, please check the directory of %s",
                            os.path.basename(self._sample_config.get(StrConstant.SAMPLE_CONFIG_PROJECT_PATH)))
            return
        self._file_list = file_list

        _chip_model = self._get_chip_model()
        if _chip_model:
            parser_factory = ParserFactory(self._file_list, self._sample_config, str(_chip_model.value))
            parser_factory.run()

    def create_calculator_factory(self: any, file_list: dict) -> None:
        """
        entry for creating data calculator factory.
        :return: NA
        """
        _chip_model = self._get_chip_model()
        if _chip_model:
            calculator_factory = CalculatorFactory(file_list, self._sample_config, str(_chip_model.value))
            calculator_factory.run()
