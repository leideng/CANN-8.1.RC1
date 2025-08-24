#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.

from common_func.info_conf_reader import InfoConfReader
from common_func.platform.chip_manager import ChipManager
from common_func.profiling_scene import ProfilingScene


class LoadInfoManager:
    """
    class used to load config
    """

    @staticmethod
    def load_manager() -> None:
        """
        manager load info
        :return:
        """

    @classmethod
    def load_info(cls: any, result_dir: str) -> None:
        """
        load info.json and init profiling scene
        :param result_dir:
        :return: None
        """
        InfoConfReader().load_info(result_dir)
        ChipManager().load_chip_info()
        ProfilingScene().init(result_dir)
