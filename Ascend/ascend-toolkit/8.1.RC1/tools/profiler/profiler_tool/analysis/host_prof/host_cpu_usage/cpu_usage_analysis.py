#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.

import logging
import os

from common_func.file_name_manager import get_file_name_pattern_match
from common_func.file_name_manager import get_host_cpu_usage_compiles
from common_func.ms_multi_process import MsMultiProcess
from common_func.msvp_common import is_valid_original_data
from common_func.path_manager import PathManager
from host_prof.host_cpu_usage.presenter.host_cpu_usage_presenter import HostCpuUsagePresenter


class CpuUsageAnalysis(MsMultiProcess):
    """
    analysis host cpu usage
    """

    def __init__(self: any, sample_config: dict) -> None:
        super().__init__(sample_config)
        self.sample_config = sample_config
        self.result_dir = self.sample_config.get("result_dir", "")

    @staticmethod
    def class_name() -> str:
        """
        class name
        """
        return CpuUsageAnalysis.__name__

    def ms_run(self: any) -> None:
        """
        run function
        """
        if not os.path.exists(self.result_dir):
            logging.error("Data project doesn't exist.")
            return
        file_list = os.listdir(PathManager.get_data_dir(self.result_dir))
        host_cpu_usage_file_patterns = get_host_cpu_usage_compiles()
        for file_name in file_list:
            host_cpu_usage_result = get_file_name_pattern_match(file_name,
                                                                *host_cpu_usage_file_patterns)
            if host_cpu_usage_result and is_valid_original_data(file_name, self.result_dir):
                logging.info(
                    "start parsing host cpu usage data file: %s", file_name)
                host_cpu_usage_presenter = HostCpuUsagePresenter(self.result_dir, file_name)
                host_cpu_usage_presenter.run()
