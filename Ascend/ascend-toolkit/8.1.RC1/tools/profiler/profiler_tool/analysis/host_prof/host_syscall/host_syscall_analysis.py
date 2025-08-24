#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.

import logging
import os

from common_func.common import get_data_dir_sorted_files
from common_func.file_name_manager import get_file_name_pattern_match
from common_func.file_name_manager import get_host_pthread_call_compiles
from common_func.file_name_manager import get_host_syscall_compiles
from common_func.ms_multi_process import MsMultiProcess
from common_func.msvp_common import is_valid_original_data
from common_func.path_manager import PathManager
from host_prof.host_syscall.presenter.host_syscall_presenter import HostSyscallPresenter


class HostSyscallAnalysis(MsMultiProcess):
    """
    host os runtime api analysis class
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
        return HostSyscallAnalysis.__name__

    def ms_run(self: any) -> None:
        """
        run function
        """
        if not os.path.exists(self.result_dir):
            logging.error("Data project doesn't exist.")
            return
        file_list = get_data_dir_sorted_files(PathManager.get_data_dir(self.result_dir))
        host_syscall_file_patterns = get_host_syscall_compiles()
        host_pthread_call_file_patterns = get_host_pthread_call_compiles()
        for file_name in file_list:
            host_syscall_result = get_file_name_pattern_match(file_name,
                                                              *host_syscall_file_patterns)
            host_pthread_call_result = get_file_name_pattern_match(file_name,
                                                                   *host_pthread_call_file_patterns)
            if (host_syscall_result or host_pthread_call_result) and is_valid_original_data(
                    file_name, self.result_dir):
                logging.info(
                    "start parsing os runtime api data file: %s", file_name)
                host_syscall_presenter = HostSyscallPresenter(self.result_dir, file_name)
                host_syscall_presenter.run()
