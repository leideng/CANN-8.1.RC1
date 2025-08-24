#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.

import os

from common_func.constant import Constant
from common_func.info_conf_reader import InfoConfReader
from common_func.ms_constant.number_constant import NumberConstant
from common_func.ms_constant.str_constant import StrConstant
from profiling_bean.basic_info.base_info import BaseInfo


class CollectInfo(BaseInfo):
    """
    collect info
    """

    def __init__(self: any) -> None:
        super(CollectInfo, self).__init__()
        self.collection_start_time = ""
        self.collection_end_time = ""
        self.result_size = ""

    def merge_data(self: any) -> any:
        """
        merge data
        :return:
        """
        self.get_collect_time()

    def get_collect_time(self: any) -> None:
        """
        get the time of collect task.
        :return: collect time
        """
        # Compatible for real time and Monotonic time
        if InfoConfReader().get_root_data(StrConstant.COLLECT_RAW_TIME_BEGIN) \
                and InfoConfReader().get_root_data(StrConstant.COLLECT_RAW_TIME_END):
            self.collection_start_time, self.collection_end_time = \
                InfoConfReader().get_collect_raw_time()
            return
        self.collection_start_time, self.collection_end_time = \
            InfoConfReader().get_collect_time()

    def get_project_size(self: any, project_path: str) -> None:
        """
        get the project size after analysis.
        :param project_path: project path
        :return: the size of the project
        """
        size = 0
        for dir_path, _, file_names in os.walk(project_path):
            for file_name in file_names:
                file_path = os.path.join(dir_path, file_name)
                size += os.path.getsize(file_path)
        self.result_size = str(
            round(size / (Constant.KILOBYTE * Constant.KILOBYTE), NumberConstant.DECIMAL_ACCURACY)) + " MB"

    def run(self: any, project_path: str) -> None:
        """
        run data
        :param project_path: project path
        :return: None
        """
        self.merge_data()
        self.get_project_size(project_path)
