#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.

from msparser.stars.stars_log_parser import StarsLogCalCulator
from profiling_bean.prof_enum.data_tag import DataTag


class SocProfilerParser(StarsLogCalCulator):
    """
    to read and parse stars data in soc_profiler buffer
    """

    def __init__(self: any, file_list: dict, sample_config: dict) -> None:
        super().__init__(file_list, sample_config)
        self._file_list = file_list.get(DataTag.SOC_PROFILER, [])
        self._file_list.sort(key=lambda x: int(x.split("_")[-1]))

