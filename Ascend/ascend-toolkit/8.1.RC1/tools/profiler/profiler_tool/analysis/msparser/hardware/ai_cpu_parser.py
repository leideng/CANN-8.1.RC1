#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
from common_func.db_name_constant import DBNameConstant
from common_func.file_name_manager import get_ai_cpu_compiles

from msparser.hardware.cpu_parser import ParsingCPUData
from profiling_bean.prof_enum.data_tag import DataTag


class ParsingAICPUData(ParsingCPUData):
    """
    parsing ai cpu data file
    """

    def __init__(self: any, file_list: dict, sample_config: dict) -> None:
        super(ParsingAICPUData, self).__init__(sample_config)
        self.sample_config = sample_config
        self.curs = None
        self.conn = None
        self.type = 'ai'
        self.dbname = DBNameConstant.DB_NAME_AICPU
        self.patterns = get_ai_cpu_compiles()
        self._file_list = file_list.get(DataTag.AICPU, [])
        self._file_list.sort(key=lambda x: int(x.split("_")[-1]))

    @property
    def cpu_type(self: any) -> str:
        """
        cpu type
        """
        return self.type

    @staticmethod
    def class_name() -> str:
        """
        class name
        """
        return ParsingAICPUData.__name__
