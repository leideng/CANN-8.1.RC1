#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 
import logging
import sqlite3

from common_func.constant import Constant
from common_func.ms_constant.str_constant import StrConstant
from common_func.ms_multi_process import MsMultiProcess
from msmodel.ge.ge_logic_stream_model import GeLogicStreamInfoModel
from msparser.data_struct_size_constant import StructFmt
from msparser.ge.ge_logic_stream_info_bean import GeLogicStreamInfoBean
from msparser.interface.data_parser import DataParser
from profiling_bean.prof_enum.data_tag import DataTag


class GeLogicStreamParser(DataParser, MsMultiProcess):
    """
    ge logic stream data parser
    """
 
    def __init__(self: any, file_list: dict, sample_config: dict) -> None:
        super().__init__(sample_config)
        super(DataParser, self).__init__(sample_config)
        self._file_list = file_list
        self._sample_config = sample_config
        self._project_path = sample_config.get(StrConstant.SAMPLE_CONFIG_PROJECT_PATH)
        self._ge_logic_stream_data = []
 
    @staticmethod
    def _get_ge_logic_stream_data(bean_data: any) -> list:
        if not bean_data:
            return []
        return bean_data.physic_logic_stream_id
 
    def parse(self: any) -> None:
        """
        parse ge logic stream data
        """
        logic_stream_info_file = self._file_list.get(DataTag.GE_LOGIC_STREAM_INFO, [])
        if logic_stream_info_file:
            self._ge_logic_stream_data = self.parse_bean_data(logic_stream_info_file, 
                                                              StructFmt.GE_LOGIC_STREAM_INFO_SIZE,
                                                              GeLogicStreamInfoBean, 
                                                              format_func=self._get_ge_logic_stream_data)
 
    def save(self: any) -> None:
        """
        save data
        """
        if not self._ge_logic_stream_data:
            return
        format_data = self.format_stream_data()
        model = GeLogicStreamInfoModel(self._project_path)
        with model as _model:
            _model.flush(format_data)
 
    def format_stream_data(self) -> list:
        format_data = []
        for stream_list in self._ge_logic_stream_data:
            format_data.extend(stream_list)
        return format_data

    def ms_run(self: any) -> None:
        """
        parse and save ge hash data
        :return:
        """
        if not self._file_list.get(DataTag.GE_LOGIC_STREAM_INFO, []):
            return
        try:
            self.parse()
        except (OSError, IOError, SystemError, ValueError, TypeError, RuntimeError) as err:
            logging.error(str(err), exc_info=Constant.TRACE_BACK_SWITCH)
            return
        try:
            self.save()
        except sqlite3.Error as err:
            logging.error(str(err), exc_info=Constant.TRACE_BACK_SWITCH)

