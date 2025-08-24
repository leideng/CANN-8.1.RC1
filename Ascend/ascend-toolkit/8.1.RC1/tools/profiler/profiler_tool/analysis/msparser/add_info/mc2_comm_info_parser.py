#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
import logging

from common_func.ms_constant.str_constant import StrConstant
from common_func.ms_multi_process import MsMultiProcess
from msparser.data_struct_size_constant import StructFmt
from msparser.interface.data_parser import DataParser
from msparser.add_info.mc2_comm_info_bean import Mc2CommInfoBean
from profiling_bean.prof_enum.data_tag import DataTag
from msmodel.add_info.mc2_comm_info_model import Mc2CommInfoModel


class Mc2CommInfoParser(DataParser, MsMultiProcess):
    def __init__(self: any, file_list: dict, sample_config: dict) -> None:
        super().__init__(sample_config)
        super(DataParser, self).__init__(sample_config)
        self._file_list = file_list
        self.project_path = sample_config.get(StrConstant.SAMPLE_CONFIG_PROJECT_PATH)
        self._communication_info = []

    def reformat_data(self) -> list:
        reformat_data = []
        for data in self._communication_info:
            reformat_data.append(
                [data.group_name, data.rank_size, data.rank_id,
                 data.usr_rank_id, data.stream_id, data.comm_stream_ids])
        return reformat_data

    def ms_run(self: any) -> None:
        """
        parse and save ge fusion data
        :return:
        """
        if not self._file_list.get(DataTag.MC2_COMM_INFO, []):
            return
        logging.info("start parsing mc2 comm info, files: %s", str(self._file_list.get(DataTag.MC2_COMM_INFO)))
        self.parse()
        self.save()

    def parse(self: any) -> None:
        """
        parse mc2 comm info data
        """
        mc2_comm_info_files = self._file_list.get(DataTag.MC2_COMM_INFO, [])
        mc2_comm_info_files = self.group_aging_file(mc2_comm_info_files)
        for file_list in mc2_comm_info_files.values():
            self._communication_info += self.parse_bean_data(file_list,
                                                             StructFmt.MC2_COMM_INFO_SIZE,
                                                             Mc2CommInfoBean,
                                                             check_func=self.check_magic_num,
                                                             )

    def save(self: any) -> None:
        """
        save mc2 comm info parser data to db
        :return: None
        """
        if not self._communication_info:
            return
        with Mc2CommInfoModel(self._project_path) as _model:
            _model.flush(self.reformat_data())
