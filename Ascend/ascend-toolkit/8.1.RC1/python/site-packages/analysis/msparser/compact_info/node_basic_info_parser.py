#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

import logging
import sqlite3
from enum import IntEnum

from common_func.constant import Constant
from common_func.hash_dict_constant import HashDictData
from common_func.ms_constant.str_constant import StrConstant
from common_func.ms_multi_process import MsMultiProcess
from msmodel.compact_info.node_basic_info_model import NodeBasicInfoModel
from msparser.compact_info.node_basic_info_bean import NodeBasicInfoBean
from msparser.data_struct_size_constant import StructFmt
from msparser.interface.data_parser import DataParser
from profiling_bean.prof_enum.data_tag import DataTag


class NodeBasicInfoParser(DataParser, MsMultiProcess):
    """
    Node Basic Info Data Parser
    """

    def __init__(self: any, file_list: dict, sample_config: dict) -> None:
        super().__init__(sample_config)
        super(DataParser, self).__init__(sample_config)
        self._file_list = file_list
        self._node_basic_info_data = {}
        self._project_path = sample_config.get(StrConstant.SAMPLE_CONFIG_PROJECT_PATH)

    @staticmethod
    def _get_node_basic_data(bean_data: any) -> list:
        if not bean_data:
            return []
        return [bean_data.level, bean_data.struct_type, bean_data.thread_id, bean_data.timestamp,
                bean_data.node_id, bean_data.task_type, bean_data.op_type, bean_data.block_dim,
                bean_data.mix_block_dim, bean_data.op_flag]

    def parse(self: any) -> None:
        """
        parse node basic data
        """
        basic_info_files = self._file_list.get(DataTag.NODE_BASIC_INFO, [])
        basic_info_files = self.group_aging_file(basic_info_files)
        for mode, file_list in basic_info_files.items():
            self._node_basic_info_data[mode] = self.parse_bean_data(file_list, StructFmt.NODE_BASIC_INFO_SIZE,
                                                                    NodeBasicInfoBean,
                                                                    format_func=self._get_node_basic_data,
                                                                    check_func=self.check_magic_num,
                                                                    )

    def save(self: any) -> None:
        """
        save
        :return:
        """
        if not self._node_basic_info_data:
            return
        model = NodeBasicInfoModel(self._project_path)
        with model as _model:
            _model.flush(self._transform_node_basic_info_data(self._node_basic_info_data))

    def ms_run(self: any) -> None:
        """
        parse and save node basic data
        :return:
        """
        if not self._file_list.get(DataTag.NODE_BASIC_INFO, []):
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

    def _transform_node_basic_info_data(self: any, data_dict: dict) -> list:
        hash_dict_data = HashDictData(self._project_path)
        type_hash_dict = hash_dict_data.get_type_hash_dict()
        ge_hash_dict = hash_dict_data.get_ge_hash_dict()
        node_basic_info_data = []
        for mode, data_list in data_dict.items():
            for data in data_list:
                # 1 type hash, 4 node hash, 6 op type
                data[1] = type_hash_dict.get('node', {}).get(data[1], data[1])
                data[4] = ge_hash_dict.get(data[4], data[4])
                data[6] = ge_hash_dict.get(data[6], data[6])
                # aging_file -> dynamic op, unaging_file -> static op
                data.append(OPType.DYNAMIC_OP.value if mode == 'aging_file' else OPType.STATIC_OP.value)
            node_basic_info_data.extend(data_list)
        return node_basic_info_data


class OPType(IntEnum):
    DYNAMIC_OP = 1
    STATIC_OP = 0
