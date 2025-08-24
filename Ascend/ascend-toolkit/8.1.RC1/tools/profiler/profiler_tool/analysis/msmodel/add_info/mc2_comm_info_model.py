#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
from collections import namedtuple

from common_func.db_name_constant import DBNameConstant
from common_func.db_manager import DBManager
from common_func.info_conf_reader import InfoConfReader
from common_func.msprof_object import CustomizedNamedtupleFactory
from msmodel.interface.parser_model import ParserModel
from msmodel.interface.view_model import ViewModel


class Mc2CommInfoModel(ParserModel):
    """
    mc2 comm info model class
    """

    def __init__(self: any, result_dir: str) -> None:
        super().__init__(result_dir, DBNameConstant.DB_MC2_COMM_INFO, [DBNameConstant.TABLE_MC2_COMM_INFO])

    def flush(self: any, data_list: list, table_name: str = DBNameConstant.TABLE_MC2_COMM_INFO) -> None:
        """
        insert data to table
        :param data_list: hccl information data
        :param table_name: table name
        :return:
        """
        self.insert_data_to_db(table_name, data_list)


class Mc2CommInfoViewModel(ViewModel):
    MC2_COMM_INFO_TYPE = CustomizedNamedtupleFactory.enhance_namedtuple(
        namedtuple("Mc2CommInfo",
                   ["aicpu_kfc_stream_id", "comm_stream_ids", "group_name"]),
        {})

    def __init__(self, result_dir: str, table_list: list):
        super().__init__(result_dir, DBNameConstant.DB_MC2_COMM_INFO, table_list)

    def get_kfc_stream(self: any, table_name: str) -> list:
        if not DBManager.judge_table_exist(self.cur, table_name):
            return []
        sql = "select aicpu_kfc_stream_id, comm_stream_ids, group_name from {0} ".format(table_name)
        mc2_comm_info_data = DBManager.fetch_all_data(self.cur, sql)
        return [self.MC2_COMM_INFO_TYPE(*data) for data in mc2_comm_info_data]
