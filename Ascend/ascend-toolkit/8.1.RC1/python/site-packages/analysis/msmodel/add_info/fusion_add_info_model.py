#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

from common_func.db_name_constant import DBNameConstant
from msmodel.interface.parser_model import ParserModel


class FusionAddInfoModel(ParserModel):
    """
    Model for Ge Fusion Info Parser
    """
    def __init__(self: any, result_dir: str) -> None:
        super().__init__(result_dir, DBNameConstant.DB_FUSION_ADD_INFO, [DBNameConstant.TABLE_FUSION_ADD_INFO])

    def flush(self: any, data_list: list, table_name: str = DBNameConstant.TABLE_FUSION_ADD_INFO) -> None:
        """
        insert data to table
        :param data_list: ge fusion info data
        :param table_name: table name
        :return:
        """
        self.insert_data_to_db(table_name, data_list)
