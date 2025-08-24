#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

from common_func.db_name_constant import DBNameConstant
from msmodel.interface.parser_model import ParserModel


class StaticOpMemModel(ParserModel):
    """
    Model for StaticOpMemParser
    """
    def __init__(self: any, result_dir: str) -> None:
        super().__init__(result_dir, DBNameConstant.DB_STATIC_OP_MEM, [DBNameConstant.TABLE_STATIC_OP_MEM])

    def flush(self: any, data_list: list, table_name: str = DBNameConstant.TABLE_STATIC_OP_MEM) -> None:
        """
        insert data to table
        :param data_list: static_op_mem data
        :param table_name: table name
        :return:
        """
        self.insert_data_to_db(table_name, data_list)
