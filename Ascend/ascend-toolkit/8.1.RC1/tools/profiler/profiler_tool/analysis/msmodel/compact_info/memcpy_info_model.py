#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

from common_func.db_name_constant import DBNameConstant
from msmodel.interface.parser_model import ParserModel


class MemcpyInfoModel(ParserModel):
    """
    db operator for memcpy info
    """

    def __init__(self: any, result_dir: str):
        super().__init__(result_dir, DBNameConstant.DB_RUNTIME, [DBNameConstant.TABLE_MEMCPY_INFO])

    def flush(self: any, data_list: list) -> None:
        """
        insert data into database
        """
        if not self.table_list:
            return
        self.insert_data_to_db(self.table_list[0], data_list)
