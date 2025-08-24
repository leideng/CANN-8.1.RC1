#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.

from common_func.db_name_constant import DBNameConstant
from msmodel.interface.parser_model import ParserModel


class LowPowerModel(ParserModel):
    """
    lowpower sample model class
    """

    def __init__(self: any, result_dir: str, db: str, table_list: list) -> None:
        super().__init__(result_dir, db, table_list)

    def flush(self: any, data_dict: dict) -> None:
        """
        insert lowpower sample data into database
        """
        self.insert_data_to_db(DBNameConstant.TABLE_LOWPOWER, data_dict.get('data_list', []))

    def get_timeline_data(self):
        """
        get lowpower sample timeline data from database
        """
        return self.get_all_data(DBNameConstant.TABLE_LOWPOWER)
