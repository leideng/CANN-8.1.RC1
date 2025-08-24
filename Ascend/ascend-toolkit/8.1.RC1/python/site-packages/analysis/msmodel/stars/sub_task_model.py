#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

from common_func.db_name_constant import DBNameConstant
from msmodel.interface.parser_model import ParserModel


class SubTaskTimeModel(ParserModel):
    """
    subtask time model class
    """
    def __init__(self: any, result_dir: str):
        super().__init__(result_dir, DBNameConstant.DB_SOC_LOG, [DBNameConstant.TABLE_SUBTASK_TIME])

    def flush(self: any, data_list: list) -> None:
        """
        flush subtask time data to db
        :param data_list:subtask time data list
        :return: None
        """
        self.insert_data_to_db(DBNameConstant.TABLE_SUBTASK_TIME, data_list)
