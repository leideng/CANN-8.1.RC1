#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.

from common_func.db_name_constant import DBNameConstant
from msmodel.interface.parser_model import ParserModel


class MsTimeModel(ParserModel):
    """
    class used to operate time db
    """

    def __init__(self: any, result_dir: str) -> None:
        super().__init__(result_dir, DBNameConstant.DB_TIME, [DBNameConstant.TABLE_TIME])

    def flush(self: any, data_list: list) -> None:
        """
        flush data to db
        :param data_list:
        :return:
        """
        self.insert_data_to_db(DBNameConstant.TABLE_TIME, data_list)