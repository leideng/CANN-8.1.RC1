#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.

from common_func.db_name_constant import DBNameConstant
from msmodel.interface.parser_model import ParserModel


class GeHostParserModel(ParserModel):
    """
    Model for Ge Host Parser
    """

    @staticmethod
    def class_name() -> str:
        """
        class name for ge op execute
        """
        return GeHostParserModel.__name__

    def flush(self: any, data_list: list) -> None:
        """
        data flush to db
        :param data_list:
        :return:
        """
        self.insert_data_to_db(DBNameConstant.TABLE_GE_HOST, data_list)
