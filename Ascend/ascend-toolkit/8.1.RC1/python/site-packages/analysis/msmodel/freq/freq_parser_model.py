#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

import logging

from common_func.db_manager import DBManager
from common_func.db_name_constant import DBNameConstant
from common_func.path_manager import PathManager
from msmodel.interface.parser_model import ParserModel


class FreqParserModel(ParserModel):
    """
    db operator for frequency parser
    """

    def __init__(self: any, result_dir: str, table_list: list) -> None:
        super(FreqParserModel, self).__init__(result_dir, DBNameConstant.DB_FREQ, table_list)

    @staticmethod
    def get_freq_data(project_path: str) -> list:
        """
        get frequency data
        """
        freq_db_path = PathManager.get_db_path(project_path, DBNameConstant.DB_FREQ)
        freq_conn, freq_curs = DBManager.check_connect_db_path(freq_db_path)
        if not (freq_conn and freq_curs) or not DBManager.judge_table_exist(freq_curs, DBNameConstant.TABLE_FREQ_PARSE):
            DBManager.destroy_db_connect(freq_conn, freq_curs)
            return []
        sql = "select syscnt, freq from {}".format(DBNameConstant.TABLE_FREQ_PARSE)
        freq_data = DBManager.fetch_all_data(freq_curs, sql)
        DBManager.destroy_db_connect(freq_conn, freq_curs)
        if not isinstance(freq_data, list) or not freq_data or len(freq_data[0]) != 2:
            logging.error("The freq data format is error!")
            return []
        freq_data.sort(key=lambda item: item[0])
        return freq_data

    def flush(self: any, data_list: list) -> None:
        """
        insert data into database
        """
        if self.table_list:
            self.insert_data_to_db(self.table_list[0], data_list)
