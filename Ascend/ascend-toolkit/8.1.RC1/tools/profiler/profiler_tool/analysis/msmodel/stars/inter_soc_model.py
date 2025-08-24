#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.

from common_func.db_manager import DBManager
from common_func.db_name_constant import DBNameConstant
from msmodel.interface.parser_model import ParserModel


class InterSocModel(ParserModel):
    """
    inter soc transmission model
    """

    def flush(self: any, data: list) -> None:
        """
        insert data to database
        """
        self.insert_data_to_db(DBNameConstant.TABLE_SOC_DATA, data)

    def get_timeline_data(self: any) -> list:
        """
        get soc timeline data
        :return: list
        """
        if not DBManager.judge_table_exist(self.cur, DBNameConstant.TABLE_SOC_DATA):
            return []
        sql = "select l2_buffer_bw_level, mata_bw_level, sys_time from {}".format(DBNameConstant.TABLE_SOC_DATA)
        data_list = DBManager.fetch_all_data(self.cur, sql)
        return data_list

    def get_summary_data(self: any) -> list:
        """
        get soc summary data
        :return:tuple
        """
        sql = "select 'average', round(sum(buffer_bw_level)/(max(time_stamp)-min(time_stamp)), 4), " \
              "round(sum(mata_bw_level)/(max(time_stamp)-min(time_stamp)), 4) from {}".format(
            DBNameConstant.TABLE_SOC_DATA)
        return DBManager.fetch_all_data(self.cur, sql)
