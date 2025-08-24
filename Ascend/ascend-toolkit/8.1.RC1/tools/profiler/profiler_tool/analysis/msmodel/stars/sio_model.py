#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

from common_func.db_manager import DBManager
from common_func.db_name_constant import DBNameConstant
from common_func.ms_constant.number_constant import NumberConstant
from msmodel.interface.parser_model import ParserModel


class SioModel(ParserModel):
    """
    sio model
    """

    def flush(self: any, data: list) -> None:
        """
        insert data to database
        时间戳单位：ns
        带宽单位：MB/s
        """
        self.insert_data_to_db(DBNameConstant.TABLE_SIO, data)

    def get_timeline_data(self: any) -> list:
        """
        get sio timeline data
        :return: list
        """
        if not DBManager.judge_table_exist(self.cur, DBNameConstant.TABLE_SIO):
            return []
        sql = "select acc_id, req_rx, rsp_rx, snp_rx, dat_rx, req_tx, rsp_tx, snp_tx, dat_tx, " \
              "timestamp/{NS_TO_US} as timestamp from {}".format(DBNameConstant.TABLE_SIO,
                                                                 NS_TO_US=NumberConstant.NS_TO_US)
        return DBManager.fetch_all_data(self.cur, sql)
