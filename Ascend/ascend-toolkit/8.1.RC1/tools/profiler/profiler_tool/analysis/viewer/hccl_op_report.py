#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.

import logging

from common_func.db_manager import DBManager
from common_func.db_name_constant import DBNameConstant
from common_func.ms_constant.number_constant import NumberConstant


class ReportHcclStatisticData:
    """
    class to report hccl op data
    """

    @staticmethod
    def _get_hccl_op_report_sql() -> str:
        sql = "select op_type,  occurrences, round(total_time/{NS_TO_US}, {accuracy}), " \
              "round(min/{NS_TO_US}, {accuracy}), round(avg/{NS_TO_US}, {accuracy}), round(max/{NS_TO_US}, " \
              "{accuracy}), round(ratio, {accuracy}) from {0} order by total_time desc " \
            .format(DBNameConstant.TABLE_HCCL_OP_REPORT, NS_TO_US=NumberConstant.NS_TO_US,
                    accuracy=NumberConstant.ROUND_THREE_DECIMAL)
        return sql

    @classmethod
    def report_hccl_op(cls: any, db_path: str, headers: list) -> tuple:
        """
        report hccl op
        :param db_path: DB path
        :param headers: table headers
        :return: headers, data, data length
        """
        conn, curs = DBManager.check_connect_db_path(db_path)
        if not (conn and curs) or not DBManager.judge_table_exist(curs, DBNameConstant.TABLE_HCCL_OP_REPORT):
            logging.warning("Failed to connect to the database %s or the table %s does not exist.",
                            DBNameConstant.DB_HCCL_SINGLE_DEVICE,
                            DBNameConstant.TABLE_HCCL_OP_REPORT)
            return [], [], 0
        sql = cls._get_hccl_op_report_sql()
        data = DBManager.fetch_all_data(curs, sql)
        DBManager.destroy_db_connect(conn, curs)
        return headers, data, len(data)
