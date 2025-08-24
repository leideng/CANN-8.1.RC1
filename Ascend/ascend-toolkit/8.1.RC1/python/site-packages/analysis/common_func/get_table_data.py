#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.

from common_func.db_manager import DBManager
from common_func.info_conf_reader import InfoConfReader
from common_func.ms_constant.number_constant import NumberConstant


class GetTableData:
    """
    class to manage DB operation
    """

    @staticmethod
    def get_table_data_for_device(cursor: any, table_name: str, device_id: str) -> list:
        """
        get table data for certain device
        """
        if not cursor:
            return []
        search_data_sql = "select duration, bandwidth, " \
                          "rxBandwidth, rxPacket, rxErrorRate, " \
                          "rxDroppedRate, txBandwidth, txPacket, txErrorRate, txDroppedRate, funcId " \
                          "from {0} where device_id={1} order by rowid"\
                          .format(table_name, device_id)
        result = DBManager.fetch_all_data(cursor, search_data_sql)
        if result:
            return result
        return []

    @staticmethod
    def get_data_count_for_device(cursor: any, table_name: str, device_id: str) -> any:
        """
        get table data for certain device
        """
        if not cursor:
            return 0
        sql = "select count(*) from {} where device_id={}".format(table_name, device_id)
        return cursor.execute(sql).fetchone()[0]
