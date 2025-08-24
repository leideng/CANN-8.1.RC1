#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.

from common_func.db_manager import DBManager
from common_func.db_name_constant import DBNameConstant
from common_func.info_conf_reader import InfoConfReader
from common_func.ms_constant.number_constant import NumberConstant
from host_prof.host_prof_base.host_prof_data_base import HostProfDataBase


class HostNetworkUsage(HostProfDataBase):
    """
    host network usage model
    """

    def __init__(self: any, result_dir: str) -> None:
        super().__init__(result_dir, DBNameConstant.DB_HOST_NETWORK_USAGE,
                         [DBNameConstant.TABLE_HOST_NETWORK_USAGE])

    def flush_data(self: any) -> None:
        """
        flush all cache data
        :return: None
        """
        self.insert_network_usage(self.cache_data)

    def insert_network_usage(self: any, network_usage_info: list) -> None:
        """
        insert network usage info to table
        :param network_usage_info: network usage info
        :return: None
        """
        insert_sql = "INSERT INTO {} VALUES(?, ?, ?, ?)".format(DBNameConstant.TABLE_HOST_NETWORK_USAGE)
        DBManager.executemany_sql(self.conn, insert_sql, network_usage_info)

    def has_network_usage_data(self: any) -> bool:
        """
        check has network usage data
        :return: check result
        """
        return DBManager.judge_table_exist(self.cur, DBNameConstant.TABLE_HOST_NETWORK_USAGE)

    def get_network_usage_data(self: any) -> dict:
        """
        get network usage data
        :return: network usage data
        """
        network_info_sql = "SELECT * from {}".format(DBNameConstant.TABLE_HOST_NETWORK_USAGE)
        network_info_list = DBManager.fetch_all_data(self.cur, network_info_sql)

        result = []
        for network_item in network_info_list:
            time_network = {
                "start": InfoConfReader().trans_into_local_time(network_item[0], is_host=True),
                "end": InfoConfReader().trans_into_local_time(network_item[1], is_host=True),
                "usage": network_item[2]
            }
            result.append(time_network)

        return {"data": result}
