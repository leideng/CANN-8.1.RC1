#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.

from common_func.db_manager import DBManager
from common_func.db_name_constant import DBNameConstant
from common_func.info_conf_reader import InfoConfReader
from common_func.ms_constant.number_constant import NumberConstant
from host_prof.host_prof_base.host_prof_data_base import HostProfDataBase


class HostMemUsage(HostProfDataBase):
    """
    host mem usage model
    """

    def __init__(self: any, result_dir: str) -> None:
        super().__init__(result_dir, DBNameConstant.DB_HOST_MEM_USAGE, [DBNameConstant.TABLE_HOST_MEM_USAGE])

    def flush_data(self: any) -> None:
        """
        flush all cache data to db
        :return: None
        """
        self.insert_mem_usage_data(self.cache_data)

    def insert_mem_usage_data(self: any, mem_usage_info: list) -> None:
        """
        insert mem usage info to table
        :param mem_usage_info: mem usage info
        :return: None
        """
        mem_usage_sql = "INSERT INTO {} VALUES(?, ?, ?)".format(DBNameConstant.TABLE_HOST_MEM_USAGE)
        DBManager.executemany_sql(self.conn, mem_usage_sql, mem_usage_info)

    def has_mem_usage_data(self: any) -> bool:
        """
        check has mem usage data
        :return: check result
        """
        return DBManager.judge_table_exist(self.cur, DBNameConstant.TABLE_HOST_MEM_USAGE)

    def get_mem_usage_data(self: any) -> dict:
        """
        get mem usage data
        :return: mem usage data
        """
        mem_info_sql = "SELECT * from {}".format(DBNameConstant.TABLE_HOST_MEM_USAGE)
        mem_info_list = DBManager.fetch_all_data(self.cur, mem_info_sql)

        result = []
        for mem_item in mem_info_list:
            time_mem = {
                "start": InfoConfReader().trans_into_local_time(mem_item[0], is_host=True),
                "end": InfoConfReader().trans_into_local_time(mem_item[1], is_host=True),
                "usage": mem_item[2]
            }
            result.append(time_mem)

        return {"data": result}

