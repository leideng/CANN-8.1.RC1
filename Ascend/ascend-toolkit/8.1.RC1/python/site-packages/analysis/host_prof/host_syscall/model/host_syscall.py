#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.

from common_func.db_manager import DBManager
from common_func.db_name_constant import DBNameConstant
from host_prof.host_prof_base.host_prof_data_base import HostProfDataBase


class HostSyscall(HostProfDataBase):
    """
    class used to operate os runtime api db
    """

    def __init__(self: any, result_dir: str) -> None:
        super().__init__(result_dir, DBNameConstant.DB_HOST_RUNTIME_API, [DBNameConstant.TABLE_HOST_RUNTIME_API])

    def flush_data(self: any) -> None:
        """
        flush all cache data to db
        :return: None
        """
        self.insert_runtime_api(self.cache_data)

    def insert_runtime_api(self: any, runtime_api_info: list) -> None:
        """
        insert host syscall info to table
        :param runtime_api_info: runtime api info
        :return: None
        """
        insert_sql = "INSERT INTO {} VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)".format(
            DBNameConstant.TABLE_HOST_RUNTIME_API)
        DBManager.executemany_sql(self.conn, insert_sql, runtime_api_info)

    def has_runtime_api_data(self: any) -> bool:
        """
        check has host syscall data
        :return: check result
        """
        return DBManager.judge_table_exist(self.cur, DBNameConstant.TABLE_HOST_RUNTIME_API)

    def get_summary_runtime_api_data(self: any) -> list:
        """
        get host runtime summary data
        :return: list of runtime api
        """
        runtime_api_sql = "SELECT runtime_pid, runtime_tid, runtime_api_name, " \
                          "sum(runtime_duration) / (select sum(runtime_duration) from {0}) as " \
                          "Percentage, " \
                          "sum(runtime_duration) as Time, count(1), avg(runtime_duration), " \
                          "max(runtime_duration), min(runtime_duration) " \
                          "from {0} group by runtime_api_name, runtime_tid " \
                          "order by Time desc".format(DBNameConstant.TABLE_HOST_RUNTIME_API)
        runtime_api_list = DBManager.fetch_all_data(self.cur, runtime_api_sql)
        return runtime_api_list

    def get_runtime_api_data(self: any) -> list:
        """
        get host syscall data
        :return: runtime api data
        """
        runtime_api_sql = "SELECT * from {0}".format(DBNameConstant.TABLE_HOST_RUNTIME_API)
        return DBManager.fetch_all_data(self.cur, runtime_api_sql)

    def get_all_tid(self: any) -> list:
        """
        get all tid
        :return: tid list
        """
        tid_sql = "select distinct(runtime_tid) from {0}".format(DBNameConstant.TABLE_HOST_RUNTIME_API)
        return DBManager.fetch_all_data(self.cur, tid_sql)
