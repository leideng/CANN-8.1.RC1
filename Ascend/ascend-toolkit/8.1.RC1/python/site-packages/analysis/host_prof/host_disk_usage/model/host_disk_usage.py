#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.

from abc import abstractmethod

from common_func.db_manager import DBManager
from common_func.db_name_constant import DBNameConstant
from common_func.info_conf_reader import InfoConfReader
from common_func.ms_constant.number_constant import NumberConstant
from host_prof.host_prof_base.host_prof_data_base import HostProfDataBase


class HostDiskUsage(HostProfDataBase):
    """
    host disk usage model
    """

    def __init__(self: any, result_dir: str) -> None:
        super().__init__(result_dir, DBNameConstant.DB_HOST_DISK_USAGE, [DBNameConstant.TABLE_HOST_DISK_USAGE])

    def insert_disk_usage_data(self: any, disk_usage_info: list) -> None:
        """
        insert disk usage info to table
        :param disk_usage_info: disk usage info
        :return: None
        """
        disk_usage_sql = "INSERT INTO {} VALUES(?, ?, ?, ?, ?, ?)".format(DBNameConstant.TABLE_HOST_DISK_USAGE)
        DBManager.executemany_sql(self.conn, disk_usage_sql, disk_usage_info)

    def has_disk_usage_data(self: any) -> bool:
        """
        check has disk usage data
        :return: check result
        """
        return DBManager.judge_table_exist(self.cur, DBNameConstant.TABLE_HOST_DISK_USAGE)

    def get_disk_usage_data(self: any) -> dict:
        """
        disk usage data
        :return: disk usage data
        """
        disk_info_sql = "SELECT * from {}".format(DBNameConstant.TABLE_HOST_DISK_USAGE)
        disk_info_list = DBManager.fetch_all_data(self.cur, disk_info_sql)

        result = []
        for disk_item in disk_info_list:
            time_mem = {
                "start": InfoConfReader().trans_into_local_time(disk_item[0], is_host=True),
                "end": InfoConfReader().trans_into_local_time(disk_item[1], is_host=True),
                "usage": disk_item[5]
            }
            result.append(time_mem)

        return {"data": result}

    @abstractmethod
    def flush_data(self: any) -> any:
        """
        flush all host disk data
        """
