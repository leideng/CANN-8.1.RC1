#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.

from abc import ABC

from common_func.db_manager import DBManager
from common_func.db_name_constant import DBNameConstant
from msconfig.config_manager import ConfigManager
from msmodel.interface.base_model import BaseModel


class PcieModel(BaseModel, ABC):
    """
    acsq task model class
    """
    TABLES_PATH = ConfigManager.TABLES_TRAINING

    def flush(self: any, data_list: list) -> None:
        """
        flush acsq task data to db
        :param data_list:acsq task data list
        :return: None
        """
        self.insert_data_to_db(DBNameConstant.TABLE_PCIE, data_list)

    def create_table(self: any) -> None:
        """
        create table
        """
        for table_name in self.table_list:
            if DBManager.judge_table_exist(self.cur, table_name):
                continue
            sql = DBManager.sql_create_general_table("PCIeDataMap", table_name, self.TABLES_PATH)
            DBManager.execute_sql(self.conn, sql)
