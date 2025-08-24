#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.

from common_func.db_manager import DBManager
from common_func.db_name_constant import DBNameConstant
from msmodel.interface.view_model import ViewModel


class QosViewModel(ViewModel):
    """
    QoS model class
    """
    def __init__(self: any, result_dir: str, db_name: str, table_list: list) -> None:
        super().__init__(result_dir, db_name, table_list)

    def get_timeline_data(self: any) -> list:
        """
        get qos bandwidth data
        :return: list
        """
        sql = "select * from {};".format(DBNameConstant.TABLE_QOS_BW)
        return DBManager.fetch_all_data(self.cur, sql)
