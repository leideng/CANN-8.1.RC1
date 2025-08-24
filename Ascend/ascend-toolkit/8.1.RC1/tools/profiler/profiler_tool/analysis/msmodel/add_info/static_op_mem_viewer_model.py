#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

from common_func.db_manager import DBManager
from common_func.db_name_constant import DBNameConstant
from msmodel.interface.view_model import ViewModel


class StaticOpMemViewModel(ViewModel):
    """
    Model for StaticOpMemParser
    """
    def get_summary_data(self: any) -> list:
        """
        get static_op_mem data
        :return: list
        """
        sql = "select * from {};".format(DBNameConstant.TABLE_STATIC_OP_MEM)
        return DBManager.fetch_all_data(self.cur, sql)
