#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.

from common_func.db_name_constant import DBNameConstant
from msconfig.config_manager import ConfigManager
from msmodel.interface.base_model import BaseModel


class RtsModel(BaseModel):
    """
    db operator for acl parser
    """
    TABLES_PATH = ConfigManager.TABLES

    @staticmethod
    def class_name() -> str:
        """
        class name
        """
        return "RtsModel"

    def create_rts_db(self: any, data: list) -> None:
        """
        create rts_track.db
        :param data:
        :return:
        """
        self.init()
        self.check_db()
        self.create_table()
        self.insert_data_to_db(DBNameConstant.TABLE_TASK_TRACK, data)
        self.finalize()
