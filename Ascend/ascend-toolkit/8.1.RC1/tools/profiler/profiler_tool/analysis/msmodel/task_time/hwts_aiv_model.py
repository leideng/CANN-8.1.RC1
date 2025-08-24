#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.

import logging
import os

from common_func.db_manager import DBManager
from common_func.db_name_constant import DBNameConstant
from common_func.ms_constant.number_constant import NumberConstant
from common_func.path_manager import PathManager
from msmodel.interface.parser_model import ParserModel
from msmodel.interface.sql_helper import SqlWhereCondition
from mscalculate.ascend_task.ascend_task import DeviceTask


class HwtsAivModel(ParserModel):
    """
    class used to operate hwts log model
    """

    def __init__(self: any, result_dir: str, table_list: list) -> None:
        super(HwtsAivModel, self).__init__(result_dir, DBNameConstant.DB_HWTS_AIV,
                                           [DBNameConstant.TABLE_HWTS_TASK, DBNameConstant.TABLE_HWTS_TASK_TIME])
        self.table_list = table_list

    def flush(self: any, data_list: list) -> tuple:
        """
        flush to db
        :param data_list: data
        :return:
        """
        return data_list, self.table_list

    def flush_data(self: any, data_list: list, table_name: str) -> None:
        """
        flush to db
        :param data_list: data
        :param table_name: table name
        :return:
        """
        self.insert_data_to_db(table_name, data_list)

    def clear(self: any) -> None:
        """
        clear db file
        :return:
        """
        if os.path.exists(PathManager.get_db_path(self.result_dir, DBNameConstant.DB_HWTS_AIV)):
            os.remove(PathManager.get_db_path(self.result_dir, DBNameConstant.DB_HWTS_AIV))

    def get_hwts_aiv_data_within_time_range(self: any, start_time: float, end_time: float) -> list:
        # in this chip subtask_id is always 0xffffffff
        sql = "select {1}.stream_id, {1}.task_id, {0} as context_id, {1}.running as timestamp, " \
              "{1}.complete - {1}.running as duration, {1}.task_type from {1} " \
              "{2} order by timestamp" \
            .format(NumberConstant.DEFAULT_GE_CONTEXT_ID, DBNameConstant.TABLE_HWTS_TASK_TIME,
                    SqlWhereCondition.get_interval_intersection_condition(
                        start_time, end_time, DBNameConstant.TABLE_HWTS_TASK_TIME, "running", "complete"))
        device_tasks = DBManager.fetch_all_data(self.cur, sql, dto_class=DeviceTask)
        if not device_tasks:
            logging.error("get device task from %s.%s error",
                          DBNameConstant.DB_HWTS_AIV, DBNameConstant.TABLE_HWTS_TASK_TIME)
        return device_tasks
