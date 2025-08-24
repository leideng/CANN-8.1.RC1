#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.

import logging
import os

from common_func.db_manager import DBManager
from common_func.db_name_constant import DBNameConstant
from common_func.ms_constant.number_constant import NumberConstant
from common_func.msvp_common import error
from common_func.path_manager import PathManager
from msmodel.interface.parser_model import ParserModel
from msmodel.interface.sql_helper import SqlWhereCondition
from mscalculate.ascend_task.ascend_task import DeviceTask


class HwtsLogModel(ParserModel):
    """
    class used to operate hwts log model
    """

    def __init__(self: any, result_dir: str) -> None:
        super(HwtsLogModel, self).__init__(result_dir, DBNameConstant.DB_HWTS,
                                           [DBNameConstant.TABLE_HWTS_TASK, DBNameConstant.TABLE_HWTS_TASK_TIME])

    def flush(self: any, data_list: list, table_name: str) -> None:
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
        try:
            if os.path.exists(PathManager.get_db_path(self.result_dir, DBNameConstant.DB_HWTS)):
                os.remove(PathManager.get_db_path(self.result_dir, DBNameConstant.DB_HWTS))
        except (OSError, SystemError) as err:
            error(os.path.basename(__file__), str(err))

    def get_hwts_data_within_time_range(self: any, start_time: float, end_time: float) -> list:
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
                          DBNameConstant.DB_HWTS, DBNameConstant.TABLE_HWTS_TASK_TIME)
        return device_tasks
