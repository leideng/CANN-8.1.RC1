#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

import logging

from common_func.db_manager import DBManager
from common_func.db_name_constant import DBNameConstant
from common_func.ms_constant.number_constant import NumberConstant
from msmodel.interface.base_model import BaseModel
from msmodel.interface.sql_helper import SqlWhereCondition
from mscalculate.ascend_task.ascend_task import DeviceTask


class RuntimeTaskTimeModel(BaseModel):

    def __init__(self: any, result_dir: str) -> None:
        super(RuntimeTaskTimeModel, self).__init__(
            result_dir, DBNameConstant.DB_RUNTIME, [DBNameConstant.TABLE_RUNTIME_TASK_TIME])

    def get_runtime_task_data_within_time_range(self: any, start_time: float, end_time: float) -> list:
        # in this chip subtask_id is always 0xffffffff
        # in this chip task time is uploaded by ts, which is stored in ts_track.data
        sql = "select {1}.stream_id, {1}.task_id, {0} as context_id, {1}.running as timestamp, " \
              "{1}.complete - {1}.running as duration, {1}.tasktype as task_type from {1} " \
              "{2} order by timestamp" \
            .format(NumberConstant.DEFAULT_GE_CONTEXT_ID, DBNameConstant.TABLE_RUNTIME_TASK_TIME,
                    SqlWhereCondition.get_interval_intersection_condition(
                        start_time, end_time, DBNameConstant.TABLE_RUNTIME_TASK_TIME, "running", "complete"))
        device_tasks = DBManager.fetch_all_data(self.cur, sql, dto_class=DeviceTask)
        if not device_tasks:
            logging.error("get device task from %s.%s error",
                          DBNameConstant.DB_HWTS, DBNameConstant.TABLE_HWTS_TASK_TIME)
        return device_tasks
