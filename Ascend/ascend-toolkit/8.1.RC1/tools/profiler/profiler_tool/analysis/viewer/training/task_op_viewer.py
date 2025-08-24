#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.

import logging
import sqlite3
from typing import List, Tuple

from common_func.constant import Constant
from common_func.db_manager import DBManager
from common_func.db_name_constant import DBNameConstant
from common_func.info_conf_reader import InfoConfReader
from common_func.msvp_common import format_high_precision_for_csv
from common_func.ms_constant.number_constant import NumberConstant
from profiling_bean.db_dto.ge_task_dto import GeTaskDto
from viewer.memory_copy.memory_copy_viewer import MemoryCopyViewer
from viewer.task_time_viewer import TaskTimeViewer


class TaskOpViewer:
    """
    viewer of training trace data
    """
    INVALID_CONTEXT_ID = 4294967295

    @staticmethod
    def get_task_op_summary(message: dict) -> Tuple[List[str], List, int]:
        """
        @param message
        Rewrite gRPC task op method.
        """
        headers = [
            "kernel_name", "kernel_type", "stream_id", "task_id",
            "task_time(us)", "task_start(us)", "task_stop(us)"
        ]
        if not message:
            logging.error("get_task_op_summary message empty")
            return headers, [], 0
        ascend_conn, ascend_curs = DBManager.check_connect_db(message.get("result_dir"), DBNameConstant.DB_ASCEND_TASK)
        ge_conn, ge_curs = DBManager.check_connect_db(message.get("result_dir"), DBNameConstant.DB_GE_INFO)
        data, _ = TaskOpViewer.get_task_data_summary(ascend_curs, ge_curs)
        DBManager.destroy_db_connect(ascend_conn, ascend_curs)
        DBManager.destroy_db_connect(ge_conn, ge_curs)
        if not data:
            return headers, [], 0
        start_ts, _ = InfoConfReader().get_collect_time()
        task_start_index = 5
        logging.info("There are %d records before task_time data filtering, timestamp is %s", len(data), start_ts)
        filtered_data = [item for item in data if item[task_start_index] > start_ts]
        logging.info("There are %d records after task_time data filtering.", len(filtered_data))
        data = TaskOpViewer._add_memcpy_data(message.get("result_dir"), filtered_data)
        return headers, data, len(data)

    @staticmethod
    def get_task_data_summary(ascend_curs: sqlite3.Cursor, ge_curs: sqlite3.Cursor) -> Tuple[List, int]:
        """
        get task info csv
        """
        if not DBManager.judge_table_exist(ge_curs, DBNameConstant.TABLE_GE_TASK):
            logging.warning(
                "No ge data collected, maybe the TaskInfo table is not created, try to export data with no ge data")

        if not DBManager.judge_table_exist(ascend_curs, DBNameConstant.TABLE_ASCEND_TASK):
            logging.warning("table AscendTask not exists, can't export device tasks.")
            return [], 0
        # select device & non-ffts-plus task only
        sql = (
            f"SELECT stream_id, task_id, batch_id, host_task_type, CAST(start_time AS INTEGER), " 
            f"CAST(duration AS INTEGER), device_task_type "
            f"from {DBNameConstant.TABLE_ASCEND_TASK} "
            f"where device_task_type != '{Constant.TASK_TYPE_UNKNOWN}'"
            f" and context_id = {TaskOpViewer.INVALID_CONTEXT_ID}"
        )
        device_tasks = DBManager.fetch_all_data(ascend_curs, sql)
        if not device_tasks:
            logging.error("No device task fetched, can't export task_time.")
            return [], 0

        try:
            task_info = TaskOpViewer._reformat_task_info(device_tasks, ge_curs)
        except (OSError, SystemError, ValueError, TypeError, RuntimeError) as error:
            logging.error(str(error), exc_info=Constant.TRACE_BACK_SWITCH)
            return [], 0
        return task_info, len(task_info)

    @staticmethod
    def _add_memcpy_data(result_dir: str, data: List) -> List:
        memcpy_viewer = MemoryCopyViewer(result_dir)
        memcpy_data = memcpy_viewer.get_memory_copy_non_chip0_summary()
        data.extend(memcpy_data)

        return data

    @staticmethod
    def _reformat_task_info(task_data: List, ge_curs: sqlite3.Cursor) -> List:
        ge_sql = f"SELECT op_name, task_type, stream_id, task_id, batch_id from {DBNameConstant.TABLE_GE_TASK}"
        op_name_list = DBManager.fetch_all_data(ge_curs, ge_sql, dto_class=GeTaskDto)
        op_info_dict = {
            (row.stream_id, row.task_id, row.batch_id): {
                "op_name": row.op_name,
                "task_type": row.task_type
            }
            for row in op_name_list
        }
        task_info = []
        for stream_id, task_id, batch_id, host_task_type, start_time, duration, device_task_type in task_data:
            op_info = op_info_dict.get((stream_id, task_id, batch_id), {})
            op_name: str = op_info.get("op_name", "N/A")
            task_type: str = TaskTimeViewer.get_task_type(host_task_type, device_task_type)
            task_type = op_info.get("task_type", task_type)
            task_time: float = round(duration / DBManager.NSTOUS, NumberConstant.ROUND_THREE_DECIMAL)
            task_start = format_high_precision_for_csv(
                InfoConfReader().trans_into_local_time(start_time))
            task_stop = format_high_precision_for_csv(
                InfoConfReader().trans_into_local_time(start_time + duration))
            task_info.append((
                op_name, task_type, stream_id, task_id,
                task_time, task_start, task_stop,
            ))
        return task_info
