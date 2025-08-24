#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.

from common_func.db_manager import DBManager
from common_func.db_name_constant import DBNameConstant
from msconfig.config_manager import ConfigManager
from msmodel.interface.parser_model import ParserModel
from profiling_bean.db_dto.msproftx_dto import MsprofTxDto, MsprofTxExDto
from profiling_bean.db_dto.step_trace_dto import MsproftxMarkDto


class MsprofTxModel(ParserModel):
    """
    db operator for msproftx parser
    """
    TABLES_PATH = ConfigManager.TABLES

    def __init__(self: any, result_dir: str, db_name: str, table_list: list) -> None:
        super().__init__(result_dir, db_name, table_list)

    def flush(self: any, data_list: list) -> None:
        """
        flush msproftx data to db
        :param data_list:msproftx data list
        :return: None
        """
        self.insert_data_to_db(DBNameConstant.TABLE_MSPROFTX, data_list)

    def get_timeline_data(self: any) -> list:
        """
        get timeline data
        """
        if not DBManager.judge_table_exist(self.cur, DBNameConstant.TABLE_MSPROFTX):
            return []
        all_data_sql = f"select category, pid, tid, start_time, (end_time-start_time) as dur_time, payload_type, " \
                       f"payload_value, message_type, message, event_type " \
                       f"from {DBNameConstant.TABLE_MSPROFTX}"
        return DBManager.fetch_all_data(self.cur, all_data_sql, dto_class=MsprofTxDto)

    def get_summary_data(self: any) -> list:
        if not DBManager.judge_table_exist(self.cur, DBNameConstant.TABLE_MSPROFTX):
            return []
        all_data_sql = f"select pid, tid, category, event_type, payload_type, payload_value, start_time, " \
                       f"end_time, message_type, message from {DBNameConstant.TABLE_MSPROFTX}"
        return DBManager.fetch_all_data(self.cur, all_data_sql)


class MsprofTxExModel(ParserModel):
    """
    db operator for msproftx ex parser
    """
    def __init__(self: any, result_dir: str, db_name: str, table_list: list):
        super().__init__(result_dir, db_name, table_list)
        self.default_task_duration = 0

    def flush(self: any, data_list: list) -> None:
        """
        flush msproftx ex data to db
        :param data_list: msproftx ex data list
        :return: None
        """
        self.insert_data_to_db(DBNameConstant.TABLE_MSPROFTX_EX, data_list)

    def get_timeline_data(self) -> list:
        """
        get timeline data
        """
        if not DBManager.judge_table_exist(self.cur, DBNameConstant.TABLE_MSPROFTX_EX):
            return []
        all_data_sql = f"select pid, tid, event_type, start_time, (end_time-start_time) as dur_time, " \
                       f"mark_id, message, domain from {DBNameConstant.TABLE_MSPROFTX_EX}"
        return DBManager.fetch_all_data(self.cur, all_data_sql, dto_class=MsprofTxExDto)

    def get_summary_data(self) -> list:
        """
        get timeline data
        """
        if not DBManager.judge_table_exist(self.cur, DBNameConstant.TABLE_MSPROFTX_EX):
            return []
        all_data_sql = f"select pid, tid, event_type, start_time, end_time, " \
                       f"message, domain, mark_id from {DBNameConstant.TABLE_MSPROFTX_EX}"
        return DBManager.fetch_all_data(self.cur, all_data_sql)

    def get_device_data(self) -> list:

        if not DBManager.judge_table_exist(self.cur, DBNameConstant.TABLE_STEP_TRACE):
            return []
        all_data_sql = 'select index_id, timestamp, stream_id, task_id from {} ' \
              'where tag_id = 11'.format(DBNameConstant.TABLE_STEP_TRACE)
        task_list = DBManager.fetch_all_data(self.cur, all_data_sql, dto_class=MsproftxMarkDto)
        if not task_list:
            return []
        res_task_data = []
        task_list.sort(key=lambda x: (x.index_id, x.timestamp))
        res_task_data.append([task_list[0].index_id, task_list[0].timestamp,
                              task_list[0].stream_id, task_list[0].task_id, self.default_task_duration])
        for i in range(1, len(task_list)):
            if task_list[i].index_id == task_list[i - 1].index_id:
                # set range data duration
                res_task_data[-1][4] = task_list[i].timestamp - res_task_data[-1][1]
            else:
                res_task_data.append([task_list[i].index_id, task_list[i].timestamp,
                                      task_list[i].stream_id, task_list[i].task_id, self.default_task_duration])

        return res_task_data
