#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

import logging

from common_func.db_manager import DBManager
from common_func.db_name_constant import DBNameConstant
from msmodel.interface.parser_model import ParserModel
from msmodel.interface.sql_helper import SqlWhereCondition


class RuntimeHostTaskModel(ParserModel):
    """
    class used to operate all runtime host task
    """

    def __init__(self: any, result_dir: str) -> None:
        super(RuntimeHostTaskModel, self).__init__(result_dir, DBNameConstant.DB_RUNTIME,
                                                   [DBNameConstant.TABLE_HOST_TASK])

    def flush(self: any, data_list: list, table_name: str = DBNameConstant.TABLE_HOST_TASK) -> None:
        """
        flush to db
        :param data_list: data
        :param table_name: table name
        :return:
        """
        self.insert_data_to_db(table_name, data_list)

    def get_host_tasks(self: any, is_all: bool, model_id: int, iter_id: int, device_id : int) -> list:
        sql = "select {0}.model_id, {0}.request_id as index_id, {0}.stream_id, {0}.task_id, " \
              "{0}.context_ids, {0}.batch_id, {0}.task_type, {0}.device_id,{0}.timestamp, " \
              "{0}.connection_id from {0} {1} order by {0}.timestamp" \
            .format(DBNameConstant.TABLE_HOST_TASK,
                    SqlWhereCondition.get_host_select_condition(is_all, model_id, iter_id, device_id))
        host_tasks = DBManager.fetch_all_data(self.cur, sql)
        if not host_tasks:
            logging.error("get host task from %s.%s error",
                          DBNameConstant.DB_RUNTIME, DBNameConstant.TABLE_HOST_TASK)
        return host_tasks
