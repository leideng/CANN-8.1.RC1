#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
from collections import namedtuple

from common_func.db_name_constant import DBNameConstant
from common_func.constant import Constant
from common_func.db_manager import DBManager
from common_func.msprof_object import CustomizedNamedtupleFactory
from msmodel.interface.parser_model import ParserModel
from msmodel.interface.view_model import ViewModel


class GeModel(ParserModel):
    """
    ge info model class
    """

    def __init__(self: any, result_dir: str, table_list: list) -> None:
        super().__init__(result_dir, DBNameConstant.DB_GE_INFO, table_list)
        self._current_table_name = None

    def flush_all(self: any, data_dict: dict) -> None:
        """
        insert all ge data to table
        :param data_dict: ge data
        :return:
        """
        for table_name in data_dict.keys():
            self._current_table_name = table_name
            self.flush(data_dict.get(table_name, []))

    def flush(self: any, data_list: list) -> None:
        """
        insert ge data into database
        """
        self.insert_data_to_db(self._current_table_name, data_list)

    def delete_table(self: any, table_name: str) -> None:
        """
        delete ge data
        """
        self.cur.execute('delete from {}'.format(table_name))

    def get_ge_model_name(self: any) -> any:
        """
        get ge model name
        """
        return self.__class__.__name__


class GeInfoViewModel(ViewModel):
    TASK_INFO_TYPE = CustomizedNamedtupleFactory.enhance_namedtuple(
        namedtuple("TaskInfo",
                   ["model_id", "op_name", "stream_id", "task_id", "block_dim", "mix_block_dim",
                    "op_stat", "task_type", "op_type", "index_id", "thread_id", "timestamp", "batch_id",
                    "tensor_num", "input_formats", "input_data_types", "input_shapes",
                    "output_formats", "output_data_types", "output_shapes", "device_id", "context_id",
                    "op_flag", "hashid"]),
        {})

    def __init__(self, result_dir: str, table_list: list):
        super().__init__(result_dir, DBNameConstant.DB_GE_INFO, table_list)

    def get_ge_info_by_device_id(self: any, table_name: str, device_id: str):
        ge_sql = "select * from {0} where device_id={1} " \
                 "and task_type != '{2}' and task_type != '{3}'" \
            .format(table_name, device_id,
                    Constant.TASK_TYPE_HCCL, Constant.TASK_TYPE_HCCL_AI_CPU)
        task_info_data = DBManager.fetch_all_data(self.cur, ge_sql)
        return [self.TASK_INFO_TYPE(*data) for data in task_info_data]
