#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.

from common_func.db_name_constant import DBNameConstant
from profiling_bean.struct_info.task_type import TaskTypeBean


class TaskTypeReader:
    """
    class for task type reader
    """

    def __init__(self: any) -> None:
        self._data = []
        self._table_name = DBNameConstant.TABLE_TASK_TYPE

    @property
    def data(self: any) -> list:
        """
        get data
        :return: data
        """
        return self._data

    @property
    def table_name(self: any) -> str:
        """
        get table_name
        :return: table_name
        """
        return self._table_name

    def read_binary_data(self: any, bean_data: any) -> None:
        """
        read step trace binary data and store them into list
        :param bean_data: binary data
        :return: None
        """
        task_type_bean = TaskTypeBean.decode(bean_data)
        if task_type_bean:
            self._data.append(
                (task_type_bean.timestamp, task_type_bean.stream_id,
                 task_type_bean.task_id, task_type_bean.task_type, task_type_bean.task_state))
