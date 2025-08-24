#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
from common_func.db_name_constant import DBNameConstant
from msparser.step_trace.ts_binary_data_reader.task_flip_bean import TaskFlipBean
from common_func.info_conf_reader import InfoConfReader


class TaskFlipReader:
    """
    class for task flip reader
    """

    def __init__(self: any) -> None:
        self._data = []
        self._table_name = DBNameConstant.TABLE_DEVICE_TASK_FLIP

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
        read task flip binary data and store them into list
        :param bean_data: binary data
        :return: None
        """
        task_flip_bean = TaskFlipBean.decode(bean_data)
        if task_flip_bean:
            self._data.append(
                [task_flip_bean.stream_id,
                 InfoConfReader().time_from_syscnt(task_flip_bean.timestamp),
                 task_flip_bean.task_id,
                 task_flip_bean.flip_num,
                 ]
            )
