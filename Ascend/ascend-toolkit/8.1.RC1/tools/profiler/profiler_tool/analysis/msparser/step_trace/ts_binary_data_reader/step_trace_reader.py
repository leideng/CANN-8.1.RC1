#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.

from common_func.db_name_constant import DBNameConstant
from profiling_bean.struct_info.step_trace import StepTrace


class StepTraceReader:
    """
    class for step trace reader
    """

    def __init__(self: any) -> None:
        self._data = []
        self._table_name = DBNameConstant.TABLE_STEP_TRACE

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
        step_trace_bean = StepTrace.decode(bean_data)
        if step_trace_bean:
            self._data.append(
                (step_trace_bean.index_id, step_trace_bean.model_id,
                 step_trace_bean.timestamp,
                 step_trace_bean.stream_id, step_trace_bean.task_id, step_trace_bean.tag_id))
