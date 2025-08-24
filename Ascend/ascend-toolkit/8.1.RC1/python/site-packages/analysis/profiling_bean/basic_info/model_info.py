#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.

from common_func.msprof_query_data import MsprofQueryData
from profiling_bean.basic_info.query_data_bean import QueryDataBean


class ModelInfo:
    """
    model info class
    """

    def __init__(self: any) -> None:
        self.iterations = []

    def run(self: any, project_path: str) -> None:
        """
        run model info
        :return: None
        """
        query_data = MsprofQueryData(project_path).query_data()
        for data in query_data:
            if data.device_id.isdigit():
                self.iterations.append(IterationInfo(data))


class IterationInfo:
    """
    iteration info class
    """

    def __init__(self: any, data: QueryDataBean) -> None:
        self._device_id = int(data.device_id)
        self._model_id = data.model_id
        self._iteration_num = data.iteration_id
