#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.

import logging
from collections import deque
from typing import Tuple

from common_func.ms_constant.stars_constant import StarsConstant
from msmodel.sqe_type_map import SqeType
from msmodel.stars.acsq_task_model import AcsqTaskModel
from msparser.interface.istars_parser import IStarsParser
from profiling_bean.stars.acsq_task import AcsqTask


class AcsqTaskParser(IStarsParser):
    """
    class used to parser acsq task log
    """

    def __init__(self: any, result_dir: str, db: str, table_list: list) -> None:
        super().__init__()
        self._model = AcsqTaskModel(result_dir, db, table_list)
        self._decoder = AcsqTask
        self._data_list = []
        self._mismatch_task = []

    def preprocess_data(self: any) -> None:
        """
        preprocess data list
        :return: NA
        """
        self._data_list, self._mismatch_task = self.get_task_time()

    def get_task_time(self: any) -> Tuple[list, list]:
        """
        Categorize data_list into start log and end log, and calculate the task time
        :return: result data list
        """
        task_map = {}
        # task id stream id func type
        self._data_list.sort(key=lambda x: x.sys_cnt)
        for data in self._data_list:
            task_key = "{0},{1}".format(str(data.task_id), str(data.stream_id))
            task_map.setdefault(task_key, {}).setdefault(data.func_type, deque([])).append(data)

        matched_result = []
        remaining_data = []
        mismatch_start_count = 0
        mismatch_end_count = 0
        for data_key, data_dict in task_map.items():
            start_que = data_dict.get(StarsConstant.ACSQ_START_FUNCTYPE, [])
            end_que = data_dict.get(StarsConstant.ACSQ_END_FUNCTYPE, [])
            while start_que and end_que:
                start_task = start_que[0]
                end_task = end_que[0]
                if start_task.sys_cnt > end_task.sys_cnt:
                    mismatch_end_count += 1
                    _ = end_que.popleft()
                    continue
                start_task = start_que.popleft()
                end_task = end_que.popleft()
                matched_result.append(
                    [start_task.stream_id, start_task.task_id, start_task.acc_id, \
                        SqeType().instance(start_task.task_type).name,
                     # start timestamp end timestamp duration
                     start_task.sys_cnt, end_task.sys_cnt, end_task.sys_cnt - start_task.sys_cnt])
            if len(start_que) > 1 or end_que:
                logging.debug("Acsq task mismatch happen in %s, start_que size: %d, end_que size: %d",
                              data_key, len(start_que), len(end_que))
                mismatch_start_count += len(start_que)
                mismatch_end_count += len(end_que)
                # when mismatched, discards illegal task
                continue
            while start_que:
                start_task = start_que.popleft()
                remaining_data.append(start_task)
        if mismatch_end_count > 0:
            logging.warning("There are %d acsq end logs mismatching.", mismatch_end_count)
        if mismatch_start_count > 0:
            logging.error("There are %d acsq start logs mismatching.", mismatch_start_count)

        return sorted(matched_result, key=lambda data: data[4]), remaining_data

    def flush(self: any) -> None:
        """
        flush all buffer data to db
        :return: NA
        """
        if not self._data_list:
            return
        if self._model.init():
            self.preprocess_data()
            self._model.flush(self._data_list)
            self._model.finalize()
            self._data_list.clear()
            self._data_list.extend(self._mismatch_task)
            self._mismatch_task.clear()
