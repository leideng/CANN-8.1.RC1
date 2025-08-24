#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2021-2023. All rights reserved.

from common_func.ms_constant.number_constant import NumberConstant
from common_func.profiling_scene import ProfilingScene
from common_func.utils import Utils


class OpCommonFunc:
    """
    common function of op summary and op statistic
    """

    TASK_TIME_COL_NUM = 8
    TASK_ID = "task_id"
    BATCH_ID = "batch_id"
    DEFAULT_NULL_NUMBER = NumberConstant.NULL_NUMBER

    @classmethod
    def calculate_task_time(cls: any, data: list) -> list:
        """
        calculate task time data
        Return: task time data
        """
        res = Utils.generator_to_list(Utils.generator_to_list(0 for _ in range(cls.TASK_TIME_COL_NUM))
                                      for _ in range(len(data)))
        previous_complete_time = 0
        for row_num, content in enumerate(data):
            # each row contains task id, stream id, start time,
            # duration time, wait time, task_type, index_id, model_id, batch_id, subtask_id
            if not Utils.is_valid_num(float(content[2])) or \
                    not Utils.is_valid_num(float(content[3])):
                continue
            res[row_num][0] = content[0]  # task id
            res[row_num][1] = content[1]  # stream id
            res[row_num][2] = float(content[2])  # start time
            res[row_num][3] = float(content[3])  # duration time
            # wait time
            res[row_num][4] = 0
            res[row_num][5] = content[4]
            res[row_num][6] = content[5]  # index_id
            if not ProfilingScene().is_operator():
                res[row_num][7] = content[6]  # model_id
                res[row_num].append(content[7])  # batch_id
            else:
                res[row_num][7] = content[7]  # batch_id
            # index -1 is subtask_id
            res[row_num].append(content[-1])
            previous_complete_time = float(content[2]) + float(content[3])
        return res

    @classmethod
    def deal_batch_id(cls: any, stream_index: int, task_index: int, merge_data: list) -> list:
        """
        add batch id for op summary
        :param stream_index: index for stream id
        :param task_index: index for task id
        :param merge_data: data to add batch id
        :return: result
        """
        stream_max_value = {}
        result = [0] * len(merge_data)
        for index, ge_data in enumerate(merge_data):
            stream_id = str(ge_data[stream_index])
            task_id = ge_data[task_index]
            if stream_id not in stream_max_value:
                stream_max_value.setdefault(stream_id, {cls.TASK_ID: task_id, cls.BATCH_ID: 0})
            else:
                current_max_value = stream_max_value.pop(stream_id)
                if int(current_max_value.get(cls.TASK_ID, -1)) >= int(task_id):
                    current_max_value[cls.BATCH_ID] += 1
                current_max_value[cls.TASK_ID] = task_id
                stream_max_value.setdefault(stream_id, current_max_value)
            result[index] = list(ge_data) + [stream_max_value.get(stream_id).get(cls.BATCH_ID)]
        return result

    @classmethod
    def _get_wait_time(cls: any, row_num: int, time_data: float, previous_complete_time: int) -> float:
        """
        get wait time
        :param time_data:
        :param per:
        :return:
        """
        if row_num == 0 or (int(time_data) - previous_complete_time) < 0:
            return cls.DEFAULT_NULL_NUMBER
        return int(time_data) - previous_complete_time
