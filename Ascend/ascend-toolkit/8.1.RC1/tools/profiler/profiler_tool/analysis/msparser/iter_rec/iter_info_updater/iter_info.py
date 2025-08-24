#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.

from msmodel.ge.ge_info_calculate_model import GeInfoModel


class IterInfo:
    """
    class used to record iter info.
    """

    def __init__(self: any, model_id: int = -1,
                 index_id: int = -1,
                 iter_id: int = -1,
                 start_time: int = -1,
                 end_time: int = -1) -> None:
        self.model_id = model_id
        self.index_id = index_id
        self.iter_id = iter_id
        self.start_time = start_time
        self.end_time = end_time
        self.behind_parallel_iter = set([])
        self.aic_count = 0
        self.hwts_count = 0
        self.aic_offset = 0
        self.hwts_offset = 0
        self.static_aic_task_set = set([])
        self.dynamic_aic_task_set = set([])

    @staticmethod
    def class_name() -> str:
        """
        class name
        """
        return "IterInfo"

    @staticmethod
    def file_name() -> str:
        """
        file name
        """
        return "iter_rec_parser"

    def is_aicore(self: any, task: any) -> bool:
        """
        judge is aicore
        """
        stream_task_batch = GeInfoModel.STREAM_TASK_BATCH_KEY_FMT.format(task.stream_id, task.task_id, task.batch_id)
        return stream_task_batch in self.static_aic_task_set or stream_task_batch in self.dynamic_aic_task_set
