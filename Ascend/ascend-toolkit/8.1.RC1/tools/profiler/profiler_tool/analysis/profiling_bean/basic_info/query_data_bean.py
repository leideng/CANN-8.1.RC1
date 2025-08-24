#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.

from common_func.msprof_common import MsProfCommonConstant


class QueryDataBean:
    """
    per data to show under the command of query
    """

    def __init__(self: any, **args: dict) -> None:
        self._job_info = args.get(MsProfCommonConstant.JOB_INFO)
        self._job_name = args.get(MsProfCommonConstant.JOB_NAME)
        self._model_id = args.get(MsProfCommonConstant.MODEL_ID)
        self._device_id = args.get(MsProfCommonConstant.DEVICE_ID)
        self._iteration_nums = args.get(MsProfCommonConstant.ITERATION_ID)
        self._collection_time = args.get(MsProfCommonConstant.COLLECTION_TIME)
        self._top_time_iteration = args.get(MsProfCommonConstant.TOP_TIME_ITERATION)
        self._rank_id = args.get(MsProfCommonConstant.RANK_ID)

    @property
    def job_info(self: any) -> str:
        """
        job info
        :return: job info
        """
        return self._job_info

    @property
    def job_name(self: any) -> str:
        """
        job name
        :return:job name
        """
        return self._job_name

    @property
    def model_id(self: any) -> str:
        """
        model id
        :return:model id
        """
        return self._model_id

    @property
    def device_id(self: any) -> str:
        """
        device id
        :return: device id
        """
        return self._device_id

    @property
    def iteration_id(self: any) -> str:
        """
        iteration id
        :return: iteration id
        """
        return self._iteration_nums

    @property
    def collection_time(self: any) -> str:
        """
        collection time
        :return: collection time
        """
        return self._collection_time

    @property
    def top_time_iteration(self: any) -> str:
        """
        top time iteration
        :return: top time iteration
        """
        return self._top_time_iteration

    @property
    def rank_id(self: any) -> str:
        """
        rank id
        :return: rank id
        """
        return self._rank_id
