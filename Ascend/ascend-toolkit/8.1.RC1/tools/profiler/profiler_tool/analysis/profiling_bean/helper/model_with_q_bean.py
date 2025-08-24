#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.

from profiling_bean.struct_info.struct_decoder import StructDecoder


class ModelWithQBean(StructDecoder):
    """
    step trace
    """

    def __init__(self: any, *args: any) -> None:
        data = args[0]
        self._timestamp = data[3]
        self._index_id = data[4]
        self._model_id = data[5]
        self._tag_id = data[6]
        self._event_id = data[8]

    @property
    def index_id(self: any) -> int:
        """
        get index id
        :return: index_id
        """
        return self._index_id

    @property
    def model_id(self: any) -> int:
        """
        get model id
        :return: model id
        """
        return self._model_id

    @property
    def timestamp(self: any) -> int:
        """
        get timestamp
        :return: timestamp
        """
        return self._timestamp

    @property
    def event_id(self: any) -> int:
        """
        get event id
        :return: event_id
        """
        return self._event_id

    @property
    def tag_id(self: any) -> int:
        """
        get tag id
        :return: tag id
        """
        return self._tag_id
