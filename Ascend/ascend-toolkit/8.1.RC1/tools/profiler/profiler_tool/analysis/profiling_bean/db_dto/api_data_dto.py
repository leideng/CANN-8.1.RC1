#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
from collections import namedtuple
from dataclasses import dataclass
from profiling_bean.db_dto.event_data_dto import EventDataDto
from profiling_bean.db_dto.dto_meta_class import InstanceCheckMeta

INVALID_LEVEL = -1
INVALID_THREAD = -1


@dataclass
class ApiDataDto(metaclass=InstanceCheckMeta):
    connection_id: int = None
    end: int = None
    id: str = ""
    item_id: str = ""
    level: str = ""
    request_id: int = None
    start: int = None
    struct_type: str = ""
    thread_id: int = None


ApiDataDtoTuple = namedtuple("ApiDataDto",
                             [
                                 "connection_id", "end", "id", "item_id", "level",
                                 "request_id", "start", "struct_type",
                                 "thread_id"
                             ],
                             defaults=[None, None, None, None, None, None, None, None, None])


def invalid_dto(level=INVALID_LEVEL, thread=INVALID_THREAD, start=-1, end=-1, struct_type="", ):
    return ApiDataDtoTuple(struct_type=struct_type, level=level, thread_id=thread, start=start, end=end)


def generate_api_data_from_event(begin_time=None, end_event_data_dto: EventDataDto = None):
    if EventDataDto:
        return ApiDataDtoTuple(struct_type=end_event_data_dto.struct_type,
                               level=end_event_data_dto.level,
                               thread_id=end_event_data_dto.thread_id,
                               start=begin_time,
                               end=end_event_data_dto.timestamp,
                               item_id=end_event_data_dto.item_id,
                               request_id=end_event_data_dto.request_id,
                               connection_id=end_event_data_dto.connection_id,
                               id=0)
    else:
        return ApiDataDtoTuple()
