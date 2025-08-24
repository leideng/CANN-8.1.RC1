#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.


class AdditionalRecord:
    """
    This class is as an abstraction of additional information
    """
    _ID = 0

    def __init__(self, add_dto: any = None, timestamp: float = -1, struct_type: str = ""):
        self.dto = add_dto
        self.timestamp = timestamp
        self.id = self._ID
        self.struct_type = struct_type
        AdditionalRecord._ID += 1

    def __lt__(self, other):
        return self.timestamp > other.timestamp

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == self.id

    def __str__(self):
        return self.struct_type + "-" + str(self.id)
