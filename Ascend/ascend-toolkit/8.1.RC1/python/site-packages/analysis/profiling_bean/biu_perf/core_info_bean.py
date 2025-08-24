#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.


class CoreInfo:
    AI_CUBE = "aic"
    AI_VECTOR0 = "aiv0"
    AI_VECTOR1 = "aiv1"

    def __init__(self: any, core_name: str) -> None:
        self._core_name = core_name
        self._group_id = int(core_name.split("_")[1])
        self._core_type = core_name.split("_")[2]
        self.core_id = self.calculate_core_id()
        self.file_list = []

    @property
    def core_name(self: any) -> str:
        """
        get core name
        """
        return self._core_name

    @property
    def group_id(self: any) -> int:
        """
        get group id
        """
        return self._group_id

    @property
    def core_type(self: any) -> str:
        """
        get core type
        """
        return self._core_type

    def calculate_core_id(self: any) -> int:
        """
        calculate core id
        """
        # aic core id is starting from 0
        if self._core_type == self.AI_CUBE:
            return self._group_id

        # aiv0 core id is starting from 25
        if self._core_type == self.AI_VECTOR0:
            return self._group_id * 2 + 25

        # aiv1 core id is starting from 26
        if self._core_type == self.AI_VECTOR1:
            return self._group_id * 2 + 26

        return -1
