#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.


class EmptyClass:
    """
    Empty class
    """

    def __init__(self: any, info: str = "") -> None:
        self._info = info

    @classmethod
    def __bool__(cls: any) -> bool:
        return False

    @classmethod
    def __str__(cls: any) -> str:
        return ""

    @property
    def info(self: any) -> str:
        """
        get info
        :return: _info
        """
        return self._info

    @staticmethod
    def is_empty() -> bool:
        """
        return this is a empty class
        """
        return True
