#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.


def singleton(cls: any) -> any:
    """
    singleton Decorators
    """
    _instance = {}

    def _singleton(*args: any, **kw: any) -> any:
        if cls not in _instance:
            _instance[cls] = cls(*args, **kw)
        return _instance.get(cls)

    return _singleton
