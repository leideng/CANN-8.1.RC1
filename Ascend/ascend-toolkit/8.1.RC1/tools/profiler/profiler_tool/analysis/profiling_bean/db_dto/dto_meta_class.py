#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.

class InstanceCheckMeta(type):
    def __instancecheck__(cls, obj):
        if obj is not None:
            return obj.__class__.__name__ == cls.__name__
        return False
