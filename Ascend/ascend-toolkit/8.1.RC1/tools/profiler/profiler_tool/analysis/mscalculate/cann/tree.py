#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

from typing import List

from mscalculate.cann.event import Event


class TreeNode:
    def __init__(self, event: Event):
        self.event = event
        self.children: List[TreeNode] = list()

    def __str__(self):
        return str(self.event)

    def add_child(self, child):
        self.children.append(child)
