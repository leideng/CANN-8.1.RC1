#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.

from abc import abstractmethod


class MetaCalculator:
    """
    abstract class for cluster communication and optimization suggestion
    """
    def __init__(self):
        self.suggestions = []

    @abstractmethod
    def run(self):
        self.calculate()

    @abstractmethod
    def calculate(self):
        """
        according to specific rules to give suggestions
        """
        return
