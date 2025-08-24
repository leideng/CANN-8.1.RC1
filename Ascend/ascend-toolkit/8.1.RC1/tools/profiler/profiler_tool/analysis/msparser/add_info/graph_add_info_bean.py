#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

from msparser.add_info.add_info_bean import AddInfoBean


class GraphAddInfoBean(AddInfoBean):
    """
    Graph Add Info Bean
    """

    def __init__(self: any, *args) -> None:
        super().__init__(*args)
        data = args[0]
        self._model_name = data[6]
        self._graph_id = data[7]

    @property
    def graph_id(self: any) -> str:
        """
        for graph id
        """
        return str(self._graph_id)

    @property
    def model_name(self: any) -> str:
        """
        for model name
        """
        return str(self._model_name)
