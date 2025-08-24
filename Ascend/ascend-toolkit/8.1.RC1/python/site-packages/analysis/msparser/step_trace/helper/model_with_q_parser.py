#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.

from common_func.db_name_constant import DBNameConstant
from profiling_bean.helper.model_with_q_bean import ModelWithQBean


class ModelWithQParser:
    """
    class for Model with Q
    """

    def __init__(self: any) -> None:
        self._data = []
        self._table_name = DBNameConstant.TABLE_MODEL_WITH_Q

    @property
    def data(self: any) -> list:
        """
        get data
        :return: data
        """
        return self._data

    @property
    def table_name(self: any) -> str:
        """
        get table_name
        :return: table_name
        """
        return self._table_name

    def read_binary_data(self: any, bean_data: any) -> None:
        """
        read model with q binary data and store them into list
        :param bean_data: binary data
        :return: None
        """
        model_with_q_bean = ModelWithQBean.decode(bean_data)
        if model_with_q_bean:
            self._data.append(
                [model_with_q_bean.index_id, model_with_q_bean.model_id,
                 model_with_q_bean.timestamp, model_with_q_bean.tag_id,
                 model_with_q_bean.event_id])
