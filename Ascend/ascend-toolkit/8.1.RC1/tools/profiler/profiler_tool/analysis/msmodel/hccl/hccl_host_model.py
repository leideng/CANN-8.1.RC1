#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

from common_func.db_name_constant import DBNameConstant
from msmodel.interface.base_model import BaseModel


class HCCLHostModel(BaseModel):
    """hccl host model class"""

    def __init__(self: any, result_dir: str) -> None:
        super().__init__(result_dir, DBNameConstant.DB_HCCL,
                         [DBNameConstant.TABLE_HCCL_OP, DBNameConstant.TABLE_HCCL_TASK])
