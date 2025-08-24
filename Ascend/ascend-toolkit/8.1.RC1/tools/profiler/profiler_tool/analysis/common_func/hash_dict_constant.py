#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
import os

from common_func.singleton import singleton
from common_func.path_manager import PathManager
from common_func.db_name_constant import DBNameConstant
from msmodel.ge.ge_hash_model import GeHashViewModel


@singleton
class HashDictData:
    """
    class for mapping hash relationship
    """

    def __init__(self: any, project_path: str) -> None:
        self._type_hash_dict = {}
        self._ge_hash_dict = {}
        if os.path.exists(PathManager.get_db_path(project_path, DBNameConstant.DB_GE_HASH)):
            self.load_hash_data(project_path)

    def load_hash_data(self: any, project_path: str) -> None:
        """
        load hash dict data
        """
        with GeHashViewModel(project_path) as _model:
            self._type_hash_dict = _model.get_type_hash_data()
            self._ge_hash_dict = _model.get_ge_hash_data()

    def get_type_hash_dict(self: any) -> dict:
        """
        get type hash dict data
        """
        return self._type_hash_dict

    def get_ge_hash_dict(self: any) -> dict:
        """
        get ge hash dict data
        """
        return self._ge_hash_dict
