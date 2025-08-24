# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.

from abc import abstractmethod
from typing import List, Dict
from concurrent.futures import ProcessPoolExecutor


class ExporterBase:
    name: str = 'base'

    @classmethod
    @abstractmethod
    def initialize(cls, args):
        pass

    @classmethod
    @abstractmethod
    def export(cls, data: Dict) -> None:
        pass
