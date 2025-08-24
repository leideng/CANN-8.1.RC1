# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.
 
from abc import abstractmethod
from typing import List, Dict


class PluginBase:
    name: str = 'plugin_base'
    depends: List[str] = []

    @classmethod
    @abstractmethod
    def parse(cls, data: Dict) -> Dict:
        pass


