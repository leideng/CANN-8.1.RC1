#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.

from abc import abstractmethod

from common_func.ms_constant.str_constant import StrConstant
from mscalculate.cluster.communication_matrix_calculator import CommunicationMatrixCalculator
from mscalculate.cluster.slow_link_calculator import SlowLinkCalculator
from mscalculate.cluster.slow_rank_calculator import SlowRankCalculator


class ClusterCalculatorFactory:
    """
    calculator factory interface for cluster scene
    """
    def __init__(self: any) -> None:
        pass

    @abstractmethod
    def generate_calculator(self):
        """
        generate calculator method
        """


class SlowRankCalculatorFactory(ClusterCalculatorFactory):
    """
    calculator factory for looking for slow rank
    """

    def __init__(self: any, data: dict) -> None:
        super().__init__()
        self.op_info = data
        self.calculate_data = []
        self.op_name_list = []

    def data_dispatch(self: any) -> None:
        """
        input data contains iteration and different ops data
        we need dispatch first then process
        """
        for op_name, rank_dict in self.op_info.items():
            self.op_name_list.append(op_name)
            self.calculate_data.append(rank_dict)

    def generate_calculator(self: any) -> SlowRankCalculator:
        self.data_dispatch()
        return SlowRankCalculator(self.calculate_data, self.op_name_list)


class SlowLinkCalculatorFactory(ClusterCalculatorFactory):
    """
    calculator factory for looking for slow link
    """

    def __init__(self: any, data: dict) -> None:
        super().__init__()
        self.op_rank_list = []
        self.calculate_data = []
        self.op_info = data

    def data_dispatch(self: any) -> None:
        """
        input data contains iteration and different ops data
        we need dispatch first then process
        """
        for op_name, rank_dict in self.op_info.items():
            for rank_id, com_dict in rank_dict.items():
                if StrConstant.SUGGESTION not in str(rank_id):
                    self.op_rank_list.append((op_name, rank_id))
                    self.calculate_data.append(com_dict[StrConstant.COMMNUNICATION_BANDWIDTH_INFO])

    def generate_calculator(self):
        self.data_dispatch()
        return SlowLinkCalculator(self.calculate_data, self.op_rank_list)


class MatrixCalculatorFactory(ClusterCalculatorFactory):
    """
    calculator factory for looking for slow link
    """

    def __init__(self: any, data: dict) -> None:
        super().__init__()
        self.op_info = data
        self.calculate_data = []
        self.op_name_list = []

    def data_dispatch(self: any) -> None:
        """
        input data contains iteration and different ops data
        we need dispatch first then process
        """
        for op_dict in self.op_info:
            self.op_name_list.append(op_dict.get(StrConstant.OP_NAME, ''))
            self.calculate_data.append(op_dict.get(StrConstant.LINK_INFO, []))

    def generate_calculator(self: any) -> CommunicationMatrixCalculator:
        self.data_dispatch()
        return CommunicationMatrixCalculator(self.calculate_data, self.op_name_list)
