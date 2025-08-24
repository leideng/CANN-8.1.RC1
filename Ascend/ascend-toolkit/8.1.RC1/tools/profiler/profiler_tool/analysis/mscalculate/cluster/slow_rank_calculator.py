#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.

from common_func.ms_constant.number_constant import NumberConstant
from common_func.ms_constant.str_constant import OpAnalysisType
from common_func.ms_constant.str_constant import StrConstant
from mscalculate.cluster.meta_calculator import MetaCalculator


class SlowRankProf:
    PROF_COMMUNICATION_ISSUE = "There is a low efficiency issue in the current cluster communication, " \
                     "Waiting take up most of the time in communication action for rank {} ." \
                               " The max wait ratio is: {:.2f} \n "
    PROF_NO_SLOW_RANK = "There is no slow rank in the current cluster communication \n"
    PROF_REASON_SLOW_RANK = "The reason is that the the ranks are not synchronized. " \
                            "Rank {} has a lot longer computation time than other ranks.\n"
    PROF_REASON_SLOW_LINK = "The possible reason is that the communication bandwidths between ranks are inconsistent.\n"


class SlowRankCalculator(MetaCalculator):
    def __init__(self, data: list, op_name_list: list):
        super().__init__()
        self.data = data
        self.op_name_list = op_name_list

    @staticmethod
    def calculate(rank_dict: dict) -> str:
        max_item = max(rank_dict.items(),
                       key=lambda x: x[1][StrConstant.COMMUNICATION_TIME_INFO][OpAnalysisType.WAIT_TIME_RATIO])
        min_item = min(rank_dict.items(),
                       key=lambda x: x[1][StrConstant.COMMUNICATION_TIME_INFO][OpAnalysisType.WAIT_TIME_RATIO])
        rank_with_max_time_ratio = max_item[0]
        rank_with_min_time_ratio = min_item[0]
        max_wait_time_ratio = max_item[1][StrConstant.COMMUNICATION_TIME_INFO][OpAnalysisType.WAIT_TIME_RATIO]
        suggestion = ''
        if max_wait_time_ratio > NumberConstant.WAIT_TIME_THRESHOLD:
            suggestion += SlowRankProf.PROF_COMMUNICATION_ISSUE.format(rank_with_max_time_ratio, max_wait_time_ratio)
            synchronization_ratio = rank_dict[rank_with_max_time_ratio][StrConstant.COMMUNICATION_TIME_INFO][
                OpAnalysisType.SYNCHRONIZATION_TIME_RATIO]
            if synchronization_ratio > NumberConstant.WAIT_TIME_THRESHOLD:
                suggestion += SlowRankProf.PROF_REASON_SLOW_RANK.format(rank_with_min_time_ratio)
            else:
                suggestion += SlowRankProf.PROF_REASON_SLOW_LINK
        else:
            suggestion += SlowRankProf.PROF_NO_SLOW_RANK
        return suggestion

    def run(self):
        for rank_dict in self.data:
            self.suggestions.append(self.calculate(rank_dict))

    def add_suggestions(self: any, op_info: dict) -> None:
        """
        add suggestion to dict
        """
        for idx, hccl_name in enumerate(self.op_name_list):
            rank_dict = op_info.get(hccl_name, {})
            rank_dict[StrConstant.SLOW_RANK_SUGGESTION] = self.suggestions[idx]






