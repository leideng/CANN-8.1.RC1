#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
import copy
from collections import OrderedDict
from decimal import Decimal

from common_func.info_conf_reader import InfoConfReader
from common_func.ms_constant.number_constant import NumberConstant
from common_func.ms_constant.str_constant import StrConstant
from common_func.platform.chip_manager import ChipManager
from common_func.trace_view_header_constant import TraceViewHeaderConstant
from common_func.trace_view_manager import TraceViewManager
from msmodel.api.api_data_viewer_model import ApiDataViewModel
from msmodel.freq.freq_data_viewer_model import FreqDataViewModel


class AiCoreFreqViewer:
    """
    Read aicore freq and generate aicore freq view
    """

    def __init__(self, params: dict):
        self._params = params
        self._project_path = params.get(StrConstant.PARAM_RESULT_DIR)
        self._pid = InfoConfReader().get_json_pid_data()
        self.freq_model = FreqDataViewModel(params)
        self.apidata_model = ApiDataViewModel(params)

    @staticmethod
    def _split_and_fill_freq(freq_lists):
        """
        将freq数据进行填充，填充频率以前一条记录为准
        """
        if not freq_lists:
            return []

        result = [freq_lists[0]]  # 初始化结果列表，包含第一个元素

        for next_item in freq_lists[1:]:
            current = result[-1]  # 当前处理的元素是结果列表的最后一个元素
            current_time = float(current[1])
            next_time = float(next_item[1])
            time_diff = next_time - current_time

            # 如果时间差超过100，填充中间数据
            if time_diff > 100:
                # 确保填充时间不超过next_time
                num_fill = min(int(time_diff // 100), int((next_time - 1 - current_time) // 100))
                for slice_time in range(1, num_fill + 1):
                    fill_time = current_time + slice_time * 100
                    new_item = copy.deepcopy(current)
                    new_item[1] = str(fill_time)
                    result.append(new_item)

            # 将下一个元素添加到结果列表
            result.append(next_item)
        return result

    def get_all_data(self):
        """
        1、read the fixed aicore freq from info.json
        2、read the floating aicore freq from freq.db
        3、concat 1/2 freq
        """
        result = []
        if not ChipManager().is_chip_v4() and not ChipManager().is_chip_v1_1_1() \
                or InfoConfReader().is_host_profiling():
            return result

        # add header for freq view
        result.extend(TraceViewManager.metadata_event([["process_name", self._pid,
                                                        InfoConfReader().get_json_tid_data(),
                                                        TraceViewHeaderConstant.PROCESS_AI_CORE_FREQ]]))
        # freq unit is MHZ
        freq_lists = []
        with self.freq_model as _model:
            freq_rows = _model.get_data()
            if not freq_rows:
                freq_rows.append((InfoConfReader().get_dev_cnt(),
                                  InfoConfReader().get_freq(StrConstant.AIC) / NumberConstant.FREQ_TO_MHz))
            for row in freq_rows:
                # row index 0 is syscnt, and row index 1 is frequency
                to_local_ts = InfoConfReader().trans_syscnt_into_local_time(row[0])
                data_list = [
                    TraceViewHeaderConstant.PROCESS_AI_CORE_FREQ,
                    to_local_ts, self._pid, 0,
                    OrderedDict({"MHz": row[1]})
                ]
                freq_lists.append(data_list)
        _, end_ts = InfoConfReader().get_collect_time()
        if not end_ts:
            end_ts = str(Decimal(freq_lists[-1][1]) + 1)
        final_data = copy.deepcopy(freq_lists[-1])
        # 增加一条记录，时间替换为采集结束时间，截断柱状图
        final_data[1] = end_ts
        freq_lists.append(final_data)
        filled_freq = self._split_and_fill_freq(freq_lists)
        changed_frequency = TraceViewManager.column_graph_trace(
            TraceViewHeaderConstant.COLUMN_GRAPH_HEAD_LEAST, filled_freq)
        result.extend(changed_frequency)

        return result
