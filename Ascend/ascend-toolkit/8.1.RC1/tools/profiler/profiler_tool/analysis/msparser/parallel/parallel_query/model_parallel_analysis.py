#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.

import logging

from msmodel.parallel.cluster_parallel_model import ClusterParallelViewModel
from msparser.parallel.parallel_query.suggestion_constant import SuggestionConstant


class ModelParallelAnalysis:
    def __init__(self: any, params: dict):
        self._params = params

    def get_parallel_data(self: any) -> dict:
        with ClusterParallelViewModel(self._params["collection_path"]) as _model:
            first_field_name, first_header_name = _model.get_first_field_name(self._params)
            condition, query_params = _model.get_parallel_condition_and_query_params(self._params)
            parallel_data = _model.get_model_parallel_data(first_field_name, condition, query_params)
        return {"parallel_mode": "Model Parallel",
                "headers": [first_header_name, "Computation Time(us)", "Pure Communication Time(us)"],
                "data": parallel_data}

    def get_tuning_suggestion(self: any) -> dict:
        with ClusterParallelViewModel(self._params["collection_path"]) as _model:
            tuning_data = _model.get_model_parallel_tuning_data()
            parallel_type = _model.get_parallel_type()
        suggestion = {
            "parallel_mode": "Model Parallel",
            "suggestion": []
        }
        if not tuning_data:
            return suggestion
        if not tuning_data[0]:
            return suggestion
        if tuning_data[0][0] is None:
            logging.error("Invalid tuning data from ClusterModelParallel table. %s", tuning_data[0])
            return suggestion
        if tuning_data[0][0] > 0.1 and parallel_type == 'auto_parallel':
            suggestion.get("suggestion").append(
                SuggestionConstant.SUGGESTIONS.get("model-parallel").get("bad_operator_tiling_with_auto").format(
                    '{:.1%}'.format(tuning_data[0][0])))
        elif tuning_data[0][0] > 0.1 and parallel_type != 'auto_parallel':
            suggestion.get("suggestion").append(
                SuggestionConstant.SUGGESTIONS.get("model-parallel").get("bad_operator_tiling_with_manual").format(
                    '{:.1%}'.format(tuning_data[0][0])))
        else:
            suggestion.get("suggestion").append(
                SuggestionConstant.SUGGESTIONS.get("model-parallel").get("optimal_operator_tiling"))
        return suggestion
