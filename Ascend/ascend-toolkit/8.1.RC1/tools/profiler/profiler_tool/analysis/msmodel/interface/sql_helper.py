#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

from common_func.ms_constant.number_constant import NumberConstant


class SqlWhereCondition:
    @staticmethod
    def get_interval_intersection_condition(start: float, end: float, table_name: str,
                                            start_col_name: str, end_col_name: str) -> str:
        """
        This function generates a conditon to filter the data that intersects the interval.
        """
        condition = ""
        if start != float("-inf"):
            condition += f"{table_name}.{end_col_name} < {start}"
            if end != float("inf"):
                condition += f" or {table_name}.{start_col_name} > {end}"
        elif end != float("inf"):
            condition += f"{table_name}.{start_col_name} > {end}"
        if condition:
            return f"where not ({condition})"
        return condition

    @staticmethod
    def get_host_select_condition(is_all_model_iter: bool, model_id: int, iter_id: int, device_id: int):
        if is_all_model_iter:
            return f"where device_id={device_id}"
        condition = f"where model_id={model_id} and (index_id={iter_id} or" \
                    f" index_id={NumberConstant.STATIC_GRAPH_INDEX}) and device_id={device_id}"
        return condition
