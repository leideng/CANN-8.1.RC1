#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

from common.const import MAX_CHAR_LINE

__all__ = ["generate_report"]

VERTICAL = " | "
ADD_SUB = " +-"
SUB_ADD = "-+ "


def write_header(table_string, table_header, column_widths):
    # Table header data
    for header in table_header:
        # Align the cells in each column print header
        table_string += VERTICAL
        for i, cell in enumerate(header):
            # Align the cells in each column
            table_string += str(cell).ljust(column_widths[i]) + VERTICAL
        table_string += "\n"
    return table_string


def write_data(table_string, data_value, column_widths, split_line):
    # Align the each column print data
    for row in data_value:
        table_string += VERTICAL
        for i, cell in enumerate(row):
            # Align the cells in each column print data
            table_string += str(cell).ljust(column_widths[i]) + VERTICAL
        table_string += "\n"
        if split_line:
            table_string += ADD_SUB
            for i, cell in enumerate(column_widths):
                # Align the cells in each column
                table_string += str().ljust(column_widths[i], '-') + SUB_ADD
            table_string += "\n"
    return table_string


def generate_report(table_header, table_data, split_line=False):
    """
    Generate a formatted table string.

    Args:
        table_header (list): The header of the table [[1, 2, 3]].
        table_data (dict): The data of the table
            {
                'none': [
                    [1, 2, 3],
                    [4, 5, 6]
                ],
                'test': [
                    [1, 2, 3],
                    [4, 5, 6]
                ]
            }.
        split_line (bool):  Indicates whether to display the data split line.
    Returns:
        str: The formatted table string.
    """
    # Calculate the width of each column
    table_row = list()
    table_row += table_header
    for data_key, data_value in table_data.items():
        key_list = ["0" for _ in range(len(table_row[0]))]
        key_list[0] = data_key
        table_row.append(key_list)
        table_row += data_value
    column_widths = [max(len(str(row[i])) for row in table_row) for i in range(len(table_row[0]))]
    # Create the formatted table string
    table_string = "\n"
    table_string += ADD_SUB
    for i, cell in enumerate(column_widths):
        if column_widths[i] > MAX_CHAR_LINE:
            column_widths[i] = MAX_CHAR_LINE
        column_widths[i] += 5
        # Align the cells in each column
        table_string += str().ljust(column_widths[i], '-') + SUB_ADD
    table_string += "\n"

    table_string = write_header(table_string, table_header, column_widths)

    # Splitting line between header and data
    table_string += " +="
    for i, cell in enumerate(column_widths):
        # Align the cells in each column
        table_string += str().ljust(column_widths[i], '=') + "=+ "
    table_string += "\n"
    # Table content data
    for data_key, data_value in table_data.items():
        if data_key != "none":
            table_string += ADD_SUB
            for i, cell in enumerate(column_widths):
                # Align the cells in each column
                table_string += str("--" + data_key if i == 0 else "").ljust(column_widths[i], '-') + SUB_ADD
            table_string += "\n"
        table_string = write_data(table_string, data_value, column_widths, split_line)

    if not split_line:
        table_string += ADD_SUB
        for i, cell in enumerate(column_widths):
            # Align the cells in each column
            table_string += str().ljust(column_widths[i], '-') + SUB_ADD
        table_string += "\n"
    return table_string.replace("-+ -", "-+--").replace("=+ =", "=+==")
