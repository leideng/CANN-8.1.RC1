#!/usr/bin/python
# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.

import os
import logging
from .output_tool import SAVE_DATA_FILE_AUTHORITY


class DataVisualization():
    @staticmethod
    def cycle_info_visualization(label_list, cycle_list, title_name, data_file_name):
        try:
            import plotly
            import plotly.graph_objs as go
        except ImportError:
            logging.warning("plotly is not installed, if you want to visualize data, please install it")
            return
        else:
            fig = go.Figure()
            fig.add_trace(go.Pie(labels=label_list, values=cycle_list))
            fig.update_layout(title=go.layout.Title(text=title_name, x=0.5), width=500, height=500)
            plotly.offline.plot(fig, auto_open=False, filename=data_file_name)
            os.chmod(path=data_file_name, mode=SAVE_DATA_FILE_AUTHORITY)