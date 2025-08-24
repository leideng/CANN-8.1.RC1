# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.

from ms_service_profiler.plugins.plugin_common import PluginCommon
from ms_service_profiler.plugins.plugin_timestamp import PluginTimeStamp
from ms_service_profiler.plugins.plugin_metric import PluginMetric
from ms_service_profiler.plugins.plugin_req_status import PluginReqStatus
from ms_service_profiler.plugins.plugin_concat import PluginConcat
from ms_service_profiler.plugins.plugin_trace import PluginTrace
from ms_service_profiler.plugins.plugin_process_name import PluginProcessName

builtin_plugins = [PluginTimeStamp, PluginConcat, PluginCommon, PluginMetric, PluginTrace, PluginProcessName]

custom_plugins = [PluginReqStatus]

