# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.

import json
from enum import Enum
from ms_service_profiler.mstx import service_profiler


class MarkType(int, Enum):
    TYPE_EVENT = 0
    TYPE_METRIC = 1
    TYPE_SPAN = 2
    TYPE_LINK = 3


class Level(int, Enum):
    ERROR = 10
    INFO = 20
    DETAILED = 30
    VERBOSE = 40


class Profiler:
    def __init__(self, profiler_level) -> None:
        self._enable = service_profiler.is_enable(profiler_level)
        self._attr = dict()
        self._span_handle = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.span_end()

    def attr(self, key, value):
        self._attr[key] = value
        return self

    def domain(self, domain):
        return self.attr("domain", domain)

    def res(self, res):
        return self.attr("rid", res)

    def metric(self, metric_name, metric_value):
        return self.attr(f"{metric_name}=", metric_value)

    def metric_inc(self, metric_name, metric_value):
        return self.attr(f"{metric_name}+", metric_value)

    def metric_scope(self, scope_name, scope_value=0):
        return self.attr(f"scope#{scope_name}", scope_value)

    def metric_scope_as_req_id(self):
        return self.attr(f"scope#", "req")

    def launch(self):
        if self._enable:
            service_profiler.mark_event(self.get_msg())

    def get_msg(self):
        return json.dumps(self._attr).replace("\"", "^")

    def link(self, from_rid, to_rid):
        if self._enable:
            self.attr("type", MarkType.TYPE_LINK).attr("from", from_rid).attr("to", to_rid)
            service_profiler.mark_event(self.get_msg())

    def event(self, event_name):
        if self._enable:
            self.attr("type", MarkType.TYPE_EVENT).attr("name", event_name)
            service_profiler.mark_event(self.get_msg())

    def span_start(self, span_name):
        if self._enable:
            self.attr("name", span_name).attr("type", MarkType.TYPE_SPAN)
            self._span_handle = service_profiler.start_span(span_name)
        return self

    def span_end(self):
        if self._enable:
            service_profiler.mark_span_attr(self.get_msg(), self._span_handle)
            service_profiler.end_span(self._span_handle)
