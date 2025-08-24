#!/usr/bin/python
# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.

import logging
from abc import ABC, abstractmethod
from .metric import Metrics
from .trace import Trace, TraceEvent


class RawTask(ABC):
    '''
    裸事件信息，主要用于承载调度内容信息
    '''
    def __init__(self, pipe_name, name):
        '''
        :param pipe_name: 对应的管道
        :param name:
        '''
        self.name = name
        self.owner = pipe_name

    def __str__(self):
        return "Task({} - {})".format(self.owner, self.name)

    @abstractmethod
    def cost_time(self):
        raise NotImplementedError("this function(cost_time) need to impl.")

    @abstractmethod
    def is_ready(self):
        '''
        检查任务的依赖是否已经ok
        :return:
        '''
        raise NotImplementedError("this function(is_ready) need to impl.")

    @abstractmethod
    def pre_func(self):
        raise NotImplementedError("this function(pre_func) need to impl.")

    @abstractmethod
    def post_func(self):
        raise NotImplementedError("this function(post_func) need to impl.")


class InstrTask(RawTask):
    '''
    该类继承了task schedule模块中的标准类接口，需要实现指令到标准任务的转换
    '''
    def __init__(self, pipe_name, instr_obj):
        '''
        :param pipe_name:  标准资源名
        :param name:       指令名称
        '''
        super(InstrTask, self).__init__(pipe_name, instr_obj.name)
        self.name = instr_obj.task_name
        self.owner = pipe_name
        self.instr_obj = instr_obj
        self.start_time = 0
        self.end_time = 0

    def cost_time(self):
        return self.instr_obj.cost_time()
    
    def size(self):
        if self.instr_obj.name == "MOV":
            return self.instr_obj.move_size()
        return self.instr_obj.cal_size()

    def is_ready(self):
        return self.instr_obj.is_ready()

    def pre_func(self):
        pass

    def post_func(self):
        from mskpp._C import task_schedule
        if not task_schedule.Schedule().get_debug_mode() and self.start_time == self.end_time:
            raise Exception("Pipe %s instruction %s do not run at %s",
                            self.owner, self.name, self.start_time)
        if Trace().is_enable:
            trace_event = TraceEvent(self.instr_obj.name, self.owner, self.name, self.start_time, self.end_time,
                                     self.size())
            Trace().add_event(trace_event)
        if Metrics().is_enable:
            Metrics().add_event(self)
        self.instr_obj.schedule_post()
