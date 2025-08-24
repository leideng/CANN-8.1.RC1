#!/usr/bin/python
# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2025. All rights reserved.

from abc import abstractmethod
from .instruction_base import InstructionBase
from .common.checker import is_required_type
from .tensor import Tensor


class ComputationInstruction(InstructionBase):
    def __init__(self, name, inputs, outputs, attr=None, instr_type=None):
        super(ComputationInstruction, self).__init__(name, inputs, outputs, attr)
        self.task_name = name  # 当这个指令任务执行时，使用task_name
        self.instr_type = instr_type
        self.init_param_check()

    @abstractmethod
    def infer_shape(self, inputs, outputs, attr):
        '''
        由输入的Tensor信息推断输出的Tensor
        :return:
        '''
        raise NotImplementedError("infer_shape should impl.")

    def instr_check(self, inputs, outputs, attr):
        self.infer_shape(inputs, outputs, attr)

    def init_param_check(self):
        for tensor in self.inputs:
            if not is_required_type(tensor, Tensor):
                raise Exception("Computational instruction inputs should be Tensor, but got {}".format(type(tensor)))
            if not tensor.is_complete():
                raise Exception("{}'s input(Tensor:{}) is not complete.".format(self.task_name, tensor))
        for tensor in self.outputs:
            if not is_required_type(tensor, Tensor):
                raise Exception("Computational instruction output should be Tensor, but got {}".format(type(tensor)))

    def cal_size(self):
        from .prof_data import ProfDataRegister
        return ProfDataRegister.get(self.name)(self.inputs, self.outputs).size()

    def cost_time(self):
        '''
        依据表格查询具体耗时
        :return:
        '''
        from .prof_data import ProfDataRegister
        return ProfDataRegister.get(self.name)(self.inputs, self.outputs).time()

    def launch(self, inputs, outputs, attr):
        from mskpp.core.instr_task import InstrTask
        from mskpp._C import task_schedule
        from mskpp.core.aicore import Core
        pipe_name = Core.get_instr_pipe_name(instr_type=self.instr_type)
        task = InstrTask(pipe_name, self)
        task_schedule.Schedule().add_task(task)
