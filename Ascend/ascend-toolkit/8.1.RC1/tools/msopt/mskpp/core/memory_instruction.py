#!/usr/bin/python
# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2025. All rights reserved.

from mskpp._C import arch
from .instruction_base import InstructionBase


class MemoryInstruction(InstructionBase):
    def __init__(self, src, dst, trans_enable, repeat, set_value_input=-1, expect_value_input=-1):
        super(MemoryInstruction, self).__init__("MOV", (src,), (dst,))
        self.task_name = "{}-{}_TO_{}".format("MOV", src.mem_type, dst.mem_type)
        self.trans_enable = trans_enable
        self.repeat = repeat
        self.set_value = set_value_input
        self.expect_value = expect_value_input

    def launch(self, inputs, outputs, attr):
        from .instr_task import InstrTask
        from mskpp._C import task_schedule
        from mskpp.core.aicore import Core
        pipe_name = Core.get_instr_pipe_name(src=inputs[0].mem_type, dst=outputs[0].mem_type)
        task = InstrTask(pipe_name, self)
        task_schedule.Schedule().add_task(task)

    def cost_time(self):
        from .prof_data import ProfDataRegister
        return ProfDataRegister.get(self.name)(self.inputs, self.outputs, self.trans_enable, self.repeat).time()

    def move_size(self):
        from .prof_data import ProfDataRegister
        return ProfDataRegister.get(self.name)(self.inputs, self.outputs, self.trans_enable, self.repeat).size()

    def instr_check(self, inputs, outputs, attr):
        src = inputs[0]
        dst = outputs[0]
        if not arch.mte_is_valid(src.mem_type, dst.mem_type):
            raise Exception("chip is not support move data from {} to {}".format(src.mem_type, dst.mem_type))

    def is_ready(self):
        # is_ready用于判断tensor是否处于valid状态
        # 如果有tensor依赖其他tensor，需要去查询依赖的tensor是否是自己期望的值
        # 判断MOV指令input的tensor的值是否符合预期
        if self.inputs[0].tensor_value != self.expect_value:
            return False
        return super().is_ready()

    def schedule_post(self):
        # 对于MOV指令task结束之后，给输出的tensor设置值
        if self.set_value != -1:
            self.outputs[0].tensor_value = self.set_value
        super().schedule_post()
