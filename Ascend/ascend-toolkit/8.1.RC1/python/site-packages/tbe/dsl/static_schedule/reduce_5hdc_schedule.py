#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright 2019-2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
reduce 5hd c axis schedule
"""
import math

from tbe import tvm
from tbe.common.platform import scope_ubuf
from tbe.common.utils import log
from tbe.common.utils.errormgr import get_error_message
from tbe.common.platform.platform_info import get_soc_spec

from .util import dfs_tensor_graph
from .util import get_all_axes
from .util import get_reduce_axes
from .util import get_align_factor

SINGLE_CORE_BYTE_SIZE_THRESHOLD = 32


class Reduce5HDCSchedule:  # 'pylint: disable=R0902
    """Schedule for reduce 5HD C axis"""

    def __init__(self):
        # Schedule object
        self._schedule = None
        # Output tensor
        self._out_tensor = None
        self._out_tensor_ub = None
        # All Tensors
        self._all_tensors = None
        # Input tensors and the only one input tensor
        self._input_tensors = None
        self._input_tensor = None
        self._input_tensor_ub = None
        # Middle tensors
        self._mid_tensors = None
        # in2out tensor map
        self.tensor_map = None
        # Original input shape and real input shape in 5HD
        self.difference = None
        self.ori_shape = None
        self.ori_format = None
        self.in_shape = None
        self.out_shape = None
        self.is_keepdims = None
        # 5HD invalid data info
        self.invalid_num = None
        self.invalid_idx = None
        # The one and only reduce node
        self.reduce_node = None
        # Var of reduce Itervar of reduce node and all var of reduce node
        self.reduce_var = None
        self.all_var = None
        # Idx of reduce var and Idx of normal var
        self.reduce_idx = []
        self.normal_idx = []
        # Itervar of outermost reduce axis for all tensors except in_tensor
        self.all_reduce_var_dict = {}
        # Device info
        self.device_ub_size = int(get_soc_spec("UB_SIZE"))
        self.device_core_num = int(get_soc_spec("CORE_NUM"))
        # Tiling info
        self.tiling_calculation_unit_axis = None
        self.tiling_calculation_unit_factor = None
        self.tiling_block_axis = []
        self.tiling_block_nparts = None
        self.ub_inner = None
        self.ub_outer = None
        self.block_inner = None
        self.block_outer = None

    def do_schedule(self, out_tensors, sch_list, spec_node_list):
        """Do 5HD Reduce C axis schedule"""
        [spec_node_list].clear()  # Use it once to avoid static checks

        # Collecting info
        self.collect_info(out_tensors, sch_list)
        # Check ori_shape and current_shape, ensure they fulfill 5HD definition
        self.format_check()
        self.obtain_tensor_info()
        self.calculate_reorder()
        # Information report for debug mode
        self.print_debug("Schedule", self._schedule)
        self.print_debug("Input tensors", self._input_tensor)
        self.print_debug("Mid tensors", self._mid_tensors)
        self.print_debug("Output tensor", self._out_tensor)
        self.print_debug("Reduce node", self.reduce_node)
        self.print_debug("Reduce_axes", self.reduce_idx)
        self.print_debug("Original shape", self.ori_shape)
        self.print_debug("Reduce Var", self.reduce_var)
        self.print_debug("All Var", self.all_var)
        self.print_debug("Keep dims", self.is_keepdims)
        # Tiling calculation
        tiling_result = self.calculate_tiling()
        if not tiling_result:
            return False
        self.calculate_block_tiling()
        # Information report for debug mode
        self.print_debug("Calculation unit axis", self.tiling_calculation_unit_axis)
        self.print_debug("Calculation unit factor", self.tiling_calculation_unit_factor)
        self.print_debug("Block tiling axes", self.tiling_block_axis)
        self.print_debug("Block tiling factor", self.tiling_block_nparts)
        # Action stage
        self.data_flow_control()
        # Mark: Do reorder has been removed from here
        self.do_tiling()
        self.do_compute_at()
        self.do_emit_insn()
        self._schedule[self._input_tensor_ub].double_buffer()

        return True

    def format_check(self):
        """Check 5HD format"""
        expected_c1 = math.ceil(self.ori_shape[-1] / 16)
        self.invalid_num = expected_c1 * 16 - self.ori_shape[-1]
        self.invalid_idx = expected_c1 - 1
        expected_5hd_shape = [self.ori_shape[0], expected_c1,
                              self.ori_shape[1], self.ori_shape[2], 16]
        if self.in_shape != expected_5hd_shape:
            dict_args = {"errCode": "E90001",
                         "detailed_cause": f"Expected 5HD shape not match, please check input parameters: "
                                           f"Ori_shape: [{self.ori_shape}], Expect_shape: [{expected_5hd_shape}]"}
            raise RuntimeError(dict_args, get_error_message(dict_args))

    def collect_info(self, out_tensors, sch_list):
        """Collect necessary information"""
        self._schedule = sch_list[0]
        self._out_tensor = out_tensors[0]
        self._all_tensors, self._input_tensors, \
            self._mid_tensors, self.tensor_map = dfs_tensor_graph(self._out_tensor)
        if len(self._input_tensors) != 1:
            dict_args = {"errCode": "E90001",
                         "detailed_cause": f"[Reduce5HDCSchedule] Reduce 5HD C axis should have only one input "
                                           f"tensor, while tensors num is [{len(self._input_tensors)}]"}
            raise RuntimeError(dict_args, get_error_message(dict_args))
        self._input_tensor = self._input_tensors[0]
        self.in_shape = list(map(int, self._input_tensor.shape))
        self.out_shape = list(map(int, self._out_tensor.shape))
        temp_ori_shape = getattr(self._out_tensor, "ori_shape", None)
        if temp_ori_shape is None:
            log.warn("[Reduce5HDCSchedule] " +
                      "Failed to get 5HDC Mode original shape")
            temp_ori_shape = [self.in_shape[0], self.in_shape[2], self.in_shape[3],
                              self.in_shape[1] * self.in_shape[4]]
        self.ori_shape = temp_ori_shape
        self.ori_format = getattr(self._out_tensor, "ori_format", None)
        if self.ori_format is None:
            log.warn("[Reduce5HDCSchedule] " +
                      "Failed to get 5HDC Mode original format")
            self.ori_format = "NHWC"
        n_idx = self.ori_format.index("N")
        h_idx = self.ori_format.index("H")
        w_idx = self.ori_format.index("W")
        c_idx = self.ori_format.index("C")
        self.ori_shape = [self.ori_shape[n_idx],
                          self.ori_shape[h_idx],
                          self.ori_shape[w_idx],
                          self.ori_shape[c_idx]]

    @staticmethod
    def print_debug(info_name, info):
        """Debug info print"""
        log.debug("[Reduce5HDCSchedule] " + str(info_name) + ":" + str(info))

    def obtain_tensor_info(self):
        """Get reduce node info"""
        for tensor in self._all_tensors:
            # Only one reduce tensor is supported
            if "reduce" in tensor.op.tag:
                if self.reduce_node is None:
                    self.reduce_node = tensor
                else:
                    dict_args = {"errCode": "E90003",
                                 "detailed_cause": f"5HDC Schedule detected multiple"
                                                   f" reduce node, reduce_node is [{self.reduce_node}]"}
                    raise RuntimeError(dict_args, get_error_message(dict_args))
        reduce_in_shape = list(map(int, self.reduce_node.op.input_tensors[0].shape))
        reduce_out_shape = list(map(int, self.reduce_node.shape))
        if len(reduce_in_shape) == len(reduce_out_shape):
            self.is_keepdims = True
        else:
            self.is_keepdims = False
        self.reduce_var = get_reduce_axes(self.reduce_node)
        self.all_var = get_all_axes(self.reduce_node)

    def data_flow_control(self):
        """Do data flow control"""
        # Read GM data into UB
        self._input_tensor_ub = self._schedule.cache_read(self._input_tensor,
                                                          scope_ubuf,
                                                          self.tensor_map[self._input_tensor])
        # Set UB tensors
        for tensor in self._mid_tensors:
            self._schedule[tensor].set_scope(scope_ubuf)
        # Move UB data into GM
        self._out_tensor_ub = self._schedule.cache_write(self._out_tensor,
                                                         scope_ubuf)

    def calculate_reorder(self):
        """Get normal and reduce axes indexes for reordering and tiling"""
        self.normal_idx = []
        self.reduce_idx = []
        for idx, var in enumerate(self.all_var):
            if var not in self.reduce_var:
                self.normal_idx.append(idx)
            else:
                self.reduce_idx.append(idx)

    def calculate_tiling(self, intermediate=None):  # 'pylint: disable=R0912
        """Get calculation unit info"""

        # Simple liveness simulator
        available_axis, available_axis_size, intermediate, total_size, \
            current_out_size, current_out_tail_size = self._get_tiling_basic_info(intermediate)
        # Rule 1: if ub/2 is not full, try to use more available_axis but keep it more than core num
        optimal_need = self.device_ub_size // 2 // total_size
        rule_one_flag = optimal_need > 1 and available_axis_size > self.device_core_num
        satisfiable_need = None
        satisfiable_need = self._apply_rule_one(available_axis, available_axis_size, current_out_size, optimal_need,
                                                rule_one_flag, satisfiable_need, total_size)
        if satisfiable_need is not None:
            intermediate.insert(0, available_axis[-1])
            self.tiling_calculation_unit_factor = satisfiable_need
            return self.calculate_tiling(intermediate)

        # Rule 2:
        # conditions:
        # 1. Current calculation unit output is smaller than 1 block
        # 2. There are usable free axis on ub_tiling_axis's higher dimension
        # actions:
        # move ub_tiling_axis to higher dimension and completely consume the higher dim
        # to solve multicore trample
        dtype_size = get_align_factor(self._out_tensor.dtype)[1]
        rule_two_flag = current_out_size < SINGLE_CORE_BYTE_SIZE_THRESHOLD // dtype_size
        rule_two_tail_flag = 0 < current_out_tail_size < SINGLE_CORE_BYTE_SIZE_THRESHOLD // dtype_size
        if rule_two_flag:
            if available_axis:
                last_available_axis_size = self.in_shape[available_axis[-1]]
                satisfiable_need_size = math.ceil(SINGLE_CORE_BYTE_SIZE_THRESHOLD / (current_out_size * dtype_size))
                intermediate.insert(0, available_axis[-1])
                satisfiable_factor = min(last_available_axis_size, satisfiable_need_size)
                self.tiling_calculation_unit_factor = satisfiable_factor
                return self.calculate_tiling(intermediate)
            else:
                self.device_core_num = 1
        if rule_two_tail_flag:
            self.tiling_calculation_unit_factor -= 1
            return self.calculate_tiling(intermediate)
        # Rule 3: if ub exceeded, check if intermediate tiling is available, if not, schedule fails
        if total_size > self.device_ub_size:
            if len(intermediate) > 1:
                # Disable multi-core
                self.device_core_num = 1
                split_axis = intermediate[1]
                if self.tiling_calculation_unit_factor:
                    split_axis_size = self.tiling_calculation_unit_factor
                else:
                    split_axis_size = self.in_shape[split_axis]
                unit_size = total_size // split_axis_size
                for factor in range(1, split_axis_size, 1):
                    if (factor + 1) * unit_size > self.device_ub_size // 2:
                        self.tiling_calculation_unit_factor = factor
                        self.tiling_calculation_unit_axis = split_axis
                        return True
            else:
                return False
        if intermediate:
            self.tiling_calculation_unit_axis = intermediate[0]
        return True

    def _apply_rule_one(self, available_axis, available_axis_size, current_out_size, optimal_need, rule_one_flag,
                        satisfiable_need, total_size):
        if rule_one_flag:
            last_available_axis_size = self.in_shape[available_axis[-1]]
            for current_need in range(min(optimal_need, last_available_axis_size), 1, -1):
                # available_axis_size must be more than core num
                # last_available_axis_size must be divisible by current_need
                # size must be smaller than ub/2
                continue_flag = (available_axis_size // current_need < self.device_core_num or
                                 not last_available_axis_size % current_need == 0 or
                                 total_size * current_need > self.device_ub_size // 2) and \
                                current_out_size >= get_align_factor(self._input_tensor.dtype)[0]
                if continue_flag:
                    continue
                satisfiable_need = current_need
                break
        return satisfiable_need

    def _get_tiling_basic_info(self, intermediate):
        """Get basic info"""
        # Available axis: All axes besides axes inside calculation unit
        # Availalbe axis size: Product of extents of all available axes
        # intermediate: non-reduce axes inside calculation unit
        # total_size: calculation unit ub buffer usage
        if intermediate is None:
            intermediate = []
        total_size = self.get_calculation_unit_size(intermediate)
        available_axis = [axis for axis in range(len(self.in_shape)) if axis not in self.reduce_idx + intermediate]
        available_axis_size = 1
        for axis in available_axis:
            available_axis_size *= self.in_shape[axis]
        current_out_size = 1
        for axis in intermediate:
            axis_size = self.in_shape[axis]
            current_out_size *= axis_size
        if self.tiling_calculation_unit_factor is None or not intermediate:
            current_out_tail_size = 0
        else:
            current_out_tail_size = 1
            axes_size = [self.in_shape[axis] for axis in intermediate]
            axes_size[0] %= self.tiling_calculation_unit_factor
            for axis_size in axes_size:
                current_out_tail_size *= axis_size
        return available_axis, available_axis_size, intermediate, total_size, current_out_size, current_out_tail_size

    def get_calculation_unit_size(self, intermediate):
        """Get current calculation unit size"""
        current_tensor = self._input_tensor
        total_size = 0
        last_size = -1
        while current_tensor in self.tensor_map:
            if current_tensor is self.reduce_node:
                break
            # Reduce axes must be in the calculation unit
            my_size = 1
            for axis in self.reduce_idx:
                axis_size = int(current_tensor.shape[axis])
                my_size *= axis_size
            # Add intermediates
            for axis in intermediate[1:]:
                axis_size = int(current_tensor.shape[axis])
                my_size *= axis_size
            if self.tiling_calculation_unit_factor:
                my_size *= self.tiling_calculation_unit_factor
            # Convert to byte size
            dtype_size = get_align_factor(current_tensor.dtype)[1]
            my_size *= dtype_size
            if my_size != last_size:
                total_size += my_size
            last_size = my_size
            current_tensor = self.tensor_map[current_tensor][0]
        return total_size

    def calculate_block_tiling(self):
        """Get block tiling info"""
        # Get all available block tiling axes
        available_axes = self.normal_idx[:]
        total_output_num = 1
        available_axis_size = 1
        dtype_size = get_align_factor(self._out_tensor.dtype)[1]
        # Get total output size
        for axis in available_axes:
            total_output_num *= self.in_shape[axis]
        total_output_dtype_size = total_output_num * dtype_size
        # Calculate block tiling nparts maximum
        maximum_core_num = math.ceil(total_output_dtype_size / SINGLE_CORE_BYTE_SIZE_THRESHOLD)
        maximum_core_num = min(maximum_core_num, self.device_core_num)
        # Get actual available block tiling axes
        for idx, axis in enumerate(available_axes):
            if axis == self.tiling_calculation_unit_axis:
                available_axes = available_axes[0:idx]
        for axis in available_axes:
            available_axis_size *= self.in_shape[axis]
        # Inject extra available axis
        if self.tiling_calculation_unit_axis is not None and \
                self.tiling_calculation_unit_factor != \
                self.in_shape[self.tiling_calculation_unit_axis]:
            available_axes.append(self.tiling_calculation_unit_axis)
            available_axis_size *= self.in_shape[self.tiling_calculation_unit_axis] // \
                self.tiling_calculation_unit_factor
        if not available_axes:
            # Unable to do block tiling
            return
        # Apply rules
        self.apply_block_tiling_rules(available_axes, available_axis_size, maximum_core_num)
        # If keepdims, tiling_block_axis should always be consecutive
        if self.is_keepdims:
            self.keepdims_fix_block_tiling_axes()

    def keepdims_fix_block_tiling_axes(self):
        """Fix tiling block axis for keepdims situation"""
        temp_tiling_block_axis = []
        for axis in self.tiling_block_axis:
            if not temp_tiling_block_axis:
                temp_tiling_block_axis.append(axis)
            else:
                while temp_tiling_block_axis[-1] + 1 != axis:
                    temp_tiling_block_axis.append(temp_tiling_block_axis[-1] + 1)
                temp_tiling_block_axis.append(axis)
        self.tiling_block_axis = temp_tiling_block_axis

    def apply_block_tiling_rules(self, available_axes, available_axis_size, maximum_core_num):
        """
        Rule 1: available block tiling axes are smaller than device core, use them all
        Rule 2: push forward from outermost axis to innermost non-reduce axis
        """
        if available_axis_size < self.device_core_num:
            self.tiling_block_axis = available_axes
            self.tiling_block_nparts = min(maximum_core_num, available_axis_size)
        else:
            current_core_num = 1
            for axis in available_axes:
                axis_size = self.in_shape[axis]
                if axis == self.tiling_calculation_unit_axis:
                    axis_size //= self.tiling_calculation_unit_factor
                if current_core_num * axis_size < maximum_core_num:
                    self.tiling_block_axis.append(axis)
                    current_core_num *= axis_size
                    self.tiling_block_nparts = current_core_num
                else:
                    max_need = maximum_core_num // current_core_num
                    max_obtainable = min(axis_size, max_need)
                    for current_use in range(max_obtainable, 1, -1):
                        # Mark: Divisible tiling has been removed from here
                        if current_core_num * current_use <= maximum_core_num:
                            self.tiling_block_axis.append(axis)
                            current_core_num *= current_use
                            self.tiling_block_nparts = current_core_num

    def do_reorder(self):
        """It it necessary do reorder all tensor's axes"""
        def __reorder(_tensor):
            iter_vars = [self._schedule[_tensor].op.axis[i] for i in [*self.normal_idx, *self.reduce_idx]]
            self._schedule[_tensor].reorder(*iter_vars)
            if _tensor not in self.all_reduce_var_dict:
                self.all_reduce_var_dict[_tensor] = \
                    self._schedule[_tensor].op.axis[self.reduce_idx[0]]
        # Do reorder and save its outermost reduce Itervar
        tensor = self._input_tensor_ub
        if self.is_keepdims:
            __reorder(tensor)
            tensor = self.tensor_map[self._input_tensor][0]
            while tensor in self.tensor_map:
                if tensor != self.reduce_node:
                    __reorder(tensor)
                tensor = self.tensor_map[tensor][0]
            if self._out_tensor != self.reduce_node:
                tensor = self._out_tensor_ub
                __reorder(tensor)
        else:
            __reorder(tensor)
            tensor = self.tensor_map[self._input_tensor][0]
            while tensor in reversed(self.tensor_map):
                if tensor != self.reduce_node:
                    __reorder(tensor)
                else:
                    break
                tensor = self.tensor_map[tensor][0]

    def do_tiling(self):
        """Tiling action stage"""
        out_stage = self._schedule[self._out_tensor]
        # Do ub tiling
        self._do_ub_tiling(out_stage)
        # Do block tiling
        if self.tiling_block_axis and self.tiling_block_nparts > 1:
            self._do_block_tiling(out_stage)
        if self.block_inner is None:
            self.block_inner = self.ub_outer
        if self.block_outer is None:
            self.block_outer = self.block_inner

    def _do_block_tiling(self, out_stage):  # 'pylint: disable=R0912
        """Apply block tiling"""
        block_tiling_axis = None
        core_num = 1
        for axis in self.tiling_block_axis:
            if self.is_keepdims:
                target_idx = axis
            else:
                target_idx = self.normal_idx.index(axis)
            if axis != self.tiling_calculation_unit_axis:
                target_size = self.out_shape[target_idx]
                # First Cut
                if block_tiling_axis is None:
                    if target_size * core_num <= self.tiling_block_nparts:
                        block_tiling_axis = out_stage.op.axis[target_idx]
                        core_num *= target_size
                        self.block_outer = block_tiling_axis
                        continue
                    # First Cut not enough
                    block_tiling_axis, self.block_inner = out_stage.split(
                        out_stage.op.axis[target_idx],
                        nparts=self.tiling_block_nparts // core_num)
                    self.block_outer = block_tiling_axis
                    break
                # Mid Cut
                if target_size * core_num <= self.tiling_block_nparts:
                    block_tiling_axis = out_stage.fuse(out_stage.op.axis[target_idx],
                                                       block_tiling_axis)
                    core_num *= target_size
                    self.block_outer = block_tiling_axis
                    continue
                # Mid Cut not enough
                outer, self.block_inner = out_stage.split(
                    out_stage.op.axis[target_idx],
                    nparts=self.tiling_block_nparts // core_num)
                block_tiling_axis = out_stage.fuse(outer,
                                                   block_tiling_axis)
                self.block_outer = block_tiling_axis
                break
            target_size = self.out_shape[target_idx]
            target_size //= self.tiling_calculation_unit_factor
            # First Cut
            if block_tiling_axis is None:
                if target_size * core_num <= self.tiling_block_nparts:
                    block_tiling_axis = self.ub_outer
                    core_num *= target_size
                    self.block_outer = block_tiling_axis
                    continue
                # First Cut not enough
                block_tiling_axis, self.ub_outer = out_stage.split(
                    self.ub_outer,
                    nparts=self.tiling_block_nparts // core_num)
                self.block_outer = block_tiling_axis
                break
            # Mid Cut
            if target_size * core_num <= self.tiling_block_nparts:
                block_tiling_axis = out_stage.fuse(self.ub_outer,
                                                   block_tiling_axis)
                self.ub_outer = block_tiling_axis
                self.block_outer = block_tiling_axis
                core_num *= target_size
                continue
            # Mid Cut not enough
            outer, self.ub_outer = out_stage.split(
                self.ub_outer,
                nparts=self.tiling_block_nparts // core_num)
            block_tiling_axis = out_stage.fuse(outer,
                                               block_tiling_axis)
            self.block_outer = block_tiling_axis
            break
        if self.block_outer is None:
            self.block_outer = block_tiling_axis
        self._schedule[self._out_tensor].bind(block_tiling_axis,
                                              tvm.thread_axis("blockIdx.x"))

    def _do_ub_tiling(self, out_stage):
        """Apply ub tiling"""
        if self.tiling_calculation_unit_axis is None:
            # calculation unit is reduce unit
            self.ub_outer, self.ub_inner = out_stage.split(out_stage.op.axis[-1],
                                                           factor=1)
        else:
            if self.is_keepdims:
                target_idx = self.tiling_calculation_unit_axis
            else:
                target_idx = self.normal_idx.index(self.tiling_calculation_unit_axis)
            factor = self.tiling_calculation_unit_factor
            if factor is None:
                self.ub_outer, self.ub_inner = out_stage.split(out_stage.op.axis[target_idx],
                                                               nparts=1)
            else:
                self.ub_outer, self.ub_inner = out_stage.split(out_stage.op.axis[target_idx],
                                                               factor=factor)

    def do_compute_at(self):
        """Compute at strategy is kind of simple here"""
        final_tensor = self._out_tensor
        self._schedule[self._out_tensor_ub].compute_at(self._schedule[final_tensor],
                                                       self.ub_outer)
        for tensor in self._mid_tensors + [self._input_tensor_ub]:
            self._schedule[tensor].compute_at(self._schedule[final_tensor],
                                              self.ub_outer)

    def do_emit_insn(self):
        """Do emit instruction"""
        self._schedule[self._input_tensor_ub].emit_insn(
            self._schedule[self._input_tensor_ub].op.axis[0],
            "dma_copy")
        for tensor in self._mid_tensors:
            tag = tensor.op.tag.split("|")[0]
            if "reduce" in tag:
                tag = "5hdc_{}".format(tag)
            self._schedule[tensor].emit_insn(
                self._schedule[tensor].op.axis[0],
                tag)
            if "5hdc" in tag:
                self._schedule[tensor].pragma(self._schedule[tensor].op.axis[0],
                                              "5hdc_CAxisVar",
                                              str(self.all_var[1]))
                self._schedule[tensor].pragma(
                    self._schedule[tensor].op.axis[0],
                    "CAxisInvalidSize",
                    16 - (self.ori_shape[-1] - self.ori_shape[-1] // 16 * 16))
        final_tag_ub = self._out_tensor.op.tag.split("|")[0]
        if "reduce" in final_tag_ub:
            final_tag_ub = "5hdc_" + final_tag_ub
        self._schedule[self._out_tensor_ub].emit_insn(
            self._schedule[self._out_tensor_ub].op.axis[0],
            final_tag_ub)
        if "5hdc" in final_tag_ub:
            self._schedule[self._out_tensor_ub].pragma(
                self._schedule[self._out_tensor_ub].op.axis[0],
                "5hdc_CAxisVar",
                str(self.all_var[1]))
            self._schedule[self._out_tensor_ub].pragma(
                self._schedule[self._out_tensor_ub].op.axis[0],
                "CAxisInvalidSize",
                16 - (self.ori_shape[-1] - self.ori_shape[-1] // 16 * 16))
        self._schedule[self._out_tensor].emit_insn(self.ub_inner, "dma_copy")
