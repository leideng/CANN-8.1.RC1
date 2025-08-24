# Copyright 2023 Huawei Technologies Co., Ltd
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
three_interpolate
"""
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform as cce
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import register_operator


class Constant:
    """
    The class for constant
    """
    MAX_ELEMENT = 2 ** 31 - 1
    TILING_ARG_NUM = 16

    TILING_MODE_0 = 0
    TILING_MODE_1 = 1


class ThreeInterpolate:
    def __init__(self, input_features, input_idx, input_weight, output, kernel_name):
        self.dtype_features = input_features.get("dtype").lower()
        self.dtype_features_bytes = cce.get_bit_len(self.dtype_features) // 8

        self.dtype_idx = input_idx.get("dtype").lower()
        self.dtype_idx_bytes = cce.get_bit_len(self.dtype_idx) // 8
        
        self.dtype_weight = input_weight.get("dtype").lower()
        self.dtype_weight_bytes = cce.get_bit_len(self.dtype_weight) // 8

        self.dtype_output = output.get("dtype").lower()
        self.dtype_output_bytes = cce.get_bit_len(self.dtype_output) // 8

        self.kernel_name = kernel_name
        self.tik_instance = tik.Tik(tik.Dprofile())

        self.dtype_tiling = "uint32"
        self.dtype_tiling_bytes_size = cce.get_bit_len(self.dtype_tiling) // 8

        self.block_byte_size = int(32)
        self.ub_byte_size = cce.get_soc_spec(cce.UB_SIZE)
        self.total_aicore_num = cce.get_soc_spec(cce.CORE_NUM)

        self.output_mask = 256 // self.dtype_output_bytes
        self.features_mask = 256 // self.dtype_features_bytes

        self.features_gm = self.tik_instance.Tensor(self.dtype_features, (Constant.MAX_ELEMENT, ), name="features_gm",
                                                    scope=tik.scope_gm)
        self.idx_gm = self.tik_instance.Tensor(self.dtype_idx, (Constant.MAX_ELEMENT, ), name="idx_gm",
                                              scope=tik.scope_gm)
        self.weight_gm = self.tik_instance.Tensor(self.dtype_weight, (Constant.MAX_ELEMENT, ), name="weight_gm",
                                                 scope=tik.scope_gm)
        self.tiling_gm = self.tik_instance.Tensor(self.dtype_tiling, (Constant.TILING_ARG_NUM, ), name="tiling_gm",
                                                scope=tik.scope_gm)
        self.output_gm = self.tik_instance.Tensor(self.dtype_output, (Constant.MAX_ELEMENT, ), name="output_gm",
                                                scope=tik.scope_gm, is_atomic_add=True)
        
        self.bs = self.tik_instance.Scalar(self.dtype_tiling, name="bs")
        self.cs = self.tik_instance.Scalar(self.dtype_tiling, name="cs")
        self.ns = self.tik_instance.Scalar(self.dtype_tiling, name="ns")
        self.ms = self.tik_instance.Scalar(self.dtype_tiling, name="ms")
        self.aicore_num = self.tik_instance.Scalar(self.dtype_tiling, name="aicore_num")
        self.entry_num_in_group = self.tik_instance.Scalar(self.dtype_tiling,
                                                           name="entry_num_in_group")
        self.each_core_entry_group_num = self.tik_instance.Scalar(self.dtype_tiling,
                                                                  name="each_core_entry_group_num")
        self.each_core_entry_group_padding_num = self.tik_instance.Scalar(self.dtype_tiling,
                                                                  name="each_core_entry_group_padding_num")
        self.c_fib_ub = self.tik_instance.Scalar(self.dtype_tiling,
                                                name="c_fit_ub")
        self.c_move_tims = self.tik_instance.Scalar(self.dtype_tiling,
                                                    name="c_move_times")
        self.idx_entry_group_block_size = self.tik_instance.Scalar(self.dtype_tiling,
                                                                  name="idx_entry_group_block_size")
        self.weight_entry_group_block_size = self.tik_instance.Scalar(self.dtype_tiling,
                                                                  name="weight_entry_group_block_size")
        self.used_core_num = self.tik_instance.Scalar(self.dtype_tiling,
                                                        name="used_core_num")
        self.tiling_mode_key = self.tik_instance.Scalar(self.dtype_tiling,
                                                        name="tiling_mode_key")
        self.core_padding_idx = self.tik_instance.Scalar(self.dtype_tiling,
                                                         name="core_padding_idx")
        self.features_move_block_size = self.tik_instance.Scalar(self.dtype_tiling,
                                                                 name="features_move_block_size")
        self.features_mask_block_size = self.tik_instance.Scalar(self.dtype_tiling,
                                                                 name="features_mask_block_size")
        self.output_move_block_size = self.tik_instance.Scalar(self.dtype_tiling,
                                                                 name="output_move_block_size")
        self.output_mask_block_size = self.tik_instance.Scalar(self.dtype_tiling,
                                                                 name="output_mask_block_size")
        self.output_clean_repeat_times = self.tik_instance.Scalar(self.dtype_tiling,
                                                                 name="output_clean_repeat_times")
        self.output_clean_remainder = self.tik_instance.Scalar(self.dtype_tiling,
                                                                 name="output_clean_remainder")
        self.output_clean_offset = self.tik_instance.Scalar(self.dtype_tiling,
                                                                 name="output_clean_offset")
        
    def tiling_args(self):
        tiling_ub = self.tik_instance.Tensor(self.dtype_tiling,
                                            (Constant.TILING_ARG_NUM, ), 
                                            name="tiling_ub",
                                            scope=tik.scope_ubuf)
        
        self.tik_instance.data_move(tiling_ub, self.tiling_gm, 0, 1,
                                    Constant.TILING_ARG_NUM * 
                                    self.dtype_tiling_bytes_size // self.block_byte_size, 0, 0)

        self.bs.set_as(tiling_ub[0])
        self.cs.set_as(tiling_ub[1])
        self.ns.set_as(tiling_ub[2])
        self.ms.set_as(tiling_ub[3])
        self.aicore_num.set_as(tiling_ub[4])
        self.entry_num_in_group.set_as(tiling_ub[5])
        self.used_core_num.set_as(tiling_ub[6])
        self.each_core_entry_group_num.set_as(tiling_ub[7])
        self.each_core_entry_group_padding_num.set_as(tiling_ub[8])
        self.c_fib_ub.set_as(tiling_ub[9])
        self.c_move_tims.set_as(tiling_ub[10])
        self.idx_entry_group_block_size.set_as(tiling_ub[11])
        self.weight_entry_group_block_size.set_as(tiling_ub[12])
        self.tiling_mode_key.set_as(tiling_ub[13])
        self.core_padding_idx.set_as(tiling_ub[14])
        self.features_move_block_size.set_as(self.c_fib_ub * self.dtype_features_bytes // self.block_byte_size)
        self.features_mask_block_size.set_as(self.features_mask * self.dtype_features_bytes // self.block_byte_size)
        self.output_move_block_size.set_as(self.c_fib_ub * self.dtype_output_bytes // self.block_byte_size)
        self.output_mask_block_size.set_as(self.output_mask * self.dtype_output_bytes // self.block_byte_size)
        self.output_clean_repeat_times.set_as(self.c_fib_ub // self.output_mask)
        self.output_clean_remainder.set_as(self.c_fib_ub % self.output_mask)
        self.output_clean_offset.set_as(self.output_clean_repeat_times * self.output_mask)

    def compute_core_dispatch(self):
        with self.tik_instance.for_range(0, self.used_core_num, block_num=self.used_core_num) as core_idx:
            with self.tik_instance.if_scope(core_idx < self.core_padding_idx):
                self._compute_entry_group(core_idx * self.each_core_entry_group_padding_num,
                                          self.each_core_entry_group_padding_num,
                                          core_idx)
            with self.tik_instance.else_scope():
                self._compute_entry_group(self.core_padding_idx * self.each_core_entry_group_padding_num +
                                          (core_idx - self.core_padding_idx) * self.each_core_entry_group_num,
                                          self.each_core_entry_group_num,
                                          core_idx)

    def compute_interpolate(self):
        self.tiling_args()
        self.compute_core_dispatch()

        opt_config = {"out_of_bound_sync_check" : True,
                      "enable_const_fold" : True}
        tbe_context.get_context().add_compile_info('vars', {'core_num': self.total_aicore_num,
                                                            'ub_size': self.ub_byte_size})
        
        self.tik_instance.BuildCCE(
            kernel_name=self.kernel_name,
            inputs=[self.features_gm,
                    self.idx_gm,
                    self.weight_gm],
            outputs=[self.output_gm],
            flowtable=[self.tiling_gm],
            config=opt_config
        )

        return self.tik_instance

    def _calc_output(self, gather_point_ub, weight_ub, output_ub, output_entry_num, move_idx):
        act_c = self.tik_instance.Scalar("int32", "act_c", init_value=self.c_fib_ub)
        with self.tik_instance.if_scope(move_idx == (self.c_move_tims - 1)):
            act_c.set_as(self.cs - move_idx * self.c_fib_ub)

        with self.tik_instance.for_range(0, output_entry_num) as entry_idx:
            self._clean_output_ub(output_ub, entry_idx)
            with self.tik_instance.for_range(0, 3) as loop_idx:
                weight = self.tik_instance.Scalar(self.dtype_weight, "weight",
                                                 init_value=weight_ub[entry_idx, loop_idx])
                self._weight_mul_gather_point(gather_point_ub, weight, entry_idx, loop_idx, act_c)
                self._output_add(gather_point_ub, output_ub, entry_idx, loop_idx, act_c)

    def _gather_point_form_gm(self, gather_point_ub, idx_ub, entry_num, entry_offset, move_idx):
        with self.tik_instance.for_range(0, entry_num) as idx:
            entry_idx = entry_offset + idx
            batch = entry_idx // self.ns
            with self.tik_instance.for_range(0, 3) as point_idx:
                point = self.tik_instance.Scalar(self.dtype_idx,
                                                "point",
                                                init_value=idx_ub[idx, point_idx])
                self.tik_instance.data_move(gather_point_ub[idx, point_idx, :],
                                            self.features_gm[batch * self.cs * self.ms + 
                                                             point * self.cs +
                                                             move_idx * self.c_fib_ub],
                                            0, 1,
                                            self.features_move_block_size,
                                            0, 0, 0, 0)
    
    def _output_ub_move_to_gm(self, output_ub, entry_offset, entry_num, move_idx):
        self.tik_instance.set_atomic_add(self.dtype_output)
        with self.tik_instance.for_range(0, entry_num) as idx:
            entry_idx = self.tik_instance.Scalar("int64", "entry_idx", init_value=entry_offset + idx)
            self.tik_instance.data_move(self.output_gm[entry_idx * self.cs + move_idx * self.c_fib_ub],
                                        output_ub[idx, :],
                                        0,
                                        1,
                                        self.output_move_block_size,
                                        0, 0, 0, 0)
        self.tik_instance.set_atomic_add(0)

    def _clean_output_ub(self, output_ub, entry_idx):
        with self.tik_instance.if_scope(self.output_clean_repeat_times != 0):
            self.tik_instance.vec_dup(self.output_mask,
                                     output_ub[entry_idx, :],
                                     0,
                                     self.output_clean_repeat_times,
                                     self.output_mask_block_size)
        
        with self.tik_instance.if_scope(self.output_clean_remainder != 0):
            self.tik_instance.vec_dup(self.output_clean_remainder,
                                      output_ub[entry_idx, self.output_clean_offset:],
                                      0, 1, 0)
    
    def _weight_mul_gather_point(self, gather_point_ub, weight, entry_idx, loop_idx, act_c):
        repeat_times = self.tik_instance.Scalar("int32", "repeat_times", init_value=(act_c // self.features_mask))
        with self.tik_instance.if_scope(repeat_times != 0):
            self.tik_instance.vmuls(self.features_mask,
                                    gather_point_ub[entry_idx, loop_idx, :],
                                    gather_point_ub[entry_idx, loop_idx, :],
                                    weight,
                                    repeat_times,
                                    1,
                                    1,
                                    self.features_mask_block_size,
                                    self.features_mask_block_size,
                                    0,
                                    "normal")

        remainder = self.tik_instance.Scalar("int32", "remainder", init_value=(act_c % self.features_mask))
        with self.tik_instance.if_scope(remainder != 0):
            self.tik_instance.vmuls(remainder,
                                    gather_point_ub[entry_idx, loop_idx, (repeat_times * self.features_mask):],
                                    gather_point_ub[entry_idx, loop_idx, (repeat_times * self.features_mask):],
                                    weight,
                                    1, 1, 1, 0, 0, 0, "normal")
            
    def _output_add(self, gather_point_ub, output_ub, entry_idx, loop_idx, act_c):
        repeat_times = self.tik_instance.Scalar("int32", "repeat_times", init_value=(act_c // self.output_mask))
        with self.tik_instance.if_scope(repeat_times != 0):
            self.tik_instance.vadd(self.output_mask,
                                   output_ub[entry_idx, :],
                                   output_ub[entry_idx, :],
                                   gather_point_ub[entry_idx, loop_idx, :],
                                   repeat_times,
                                   1, 1, 1,
                                   self.output_mask_block_size,
                                   self.output_mask_block_size,
                                   self.features_mask_block_size,
                                   0)
        
        remainder = self.tik_instance.Scalar("int32", "remainder", init_value=(act_c % self.output_mask))
        with self.tik_instance.if_scope(remainder != 0):
            self.tik_instance.vadd(remainder,
                                   output_ub[entry_idx, (repeat_times * self.output_mask):],
                                   output_ub[entry_idx, (repeat_times * self.output_mask):],
                                   gather_point_ub[entry_idx, loop_idx, (repeat_times * self.output_mask) :],
                                   1, 1, 1, 1, 0, 0, 0, 0)
            
    def _compute_entry_group(self, group_start_offset, group_num, core_idx):
        idx_ub = self.tik_instance.Tensor(self.dtype_idx,
                                         (self.entry_num_in_group, 3),
                                         name="idx_ub",
                                         scope=tik.scope_ubuf)

        weight_ub = self.tik_instance.Tensor(self.dtype_weight,
                                            (self.entry_num_in_group, 3),
                                            name="weight_ub",
                                            scope=tik.scope_ubuf)

        output_ub = self.tik_instance.Tensor(self.dtype_output,
                                            (self.entry_num_in_group, self.c_fib_ub),
                                            name="output_ub",
                                            scope=tik.scope_ubuf)

        gather_point_ub = self.tik_instance.Tensor(self.dtype_features,
                                                  (self.entry_num_in_group, 3, self.c_fib_ub),
                                                  name="gather_point_ub",
                                                  scope=tik.scope_ubuf)

        with self.tik_instance.if_scope(core_idx != (self.used_core_num - 1)):
            self._compute_with_tiling_mode_0_each_core(group_start_offset,
                                            group_num,
                                            idx_ub,
                                            weight_ub,
                                            output_ub,
                                            gather_point_ub)
        with self.tik_instance.else_scope():
            self._compute_with_tiling_mode_0_last_core(group_start_offset,
                                                       group_num,
                                                       idx_ub,
                                                       weight_ub,
                                                       output_ub,
                                                       gather_point_ub)

    def _compute_with_tiling_mode_0_each_core(self, group_start_offset, group_num, idx_ub, weight_ub, output_ub,
                                    gather_point_ub):
        with self.tik_instance.for_range(0, group_num) as group_idx:
            entry_offset = (group_start_offset + group_idx) * self.entry_num_in_group

            self.tik_instance.data_move(idx_ub,
                                        self.idx_gm[entry_offset * 3],
                                        0, 1, self.idx_entry_group_block_size,
                                        0, 0, 0, 0)

            self.tik_instance.data_move(weight_ub,
                                        self.weight_gm[entry_offset * 3],
                                        0, 1, self.weight_entry_group_block_size,
                                        0, 0, 0, 0)
                                    
            with self.tik_instance.for_range(0, self.c_move_tims) as move_idx:
                self._gather_point_form_gm(gather_point_ub,
                                  idx_ub,
                                  self.entry_num_in_group,
                                  entry_offset,
                                  move_idx)

                self._calc_output(gather_point_ub,
                                 weight_ub,
                                 output_ub,
                                 self.entry_num_in_group,
                                 move_idx)

                self._output_ub_move_to_gm(output_ub,
                                          entry_offset,
                                          self.entry_num_in_group,
                                          move_idx)
                
    def _compute_with_tiling_mode_0_last_core(self, group_start_offset, group_num, idx_ub, weight_ub, output_ub,
                                    gather_point_ub):
        with self.tik_instance.for_range(0, group_num) as group_idx:
            entry_offset = (group_start_offset + group_idx) * self.entry_num_in_group

            self.tik_instance.data_move(idx_ub,
                                        self.idx_gm[entry_offset * 3],
                                        0, 1, self.idx_entry_group_block_size,
                                        0, 0, 0, 0)

            self.tik_instance.data_move(weight_ub,
                                        self.weight_gm[entry_offset * 3],
                                        0, 1, self.weight_entry_group_block_size,
                                        0, 0, 0, 0)
                                    
            entry_num = self.tik_instance.Scalar("int32", "entry_num", init_value=self.entry_num_in_group)
            with self.tik_instance.if_scope(group_idx == (group_num - 1)):
                entry_num.set_as(self.bs * self.ns - entry_offset)

            with self.tik_instance.for_range(0, self.c_move_tims) as move_idx:
                self._gather_point_form_gm(gather_point_ub,
                                  idx_ub,
                                  entry_num,
                                  entry_offset,
                                  move_idx)

                self._calc_output(gather_point_ub,
                                 weight_ub,
                                 output_ub,
                                 entry_num,
                                 move_idx)

                self._output_ub_move_to_gm(output_ub,
                                          entry_offset,
                                          entry_num,
                                          move_idx)


@register_operator('ThreeInterpolate')
def three_interpolate(features, idx, weight, output, kernel_name='ThreeInterpolate'):
    """
    algorithm:ThreeInterpolate
    Operation for ThreeInterpolate

    Parameters
    ---------
    features : dict
        dict with keys(range and dtype) of features tensor
    idx : dict
        dict with keys(range and dtype) of idx tensor
    weight : dict
        dict with keys(range and dtype) of weight tensor
    output : dict
        dict with keys(range and dtype) of output tensor
    kernel_name : str
        kernel name, default value is "ThreeeInterpolate"
    Returns
    ---------
    None
    """
    resolution = ThreeInterpolate(features, idx, weight, output, kernel_name)
    tik_instance = resolution.compute_interpolate()
    return tik_instance
                        