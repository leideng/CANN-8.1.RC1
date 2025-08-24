#!/usr/bin/env python
# coding: utf-8
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
The operator name: points_in_polygons
Function: determining whether points are within the polygon
The main calculation formula:
    x = sx + (py - sy) * (tx - sx) / (ty - sy)
    (sx, sy), a endpoint of a polygons side
    (tx, ty), the other endpoint of a polygons side
"""
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import register_operator

# define const var
RESERVE_SIZE = 16 * 1024
BLOCK_SIZE = 32
DTYPE_BYTES = {"float16": 2, "float32": 4, "int32": 4}
MASK_DTYPE = {"float16": "uint16", "float32": "uint32"}
MASK_NUM = {"float16": 128, "float32": 64}
MASK_TENSOR_SHAPE = {"float16": 8, "float32": 2}
TILING_ARG_NUM = 64
TILING_MODE_DEFAULT = 0
SCALAR_TENSOR_SIZE = 32
MAX_INT32 = 2 ** 31 - 1
TILING_DTYPE = "int32"
TILING_MODE_0 = 0
TILING_MODE_1 = 1
TILING_MODE_2 = 2
MAX_REPEAT_TIME = 255
ONE_REPEAT_HANDLE_BLOCKS = 8
TENSOR_NUM = 16


# 'pylint: disable=invalid-name,too-many-arguments,useless-return
# 'pylint: disable=too-many-instance-attributes,attribute-defined-outside-init
# @para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT)
@register_operator("PointsInPolygons")
def points_in_polygons(points, polygons, output, kernel_name="points_in_polygons"):
    """
    the external interfaces of op stride_add

    Parameters
    ----------
    points: dict including shape, format and dtype
        dtype supports  float32; format supports ND
    polygons: dict including shape, format and dtype
        dtype supports float32; format supports ND
    output: dict including shape, format and dtype
        dtype supports float32; format supports ND
    kernel_name: cce kernel name

    Returns
    -------
    tik_instance: tik_instance
    """
    points_in_polygons_process = PointsInPolygons(
        points, polygons, output, kernel_name)
    tik_instance = points_in_polygons_process.compute()
    return


class PointsInPolygons():
    """
    the main class of op points_in_polygons
    """

    def __init__(self, points, polygons, output, kernel_name="points_in_polygons"):
        """
        the constructor function of class PointsInPolygons

        Parameters
        ----------

        Returns
        -------
        None
        """
        # Get data dtype
        self.points_dtype = points.get('dtype')
        self.polygons_dtype = polygons.get('dtype')
        self.output_dtype = output.get('dtype')
        self.data_type = self.points_dtype
        self.mask_type = MASK_DTYPE.get(self.points_dtype)
        self.mask_num = MASK_NUM.get(self.points_dtype)
        self.mask_tensor_shape = MASK_TENSOR_SHAPE.get(self.points_dtype)
        self.kernel_name = kernel_name
        self.tiling_dtype = TILING_DTYPE

        self.points_dsize = DTYPE_BYTES.get(self.data_type)
        self.ele_num_each_block = BLOCK_SIZE // self.points_dsize
        self.dtype_bytes_size_tiling = DTYPE_BYTES.get(self.tiling_dtype)
        self.tiling_each_block = BLOCK_SIZE // self.dtype_bytes_size_tiling

        available_ub_size = (
            tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) - RESERVE_SIZE)
        # maximum elements of each tensor on the UB
        self.ub_max_num = (available_ub_size // self.points_dsize // TENSOR_NUM
                           // self.ele_num_each_block * self.ele_num_each_block)
        self.tik_instance = tik.Tik(tik.Dprofile())

    def define_global_scalar(self):
        self.max_scalar = self.tik_instance.Scalar(self.data_type)
        self.min_scalar = self.tik_instance.Scalar(self.data_type)
        self.const_num_one = self.tik_instance.Scalar(self.points_dtype)
        self.const_num_three = self.tik_instance.Scalar(self.points_dtype)
        self.const_num_one.set_as(1)
        self.const_num_three.set_as(3)
        return

    def prepare_gm_for_data(self):
        """
        Prepare GM for inputs and outputs of points_in_polygons OP

        Parameters
        ----------

        Returns
        -------
        None
        """
        self.tiling_gm = self.tik_instance.Tensor(self.tiling_dtype, (TILING_ARG_NUM,),
                                                  name='tiling_gm', scope=tik.scope_gm)

        self.points_gm = self.tik_instance.Tensor(self.data_type,
                                                  (MAX_INT32,),
                                                  name='points_gm',
                                                  scope=tik.scope_gm)

        self.polygons_gm = self.tik_instance.Tensor(self.data_type,
                                                    (MAX_INT32,),
                                                    name='polygons_gm',
                                                    scope=tik.scope_gm)

        self.output_gm = self.tik_instance.Tensor(self.output_dtype,
                                                  (MAX_INT32,),
                                                  name='output_gm',
                                                  is_atomic_add=True,
                                                  scope=tik.scope_gm)
        return

    def prepare_ub_for_data(self):
        """
        Prepare UB for the input data of points_in_polygons OP

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.point_on_side_ub = self.tik_instance.Tensor(self.data_type,
                                                         (self.ub_max_num,),
                                                         name="point_on_side_ub",
                                                         scope=tik.scope_ubuf)
        self.index_ub = self.tik_instance.Tensor(self.data_type,
                                                 (self.ub_max_num,),
                                                 name="index_ub",
                                                 scope=tik.scope_ubuf)
        self.points_ub = self.tik_instance.Tensor(self.data_type,
                                                  (self.ub_max_num,),
                                                  name="points_ub",
                                                  scope=tik.scope_ubuf)

        self.x1_ub = self.tik_instance.Tensor(self.data_type,
                                              (self.ub_max_num,),
                                              name="x1_ub",
                                              scope=tik.scope_ubuf)
        self.y1_ub = self.tik_instance.Tensor(self.data_type,
                                              (self.ub_max_num,),
                                              name="y1_ub",
                                              scope=tik.scope_ubuf)
        self.x2_ub = self.tik_instance.Tensor(self.data_type,
                                              (self.ub_max_num,),
                                              name="x2_ub",
                                              scope=tik.scope_ubuf)
        self.y2_ub = self.tik_instance.Tensor(self.data_type,
                                              (self.ub_max_num,),
                                              name="y2_ub",
                                              scope=tik.scope_ubuf)
        self.res_ub = self.tik_instance.Tensor(self.data_type,
                                               (self.ub_max_num,),
                                               name="res_ub",
                                               scope=tik.scope_ubuf)
        self.tem_ub = self.tik_instance.Tensor(self.data_type,
                                               (self.ub_max_num,),
                                               name="tem_ub",
                                               scope=tik.scope_ubuf)
        self.compute_x_ub = self.tik_instance.Tensor(self.data_type,
                                                     (self.ub_max_num,),
                                                     name="compute_x_ub",
                                                     scope=tik.scope_ubuf)
        self.compute_y_ub = self.tik_instance.Tensor(self.data_type,
                                                     (self.ub_max_num,),
                                                     name="compute_y_ub",
                                                     scope=tik.scope_ubuf)
        self.compute_y_ub2 = self.tik_instance.Tensor(self.data_type,
                                                      (self.ub_max_num,),
                                                      name="compute_y_ub2",
                                                      scope=tik.scope_ubuf)
        self.mask_tensor = self.tik_instance.Tensor(self.mask_type,
                                                    (self.mask_tensor_shape,),
                                                    name="mask_tensor",
                                                    scope=tik.scope_ubuf)
        self.bool_mask_tensor = self.tik_instance.Tensor("bool",
                                                         (self.ub_max_num,),
                                                         name="mask_tensor",
                                                         scope=tik.scope_ubuf)
        self.dump_one_ub = self.tik_instance.Tensor(self.data_type,
                                                    (self.mask_num,),
                                                    name="dump_one_ub",
                                                    scope=tik.scope_ubuf)
        self.dump_zero_ub = self.tik_instance.Tensor(self.data_type,
                                                     (self.mask_num,),
                                                     name="dump_zero_ub",
                                                     scope=tik.scope_ubuf)
        self.select_dst_ub = self.tik_instance.Tensor(self.data_type,
                                                      (self.ub_max_num,),
                                                      name="select_dst_ub",
                                                      scope=tik.scope_ubuf)
        self.dump_ub(self.ub_max_num, self.res_ub, 0)
        self.dump_ub(self.ub_max_num, self.index_ub, 0)
        self.dump_ub(self.ub_max_num, self.point_on_side_ub, 0)
        self.dump_const_value(self.dump_zero_ub, 0)
        self.dump_const_value(self.dump_one_ub, 1)

        return
   
    def dump_ub(self, compute_num, dump_obj, dump_value):
        return_dic = self.define_const_var_for_cycle(compute_num)
        one_repeate_compute_eles = return_dic.get("one_repeate_compute_eles")
        max_numel_one_loop = return_dic.get("max_repeat_time_compute_eles")
        remain_repeat_time = return_dic.get("remain_repeat_time")
        remain_mask = return_dic.get("remain_mask")
        final_offset = return_dic.get("final_offset")
        loops = self.tik_instance.Scalar("int32")
        loops.set_as(compute_num // max_numel_one_loop)
        dump_offset = self.tik_instance.Scalar("int32")
        with self.tik_instance.if_scope(loops > 0):
            with self.tik_instance.for_range(0, loops) as loop:
                dump_offset.set_as(loop * max_numel_one_loop)
                self.tik_instance.vec_dup(one_repeate_compute_eles,
                                          dump_obj[dump_offset],
                                          dump_value, MAX_REPEAT_TIME, 8)
        with self.tik_instance.if_scope(remain_repeat_time > 0):
            dump_offset.set_as(loops * max_numel_one_loop)
            self.tik_instance.vec_dup(one_repeate_compute_eles,
                                      dump_obj[dump_offset],
                                      dump_value, remain_repeat_time, 8)
        with self.tik_instance.if_scope(remain_mask > 0):
            self.tik_instance.vec_dup(remain_mask,
                                      dump_obj[final_offset],
                                      dump_value,
                                      1,
                                      8)
        return

    def dump_const_value(self, dump_obj, dump_value):
        self.tik_instance.vec_dup(self.mask_num, dump_obj, dump_value, 1, 8)

    def get_tiling_args(self):
        """
        get tiling args from tiling_ub

        Parameters
        ----------
        tiling_ub: tensor with tiling_args in ub

        Returns
        -------
        None
        """
        tiling_ub = self.tik_instance.Tensor(self.tiling_dtype,
                                             (TILING_ARG_NUM,),
                                             name='tiling_ub',
                                             scope=tik.scope_ubuf)
        self.tik_instance.data_move(tiling_ub,
                                    self.tiling_gm,
                                    0,
                                    1,
                                    SCALAR_TENSOR_SIZE //
                                    self.tiling_each_block,
                                    0,
                                    0)
        self.batch_of_points = self.tik_instance.Scalar(
            self.tiling_dtype, name='batch_of_points')
        self.batch_of_polygons = self.tik_instance.Scalar(
            self.tiling_dtype, name='batch_of_polygons')
        self.handle_eles_per_core = self.tik_instance.Scalar(
            self.tiling_dtype, name='handle_eles_per_core')
        self.last_core_handle_eles = self.tik_instance.Scalar(
            self.tiling_dtype, name='last_core_handle_eles')
        self.not_last_core_cycles = self.tik_instance.Scalar(
            self.tiling_dtype, name='not_last_core_cycles')
        self.last_core_cycles = self.tik_instance.Scalar(
            self.tiling_dtype, name='last_core_cycles')
        self.tiling_mode = self.tik_instance.Scalar(
            self.tiling_dtype, name='tiling_mode')
        self.core_used = self.tik_instance.Scalar(
            self.tiling_dtype, name='core_used')
        self.aicore_num = self.tik_instance.Scalar(
            self.tiling_dtype, name='aicore_num')
        self.per_cor_points_num = self.tik_instance.Scalar(
            "int32", name="per_cor_points_num")
        self.last_core_point_num = self.tik_instance.Scalar(
            "int32", name="last_core_point_num")
        self.per_cycle_point_num = self.tik_instance.Scalar(
            "int32", name="per_cycle_point_num")

        self.batch_of_points.set_as(tiling_ub[0])
        self.batch_of_polygons.set_as(tiling_ub[1])
        self.handle_eles_per_core.set_as(tiling_ub[2])
        self.last_core_handle_eles.set_as(tiling_ub[3])
        self.tiling_mode.set_as(tiling_ub[4])
        self.core_used.set_as(tiling_ub[5])
        self.aicore_num.set_as(tiling_ub[6])
        self.per_cor_points_num.set_as(self.handle_eles_per_core // 2)
        self.last_core_point_num.set_as(self.last_core_handle_eles // 2)
        self.per_cycle_point_num.set_as(self.ub_max_num // 2)

    def compute(self):
        """
        the main compute function

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.prepare_gm_for_data()
        self.get_tiling_args()
        self.define_global_scalar()
        with self.tik_instance.for_range(0, self.core_used, block_num=self.core_used) as aicore_id:
            self.prepare_ub_for_data()
            self.compute_per_core(aicore_id)

        opt_config = {"enable_const_fold": True}
        tbe_context.get_context().add_compile_info(
            'vars', {'ub_size': self.ub_max_num})
        self.tik_instance.BuildCCE(
            kernel_name=self.kernel_name,
            inputs=[self.points_gm, self.polygons_gm],
            outputs=[self.output_gm],
            flowtable=[self.tiling_gm],
            config=opt_config)
        return self.tik_instance

    def compute_per_core(self, core_id):
        """
        Make logical judgments based on the number of elements to be processed by each core

        Parameters
        ----------

        Returns
        -------
        None
        """
        points_gm_offset = self.tik_instance.Scalar(
            "int32", name="points_gm_offset")
        compute_num = self.tik_instance.Scalar("int32", name="compute_num")
        cycles = self.tik_instance.Scalar("int32")
        with self.tik_instance.if_scope(core_id == self.core_used - 1):
            compute_num.set_as(self.last_core_handle_eles)
        with self.tik_instance.else_scope():
            compute_num.set_as(self.handle_eles_per_core)
        cycles.set_as(compute_num // self.ub_max_num)

        with self.tik_instance.if_scope(cycles > 0):
            with self.tik_instance.for_range(0, cycles) as cycle:
                points_gm_offset.set_as(
                    core_id * self.handle_eles_per_core + cycle * self.ub_max_num)
                compute_num.set_as(self.ub_max_num)
                self.compute_each_batch(
                    core_id, points_gm_offset, compute_num, cycle)
        compute_num.set_as(compute_num % self.ub_max_num)
        points_gm_offset.set_as(
            core_id * self.handle_eles_per_core + cycles * self.ub_max_num)
        self.compute_each_batch(core_id, points_gm_offset, compute_num, cycles)

    def compute_each_batch(self, core_id, points_gm_offset, compute_num, cycle):
        polygons_gm_offset = self.tik_instance.Scalar("int32")
        output_gm_offset = self.tik_instance.Scalar("int32")
        points_burst_len = self.tik_instance.Scalar("int32")
        polygons_burst_len = self.tik_instance.Scalar("int32")

        points_burst_len.set_as((compute_num - 1) //
                                self.ele_num_each_block + 1)
        self.tik_instance.data_move(self.points_ub,
                                    self.points_gm[points_gm_offset], 0, 1,
                                    points_burst_len, 0, 0)

        index_j = self.tik_instance.Scalar("int32", name="index_j")
        tem = self.tik_instance.Scalar("float32", name="tem")
        move_points = self.tik_instance.Scalar("int32", name="move_points")
        move_points.set_as(compute_num // 2)
        move_times_of_col = self.tik_instance.Scalar("int32")
        # 3
        move_times_of_col.set_as(self.batch_of_polygons // self.ub_max_num)

        with self.tik_instance.for_range(0, move_points) as point_i:
            point_x = self.tik_instance.Scalar(
                self.points_dtype, name="point_x")
            point_y = self.tik_instance.Scalar(
                self.points_dtype, name="point_y")
            index_scalar = self.tik_instance.Scalar(
                "int32", name="index_scalar")
            point_x.set_as(self.points_ub[point_i * 2])
            point_y.set_as(self.points_ub[point_i * 2 + 1])

            # Moving data of polygons multiple times
            with self.tik_instance.if_scope(move_times_of_col > 0):
                with self.tik_instance.for_range(0, move_times_of_col) as loop_index:
                    polygons_gm_offset.set_as(loop_index * self.ub_max_num)
                    polygons_burst_len.set_as(
                        (self.ub_max_num - 1) // self.ele_num_each_block + 1)

                    index_j.set_as(3)
                    with self.tik_instance.for_range(0, 4) as index_i:
                        self.move_data(index_j, index_i,
                                       polygons_gm_offset, polygons_burst_len)
                        index_j.set_as(index_i)
                        self.compute_each_loop(
                            point_x, point_y, self.ub_max_num)

                    with self.tik_instance.for_range(0, self.ub_max_num) as res_i:
                        index_scalar.set_as(self.index_ub[res_i])
                        with self.tik_instance.if_scope(index_scalar > 0):
                            self.res_ub[res_i].set_as(0)
                        index_scalar.set_as(self.point_on_side_ub[res_i])
                        with self.tik_instance.if_scope(index_scalar > 0):
                            self.res_ub[res_i].set_as(0)
                    with self.tik_instance.for_range(0, self.ub_max_num) as k:
                        tem.set_as(self.res_ub[k])
                        with self.tik_instance.if_scope(tik.any(tem == self.const_num_one,
                                                        tem == self.const_num_three)):
                            self.res_ub[k].set_as(1)
                        with self.tik_instance.else_scope():
                            self.res_ub[k].set_as(0)

                    output_gm_offset.set_as(core_id * self.per_cor_points_num * self.batch_of_polygons +
                                            cycle * self.per_cycle_point_num * self.batch_of_polygons +
                                            point_i * self.batch_of_polygons + loop_index * self.ub_max_num)

                    self.tik_instance.data_move(self.output_gm[output_gm_offset],
                                                self.res_ub, 0, 1,
                                                polygons_burst_len, 0, 0)

            last_num_data = self.tik_instance.Scalar("int32")
            last_num_data.set_as(self.batch_of_polygons % self.ub_max_num)
            polygons_gm_offset.set_as(move_times_of_col * self.ub_max_num)

            with self.tik_instance.if_scope(last_num_data > 0):
                polygons_burst_len.set_as(
                    (last_num_data - 1) // self.ele_num_each_block + 1)
                index_j.set_as(3)
                with self.tik_instance.for_range(0, 4) as index_i:
                    self.move_data(index_j, index_i,
                                   polygons_gm_offset, polygons_burst_len)
                    index_j.set_as(index_i)
                    self.compute_each_loop(point_x, point_y, last_num_data)

                # Determine whether the point is on a side of the polygons
                with self.tik_instance.for_range(0, last_num_data) as res_i:
                    index_scalar.set_as(self.index_ub[res_i])
                    with self.tik_instance.if_scope(index_scalar > 0):
                        self.res_ub[res_i].set_as(0)
                    index_scalar.set_as(self.point_on_side_ub[res_i])
                    with self.tik_instance.if_scope(index_scalar > 0):
                        self.res_ub[res_i].set_as(0)

                # Determine whether the number of cross points is an odd number
                with self.tik_instance.for_range(0, last_num_data) as k:
                    tem.set_as(self.res_ub[k])
                    with self.tik_instance.if_scope(tik.any(tem == self.const_num_one, tem == self.const_num_three)):
                        self.res_ub[k].set_as(1)
                    with self.tik_instance.else_scope():
                        self.res_ub[k].set_as(0)

                output_gm_offset.set_as(core_id * self.per_cor_points_num * self.batch_of_polygons +
                                        cycle * self.per_cycle_point_num * self.batch_of_polygons +
                                        point_i * self.batch_of_polygons + polygons_gm_offset)

                self.move_ub_2_gm_for_tail_block(
                    self.output_gm, self.res_ub, output_gm_offset, last_num_data)

                self.dump_ub(self.ub_max_num, self.res_ub, 0)
                self.dump_ub(self.ub_max_num, self.index_ub, 0)
                self.dump_ub(self.ub_max_num, self.point_on_side_ub, 0)

    def move_ub_2_gm_for_tail_block(self, dst_gm, src_ub, gm_offset, data_num):
        block_ub = self.tik_instance.Tensor(
            self.output_dtype,
            (self.ele_num_each_block, ),
            name='block_ub',
            scope=tik.scope_ubuf)
        output_burst_len = self.tik_instance.Scalar(
            "int32", name="output_burst_len")
        output_burst_len.set_as(data_num // self.ele_num_each_block)
        with self.tik_instance.if_scope(output_burst_len >= 1):
            self.tik_instance.data_move(self.output_gm[gm_offset],
                                        src_ub, 0, 1,
                                        output_burst_len, 0, 0)
            with self.tik_instance.if_scope(data_num % self.ele_num_each_block > 0):
                back_addr = data_num // self.ele_num_each_block * self.ele_num_each_block - \
                    (self.ele_num_each_block - data_num %
                     self.ele_num_each_block)

                with self.tik_instance.for_range(0, self.ele_num_each_block) as index:
                    block_ub[index].set_as(src_ub[back_addr + index])
                self.tik_instance.data_move(self.output_gm[gm_offset + back_addr],
                                            block_ub, 0, 1,
                                            1, 0, 0)
        with self.tik_instance.else_scope():
            self.tik_instance.set_atomic_add(self.output_dtype)
            tail_eles = data_num % self.ele_num_each_block
            with self.tik_instance.for_range(tail_eles, self.ele_num_each_block) as ub_index:
                src_ub[ub_index].set_as(0)
            self.tik_instance.data_move(self.output_gm[gm_offset],
                                        src_ub, 0, 1,
                                        1, 0, 0)
            self.tik_instance.set_atomic_add(0)

    def move_data(self, index_j, index_i, polygons_gm_offset, polygons_burst_len):
        """
        move the data of two points on a line

        Parameters
        ----------

        Returns
        -------
        None
        """

        # The coordinate x of the first point
        self.tik_instance.data_move(self.x1_ub,
                                    self.polygons_gm[polygons_gm_offset +
                                                     index_i * 2 * self.batch_of_polygons], 0, 1,
                                    polygons_burst_len, 0, 0)
        # The coordinate y of the first point
        self.tik_instance.data_move(self.y1_ub,
                                    self.polygons_gm[polygons_gm_offset +
                                                     (index_i * 2 + 1) * self.batch_of_polygons], 0, 1,
                                    polygons_burst_len, 0, 0)
        # The coordinate x of the second point
        self.tik_instance.data_move(self.x2_ub,
                                    self.polygons_gm[polygons_gm_offset +
                                                     self.batch_of_polygons * index_j * 2], 0, 1,
                                    polygons_burst_len, 0, 0)
        # The coordinate y of the second point
        self.tik_instance.data_move(self.y2_ub,
                                    self.polygons_gm[polygons_gm_offset +
                                                     self.batch_of_polygons * (index_j * 2 + 1)], 0, 1,
                                    polygons_burst_len, 0, 0)

    def compute_each_loop(self, point_x, point_y, compute_num):
        """
        Calculated according to the formula.
        NOTICE: ubs are reused in some places

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.compute_sub(self.compute_x_ub, self.x2_ub,
                         self.x1_ub, compute_num)

        self.compute_sub(self.compute_y_ub, self.y2_ub,
                         self.y1_ub, compute_num)

        self.compute_muls(self.compute_y_ub2, self.y1_ub, -1.0, compute_num)

        self.compute_adds(self.compute_y_ub2,
                          self.compute_y_ub2, point_y, compute_num)

        self.compute_mul(self.compute_x_ub, self.compute_y_ub2,
                         self.compute_x_ub, compute_num)

        self.compute_div(self.compute_y_ub, self.compute_x_ub,
                         self.compute_y_ub, compute_num)

        self.compute_vadd(self.compute_y_ub, self.x1_ub,
                          self.compute_y_ub, compute_num)

        self.dump_ub(compute_num, self.tem_ub, point_x)

        self.tik_instance.h_cmpv(
            self.bool_mask_tensor, self.compute_y_ub, self.tem_ub, "GT")
        self.dump_ub(compute_num, self.compute_x_ub, 1)
        self.dump_ub(compute_num, self.compute_y_ub2, 0)
        self.tik_instance.h_sel(
            self.select_dst_ub, self.compute_x_ub, self.compute_y_ub2, self.bool_mask_tensor)

        self.compute_vec_cmpv_gt(
            self.mask_tensor, self.compute_y_ub, self.tem_ub, compute_num)
        self.compute_vadd(self.res_ub, self.tem_ub, self.res_ub, compute_num)

        with self.tik_instance.for_range(0, compute_num) as i:
            sx = self.tik_instance.Scalar(self.points_dtype)
            sy = self.tik_instance.Scalar(self.points_dtype)
            tx = self.tik_instance.Scalar(self.points_dtype)
            ty = self.tik_instance.Scalar(self.points_dtype)
            temp_value5 = self.tik_instance.Scalar(self.points_dtype)
            cross_num = self.tik_instance.Scalar(self.points_dtype)
            sx.set_as(self.x1_ub[i])
            sy.set_as(self.y1_ub[i])
            tx.set_as(self.x2_ub[i])
            ty.set_as(self.y2_ub[i])
            temp_value5.set_as(self.compute_y_ub[i])

            # Determine whether the cross point is on the vertex
            with self.tik_instance.if_scope(sy > ty):
                with self.tik_instance.if_scope(ty == point_y):
                    with self.tik_instance.if_scope(tik.all(temp_value5 == tx, temp_value5 > point_x)):
                        cross_num.set_as(self.res_ub[i])
                        self.res_ub[i].set_as(cross_num - 1)
            with self.tik_instance.else_scope():
                with self.tik_instance.if_scope(sy == point_y):
                    with self.tik_instance.if_scope(tik.all(temp_value5 == sx, temp_value5 > point_x)):
                        cross_num.set_as(self.res_ub[i])
                        self.res_ub[i].set_as(cross_num - 1)

            # Determine whether the point(point_x, point_y) is on the vertex
            with self.tik_instance.if_scope(tik.all(sx == point_x, sy == point_y)):
                self.index_ub[i].set_as(1)
            with self.tik_instance.if_scope(tik.all(tx == point_x, ty == point_y)):
                self.index_ub[i].set_as(1)

            # Determine whether the point(point_x, point_y) is on the edge of polygons
            with self.tik_instance.if_scope(temp_value5 == point_x):
                self.point_on_side_ub[i].set_as(1)
            # Notice: resue scalar
            sx.set_as(self.y1_ub[i])
            sy.set_as(self.y2_ub[i])

            # Determine whether the cross point is on the extension line of the edge of polygons
            with self.tik_instance.if_scope(sx >= sy):
                self.max_scalar.set_as(self.y1_ub[i])
                self.min_scalar.set_as(self.y2_ub[i])
            with self.tik_instance.else_scope():
                self.max_scalar.set_as(self.y2_ub[i])
                self.min_scalar.set_as(self.y1_ub[i])
            with self.tik_instance.if_scope(tik.any(point_y > self.max_scalar, point_y < self.min_scalar)):
                sx.set_as(self.select_dst_ub[i])
                sy.set_as(self.res_ub[i])
                with self.tik_instance.if_scope(sx == self.const_num_one):
                    self.res_ub[i].set_as(sy - self.const_num_one)

    def define_const_var_for_cycle(self, compute_num):
        max_repeat_time = MAX_REPEAT_TIME
        one_repeate_compute_eles = ONE_REPEAT_HANDLE_BLOCKS * self.ele_num_each_block
        max_repeat_time_compute_eles = one_repeate_compute_eles * max_repeat_time
        remain_repeat_time = compute_num % max_repeat_time_compute_eles // one_repeate_compute_eles
        remain_mask = compute_num % max_repeat_time_compute_eles % one_repeate_compute_eles
        final_offset = compute_num // max_repeat_time_compute_eles * max_repeat_time_compute_eles + \
            compute_num % max_repeat_time_compute_eles // one_repeate_compute_eles * \
            one_repeate_compute_eles
        return {
                "one_repeate_compute_eles": one_repeate_compute_eles,
                "max_repeat_time_compute_eles": max_repeat_time_compute_eles,
                "remain_repeat_time": remain_repeat_time,
                "remain_mask": remain_mask,
                "final_offset": final_offset
                }

    def compute_sub(self, dst, src0, src1, compute_num):
        return_dic = self.define_const_var_for_cycle(compute_num)
        one_repeate_compute_eles = return_dic.get("one_repeate_compute_eles")
        max_numel_one_loop = return_dic.get("max_repeat_time_compute_eles")
        remain_repeat_time = return_dic.get("remain_repeat_time")
        remain_mask = return_dic.get("remain_mask")
        final_offset = return_dic.get("final_offset")
        loops = self.tik_instance.Scalar("int32")
        loops.set_as(compute_num // max_numel_one_loop)
        with self.tik_instance.if_scope(loops > 0):
            with self.tik_instance.for_range(0, loops) as loop:
                self.tik_instance.vec_sub(one_repeate_compute_eles,
                                          dst[loop * max_numel_one_loop],
                                          src0[loop * max_numel_one_loop],
                                          src1[loop * max_numel_one_loop],
                                          MAX_REPEAT_TIME,
                                          8, 8, 8)

        with self.tik_instance.if_scope(remain_repeat_time > 0):
            self.tik_instance.vec_sub(one_repeate_compute_eles,
                                      dst[compute_num // max_numel_one_loop *
                                          max_numel_one_loop],
                                      src0[compute_num // max_numel_one_loop *
                                           max_numel_one_loop],
                                      src1[compute_num // max_numel_one_loop *
                                           max_numel_one_loop], remain_repeat_time,
                                      8, 8, 8)

        with self.tik_instance.if_scope(remain_mask > 0):
            self.tik_instance.vec_sub(remain_mask,
                                      dst[final_offset],
                                      src0[final_offset],
                                      src1[final_offset],
                                      1,
                                      8, 8, 8)

    def compute_muls(self, dst, src, scalar, compute_num):
        return_dic = self.define_const_var_for_cycle(compute_num)
        one_repeate_compute_eles = return_dic.get("one_repeate_compute_eles")
        max_numel_one_loop = return_dic.get("max_repeat_time_compute_eles")
        remain_repeat_time = return_dic.get("remain_repeat_time")
        remain_mask = return_dic.get("remain_mask")
        final_offset = return_dic.get("final_offset")
        loops = self.tik_instance.Scalar("int32")
        loops.set_as(compute_num // max_numel_one_loop)
        with self.tik_instance.if_scope(loops > 0):
            with self.tik_instance.for_range(0, loops) as loop:
                with self.tik_instance.for_range(0,
                                                 compute_num // max_numel_one_loop) as loop:
                    self.tik_instance.vec_muls(one_repeate_compute_eles,
                                               dst[loop * max_numel_one_loop],
                                               src[loop * max_numel_one_loop],
                                               scalar,
                                               MAX_REPEAT_TIME,
                                               8, 8)

        with self.tik_instance.if_scope(remain_repeat_time > 0):
            self.tik_instance.vec_muls(one_repeate_compute_eles,
                                       dst[compute_num // max_numel_one_loop *
                                           max_numel_one_loop],
                                       src[compute_num // max_numel_one_loop *
                                           max_numel_one_loop],
                                       scalar, remain_repeat_time, 8, 8)

        with self.tik_instance.if_scope(remain_mask > 0):
            self.tik_instance.vec_muls(remain_mask,
                                       dst[final_offset],
                                       src[final_offset],
                                       scalar,
                                       1,
                                       8, 8)

    def compute_mul(self, dst, src0, src1, compute_num):
        return_dic = self.define_const_var_for_cycle(compute_num)
        one_repeate_compute_eles = return_dic.get("one_repeate_compute_eles")
        max_numel_one_loop = return_dic.get("max_repeat_time_compute_eles")
        remain_repeat_time = return_dic.get("remain_repeat_time")
        remain_mask = return_dic.get("remain_mask")
        final_offset = return_dic.get("final_offset")
        loops = self.tik_instance.Scalar("int32")
        loops.set_as(compute_num // max_numel_one_loop)
        with self.tik_instance.if_scope(loops > 0):
            with self.tik_instance.for_range(0, loops) as loop:
                self.tik_instance.vec_mul(one_repeate_compute_eles,
                                          dst[loop * max_numel_one_loop],
                                          src0[loop * max_numel_one_loop],
                                          src1[loop * max_numel_one_loop], MAX_REPEAT_TIME,
                                          8, 8, 8)

        with self.tik_instance.if_scope(remain_repeat_time > 0):
            self.tik_instance.vec_mul(one_repeate_compute_eles,
                                      dst[compute_num // max_numel_one_loop *
                                          max_numel_one_loop],
                                      src0[compute_num // max_numel_one_loop *
                                           max_numel_one_loop],
                                      src1[compute_num // max_numel_one_loop *
                                           max_numel_one_loop], remain_repeat_time,
                                      8, 8, 8)

        with self.tik_instance.if_scope(remain_mask > 0):
            self.tik_instance.vec_mul(remain_mask,
                                      dst[final_offset],
                                      src0[final_offset],
                                      src1[final_offset],
                                      1,
                                      8, 8, 8)

    def compute_div(self, dst, src0, src1, compute_num):
        return_dic = self.define_const_var_for_cycle(compute_num)
        one_repeate_compute_eles = return_dic.get("one_repeate_compute_eles")
        max_numel_one_loop = return_dic.get("max_repeat_time_compute_eles")
        remain_repeat_time = return_dic.get("remain_repeat_time")
        remain_mask = return_dic.get("remain_mask")
        final_offset = return_dic.get("final_offset")
        loops = self.tik_instance.Scalar("int32")
        loops.set_as(compute_num // max_numel_one_loop)
        with self.tik_instance.if_scope(loops > 0):
            with self.tik_instance.for_range(0,
                                             compute_num // max_numel_one_loop) as loop:
                self.tik_instance.vdiv(one_repeate_compute_eles,
                                       dst[loop * max_numel_one_loop],
                                       src0[loop * max_numel_one_loop],
                                       src1[loop * max_numel_one_loop], MAX_REPEAT_TIME,  1, 1, 1,
                                       8, 8, 8)

        with self.tik_instance.if_scope(remain_repeat_time > 0):
            self.tik_instance.vdiv(one_repeate_compute_eles,
                                   dst[compute_num // max_numel_one_loop *
                                       max_numel_one_loop],
                                   src0[compute_num // max_numel_one_loop *
                                        max_numel_one_loop],
                                   src1[compute_num // max_numel_one_loop *
                                        max_numel_one_loop], remain_repeat_time,  1, 1, 1,
                                   8, 8, 8)

        with self.tik_instance.if_scope(remain_mask > 0):
            self.tik_instance.vdiv(remain_mask,
                                   dst[final_offset],
                                   src0[final_offset],
                                   src1[final_offset],
                                   1,
                                   1, 1, 1,
                                   8, 8, 8)

    def compute_adds(self, dst, src, scalar, compute_num):
        return_dic = self.define_const_var_for_cycle(compute_num)
        one_repeate_compute_eles = return_dic.get("one_repeate_compute_eles")
        max_numel_one_loop = return_dic.get("max_repeat_time_compute_eles")
        remain_repeat_time = return_dic.get("remain_repeat_time")
        remain_mask = return_dic.get("remain_mask")
        final_offset = return_dic.get("final_offset")
        loops = self.tik_instance.Scalar("int32")
        loops.set_as(compute_num // max_numel_one_loop)
        with self.tik_instance.if_scope(loops > 0):
            with self.tik_instance.for_range(0,
                                             compute_num // max_numel_one_loop) as loop:
                self.tik_instance.vec_adds(one_repeate_compute_eles,
                                           dst[loop * max_numel_one_loop],
                                           src[loop * max_numel_one_loop],
                                           scalar, MAX_REPEAT_TIME, 8, 8)

        with self.tik_instance.if_scope(remain_repeat_time > 0):
            self.tik_instance.vec_adds(one_repeate_compute_eles,
                                       dst[compute_num // max_numel_one_loop *
                                           max_numel_one_loop],
                                       src[compute_num // max_numel_one_loop *
                                           max_numel_one_loop],
                                       scalar, remain_repeat_time, 8, 8)
        with self.tik_instance.if_scope(remain_mask > 0):
            self.tik_instance.vec_adds(remain_mask,
                                       dst[final_offset],
                                       src[final_offset],
                                       scalar,
                                       1, 8, 8)

    def compute_vadd(self, dst, src0, src1, compute_num):
        return_dic = self.define_const_var_for_cycle(compute_num)
        one_repeate_compute_eles = return_dic.get("one_repeate_compute_eles")
        max_numel_one_loop = return_dic.get("max_repeat_time_compute_eles")
        remain_repeat_time = return_dic.get("remain_repeat_time")
        remain_mask = return_dic.get("remain_mask")
        final_offset = return_dic.get("final_offset")
        add_offset = self.tik_instance.Scalar("int32")
        loops = self.tik_instance.Scalar("int32")
        loops.set_as(compute_num // max_numel_one_loop)
        with self.tik_instance.if_scope(loops > 0):
            with self.tik_instance.for_range(0,
                                             compute_num // max_numel_one_loop) as loop:
                add_offset.set_as(loop * max_numel_one_loop)
                self.tik_instance.vec_add(64,
                                          dst[add_offset],
                                          src0[add_offset],
                                          src1[add_offset],
                                          MAX_REPEAT_TIME, 8, 8, 8)

        with self.tik_instance.if_scope(remain_repeat_time > 0):
            self.tik_instance.vec_add(one_repeate_compute_eles,
                                      dst[compute_num // max_numel_one_loop *
                                          max_numel_one_loop],
                                      src0[compute_num // max_numel_one_loop *
                                           max_numel_one_loop],
                                      src1[compute_num // max_numel_one_loop *
                                           max_numel_one_loop], remain_repeat_time,
                                      8, 8, 8)
        with self.tik_instance.if_scope(remain_mask > 0):
            self.tik_instance.vec_add(remain_mask,
                                      dst[final_offset],
                                      src0[final_offset],
                                      src1[final_offset],
                                      1, 8, 8, 8)

    def compute_vec_cmpv_gt(self, sel, src0, src1, compute_num):
        offset = self.tik_instance.Scalar("int32")
        loops = self.tik_instance.Scalar("int32")
        loops.set_as(compute_num // self.mask_num)
        with self.tik_instance.if_scope(loops > 0):
            with self.tik_instance.for_range(0, loops) as loop:
                offset.set_as(loop * self.mask_num)
                self.tik_instance.vec_cmpv_gt(sel,
                                              src0[offset],
                                              src1[offset],
                                              1, 8, 8)
                self.tik_instance.vec_sel(
                    self.mask_num, 0, src1[offset], sel, self.dump_one_ub, self.dump_zero_ub, 1, 8, 8, 8)
        left_num = self.tik_instance.Scalar("int32")
        left_num.set_as(compute_num % self.mask_num)
        with self.tik_instance.if_scope(left_num > 0):
            offset.set_as(loops * self.mask_num)
            self.tik_instance.vec_cmpv_gt(sel,
                                          src0[offset],
                                          src1[offset],
                                          1, 8, 8)
            self.tik_instance.vec_sel(
                left_num, 0, src1[offset], sel, self.dump_one_ub, self.dump_zero_ub, 1, 8, 8, 8)
