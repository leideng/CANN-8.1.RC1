#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright 2019 Huawei Technologies Co., Ltd
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
decode_wheels_target
"""
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import para_check


# 'pylint: disable=too-few-public-methods,not-use-list-comprehension
class Constant:
    """
    the class for constant
    """
    SINGLE_N_MAX = 640
    N_MAX = 65500
    ONE = 1
    TWO = 2
    FOUR = 4
    EIGHT = 8
    TEN = 10
    SIXTEEN = 16
    BLOCK_DATA = 32
    MASK = 128
    MAXTRIX = 256
    MAXTRIX_DATA = 512


# 'pylint: disable=super-with-arguments
def int_ceil_div(divisor_a, divisor_b):
    """
    round up function

    Paramater:
    :param divisor_a: int.
    :param divisor_b: int.
    :return: int
    """
    if divisor_b == 0:
        raise RuntimeError("division by zero")
    return (divisor_a + divisor_b - Constant.ONE) // divisor_b


def check_decode_wheels_target_params(boundary_predictions, anchors, boundary_encoded):
    """
    The params check function of decode_wheels_target

    Parameters:
    ----------
    Returns : All transformed params.
    ----------
    """
    shape_x = boundary_predictions.get("shape")
    shape_x_num = len(shape_x)
    shape_y = anchors.get("shape")
    shape_y_num = len(shape_y)
    shape_z = boundary_encoded.get("shape")
    shape_z_num = len(shape_z)
    dtype_x = boundary_predictions.get("dtype").lower()
    dtype_y = anchors.get("dtype").lower()
    dtype_z = boundary_encoded.get("dtype").lower()

    # Abnormality test
    if dtype_x != dtype_y or dtype_x != dtype_z or dtype_y != dtype_z:
        raise RuntimeError("dtype of inputs and output should be consistent")
    if dtype_x != 'float16':
        raise RuntimeError("dtype of inputs should be float16")
    if shape_x_num != shape_y_num or shape_x_num != shape_z_num or shape_y_num != shape_z_num:
        raise RuntimeError("dimension of inputs should be consistent")
    if shape_x_num != Constant.TWO:
        raise RuntimeError("dimension of inputs should be TWO")
    check_decode_wheels_target_params_1(shape_x, shape_y, shape_z)


def check_decode_wheels_target_params_1(shape_x, shape_y, shape_z):
    """
    The params check function of decode_wheels_target

    Parameters:
    ----------
    Returns : All transformed params.
    ----------
    """
    n_x, m_x = shape_x
    n_y, m_y = shape_y
    n_z, m_z = shape_z
    if not isinstance(n_x, int):
        raise RuntimeError("n dimension of input should be int")
    if n_x != n_y or n_x != n_z or n_y != n_z:
        raise RuntimeError("n dimension of inputs should be consistent")
    if m_x != Constant.EIGHT:
        raise RuntimeError("m dimension of boundary_predictions should be EIGHT")
    if m_y != Constant.FOUR:
        raise RuntimeError("m dimension of anchors should be FOUR")
    if m_z != Constant.EIGHT:
        raise RuntimeError("m dimension of boundary_encoded should be EIGHT")
    if n_x < Constant.ONE or n_x > Constant.N_MAX:
        raise RuntimeError("n dimension of inputs should in [1, 65500]")


class Tiling:
    """
    calculating the shape
    Returns
    -------
    None
    """
    def __init__(self, n_x, core_num):
        self.n_x = n_x
        self.core_num = core_num
        self.last_n = self.n_x % Constant.SINGLE_N_MAX
        self.last_core = self.n_x // Constant.SINGLE_N_MAX % self.core_num
        self.factor = self.n_x // Constant.SINGLE_N_MAX // self.core_num

    def set_shape_maxtrix(self, n_x):
        """
        set_input_shape
        :return:
        """
        self.n_x = n_x

    def set_n_maxtrix(self, core_num):
        """
        set_input_shape
        :return:
        """
        self.core_num = core_num


class InitShape:
    """
    calculating the shape
    Returns
    -------
    None
    """
    def __init__(self, n_x):
        self.n_x = n_x
        # number of SIXTEEN*SIXTEEN
        self.n_maxtrix = int_ceil_div(n_x * Constant.EIGHT, Constant.MAXTRIX)
        # shape of calculate
        self.shape_maxtrix = (self.n_maxtrix, Constant.SIXTEEN, Constant.SIXTEEN)
        # repeat times of rep_stride*block
        self.repeat_whxy = self.n_maxtrix * Constant.TWO // Constant.EIGHT
        # rep_stride*block of one repeat
        self.rep_stride = self.n_maxtrix * Constant.TWO % Constant.EIGHT
        # number of instruction
        self.instruction_number = Constant.TWO

        if self.repeat_whxy == 0:
            self.repeat_whxy = Constant.ONE
            self.instruction_number = Constant.ONE

        if self.rep_stride == 0:
            self.instruction_number = Constant.ONE
            self.rep_stride = Constant.EIGHT

    def set_input_shape(self, repeat_whxy):
        """
        set_input_shape
        :return:
        """
        self.repeat_whxy = repeat_whxy

    def set_output_shape(self, rep_stride):
        """
        set_input_shape
        :return:
        """
        self.rep_stride = rep_stride


class InitTensor:
    """
    init the input output tensor
    Returns
    -------
    None
    """
    def __init__(self, tik_instance, shape_x, shape_y, dtype_x):
        self.data_x = tik_instance.Tensor(
            dtype_x, shape_x, name="data_x", scope=tik.scope_gm)
        self.data_y = tik_instance.Tensor(
            dtype_x, shape_y, name="data_y", scope=tik.scope_gm)
        self.data_z = tik_instance.Tensor(
            dtype_x, shape_x, name="data_z", scope=tik.scope_gm)
        self.dump_0 = tik_instance.Scalar(dtype="float16", init_value=0.0)
        self.dump_half = tik_instance.Scalar(dtype="float16", init_value=0.5)

    def set_data_x(self, data_x):
        """
        data_x_ub
        :return:
        """
        self.data_x = data_x

    def set_data_y(self, data_y):
        """
        data_y_ub
        :return:
        """
        self.data_y = data_y


class InitSecondTensor:
    """
    init the input output tensor
    Returns
    -------
    None
    """
    def __init__(self, tik_instance, init_shape, dtype_x):
        self.data_x_ub = tik_instance.Tensor(
            dtype_x, init_shape.shape_maxtrix, name="data_x_ub", scope=tik.scope_ubuf)
        self.data_y_ub = tik_instance.Tensor(
            dtype_x, init_shape.shape_maxtrix, name="data_y_ub", scope=tik.scope_ubuf)
        self.data_x_ub_trs = tik_instance.Tensor(
            dtype_x, init_shape.shape_maxtrix, name="data_x_ub_trs", scope=tik.scope_ubuf)
        self.data_y_ub_trs = tik_instance.Tensor(
            dtype_x, init_shape.shape_maxtrix, name="data_y_ub_trs", scope=tik.scope_ubuf)
        self.data_y_ub_trs1 = tik_instance.Tensor(
            dtype_x, (init_shape.n_maxtrix, Constant.EIGHT, Constant.SIXTEEN),
            name="data_y_ub_trs1", scope=tik.scope_ubuf)

    def set_data_x_ub(self, data_x_ub):
        """
        data_y_ub_trs1
        :return:
        """
        self.data_x_ub = data_x_ub

    def set_data_y_ub(self, data_y_ub):
        """
        data_anchor_wh
        :return:
        """
        self.data_y_ub = data_y_ub


class InitThirdTensor(InitSecondTensor):
    """
    init the input output tensor
    Returns
    -------
    None
    """
    def __init__(self, tik_instance, init_shape, dtype_x):
        super(InitThirdTensor, self).__init__(tik_instance, init_shape, dtype_x)
        self.data_anchor_wh = tik_instance.Tensor(
            dtype_x, init_shape.shape_maxtrix, name="data_anchor_wh", scope=tik.scope_ubuf)
        self.data_anchor_x0y0 = tik_instance.Tensor(
            dtype_x, init_shape.shape_maxtrix, name="data_anchor_x0y0", scope=tik.scope_ubuf)
        self.data_anchor_xy = tik_instance.Tensor(
            dtype_x, init_shape.shape_maxtrix, name="data_anchor_xy", scope=tik.scope_ubuf)
        self.data_z_ub0 = tik_instance.Tensor(
            dtype_x, init_shape.shape_maxtrix, name="data_z_ub0", scope=tik.scope_ubuf)
        self.data_z_ub1 = tik_instance.Tensor(
            dtype_x, init_shape.shape_maxtrix, name="data_z_ub1", scope=tik.scope_ubuf)
        self.data_z_ub = tik_instance.Tensor(
            dtype_x, init_shape.shape_maxtrix, name="data_z_ub", scope=tik.scope_ubuf)

    def set_data_y_ub_trs1(self, data_y_ub_trs1):
        """
        data_y_ub_trs1
        :return:
        """
        self.data_y_ub_trs1 = data_y_ub_trs1

    def set_data_anchor_wh(self, data_anchor_wh):
        """
        data_anchor_wh
        :return:
        """
        self.data_anchor_wh = data_anchor_wh


def calculate_process(tik_instance, gm_tensor, shape, current_data_x, current_data_y):
    """
    :param tik_instance: tik
    :param shape: class
    :param gm_tensor:class
    :param current_data_x: int
    :param current_data_y: int
    :return:
    """
    # scalr init
    mid_tensor = InitThirdTensor(tik_instance, shape, "float16")
    tik_instance.data_move(mid_tensor.data_x_ub,
                           gm_tensor.data_x[current_data_x],
                           0,
                           Constant.ONE, int_ceil_div(shape.n_x * Constant.EIGHT, Constant.SIXTEEN),
                           0, 0)

    tik_instance.data_move(mid_tensor.data_y_ub,
                           gm_tensor.data_y[current_data_y],
                           0,
                           int_ceil_div(shape.n_x * Constant.FOUR, Constant.SIXTEEN), Constant.ONE,
                           0, Constant.ONE)
    if shape.n_x not in (Constant.ONE, Constant.TWO):
        tik_instance.data_move(mid_tensor.data_y_ub[0 + Constant.SIXTEEN],
                               gm_tensor.data_y[current_data_y + Constant.EIGHT],
                               0, int_ceil_div(shape.n_x * Constant.FOUR, Constant.SIXTEEN),
                               Constant.ONE, 0, Constant.ONE)
    with tik_instance.for_range(0, shape.n_maxtrix) as i:
        tik_instance.vtranspose(mid_tensor.data_x_ub_trs[Constant.MAXTRIX * i],
                                mid_tensor.data_x_ub[Constant.MAXTRIX * i])
        tik_instance.vtranspose(mid_tensor.data_y_ub_trs[Constant.MAXTRIX * i],
                                mid_tensor.data_y_ub[Constant.MAXTRIX * i])

    # extract tensor_y_ub_trs
    tik_instance.vadds(Constant.MASK,
                       mid_tensor.data_y_ub_trs1,
                       mid_tensor.data_y_ub_trs,
                       gm_tensor.dump_0,
                       shape.n_maxtrix,
                       Constant.ONE, Constant.ONE,
                       Constant.EIGHT, Constant.SIXTEEN)
    # calculate data_anchor_wh and data_anchor_x0y0
    with tik_instance.if_scope(shape.instruction_number == Constant.ONE):
        with tik_instance.for_range(0, Constant.FOUR) as i:
            with tik_instance.for_range(0, Constant.TWO) as j:
                tik_instance.vsub(Constant.SIXTEEN * shape.rep_stride,
                                  mid_tensor.data_anchor_wh[Constant.SIXTEEN * i * Constant.TWO + Constant.SIXTEEN * j],
                                  mid_tensor.data_y_ub_trs1[Constant.SIXTEEN * j + Constant.BLOCK_DATA],
                                  mid_tensor.data_y_ub_trs1[Constant.SIXTEEN * j],
                                  shape.repeat_whxy,
                                  Constant.EIGHT,
                                  Constant.FOUR,
                                  Constant.FOUR,
                                  Constant.EIGHT * shape.rep_stride,
                                  Constant.FOUR * shape.rep_stride,
                                  Constant.FOUR * shape.rep_stride)
                tik_instance.vadd(Constant.SIXTEEN * shape.rep_stride,
                                  mid_tensor.data_anchor_x0y0[Constant.SIXTEEN * i * Constant.TWO
                                                              + Constant.SIXTEEN * j],
                                  mid_tensor.data_y_ub_trs1[Constant.SIXTEEN * j + Constant.BLOCK_DATA],
                                  mid_tensor.data_y_ub_trs1[Constant.SIXTEEN * j],
                                  shape.repeat_whxy,
                                  Constant.EIGHT,
                                  Constant.FOUR,
                                  Constant.FOUR,
                                  Constant.EIGHT * shape.rep_stride,
                                  Constant.FOUR * shape.rep_stride,
                                  Constant.FOUR * shape.rep_stride)

    with tik_instance.else_scope():
        with tik_instance.for_range(0, Constant.FOUR) as i:
            with tik_instance.for_range(0, Constant.TWO) as j:
                tik_instance.vsub(Constant.MASK,
                                  mid_tensor.data_anchor_wh[Constant.SIXTEEN * i * Constant.TWO + Constant.SIXTEEN * j],
                                  mid_tensor.data_y_ub_trs1[Constant.SIXTEEN * j + Constant.BLOCK_DATA],
                                  mid_tensor.data_y_ub_trs1[Constant.SIXTEEN * j],
                                  shape.repeat_whxy,
                                  Constant.EIGHT, Constant.FOUR, Constant.FOUR,
                                  Constant.EIGHT * Constant.EIGHT, Constant.BLOCK_DATA, Constant.BLOCK_DATA)

                tik_instance.vsub(Constant.SIXTEEN * shape.rep_stride,
                                  mid_tensor.data_anchor_wh[Constant.SIXTEEN * i * Constant.TWO + Constant.SIXTEEN * j
                                                            + shape.repeat_whxy * Constant.FOUR * Constant.MAXTRIX],
                                  mid_tensor.data_y_ub_trs1[Constant.SIXTEEN * j + Constant.BLOCK_DATA
                                                            + shape.repeat_whxy * Constant.MAXTRIX_DATA],
                                  mid_tensor.data_y_ub_trs1[Constant.SIXTEEN * j
                                                            + shape.repeat_whxy * Constant.MAXTRIX_DATA],
                                  Constant.ONE,
                                  Constant.EIGHT, Constant.FOUR, Constant.FOUR,
                                  Constant.EIGHT * shape.rep_stride,
                                  Constant.FOUR * shape.rep_stride,
                                  Constant.FOUR * shape.rep_stride)

                tik_instance.vadd(Constant.MASK,
                                  mid_tensor.data_anchor_x0y0[Constant.SIXTEEN * i * Constant.TWO
                                                              + Constant.SIXTEEN * j],
                                  mid_tensor.data_y_ub_trs1[Constant.SIXTEEN * j + Constant.BLOCK_DATA],
                                  mid_tensor.data_y_ub_trs1[Constant.SIXTEEN * j],
                                  shape.repeat_whxy,
                                  Constant.EIGHT, Constant.FOUR, Constant.FOUR,
                                  Constant.EIGHT * Constant.EIGHT, Constant.BLOCK_DATA, Constant.BLOCK_DATA)

                tik_instance.vadd(Constant.SIXTEEN * shape.rep_stride,
                                  mid_tensor.data_anchor_x0y0[Constant.SIXTEEN * i * Constant.TWO
                                                              + Constant.SIXTEEN * j
                                                              + shape.repeat_whxy * Constant.FOUR * Constant.MAXTRIX],
                                  mid_tensor.data_y_ub_trs1[Constant.SIXTEEN * j + Constant.BLOCK_DATA
                                                            + shape.repeat_whxy * Constant.MAXTRIX_DATA],
                                  mid_tensor.data_y_ub_trs1[Constant.SIXTEEN * j
                                                            + shape.repeat_whxy * Constant.MAXTRIX_DATA],
                                  Constant.ONE,
                                  Constant.EIGHT, Constant.FOUR, Constant.FOUR,
                                  Constant.EIGHT * shape.rep_stride,
                                  Constant.FOUR * shape.rep_stride,
                                  Constant.FOUR * shape.rep_stride)

    # calculate mid_tensor.data_anchor_xy

    tik_instance.vmuls(Constant.MASK, mid_tensor.data_anchor_xy,
                       mid_tensor.data_anchor_x0y0,
                       gm_tensor.dump_half,
                       shape.n_maxtrix * Constant.TWO,
                       Constant.ONE, Constant.ONE,
                       Constant.EIGHT, Constant.EIGHT)

    # calculate input * mid_tensor.data_anchor_wh
    tik_instance.vmul(Constant.MASK,
                      mid_tensor.data_z_ub0,
                      mid_tensor.data_x_ub_trs,
                      mid_tensor.data_anchor_wh,
                      shape.n_maxtrix * Constant.TWO,
                      Constant.ONE, Constant.ONE, Constant.ONE,
                      Constant.EIGHT, Constant.EIGHT, Constant.EIGHT)

    # calculate input * mid_tensor.data_anchor_wh + mid_tensor.data_anchor_xy

    tik_instance.vadd(Constant.MASK,
                      mid_tensor.data_z_ub1,
                      mid_tensor.data_z_ub0,
                      mid_tensor.data_anchor_xy,
                      shape.n_maxtrix * Constant.TWO,
                      Constant.ONE, Constant.ONE, Constant.ONE,
                      Constant.EIGHT, Constant.EIGHT, Constant.EIGHT)
    # transpose output
    with tik_instance.for_range(0, shape.n_maxtrix) as i:
        tik_instance.vtranspose(mid_tensor.data_z_ub[Constant.MAXTRIX * i],
                                mid_tensor.data_z_ub1[Constant.MAXTRIX * i])

    # copy ub to gm
    tik_instance.data_move(gm_tensor.data_z[current_data_x],
                           mid_tensor.data_z_ub,
                           0,
                           Constant.ONE, int_ceil_div(shape.n_x * Constant.EIGHT, Constant.SIXTEEN),
                           0, 0)


@para_check.check_input_type(dict, dict, dict, str)
def decode_wheels_target(
        boundary_predictions,
        anchors,
        boundary_encoded,
        kernel_name="cce_decode_wheels_target_float16"):
    """
    calculating data

    Parameters
    ----------
    boundary_predictions : dict
        shape and dtype of boundary_predictions
    anchors : dict
        shape and dtype of anchors
    boundary_encoded : dict
        shape and dtype of output, should be same shape and type as boundary_predictions
    kernel_name : str
        kernel name, default value is "decode_wheels_target"

    Returns none
    -------

    """

    check_decode_wheels_target_params(boundary_predictions, anchors, boundary_encoded)
    para_check.check_kernel_name(kernel_name)
    shape_x = boundary_predictions.get("shape")

    tik_instance = tik.Tik(tik.Dprofile(), True)
    core_num = tik.Dprofile().get_aicore_num()

    tiling = Tiling(shape_x[0], core_num)

    # gm_tensor init
    gm_tensor = InitTensor(tik_instance, shape_x, [shape_x[0], Constant.FOUR], 'float16')
    if tiling.factor > 0:
        thread_num = Constant.TWO if tiling.factor != Constant.ONE else Constant.ONE
        with tik_instance.for_range(0, core_num, block_num=core_num) as current_core:
            with tik_instance.for_range(0, tiling.factor, thread_num=thread_num) as current_factor:
                shape = InitShape(Constant.SINGLE_N_MAX)
                current_data_x = Constant.EIGHT * Constant.SINGLE_N_MAX * (current_core + core_num * current_factor)
                current_data_y = Constant.FOUR * Constant.SINGLE_N_MAX * (current_core + core_num * current_factor)
                calculate_process(tik_instance, gm_tensor, shape, current_data_x, current_data_y)
    if tiling.last_core > 0:
        thread_num = Constant.TWO if tiling.last_core != Constant.ONE else Constant.ONE
        with tik_instance.for_range(0, tiling.last_core, thread_num=thread_num) as current_core:
            shape = InitShape(Constant.SINGLE_N_MAX)
            current_data_x = Constant.EIGHT * Constant.SINGLE_N_MAX * (core_num * tiling.factor + current_core)
            current_data_y = Constant.FOUR * Constant.SINGLE_N_MAX * (core_num * tiling.factor + current_core)
            calculate_process(tik_instance, gm_tensor, shape, current_data_x, current_data_y)
    if tiling.last_n > 0:
        shape = InitShape(tiling.last_n)
        current_data_x = Constant.EIGHT * Constant.SINGLE_N_MAX * (core_num * tiling.factor + tiling.last_core)
        current_data_y = Constant.FOUR * Constant.SINGLE_N_MAX * (core_num * tiling.factor + tiling.last_core)
        calculate_process(tik_instance, gm_tensor, shape, current_data_x, current_data_y)

    # build_cce
    tik_instance.BuildCCE(
        kernel_name=kernel_name,
        inputs=[gm_tensor.data_x, gm_tensor.data_y],
        outputs=[gm_tensor.data_z])
