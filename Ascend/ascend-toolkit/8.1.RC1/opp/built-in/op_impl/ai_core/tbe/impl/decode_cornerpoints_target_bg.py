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
decode_cornerpoints_target_bg
"""
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import para_check


# 'pylint: disable=too-few-public-methods,not-use-list-comprehension
class Constant:
    """
    the class for constant
    """
    # the max num of single copy
    SINGLE_N_MAX = 1024
    # the dim of inputs must be 4
    SHAPE_DIM = 2
    # shape_x number of m dim
    M_SHAPE_X = 4
    # shape_y number of m dim
    M_SHAPE_Y = 4
    # N must be in [1,65500]
    N_MIN = 1
    N_MAX = 65500
    CONFIG_ONE = 1
    CONFIG_TWO = 2
    CONFIG_FOUR = 4
    CONFIG_EIGHT = 8
    CONFIG_SIXTEEN = 16
    CONFIG_DATA_SIZE = 32
    CONFIG_SIXTY_FOUR = 64
    CONFIG_BLOCK_SIZE = 128
    # the matrix of hardware
    CONFIG_MATRIX = 256


# 'pylint: disable=too-many-arguments
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
    return (divisor_a + divisor_b - Constant.CONFIG_ONE) // divisor_b


def tiling_func(frame):
    """
    get tilling parameter

    Parameter: frame
    ---------------------------------------------------------
    :return: total_handling_times, last_handling_n
    ---------------------------------------------------------
    """
    total_handling_times = int_ceil_div(frame, Constant.SINGLE_N_MAX)

    last_handling_n = frame % Constant.SINGLE_N_MAX
    if last_handling_n == 0:
        last_handling_n = Constant.SINGLE_N_MAX

    return total_handling_times, last_handling_n


def check_decode_cornerpoints_target_bg_params(keypoints_prediction,
                                               anchors,
                                               keypoints_decoded):
    """
    The params check function of decode_wheels_target

    Parameters:
    ----------
    Returns : All transformed params.
    ----------
    """
    if (keypoints_prediction is None) or (anchors is None):
        raise RuntimeError("The keypoints_prediction/anchors is null!")

    if not isinstance(keypoints_prediction, dict):
        raise RuntimeError("the input parameter keypoints_prediction must be dict,"
                           " while type of input is %s" % type(keypoints_prediction))

    if not isinstance(anchors, dict):
        raise RuntimeError("the input parameter anchors must be dict,"
                           " while type of input is %s" % type(anchors))

    shape_x = keypoints_prediction.get("shape")
    shape_x_num = len(shape_x)
    shape_y = anchors.get("shape")
    shape_y_num = len(shape_y)
    shape_z = keypoints_decoded.get("shape")
    shape_z_num = len(shape_z)
    dtype_x = keypoints_prediction.get("dtype").lower()
    dtype_y = anchors.get("dtype").lower()
    dtype_z = keypoints_decoded.get("dtype").lower()

    if shape_x is None or dtype_x is None:
        raise RuntimeError("The keypoints_prediction/keypoints_prediction must include 'shape' and 'dtype'!")
    if shape_y is None or dtype_y is None:
        raise RuntimeError("The anchors/anchors must include 'shape' and 'dtype'!")

    if shape_y != shape_x:
        raise RuntimeError("The shape of input should be the same !")
    # the shape and type of the output  should be the same as the input
    if shape_x != shape_z or shape_y != shape_z:
        raise RuntimeError("The shape of output should be the same as the input!")

    # Abnormality test
    if dtype_x != dtype_y or dtype_x != dtype_z or dtype_y != dtype_z:
        raise RuntimeError("dtype of inputs and output should be consistent")
    if dtype_x != 'float16':
        raise RuntimeError("dtype of inputs should be float16")
    if shape_x_num != shape_y_num or shape_x_num != shape_z_num or shape_y_num != shape_z_num:
        raise RuntimeError("dimension of inputs should be consistent")
    if shape_x_num != Constant.SHAPE_DIM:
        raise RuntimeError("dimension of inputs should be 2")
    check_decode_cornerpoints_target_bg_params_1(shape_x, shape_y, shape_z)


def check_decode_cornerpoints_target_bg_params_1(shape_x, shape_y, shape_z):
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
    if m_x != Constant.M_SHAPE_X:
        raise RuntimeError("m dimension of keypoints_prediction should be 4")
    if m_y != Constant.M_SHAPE_Y:
        raise RuntimeError("m dimension of anchors should be 4")
    if m_z != Constant.M_SHAPE_X:
        raise RuntimeError("m dimension of keypoints_encoded should be 4")
    if n_x < Constant.N_MIN or n_x > Constant.N_MAX:
        raise RuntimeError("n dimension of inputs should in [1, 65500]")


class InitShape:
    """
    init the input output shape
    Returns
    -------
    None
    """
    def __init__(self, keypoints_prediction, anchors, keypoints_decoded):
        self.shape_x = keypoints_prediction.get("shape")
        self.shape_y = anchors.get("shape")
        self.shape_z = keypoints_decoded.get("shape")
        self.dtype_x = keypoints_prediction.get("dtype").lower()
        self.dtype_y = anchors.get("dtype").lower()
        self.dtype_z = keypoints_decoded.get("dtype").lower()

    def set_data_y_ub_trs1(self, shape_x):
        """
        data_y_ub_trs1
        :return:
        """
        self.shape_x = shape_x

    def set_data_anchor_wh(self, shape_y):
        """
        data_anchor_wh
        :return:
        """
        self.shape_y = shape_y


class InitNumber:
    """
    init the input output number
    Returns
    -------
    None
    """
    def __init__(self, n_x):
        self.n_number = n_x
        # n_maxtrix  16*16 blocks--->number
        self.n_maxtrix = int_ceil_div(self.n_number * Constant.CONFIG_FOUR, Constant.CONFIG_MATRIX)
        self.shape_maxtrix = (self.n_maxtrix, Constant.CONFIG_SIXTEEN, Constant.CONFIG_SIXTEEN)
        self.burst_xy = int_ceil_div(self.n_number * Constant.CONFIG_FOUR, Constant.CONFIG_SIXTEEN)

    def set_data_y_ub_trs1(self, shape_maxtrix):
        """
        data_y_ub_trs1
        :return:
        """
        self.shape_maxtrix = shape_maxtrix

    def set_data_anchor_wh(self, burst_xy):
        """
        data_anchor_wh
        :return:
        """
        self.burst_xy = burst_xy


class InitFirstTensor:
    """
    init the input output tensor
    Returns
    -------
    None
    """
    def __init__(self, tik_instance, init_shape):
        self.data_x = tik_instance.Tensor(init_shape.dtype_x, init_shape.shape_x,
                                          name="data_x", scope=tik.scope_gm)
        self.data_y = tik_instance.Tensor(init_shape.dtype_y, init_shape.shape_y,
                                          name="data_y", scope=tik.scope_gm)
        self.data_z = tik_instance.Tensor(init_shape.dtype_z, init_shape.shape_z,
                                          name="data_z", scope=tik.scope_gm)

    def set_data_y_ub_trs1(self, data_x):
        """
        data_y_ub_trs1
        :return:
        """
        self.data_x = data_x

    def set_data_anchor_wh(self, data_y):
        """
        data_anchor_wh
        :return:
        """
        self.data_y = data_y


class InitsecondTensor:
    """
    init the input output tensor
    Returns
    -------
    None
    """
    def __init__(self, tik_instance, init_shape, init_number):
        self.data_x_ub = tik_instance.Tensor(init_shape.dtype_x, init_number.shape_maxtrix,
                                             name="data_x_ub", scope=tik.scope_ubuf)
        self.data_y_ub = tik_instance.Tensor(init_shape.dtype_y, init_number.shape_maxtrix,
                                             name="data_y_ub", scope=tik.scope_ubuf)

        self.data_x_ub_trans = tik_instance.Tensor(init_shape.dtype_x, init_number.shape_maxtrix,
                                                   name="data_x_ub_trans", scope=tik.scope_ubuf)
        self.data_y_ub_trans = tik_instance.Tensor(init_shape.dtype_y, init_number.shape_maxtrix,
                                                   name="data_y_ub_trans", scope=tik.scope_ubuf)

        self.data_wh = tik_instance.Tensor(init_shape.dtype_y, init_number.shape_maxtrix,
                                           name="data_wh", scope=tik.scope_ubuf)

        self.data_cxcy_temp = tik_instance.Tensor(init_shape.dtype_y, init_number.shape_maxtrix,
                                                  name="data_cxcy_temp", scope=tik.scope_ubuf)

    def set_data_y_ub_trs1(self, data_wh):
        """
        data_y_ub_trs1
        :return:
        """
        self.data_wh = data_wh

    def set_data_anchor_wh(self, data_cxcy_temp):
        """
        data_anchor_wh
        :return:
        """
        self.data_cxcy_temp = data_cxcy_temp


class InitThirdTensor:
    """
    init the input output tensor
    Returns
    -------
    None
    """
    def __init__(self, tik_instance, init_shape, init_number):
        self.data_cxcy = tik_instance.Tensor(init_shape.dtype_y, init_number.shape_maxtrix,
                                             name="data_cxcy", scope=tik.scope_ubuf)
        self.dump_half = tik_instance.Scalar(dtype="float16", init_value=0.5)

        self.data_input_mul = tik_instance.Tensor(init_shape.dtype_y, init_number.shape_maxtrix,
                                                  name="data_input_mul", scope=tik.scope_ubuf)
        self.data_z_ub = tik_instance.Tensor(init_shape.dtype_y, init_number.shape_maxtrix,
                                             name="data_z_ub", scope=tik.scope_ubuf)
        self.data_z_ub_tran = tik_instance.Tensor(init_shape.dtype_y, init_number.shape_maxtrix,
                                                  name="data_z_ub_tran", scope=tik.scope_ubuf)

    def set_data_y_ub_trs1(self, data_z_ub):
        """
        data_y_ub_trs1
        :return:
        """
        self.data_z_ub = data_z_ub

    def set_data_anchor_wh(self, data_z_ub_tran):
        """
        data_anchor_wh
        :return:
        """
        self.data_z_ub_tran = data_z_ub_tran


def calculate_process(tik_instance,
                      init_number,
                      init_first_tensor,
                      init_second_tensor,
                      init_third_tensor,
                      current_handling_times):
    """
    :param tik_instance: tik
    :param init_number: class
    :param init_first_tensor:class
    :param init_second_tensor: class
    :param init_third_tensor: class
    :return:
    """
    tik_instance.data_move(
        init_second_tensor.data_x_ub,
        init_first_tensor.data_x[current_handling_times * Constant.SINGLE_N_MAX * Constant.CONFIG_FOUR],
        0, Constant.CONFIG_ONE, init_number.burst_xy, 0, 0)
    tik_instance.data_move(
        init_second_tensor.data_y_ub,
        init_first_tensor.data_y[current_handling_times * Constant.SINGLE_N_MAX * Constant.CONFIG_FOUR],
        0, Constant.CONFIG_ONE, init_number.burst_xy, 0, 0)
    with tik_instance.for_range(0, init_number.n_maxtrix) as i:
        tik_instance.vtranspose(init_second_tensor.data_x_ub_trans[(i * Constant.CONFIG_MATRIX)],
                                init_second_tensor.data_x_ub[(i * Constant.CONFIG_MATRIX)])
        tik_instance.vtranspose(init_second_tensor.data_y_ub_trans[(i * Constant.CONFIG_MATRIX)],
                                init_second_tensor.data_y_ub[(i * Constant.CONFIG_MATRIX)])

    if init_number.n_maxtrix == Constant.CONFIG_ONE:
        with tik_instance.for_range(0, Constant.CONFIG_TWO) as i:
            with tik_instance.for_range(0, Constant.CONFIG_TWO) as j:
                tik_instance.vsub(
                    Constant.CONFIG_SIXTY_FOUR, init_second_tensor.data_wh[Constant.CONFIG_DATA_SIZE * i
                                                                           + Constant.CONFIG_SIXTEEN * j],
                    init_second_tensor.data_y_ub_trans[Constant.CONFIG_DATA_SIZE + Constant.CONFIG_SIXTEEN * j],
                    init_second_tensor.data_y_ub_trans[Constant.CONFIG_SIXTEEN * j],
                    init_number.n_maxtrix, Constant.CONFIG_FOUR, Constant.CONFIG_FOUR, Constant.CONFIG_FOUR,
                    Constant.CONFIG_SIXTEEN, Constant.CONFIG_SIXTEEN, Constant.CONFIG_SIXTEEN)
                tik_instance.vadd(
                    Constant.CONFIG_SIXTY_FOUR,
                    init_second_tensor.data_cxcy_temp[Constant.CONFIG_DATA_SIZE * i + Constant.CONFIG_SIXTEEN * j],
                    init_second_tensor.data_y_ub_trans[Constant.CONFIG_DATA_SIZE + Constant.CONFIG_SIXTEEN * j],
                    init_second_tensor.data_y_ub_trans[Constant.CONFIG_SIXTEEN * j],
                    init_number.n_maxtrix, Constant.CONFIG_FOUR, Constant.CONFIG_FOUR, Constant.CONFIG_FOUR,
                    Constant.CONFIG_SIXTEEN, Constant.CONFIG_SIXTEEN, Constant.CONFIG_SIXTEEN)
    elif init_number.n_maxtrix % Constant.CONFIG_TWO == 0:
        with tik_instance.for_range(0, Constant.CONFIG_TWO) as i:
            with tik_instance.for_range(0, Constant.CONFIG_TWO) as j:
                tik_instance.vsub(
                    Constant.CONFIG_BLOCK_SIZE, init_second_tensor.data_wh[Constant.CONFIG_DATA_SIZE * i
                                                                           + Constant.CONFIG_SIXTEEN * j],
                    init_second_tensor.data_y_ub_trans[Constant.CONFIG_DATA_SIZE + Constant.CONFIG_SIXTEEN * j],
                    init_second_tensor.data_y_ub_trans[Constant.CONFIG_SIXTEEN * j],
                    init_number.n_maxtrix // Constant.CONFIG_TWO,
                    Constant.CONFIG_FOUR, Constant.CONFIG_FOUR, Constant.CONFIG_FOUR,
                    Constant.CONFIG_DATA_SIZE, Constant.CONFIG_DATA_SIZE, Constant.CONFIG_DATA_SIZE)
                tik_instance.vadd(
                    Constant.CONFIG_BLOCK_SIZE,
                    init_second_tensor.data_cxcy_temp[Constant.CONFIG_DATA_SIZE * i + Constant.CONFIG_SIXTEEN * j],
                    init_second_tensor.data_y_ub_trans[Constant.CONFIG_DATA_SIZE + Constant.CONFIG_SIXTEEN * j],
                    init_second_tensor.data_y_ub_trans[Constant.CONFIG_SIXTEEN * j],
                    init_number.n_maxtrix // Constant.CONFIG_TWO,
                    Constant.CONFIG_FOUR, Constant.CONFIG_FOUR, Constant.CONFIG_FOUR,
                    Constant.CONFIG_DATA_SIZE, Constant.CONFIG_DATA_SIZE, Constant.CONFIG_DATA_SIZE)
    else:
        with tik_instance.for_range(0, Constant.CONFIG_TWO) as i:
            with tik_instance.for_range(0, Constant.CONFIG_TWO) as j:
                tik_instance.vsub(
                    Constant.CONFIG_BLOCK_SIZE, init_second_tensor.data_wh[Constant.CONFIG_DATA_SIZE * i
                                                                           + Constant.CONFIG_SIXTEEN * j],
                    init_second_tensor.data_y_ub_trans[Constant.CONFIG_DATA_SIZE + Constant.CONFIG_SIXTEEN * j],
                    init_second_tensor.data_y_ub_trans[Constant.CONFIG_SIXTEEN * j],
                    init_number.n_maxtrix // Constant.CONFIG_TWO,
                    Constant.CONFIG_FOUR, Constant.CONFIG_FOUR, Constant.CONFIG_FOUR,
                    Constant.CONFIG_DATA_SIZE, Constant.CONFIG_DATA_SIZE, Constant.CONFIG_DATA_SIZE)
                tik_instance.vsub(
                    Constant.CONFIG_SIXTY_FOUR,
                    init_second_tensor.data_wh[
                        (init_number.n_maxtrix - Constant.CONFIG_ONE)
                        * Constant.CONFIG_MATRIX + Constant.CONFIG_DATA_SIZE * i + Constant.CONFIG_SIXTEEN * j],
                    init_second_tensor.data_y_ub_trans[
                        (init_number.n_maxtrix - Constant.CONFIG_ONE)
                        * Constant.CONFIG_MATRIX + Constant.CONFIG_DATA_SIZE + Constant.CONFIG_SIXTEEN * j],
                     init_second_tensor.data_y_ub_trans[
                         (init_number.n_maxtrix - Constant.CONFIG_ONE)
                         * Constant.CONFIG_MATRIX + Constant.CONFIG_SIXTEEN * j],
                    Constant.CONFIG_ONE, Constant.CONFIG_FOUR, Constant.CONFIG_FOUR, Constant.CONFIG_FOUR,
                    Constant.CONFIG_SIXTEEN, Constant.CONFIG_SIXTEEN, Constant.CONFIG_SIXTEEN)
                tik_instance.vadd(
                    Constant.CONFIG_BLOCK_SIZE,
                    init_second_tensor.data_cxcy_temp[Constant.CONFIG_DATA_SIZE * i + Constant.CONFIG_SIXTEEN * j],
                    init_second_tensor.data_y_ub_trans[Constant.CONFIG_DATA_SIZE + Constant.CONFIG_SIXTEEN*j],
                    init_second_tensor.data_y_ub_trans[Constant.CONFIG_SIXTEEN * j],
                    init_number.n_maxtrix // Constant.CONFIG_TWO,
                    Constant.CONFIG_FOUR, Constant.CONFIG_FOUR, Constant.CONFIG_FOUR,
                    Constant.CONFIG_DATA_SIZE, Constant.CONFIG_DATA_SIZE, Constant.CONFIG_DATA_SIZE)
                tik_instance.vadd(
                    Constant.CONFIG_SIXTY_FOUR,
                    init_second_tensor.data_cxcy_temp[
                        (init_number.n_maxtrix - Constant.CONFIG_ONE)
                        * Constant.CONFIG_MATRIX + Constant.CONFIG_DATA_SIZE * i + Constant.CONFIG_SIXTEEN * j],
                    init_second_tensor.data_y_ub_trans[
                        (init_number.n_maxtrix - Constant.CONFIG_ONE)
                        * Constant.CONFIG_MATRIX + Constant.CONFIG_DATA_SIZE + Constant.CONFIG_SIXTEEN * j],
                    init_second_tensor.data_y_ub_trans[
                        (init_number.n_maxtrix - Constant.CONFIG_ONE)
                        * Constant.CONFIG_MATRIX + Constant.CONFIG_SIXTEEN * j],
                    Constant.CONFIG_ONE, Constant.CONFIG_FOUR, Constant.CONFIG_FOUR, Constant.CONFIG_FOUR,
                    Constant.CONFIG_SIXTEEN, Constant.CONFIG_SIXTEEN, Constant.CONFIG_SIXTEEN)

    tik_instance.vmuls(Constant.CONFIG_BLOCK_SIZE, init_third_tensor.data_cxcy,
                       init_second_tensor.data_cxcy_temp,
                       init_third_tensor.dump_half,
                       init_number.n_maxtrix * Constant.CONFIG_TWO,
                       Constant.CONFIG_ONE, Constant.CONFIG_ONE,
                       Constant.CONFIG_EIGHT, Constant.CONFIG_EIGHT)

    tik_instance.vmul(Constant.CONFIG_BLOCK_SIZE, init_third_tensor.data_input_mul,
                      init_second_tensor.data_x_ub_trans,
                      init_second_tensor.data_wh,
                      init_number.n_maxtrix * Constant.CONFIG_TWO,
                      Constant.CONFIG_ONE, Constant.CONFIG_ONE, Constant.CONFIG_ONE,
                      Constant.CONFIG_EIGHT, Constant.CONFIG_EIGHT, Constant.CONFIG_EIGHT)

    tik_instance.vadd(Constant.CONFIG_BLOCK_SIZE, init_third_tensor.data_z_ub,
                      init_third_tensor.data_input_mul,
                      init_third_tensor.data_cxcy,
                      init_number.n_maxtrix * Constant.CONFIG_TWO,
                      Constant.CONFIG_ONE, Constant.CONFIG_ONE, Constant.CONFIG_ONE,
                      Constant.CONFIG_EIGHT, Constant.CONFIG_EIGHT, Constant.CONFIG_EIGHT)

    with tik_instance.for_range(0, init_number.n_maxtrix) as i:
        tik_instance.vtranspose(
            init_third_tensor.data_z_ub_tran[(i * Constant.CONFIG_MATRIX)],
            init_third_tensor.data_z_ub[(i * Constant.CONFIG_MATRIX)])

    burst_z = int_ceil_div(init_number.n_number * Constant.CONFIG_FOUR, Constant.CONFIG_SIXTEEN)
    tik_instance.data_move(
        init_first_tensor.data_z[current_handling_times * Constant.SINGLE_N_MAX * Constant.CONFIG_FOUR],
        init_third_tensor.data_z_ub_tran, 0,
        Constant.CONFIG_ONE, burst_z, 0, 0)


@para_check.check_input_type(dict, dict, dict, str)
def decode_cornerpoints_target_bg(keypoints_prediction,
                                  anchors,
                                  keypoints_decoded,
                                  kernel_name="decode_cornerpoints_target_bg"):
    """
    The params check function of decode_cornerpoints_target_bg

    Parameters:
    ----------
    Returns : All transformed params.
    ----------
    """
    tik_instance = tik.Tik(tik.Dprofile(), True)

    para_check.check_kernel_name(kernel_name)

    check_decode_cornerpoints_target_bg_params(keypoints_prediction,
                                               anchors,
                                               keypoints_decoded)
    init_shape = InitShape(keypoints_prediction, anchors, keypoints_decoded)

    total_handling_times, last_handling_n = tiling_func(init_shape.shape_x[0])

    init_first_tensor = InitFirstTensor(tik_instance, init_shape)

    with tik_instance.for_range(0, total_handling_times - Constant.CONFIG_ONE) as current_handling_times:
        n_x = Constant.SINGLE_N_MAX

        init_number = InitNumber(n_x)

        with tik_instance.new_stmt_scope():
            init_second_tensor = InitsecondTensor(tik_instance, init_shape, init_number)

            init_third_tensor = InitThirdTensor(tik_instance, init_shape, init_number)

            calculate_process(tik_instance,
                              init_number,
                              init_first_tensor,
                              init_second_tensor,
                              init_third_tensor,
                              current_handling_times)

    n_x = last_handling_n
    init_number = InitNumber(n_x)

    with tik_instance.new_stmt_scope():
        init_second_tensor = InitsecondTensor(tik_instance, init_shape, init_number)

        init_third_tensor = InitThirdTensor(tik_instance, init_shape, init_number)

        calculate_process(tik_instance,
                          init_number,
                          init_first_tensor,
                          init_second_tensor,
                          init_third_tensor,
                          total_handling_times - Constant.CONFIG_ONE)

    tik_instance.BuildCCE(kernel_name=kernel_name,
                          inputs=[init_first_tensor.data_x, init_first_tensor.data_y],
                          outputs=[init_first_tensor.data_z])
