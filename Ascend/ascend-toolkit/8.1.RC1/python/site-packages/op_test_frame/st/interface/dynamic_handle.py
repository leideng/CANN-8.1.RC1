#!/usr/bin/env python
# coding=utf-8
"""
Function:
This method mainly handle the dynamic scenario.
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved © 2021
"""
from op_test_frame.st.interface import utils
from op_test_frame.st.interface.const_manager import ConstManager


def check_typical_shape_valid(typical_shape, json_path):
    """
    check typical_shape are integers and greater than 0
    """
    for dim in typical_shape:
        if not isinstance(dim, int):
            utils.print_error_log(
                'The value(%s) of "typical_shape" is not int. '
                'Please modify it in file %s.' % (typical_shape, json_path))
            raise utils.OpTestGenException(
                ConstManager.OP_TEST_GEN_INVALID_DATA_ERROR)
        if dim < 0:
            utils.print_error_log(
                'The value(%s) of "typical_shape" must be greater than 0. '
                'Please modify it in file %s.' % (typical_shape, json_path))
            raise utils.OpTestGenException(
                ConstManager.OP_TEST_GEN_INVALID_DATA_ERROR)


def _check_dynamic_shape(dim):
    dynamic_count = 0
    if dim in (ConstManager.SHAPE_DYNAMIC_SCENARIOS_ONE,
               ConstManager.SHAPE_DYNAMIC_SCENARIOS_TWO):
        dynamic_count = 1
    return dynamic_count


def check_not_dynamic_shape(shape_list):
    """
    check whether dynamic shape, otherwise return False
    """
    if not shape_list:
        return ''
    # check -1 or -2 in shape_list value as a basis, return True.
    dynamic_shape_count = 0
    for dim in shape_list:
        if isinstance(dim, list):
            for item in dim:
                dynamic_shape_count += _check_dynamic_shape(item)
        if isinstance(dim, int):
            dynamic_shape_count += _check_dynamic_shape(dim)
    return dynamic_shape_count


def set_typical_shape_in_cur_params(cur_params, tensor, current_json_path):
    """
    update cur_params dict
    """
    shape_list = cur_params.get('shape')
    for dim in shape_list:
        if dim in (ConstManager.SHAPE_DYNAMIC_SCENARIOS_ONE,
                   ConstManager.SHAPE_DYNAMIC_SCENARIOS_TWO):
            typical_shape_list = tensor.get(ConstManager.TYPICAL_SHAPE)
            if typical_shape_list is None:
                utils.print_error_log("Please add \"typical_shape\" filed in "
                                      "%s used for executing the operator in "
                                      "dynamic shape scenarios."
                                      % current_json_path)
                raise utils.OpTestGenException(
                    ConstManager.OP_TEST_GEN_NONE_TYPICAL_SHAPE_ERROR)
            if typical_shape_list is not None:
                cur_params.update({ConstManager.TYPICAL_SHAPE: typical_shape_list[0]})
            # dynamic shape scenarios two, need to remove shape_range.
            if dim == ConstManager.SHAPE_DYNAMIC_SCENARIOS_TWO \
                    and cur_params.get(ConstManager.SHAPE_RANGE):
                cur_params.pop(ConstManager.SHAPE_RANGE)


def replace_shape_to_typical_shape(op_desc_dict):
    """
    if exist typical_shape and shape dim is -1 or -2,
    replace typical_shape as shape,
    return typical_shape
    Otherwise return initials shape
    """
    if op_desc_dict.get(ConstManager.TYPICAL_SHAPE) is not None:
        typical_shape_list = op_desc_dict.get(ConstManager.TYPICAL_SHAPE)
        if len(typical_shape_list) == 0:
            utils.print_warn_log("Please input values of typical_shape used "
                                 "for executing the operator.")
        shape_list = typical_shape_list
    else:
        shape_list = op_desc_dict.get('shape')
    return shape_list
