#!/usr/bin/env python
# coding=utf-8
"""
Function:
SubCaseDesign class
This class mainly involves parse
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved © 2020
"""

import os

from op_test_frame.st.interface.global_config_parser import GlobalConfig as GC

from op_test_frame.st.interface import utils
from op_test_frame.st.interface import st_report
from op_test_frame.st.interface import op_st_case_info
from op_test_frame.st.interface import dynamic_handle
from op_test_frame.st.interface import case_design as CD
from op_test_frame.st.interface.const_manager import ConstManager


class SubCaseDesign:
    """
    the class for design test subcase.
    """
    WHITE_LISTS = GC.instance().white_lists

    def __init__(self, current_json_path, json_obj,
                 total_case_list, report):
        self.current_json_path = current_json_path
        self.json_obj = json_obj
        self.case_idx = 1
        self.total_case_list = total_case_list
        self.report = report

    @staticmethod
    def _check_bin_valid(bin_path, dir_path):
        real_bin_path = os.path.join(dir_path, bin_path)
        if os.path.splitext(real_bin_path)[-1] != ConstManager.BIN_FILE:
            utils.print_error_log(
                'The file "%s" is invalid, only supports .bin file. '
                'Please modify it.' % bin_path)
            raise utils.OpTestGenException(
                ConstManager.OP_TEST_GEN_INVALID_PATH_ERROR)
        utils.check_path_valid(real_bin_path)
        return real_bin_path

    @staticmethod
    def _parse_expect_output_param(case, pyfile, function, err_thr):
        if not pyfile or not function:
            return
        if not pyfile and function:
            case.update({"calc_expect_func": function})
        if pyfile and function:
            case.update({"calc_expect_func_file": pyfile})
            case.update({"calc_expect_func_file_func": function})
        case.update({ConstManager.ERROR_THRESHOLD: err_thr})
        return

    @staticmethod
    def _check_cur_params_undefined(cur_params):
        if cur_params.get('format') in ConstManager.OPTIONAL_TYPE_LIST:
            cur_params['type'] = ConstManager.TYPE_UNDEFINED
        if cur_params.get('type') == ConstManager.TYPE_UNDEFINED:
            cur_params['format'] = ConstManager.TYPE_UNDEFINED

    @staticmethod
    def _check_ori_format_list_str_valid(dic_desc):
        ori_format_list = []
        if isinstance(dic_desc.get('ori_format'), str):
            ori_format_list = [dic_desc.get('ori_format')]
        if isinstance(dic_desc.get('ori_format'), list):
            ori_format_list = dic_desc.get('ori_format')
        return ori_format_list

    @staticmethod
    def _check_ori_filed_length_valid(ori_list, comapre_list, ori_filed,
                                      compare_filed):
        if len(ori_list) != len(comapre_list):
            utils.print_error_log('please checkout, teh length of %s and %s '
                                  'must be the same.' % (compare_filed,
                                                         ori_filed))
            raise utils.OpTestGenException(
                ConstManager.OP_TEST_GEN_INVALID_DATA_ERROR)

    @staticmethod
    def _check_fuzz_in_json(json_obj, key):
        if json_obj.get(key) == "fuzz":
            utils.print_error_log('The value ("fuzz") of %s is invalid. '
                                  'Configure "fuzz_impl" filed correctly.'
                                  % key)
            raise utils.OpTestGenException(
                ConstManager.OP_TEST_GEN_INVALID_DATA_ERROR)

    @staticmethod
    def _check_set_error_threshold(json_obj):
        err_thr = json_obj.get(ConstManager.ERROR_THRESHOLD)
        if err_thr is None:
            return err_thr
        if isinstance(err_thr, list) and len(err_thr) == 2:
            err_thr = utils.check_list_float(err_thr, ConstManager.ERROR_THRESHOLD)
        else:
            utils.print_error_log(
                "Error_threshold is unsupported. Example [0.01, 0.01].")
            raise utils.OpTestGenException(ConstManager.OP_TEST_GEN_INVALID_ERROR_THRESHOLD_ERROR)
        return err_thr

    def get_current_json_path(self):
        """
        get current json path
        """
        return self.current_json_path

    def get_json_obj(self):
        """
        get json obj
        """
        return self.json_obj

    def _check_expect_output_param(self, json_obj):
        expect_str = json_obj.get("calc_expect_func_file")
        real_expect_file_path = None
        function = None
        if expect_str:
            length = len(expect_str.split(":"))
            if length == 2:
                real_expect_file_path, function = expect_str.split(":")
                function = function.replace(ConstManager.SPACE, ConstManager.EMPTY)
                json_dir_path = os.path.dirname(self.current_json_path)
                real_expect_file_path = os.path.join(json_dir_path, real_expect_file_path)
                utils.print_info_log("The expect data generate python file:%s."
                                     % real_expect_file_path)
                utils.check_path_valid(real_expect_file_path)
                if not function:
                    function = json_obj.get(ConstManager.OP)
            else:
                utils.print_warn_log("The value of calc_expect_func_file is "
                                     "'%s' without function specified! If no "
                                     "need to compare output data, ignore."
                                     % expect_str)
        else:
            expect_str = json_obj.get("calc_expect_func")
            if not expect_str:
                utils.print_warn_log("There is no expect output function in "
                                     "the case json, if no need to compare "
                                     "output data, ignore.")
        return real_expect_file_path, function

    def _check_key_exist(self, json_obj, key, tensor):
        if key not in json_obj:
            utils.print_error_log(
                'There is no key "%s" for "%s". Please modify it in file %s.'
                % (key, tensor, self.current_json_path))
            raise utils.OpTestGenException(
                ConstManager.OP_TEST_GEN_INVALID_DATA_ERROR)

    def _check_shape_valid(self, shape):
        for dim in shape:
            if not isinstance(dim, int):
                utils.print_error_log(
                    'The value(%s) of "shape" is not int. Please modify it in '
                    'file %s.' % (shape, self.current_json_path))
                raise utils.OpTestGenException(
                    ConstManager.OP_TEST_GEN_INVALID_DATA_ERROR)
            if dim < ConstManager.SHAPE_DYNAMIC_SCENARIOS_TWO:
                utils.print_error_log(
                    'The value(%s) of "shape" must be greater than %s. Please '
                    'modify it in file %s.'
                    % (shape, ConstManager.SHAPE_DYNAMIC_SCENARIOS_TWO,
                       self.current_json_path))
                raise utils.OpTestGenException(
                    ConstManager.OP_TEST_GEN_INVALID_DATA_ERROR)

    def _check_range_value_valid(self, range_value, for_shape_range=False):
        if len(range_value) != 2:
            utils.print_error_log('The value(%s) of "range_value" is not [min,'
                                  'max]. Please modify it in file %s.'
                                  % (range_value, self.current_json_path))
            raise utils.OpTestGenException(ConstManager.OP_TEST_GEN_INVALID_DATA_ERROR)
        for value in range_value:
            if not isinstance(value, float) and not isinstance(value, int):
                utils.print_error_log(
                    'The value(%s) of "range_value" is not int or float'
                    '. Please modify it in file %s.'
                    % (range_value, self.current_json_path))
                raise utils.OpTestGenException(
                    ConstManager.OP_TEST_GEN_INVALID_DATA_ERROR)
        # consider shape_range allow[n, -1] or [-1,n]in dynamic shape scenario.
        if for_shape_range and (range_value[0] == -1 or range_value[1] == -1):
            return
        if range_value[1] < range_value[0]:
            utils.print_error_log(
                'In %s the maximum value is less than the minimum value '
                'Please modify it in file %s.' % (
                    range_value, self.current_json_path))
            raise utils.OpTestGenException(ConstManager.OP_TEST_GEN_INVALID_DATA_ERROR)

    def _check_value_valid(self, one_input_desc, json_obj, tensor, for_fuzz):
        # if have value, case generate by no cross
        if not for_fuzz:
            for key in one_input_desc:
                if one_input_desc[key] is not None and len(one_input_desc[key]) > 1:
                    utils.print_error_log('The "value" field (configured in "%s") is specified, '
                                          'each of the rest fields can be configured with one '
                                          'profile only. Please modify it in file %s.' % (tensor,
                                                                                          self.current_json_path))
                    raise utils.OpTestGenException(
                        ConstManager.OP_TEST_GEN_INVALID_DATA_ERROR)
        value_list = []
        case_value = json_obj.get(ConstManager.VALUE)
        if isinstance(case_value, (str, list)):
            if isinstance(case_value, str):
                dir_path = os.path.dirname(self.current_json_path)
                real_bin_path = self._check_bin_valid(case_value, dir_path)
                value_list.append(real_bin_path)
            else:
                value_list.append(case_value)
        else:
            utils.print_error_log('The value (%s) is invalid, for "%s" only supports list or bin file.'
                                  'Please modify it in file %s.' % (case_value,
                                                                    tensor, self.current_json_path))
            raise utils.OpTestGenException(
                ConstManager.OP_TEST_GEN_INVALID_DATA_ERROR)
        return value_list

    def _deal_with_value(self, input_desc, one_input_desc, for_fuzz=False):
        if input_desc.get(ConstManager.VALUE) is not None:
            value_list = self._check_value_valid(one_input_desc, input_desc, ConstManager.INPUT_DESC, for_fuzz)
            if for_fuzz:
                one_input_desc.update({"value": value_list[0]})
            else:
                one_input_desc.update({"value": value_list})
        # check is_const flag and deal with const input.
        const_input = utils.ConstInput(input_desc.get(ConstManager.IS_CONST))
        const_input.deal_with_const(one_input_desc, for_fuzz)

    def _check_list_list_valid(self, json_obj, key, tensor):
        self._check_key_exist(json_obj, key, tensor)
        self._check_fuzz_in_json(json_obj, key)
        value_list = []
        if not isinstance(json_obj[key], list):
            utils.print_error_log(
                'The value (%s) is invalid. The key "%s" for "%s" only '
                'supports [] or [[]]. Please modify it in file %s.' % (
                    json_obj[key], key, tensor, self.current_json_path))
            raise utils.OpTestGenException(
                ConstManager.OP_TEST_GEN_INVALID_DATA_ERROR)
        check_type = None
        for item in json_obj[key]:
            if isinstance(item, (tuple, list)):
                current_type = list
            else:
                current_type = int
            if check_type is None:
                check_type = current_type
            else:
                if check_type != current_type:
                    utils.print_error_log(
                        'The value (%s) is invalid. The key "%s" for "%s" '
                        'only supports [] or [[]]. Please modify it '
                        'in file %s.' % (json_obj[key], key, tensor,
                                         self.current_json_path))
                    raise utils.OpTestGenException(
                        ConstManager.OP_TEST_GEN_INVALID_DATA_ERROR)
        if check_type == list:
            value_list = json_obj[key]
        else:
            value_list.append(json_obj[key])
        return value_list

    def _check_name_type_valid(self, attr, key):
        if not isinstance(attr[key], str):
            utils.print_error_log(
                'The value (%s) is invalid. The key "%s" for "attr" only '
                'supports string. Please modify it in file %s.'
                % (attr[key], key, self.current_json_path))
            raise utils.OpTestGenException(ConstManager.OP_TEST_GEN_INVALID_DATA_ERROR)
        if attr[key] == "":
            utils.print_error_log(
                'The value of "%s" for "attr" is empty. Please modify '
                'it in file %s.' % (key, self.current_json_path))
            raise utils.OpTestGenException(ConstManager.OP_TEST_GEN_INVALID_DATA_ERROR)
        if key == 'type' and attr[key] not in list(ConstManager.ATTR_TYPE_MAP.values()):
            utils.print_error_log(
                'The value(%s) of "type" does not support. Only supports %s. '
                'Please modify it in file %s.' % (attr[key], list(ConstManager.ATTR_TYPE_MAP.values()),
                                                  self.current_json_path))
            raise utils.OpTestGenException(ConstManager.OP_TEST_GEN_INVALID_DATA_ERROR)

    def _replace_fuzz_param(self, json_obj, key, param_type, fuzz_dict):
        if json_obj.get(key) == "fuzz":
            name = json_obj.get("name")
            if name is None:
                utils.print_error_log(
                    'The "%s" not have name field in %s, '
                    'Please add it.' % (json_obj, self.current_json_path))
                raise utils.OpTestGenException(
                    ConstManager.OP_TEST_GEN_INVALID_PARAM_ERROR)
            try:
                fuzz_result_param = fuzz_dict.get(param_type).get(name).get(key)
            except AttributeError as ex:
                utils.print_error_log('Analyze the return value of the fuzz '
                                      'script, %s in %s or %s in %s may '
                                      'invalid. %s' % (name, param_type, key,
                                                       name, str(ex)))
                raise utils.OpTestGenException(ConstManager.OP_TEST_GEN_INVALID_PARAM_ERROR) from ex
            finally:
                pass
            json_obj[key] = fuzz_result_param
        return json_obj

    def _check_attr_value_valid(self, attr, fuzz_dict=None):
        if fuzz_dict is not None:
            attr = self._replace_fuzz_param(attr, 'value', 'attr', fuzz_dict)
        utils.check_attr_value_valid(attr)
        return attr

    def _check_attr_valid(self, json_obj, fuzz_dict=None):
        attr_list = []
        if ConstManager.ATTR not in json_obj:
            return attr_list
        if not isinstance(json_obj[ConstManager.ATTR], (tuple, list)):
            utils.print_error_log(
                'The value (%s) of "attr" is not list. Please modify it in file'
                ' %s.' % (json_obj[ConstManager.ATTR], self.current_json_path))
            raise utils.OpTestGenException(
                ConstManager.OP_TEST_GEN_INVALID_DATA_ERROR)
        name_list = []
        for attr in json_obj[ConstManager.ATTR]:
            CD.check_required_key_valid(attr, ConstManager.ATTR_REQUIRED_KEYS,
                                        ConstManager.ATTR,
                                        self.current_json_path)
            self._check_name_type_valid(attr, 'name')
            if attr['name'] not in name_list:
                name_list.append(attr['name'])
            else:
                utils.print_error_log(
                    'The %s already exists. Please modify or remove the '
                    'redundant key in file %s.'
                    % (attr['name'], self.current_json_path))
                raise utils.OpTestGenException(
                    ConstManager.OP_TEST_GEN_INVALID_DATA_ERROR)
            self._check_name_type_valid(attr, 'type')
            attr = self._check_attr_value_valid(attr, fuzz_dict)
            attr_list.append(attr)
        return attr_list

    def _add_case_to_total_case(self, case, case_idx, py_file_and_function,
                                case_list):
        py_file = py_file_and_function[0]
        function = py_file_and_function[1]
        err_thr = py_file_and_function[2]
        self._parse_expect_output_param(case, py_file, function, err_thr)
        case_idx += 1
        case_list.append(case)
        # deal with report
        case_info = op_st_case_info.OpSTCase(case['case_name'],
                                             case)
        st_case_trace = op_st_case_info.OpSTCaseTrace(case_info)
        case_rpt = st_report.OpSTCaseReport(st_case_trace)
        self.report.add_case_report(case_rpt)
        return case_idx, case_list

    def _check_list_str_valid(self, json_obj, key, support_list, tensor):
        self._check_key_exist(json_obj, key, tensor)
        self._check_fuzz_in_json(json_obj, key)
        value_list = []
        if isinstance(json_obj[key], str):
            value_list.append(json_obj[key])
        elif isinstance(json_obj[key], (tuple, list)):
            value_list = json_obj[key]
        else:
            utils.print_error_log(
                'The value (%s) is invalid. The key "%s" for "%s" only '
                'supports string or [string]. Please modify it in file %s.'
                % (json_obj[key], key, tensor, self.current_json_path))
            raise utils.OpTestGenException(ConstManager.OP_TEST_GEN_INVALID_DATA_ERROR)
        if len(value_list) == 0:
            utils.print_error_log(
                'The value of "%s" for "%s" is empty. Only supports %s.'
                ' Please modify it in file %s.' %
                (key, tensor, support_list, self.current_json_path))
            raise utils.OpTestGenException(
                ConstManager.OP_TEST_GEN_INVALID_DATA_ERROR)
        for item in value_list:
            if not isinstance(item, str):
                utils.print_error_log(
                    'The value (%s) is invalid. The key "%s" for "%s" only'
                    ' supports string or [string]. Please modify it in file %s.'
                    % (item, key, tensor, self.current_json_path))
                raise utils.OpTestGenException(
                    ConstManager.OP_TEST_GEN_INVALID_DATA_ERROR)
            if item == '':
                utils.print_error_log(
                    'The value(%s) of "%s" for "%s" contains empty string. '
                    'Only supports %s. Please modify it in file %s.' %
                    (json_obj[key], key, tensor, support_list,
                     self.current_json_path))
                raise utils.OpTestGenException(
                    ConstManager.OP_TEST_GEN_INVALID_DATA_ERROR)
            if item not in support_list:
                utils.print_error_log(
                    'The value(%s) of "%s" for "%s" does not support. '
                    'Only supports %s. Please modify it in file %s.' %
                    (item, key, tensor, support_list, self.current_json_path))
                raise utils.OpTestGenException(
                    ConstManager.OP_TEST_GEN_INVALID_DATA_ERROR)
        return value_list

    def _get_ori_filed_data(self, input_or_out_desc, key_desc, format_list, shape_list):
        ori_filed_list = []
        if input_or_out_desc.get('ori_format') is not None:
            ori_format_list = self._check_ori_format_list_str_valid(input_or_out_desc)
            self._check_ori_filed_length_valid(ori_format_list, format_list, "ori_format", "format")
            ori_filed_list.append(ori_format_list)
        if input_or_out_desc.get('ori_shape') is not None:
            ori_shape_list = self._check_list_list_valid(input_or_out_desc, 'ori_shape', key_desc)
            self._check_ori_filed_length_valid(ori_shape_list, shape_list, "ori_shape", "shape")
            ori_filed_list.append(ori_shape_list)
        if len(ori_filed_list) == 1:
            utils.print_error_log('please checkout, ori_format and ori_shape is exist at the same time.')
            raise utils.OpTestGenException(
                ConstManager.OP_TEST_GEN_INVALID_DATA_ERROR)
        return ori_filed_list

    def _get_dynamic_shape_info(self, op_key, op_desc, one_op_dict):
        """
        get dynamic shape info: typical_shape/shape_range
        """
        if not dynamic_handle.check_not_dynamic_shape(op_desc.get('shape')):
            return
        typical_shape_list = self._check_list_list_valid(
            op_desc, ConstManager.TYPICAL_SHAPE, op_key)
        for item in typical_shape_list:
            dynamic_handle.check_typical_shape_valid(
                item, self.current_json_path)
        one_op_dict.update({
            ConstManager.TYPICAL_SHAPE: typical_shape_list})
        if op_desc.get(ConstManager.SHAPE_RANGE):
            shape_range_list_list = []
            shape_range_list = self._check_list_list_valid(
                op_desc, ConstManager.SHAPE_RANGE, op_key)
            shape_range_list_list.append(shape_range_list)
            for item in shape_range_list:
                self._check_range_value_valid(item, for_shape_range=True)
            one_op_dict.update({
                ConstManager.SHAPE_RANGE: shape_range_list_list})
