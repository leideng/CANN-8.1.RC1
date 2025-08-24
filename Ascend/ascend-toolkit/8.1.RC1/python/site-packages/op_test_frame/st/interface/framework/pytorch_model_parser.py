#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Function:
PytorchModelParse class. This class mainly get case info from onnx model.
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved © 2020
Change History: 2020-07-11 file Created
"""

import os

import onnx
import google
from onnx import helper
from onnx import shape_inference
from onnx import TensorProto

from op_test_frame.st.interface.const_manager import ConstManager
from op_test_frame.st.interface import utils


class PytorchConstVaraible:
    """
    class PytorchConstVaraible
    """
    TMP_SHAPE_FILE = 'tmp_shape.json'
    TMP_GA_PATH_FILE = 'tmp_ga_path.json'
    ONNX_DTYPE_MAP = {
        1: 'float',
        2: 'uint8',
        3: 'int8',
        4: 'uint16',
        5: 'int16',
        6: 'int32',
        7: 'int64',
        8: 'str',
        9: 'bool',
        10: 'float16',
        11: 'double',
        12: 'uint32',
        13: 'uint64'
    }
    ONNX_TENSOR_DTYPE = {
        0: TensorProto.FLOAT,
        1: TensorProto.FLOAT,
        2: TensorProto.UINT8,
        3: TensorProto.INT8,
        4: TensorProto.UINT16,
        5: TensorProto.INT16,
        6: TensorProto.INT32,
        7: TensorProto.INT64,
        8: TensorProto.STRING,
        9: TensorProto.BOOL,
        10: TensorProto.FLOAT16,
        11: TensorProto.DOUBLE,
        12: TensorProto.UINT32,
        13: TensorProto.UINT64
    }

    def get_tmp_shape_file(self):
        """
        get tmp shape file
        :return: tmp_shape_file
        """
        return self.TMP_SHAPE_FILE

    def get_onnx_tensor_type(self):
        """
        get onnx tensor type
        :return: onnx_tensor_type
        """
        return self.ONNX_TENSOR_DTYPE


def get_shape_and_notice(shape, layer_name):
    """
    get shape and notice
    """
    if isinstance(shape, list):
        for dim in shape:
            if not isinstance(dim, int):
                utils.print_warn_log(
                    'The input shape(%s) of layer(%s) is not a number. '
                    'Try to change the input shape to fix the '
                    'problem.' % (shape, layer_name))
                return
            if dim <= 0:
                utils.print_warn_log(
                    'The input shape(%s) of layer(%s) must be greater than'
                    ' 0. Try to change the input shape '
                    'to fix the problem.' % (shape, layer_name))
                return
    else:
        utils.print_warn_log(
            'The input shape(%s) format error.Please retype.' % shape)


def check_shape_and_notice(shape):
    """
    check shape and notice
    """
    if isinstance(shape, str):
        shape = shape.split(',')
        for dim in shape:
            if not dim.isdigit():
                utils.print_warn_log(
                    'The dim(%s) in shape(%s) is not a number.' % (dim,
                                                                   shape))
                return False
            if int(dim) <= 0:
                utils.print_warn_log(
                    'The dim(%s) in shape(%s) must be greater than 0.' %
                    (dim, shape))
                return False
        return True
    utils.print_warn_log(
        'The input shape(%s) format error.Please retype.' % shape)
    return False


def _load_model(model_path):
    try:
        model = onnx.load(model_path)
    except google.protobuf.message.DecodeError as err:
        utils.print_error_log("{} is not a valid model file. "
                              "Please check the model.".format(model_path))
        raise utils.OpTestGenException(ConstManager.OP_TEST_GEN_INVALID_DATA_ERROR) from err
    finally:
        pass
    return model


def _infer_model_shape(origin_model, input_nums=None):
    graph = origin_model.graph
    tensors = origin_model.graph.initializer
    for i, tensor in enumerate(tensors):
        value_info = helper.make_tensor_value_info(tensor.name,
                                                   PytorchConstVaraible.ONNX_TENSOR_DTYPE.get(tensor.data_type),
                                                   tensor.dims)
        graph.input.insert(input_nums + i, value_info)
    try:
        infer_model = shape_inference.infer_shapes(origin_model)
    except RuntimeError:
        utils.print_warn_log("The model tensor shape cannot be inferred, "
                             "skip inference shape.")
        return origin_model
    finally:
        pass
    return infer_model


def _get_tensor_shape(tensor_names, all_tensors):
    tensor_dtypes = []
    tensor_shapes = []
    for tensor_name in tensor_names:
        if tensor_name not in all_tensors.keys():
            continue
        tensor = all_tensors.get(tensor_name)
        tensor_dtypes.append(tensor.get('dtype'))
        tensor_shapes.append(tensor.get('shape'))
    return tensor_dtypes, tensor_shapes


def _update_tensor_info(name, dtype, shape, all_tensors):
    tensor_info = {
        'shape': shape,
        'dtype': PytorchConstVaraible.ONNX_DTYPE_MAP.get(dtype)
    }
    all_tensors.update({name: tensor_info})


def _get_shape_list(tensor_type):
    shape_list = []
    for dim in tensor_type.shape.dim:
        if not str(dim):
            shape_list.append(-1)
        if hasattr(dim, 'dim_param') and dim.dim_param != '':
            shape_list.append(-1)
            continue
        if hasattr(dim, 'dim_value') and dim.dim_value != '':
            shape_list.append(dim.dim_value)
    return shape_list


def _parse_tensor_info(tensors, all_tensors):
    for tensor in tensors:
        if tensor.name in all_tensors.keys():
            continue
        tensor_type = tensor.type.tensor_type
        elem_type = tensor_type.elem_type
        if not hasattr(tensor_type, 'shape'):
            utils.print_warn_log("Cannot parse {} tensor shape".format(tensor.name))
            _update_tensor_info(tensor.name, elem_type, '', all_tensors)
            continue
        if not str(tensor_type.shape) or not hasattr(tensor_type.shape, 'dim'):
            tensor_shape = []
            _update_tensor_info(tensor.name, elem_type, tensor_shape, all_tensors)
            continue
        shape_list = _get_shape_list(tensor_type)
        _update_tensor_info(tensor.name, elem_type, shape_list, all_tensors)
    return all_tensors


def _get_all_tensors(graph):
    tensors = {}
    # parse all input tensor
    input_tensors = _parse_tensor_info(graph.input, tensors)
    # parse the input and output tensor of middle layer
    all_tensors = _parse_tensor_info(graph.value_info, input_tensors)
    # parser the tensor, which has param, such as weight, bias
    for param_tensor in graph.initializer:
        if param_tensor.name in all_tensors.keys():
            continue
        all_tensors.update({
            param_tensor.name: {
                'dtype': PytorchConstVaraible.ONNX_DTYPE_MAP.get(param_tensor.data_type),
                'shape': param_tensor.dims
            }
        })
    return all_tensors


def _get_input_shape_from_user(new_shape_map):
    result = False
    for (layer_name, value) in new_shape_map.items():
        node_shape = value["ori_shape"]
        get_shape_and_notice(node_shape, layer_name)
        utils.print_info_log('"%s" layer is a input '
                             'operator , the original input shape : %s'
                             % (layer_name, node_shape))
        while True:
            new_input_shape = input(
                'Would you like to change the input shape ? '
                'If yes, please enter a new shape like 8,224,224,3.'
                'if not, enter "n" skip:')
            if new_input_shape.lower() == 'n':
                utils.print_info_log("Skip change shape.")
                value["new_shape"] = node_shape
                break
            if check_shape_and_notice(new_input_shape):
                value["new_shape"] = new_input_shape.split(',')
                result = True
                break

            utils.print_warn_log(
                "The input shape above is invalid, please retype!")
    return result


def _get_op_attr(node):
    op_attr = []
    for attr in node.attribute:
        attr_info = {
            'name': attr.name,
            'type': PytorchConstVaraible.ONNX_DTYPE_MAP.get(attr.type)
        }
        try:
            attr_value = helper.get_attribute_value(attr)
        except ValueError:
            utils.print_warn_log("Unsupported ONNX attribute: {}, cannot "
                                 "parse the attribute value".format(attr))
            attr_value = []
        finally:
            attr_info['value'] = attr_value
        op_attr.append(attr_info)
    return op_attr


def _get_node_list(model_path, ini_op_type, input_nums=None):
    origin_model = _load_model(model_path)
    infer_model = _infer_model_shape(origin_model, input_nums)
    graph = infer_model.graph
    all_tensors = _get_all_tensors(graph)
    node_list = []
    for i, node in enumerate(graph.node):
        if node.op_type != ini_op_type:
            continue
        op_info = {"op_type": node.op_type,
                   "input_dtype": [],
                   "output_dtype": [],
                   "input_shape": [],
                   "output_shape": []
                   }
        if node.name:
            op_info['layer'] = node.name
        else:
            op_info['layer'] = 'layer{}'.format(i)
        input_dtype, input_shape = _get_tensor_shape(node.input, all_tensors)
        op_info['input_dtype'] = input_dtype
        op_info['input_shape'] = input_shape
        output_dtype, output_shape = _get_tensor_shape(node.output, all_tensors)
        op_info['output_dtype'] = output_dtype
        op_info['output_shape'] = output_shape
        # get attr
        op_info['attr'] = _get_op_attr(node)
        node_list.append(op_info)
    return node_list


class PyTorchModelParse:
    """
    the class for parse onnx model.
    """

    def __init__(self, args):
        if hasattr(args, 'input_file'):
            self.input_file = args.input_file
        if hasattr(args, 'output_path'):
            self.output_path = args.output_path
        self.op_info = {}
        self.op_name = ''
        if hasattr(args, 'model_path'):
            self.model_path = args.model_path
        if hasattr(args, 'quiet'):
            self.quiet_flag = args.quiet

    @staticmethod
    def get_infer_model(onnx_model):
        onnx.checker.check_model(onnx_model)
        infer_model = shape_inference.infer_shapes(onnx_model)
        return infer_model

    @staticmethod
    def _insert_new_shape(onnx_model, new_shape_map):
        graph = onnx_model.graph
        op_info = []
        for op_name, op_info in new_shape_map.items():
            for input_tensor in graph.input:
                if input_tensor.name != op_name:
                    continue
                if not op_info.get("new_shape"):
                    utils.print_warn_log("The input layer {} new shape is "
                                         "null, please check the new shape."
                                         .format(op_name))
                    continue
                try:
                    new_input_tensor = onnx.helper.make_tensor_value_info(
                        name=input_tensor.name,
                        elem_type=input_tensor.type.tensor_type.elem_type,
                        shape=list(map(int, op_info.get("new_shape")))
                    )
                except ValueError as err:
                    utils.print_error_log("Input {} new shape dim must be int, "
                                          "please check it.".format(input_tensor.name))
                    raise utils.OpTestGenException(ConstManager.OP_TEST_GEN_INVALID_DATA_ERROR) from err
                finally:
                    pass
                graph.input.remove(input_tensor)
                graph.input.insert(0, new_input_tensor)
        return op_info

    def get_input_shape(self):
        """
        get input shape
        """
        utils.check_path_valid(self.model_path)
        utils.check_path_valid(self.output_path, True)
        input_shape_map = self._get_model_inputs()
        json_path = os.path.join(self.output_path, PytorchConstVaraible.TMP_SHAPE_FILE)
        utils.write_json_file(json_path, input_shape_map)

    def change_shape(self):
        """
        change shape
        """
        self._check_change_shape_argument_valid()
        new_shape_map = utils.load_json_file(self.input_file)
        new_model_path = self._change_shape_fn(new_shape_map)
        json_path = os.path.realpath(os.path.join(self.output_path, PytorchConstVaraible.TMP_GA_PATH_FILE))
        utils.write_json_file(json_path, {'new_model_path': new_model_path})

    def get_model_nodes(self, ini_op_type):
        """
        get model nodes
        """
        utils.check_path_valid(self.model_path)
        utils.check_name_valid(ini_op_type)
        input_shape_map = self._get_model_inputs()
        if not self.quiet_flag:
            if _get_input_shape_from_user(input_shape_map):
                model_path = self._change_shape_fn(input_shape_map)
            else:
                model_path = self.model_path
            utils.print_info_log("Start to get the \"%s\" operator in the "
                                 "model %s." % (ini_op_type, model_path))
            nodes_list = _get_node_list(model_path, ini_op_type, input_nums=len(input_shape_map))
        else:
            nodes_list = _get_node_list(self.model_path, ini_op_type, input_nums=len(input_shape_map))
        return nodes_list

    def _check_change_shape_argument_valid(self):
        if not self.input_file.endswith(".json"):
            utils.print_error_log(
                'The file "%s" is invalid, only supports .json file. '
                'Please modify it.' % self.input_file)
            raise utils.OpTestGenException(
                ConstManager.OP_TEST_GEN_INVALID_PATH_ERROR)
        utils.check_path_valid(self.input_file)
        utils.check_path_valid(self.model_path)
        utils.check_path_valid(self.output_path, True)

    def _get_model_inputs(self):
        graph = _load_model(self.model_path).graph
        all_tensors = _parse_tensor_info(graph.input, {})
        params = generator_to_list((init.name for init in graph.initializer))
        input_shape_map = {}
        for tensor_name, tensor_info in all_tensors.items():
            if tensor_name in params:
                continue
            input_shape_map.update({
                tensor_name: {'ori_shape': tensor_info.get('shape'),
                              'new_shape': []
                              }
            })
        return input_shape_map

    def _change_shape_fn(self, new_shape_map):
        real_path = os.path.realpath(self.model_path)
        onnx_model = _load_model(real_path)
        onnx.checker.check_model(onnx_model)
        op_info = self._insert_new_shape(onnx_model, new_shape_map)

        try:
            infer_model = self.get_infer_model(onnx_model)
        except RuntimeError as err:
            utils.print_error_log("{} model input shape cannot be changed into {}"
                                  .format(self.model_path,
                                          list(map(int, op_info.get("new_shape")))))
            raise utils.OpTestGenException(ConstManager.OP_TEST_GEN_INVALID_DATA_ERROR) from err
        finally:
            pass

        utils.print_info_log("The {} input shape has been changed.".format(self.model_path))
        _, tmp_filename = os.path.split(real_path)
        prefix, suffix = os.path.splitext(tmp_filename)
        first_new_shape = '_'.join(str(i) for i in list(new_shape_map.values())[0].get('new_shape'))
        new_model_name = (prefix + '{}' + suffix).format(first_new_shape)
        new_model_path = os.path.realpath(os.path.join(self.output_path, new_model_name))
        onnx.save(infer_model, new_model_path)
        return new_model_path


def get_shape(args, op_type):
    """
    get shape
    """
    _ = op_type
    pt_parser = PyTorchModelParse(args)
    return pt_parser.get_input_shape()


def change_shape(args, op_type):
    """
    change shape
    """
    _ = op_type
    pt_parser = PyTorchModelParse(args)
    return pt_parser.change_shape()


def get_model_nodes(args, op_type):
    """
    get model nodes
    """
    pt_parser = PyTorchModelParse(args)
    return pt_parser.get_model_nodes(op_type)


def generator_to_list(gen: any) -> list:
    """
    convert generator to list
    :param gen : generator
    :return:
    """
    result = []
    try:
        for data in gen:
            result.append(data)
    except(TypeError,) as err:
        utils.print_error_log("Failed to convert generator to list. %s" % err)
    finally:
        pass
    return result
