/*
 * Copyright (C)  2020. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file decode_bbox_v2_plugin.cpp
 * \brief
 */
#include "register/register.h"
#include "graph/ascend_string.h"


namespace domi {
    namespace {
        const char *const K_BOXES_UNPACK = "/unstack";
        const char *const K_BOXES_DIV = "RealDiv";
        const size_t K_REAL_DIV_INPUT_SIZE = 2;
        const size_t K_SCALE_SIZE = 4;
    }  // namespace

    Status ParseFloatFromConstNode(const ge::Operator *node, float &value)
    {
        if (node == nullptr) {
            return FAILED;
        }
        ge::Tensor tensor;
        auto ret = node->GetAttr("value", tensor);
        if (ret != ge::GRAPH_SUCCESS) {
            ge::AscendString opName;
            ge::graphStatus getNameStatus = node->GetName(opName);
            if (getNameStatus != ge::GRAPH_SUCCESS) {
                return FAILED;
            }
            return FAILED;
        }
        uint8_t *dataAddr = tensor.GetData();
        value = *(reinterpret_cast<float *>(dataAddr));
        return SUCCESS;
    }

    Status DecodeBboxV2GetStatus(std::map <std::string, std::string> &scales_const_name_map,
                                 std::map<string, const ge::Operator *> node_map,
                                 ge::Operator &op_dest)
    {
        std::vector<float> scales_list = {1.0, 1.0, 1.0, 1.0};
        if (scales_const_name_map.size() != K_SCALE_SIZE) {
            ge::AscendString opName;
            ge::graphStatus ret = op_dest.GetName(opName);
            if (ret != ge::GRAPH_SUCCESS) {
                return FAILED;
            }
        } else {
            size_t i = 0;
            for (const auto &name_pair : scales_const_name_map) {
                float scale_value = 1.0;
                auto ret = ParseFloatFromConstNode(node_map[name_pair.second], scale_value);
                if (ret != SUCCESS) {
                    return ret;
                }
                scales_list[i++] = scale_value;
            }
        }
        op_dest.SetAttr("scales", scales_list);
        return SUCCESS;
    }

    Status DecodeBboxV2ParseParams(const std::vector <ge::Operator> &inside_nodes, ge::Operator &op_dest)
    {
        std::map <std::string, std::string> scales_const_name_map;
        std::map<string, const ge::Operator *> node_map;
        for (const auto &node : inside_nodes) {
            ge::AscendString op_type;
            ge::graphStatus ret = node.GetOpType(op_type);
            if (ret != ge::GRAPH_SUCCESS) {
                return FAILED;
            }
            ge::AscendString op_name;
            ret = node.GetName(op_name);
            string str_op_name;
            if (op_name.GetString() != nullptr) {
                str_op_name = op_name.GetString();
            }
            if (op_type == K_BOXES_DIV) {
                if (node.GetInputsSize() < K_REAL_DIV_INPUT_SIZE) {
                    return FAILED;
                }
                ge::AscendString input_unpack_name0;
                ret = node.GetInputDesc(0).GetName(input_unpack_name0);
                string str_input_unpack_name0;
                if (input_unpack_name0.GetString() != nullptr) {
                    str_input_unpack_name0 = input_unpack_name0.GetString();
                }
                ge::AscendString input_unpack_name1;
                ret = node.GetInputDesc(1).GetName(input_unpack_name1);
                string str_input_unpack_name1;
                if (input_unpack_name1.GetString() != nullptr) {
                    str_input_unpack_name1 = input_unpack_name1.GetString();
                }
                if (str_input_unpack_name0.find(K_BOXES_UNPACK) != string::npos) {
                    scales_const_name_map.insert({str_op_name, str_input_unpack_name1});
                }
            }
            node_map[str_op_name] = &node;
        }

        return DecodeBboxV2GetStatus(scales_const_name_map, node_map, op_dest);
    }

    REGISTER_CUSTOM_OP("DecodeBboxV2")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("DecodeBboxV2")
    .FusionParseParamsFn(DecodeBboxV2ParseParams)
    .ImplyType(ImplyType::TVM);
}  // namespace domi