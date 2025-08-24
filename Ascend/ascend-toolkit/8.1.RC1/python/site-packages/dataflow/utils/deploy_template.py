#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Copyright 2024-2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

import json
import dataflow.dataflow as df


def generate_deploy_template(graph, file_path):
    # extract input FlowData
    added_nodes = set()
    used_nodes = []
    input_datas = set()
    batch_deploy_info = []

    for _, output in enumerate(graph._outputs):
        used_nodes.append(output.node)

    while len(used_nodes) > 0:
        node = used_nodes.pop()
        if node.name in added_nodes:
            # skip this node
            continue
        added_nodes.add(node.name)
        deploy_node_name = node.name
        if node.alias is not None:
            deploy_node_name = node.alias
        batch_deploy_info.append(
            {"flow_node_list": [deploy_node_name], "logic_device_list": "0:0:0:0"}
        )
        for anchor in node._input_anchors:
            if not isinstance(anchor, df.FlowData):
                used_nodes.append(anchor.node)

    data_flow_deploy_info = {"batch_deploy_info": batch_deploy_info}
    with open(file_path, "w") as f:
        # use indent 4
        json.dump(data_flow_deploy_info, f, indent=4)
    print("generate deploy template success, file:", file_path)
