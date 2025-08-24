/*
* Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @brief acl interface for amct calibration.
 *
 * @file acl_graph_calibration.h
 *
 * @version 1.0
 */

#ifndef ACL_GRAPH_CALIBRATION_H
#define ACL_GRAPH_CALIBRATION_H
#include <map>

#include "graph/ascend_string.h"
#include "graph/ge_error_codes.h"
#include "graph/graph.h"

namespace ge {
    /**
    * @brief acl interface to modify the given graph to amct quantized graph.
    * @param [in] graph: ge::Graph to be quantized
    * @param [in] quantizeConfigs: std::map<ge::AscendString, ge::AscendString> quantizetion options.
    * @return amctStatus
    */
    graphStatus aclgrphCalibration(ge::Graph &graph,
                                   const std::map<ge::AscendString, ge::AscendString> &quantizeConfigs);
}
#endif
