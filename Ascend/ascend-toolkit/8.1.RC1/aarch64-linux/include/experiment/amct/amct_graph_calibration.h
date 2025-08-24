/*
* Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
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
 * @brief amct interface for calibration.
 *
 * @file amct_graph_calibration.h
 *
 * @version 1.0
 */

#ifndef AMCT_GRAPH_CALIBRATION_H
#define AMCT_GRAPH_CALIBRATION_H
#include <map>
#include <string>

#include "amct_error_code.h"
#include "graph/graph.h"

namespace amct {
    /**
    * @brief acl interface to modify the given graph to amct quantized graph.
    * @param [in] graph: ge::Graph to be quantized
    * @param [in] options: std::map<std::string, std::string> options.
    * @return amctStatus
    */
    amctStatus graphCalibration(ge::Graph &graph, const std::map<std::string, std::string> &options);
}
#endif
