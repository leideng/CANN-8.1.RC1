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
 * @brief ge::aclgrphCalibration interface options
 *
 * @file acl_calibration_configs.h
 *
 * @version 1.0
 */

#ifndef ACL_CALIBRATION_CONFIGS_H
#define ACL_CALIBRATION_CONFIGS_H

#include <set>

#include "ge/ge_api_types.h"

namespace amct {
    namespace aclCaliConfigs {
        // required configs
        const char *const INPUT_DATA_DIR = "inputDataDir";
        const char *const INPUT_SHAPE = ge::ir_option::INPUT_SHAPE;
        const char *const SOC_VERSION = ge::ir_option::SOC_VERSION;
        // optional configs
        const char *const INPUT_FORMAT = ge::ir_option::INPUT_FORMAT;
        const char *const INPUT_FP16_NODES = ge::ir_option::INPUT_FP16_NODES;
        const char *const LOG_LEVEL = "log";
        const char *const CONFIG_FILE = "configFile";
        const char *const OUT_NODES = ge::ir_option::OUT_NODES;
        const char *const INSERT_OP_FILE = ge::ir_option::INSERT_OP_FILE;
        const char *const DEVICE_ID = "deviceId";
        const char *const IP_ADDR = "ipAddr";
        const char *const PORT = "port";
        const char *const AICORE_NUM = ge::ir_option::AICORE_NUM;
        const char *const GROUP_ID = "groupId";
        const std::set<std::string> PTQ_CONFIGS = {INPUT_DATA_DIR,
                                                   INPUT_SHAPE,
                                                   INPUT_FORMAT,
                                                   INPUT_FP16_NODES,
                                                   LOG_LEVEL,
                                                   SOC_VERSION,
                                                   CONFIG_FILE,
                                                   OUT_NODES,
                                                   DEVICE_ID,
                                                   IP_ADDR,
                                                   PORT,
                                                   AICORE_NUM,
                                                   GROUP_ID,
                                                   INSERT_OP_FILE};
        const std::set<std::string> CALI_OPTIONS = {INPUT_DATA_DIR,
                                                    CONFIG_FILE};
        const std::set<std::string> DEVICE_OPTIONS = {DEVICE_ID,
                                                      IP_ADDR,
                                                      PORT,
                                                      AICORE_NUM,
                                                      GROUP_ID};
        const std::set<std::string> SUPPORTED_SOC_VERSION = {"Ascend310",
                                                             "Ascend310P3",
                                                             "Ascend310P1",
                                                             "Ascend310P5",
                                                             "Ascend310P7",
                                                             "Ascend910A",
                                                             "Ascend910B",
                                                             "Ascend610",
                                                             "SD3403",
                                                             "Ascend910B1",
                                                             "Ascend910B2",
                                                             "Ascend910B3",
                                                             "Ascend910B4",
                                                             "BS9SX1AA",
                                                             "BS9SX1AB",
                                                             "BS9SX1AC",
                                                             "Ascend310B1",
                                                             "Ascend310B2",
                                                             "Ascend310B3",
                                                             "Ascend310B4",
                                                             "Ascend910B2C",
                                                             "AS31XM1X",
                                                             "Ascend910_9391",
                                                             "Ascend910_9392",
                                                             "Ascend910_9381",
                                                             "Ascend910_9382",
                                                             "Ascend910_9372",
                                                             "Ascend910_9362",
                                                             "Ascend910B4-1",
                                                             "Ascend610Lite",
                                                             "BS9SX2AA",
                                                             "BS9SX2AB",
                                                             "MC61AM21AA",
                                                             "MC61AM21AB"};
    }
}
#endif // ACL_CALIBRATION_CONFIGS_H
