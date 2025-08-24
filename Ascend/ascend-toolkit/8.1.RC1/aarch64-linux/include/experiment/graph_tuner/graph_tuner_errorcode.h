/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: 对外接口错误码定义
 */
#ifndef GRAPH_TUNER_GRAPH_TUNER_ERRORCODE_H
#define GRAPH_TUNER_GRAPH_TUNER_ERRORCODE_H

#include <cstdint>

namespace tune {
using Status = uint32_t;

/** Assigned SYS ID */
constexpr uint8_t SYSID_TUNE = 10U;

/* * lx module ID */
constexpr uint8_t TUNE_MODID_LX = 30U; // lx fusion  pass id
} // namespace tune

/* *
 * Build error code
 */
#define TUNE_DEF_ERRORNO(sysid, modid, name, value, desc)                                                     \
    constexpr tune::Status name = ((((static_cast<uint32_t>(0xFFU & (static_cast<uint8_t>(sysid)))) << 24U) | \
        ((static_cast<uint32_t>(0xFFU & (static_cast<uint8_t>(modid)))) << 16U)) |                            \
        (0xFFFFU & (static_cast<uint16_t>(value))));

#define TUNE_DEF_ERRORNO_LX(name, value, desc) TUNE_DEF_ERRORNO(SYSID_TUNE, TUNE_MODID_LX, name, value, desc)

namespace tune {
// generic
TUNE_DEF_ERRORNO(0U,    0U,    SUCCESS, 0U,      "Success");
TUNE_DEF_ERRORNO(0xFFU, 0xFFU, FAILED,  0xFFFFU, "Failed");

TUNE_DEF_ERRORNO_LX(NO_FUSION_STRATEGY,  10U, "Lx fusion strategy is invalid.");
TUNE_DEF_ERRORNO_LX(UNTUNED,             20U, "Untuned");
TUNE_DEF_ERRORNO_LX(NO_UPDATE_BANK,      30U, "Model Bank is not updated!");
TUNE_DEF_ERRORNO_LX(HIT_FUSION_STRATEGY, 40U, "Hit lx fusion strategy.");
// inferface
} // namespace tune
#endif // GRAPH_TUNER_GRAPH_TUNER_ERRORCODE_H
