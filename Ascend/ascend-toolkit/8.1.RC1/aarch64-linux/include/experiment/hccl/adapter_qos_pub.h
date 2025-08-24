/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 * Description: adapter层重构，qos接口
 */

#ifndef HCCL_INC_ADAPTER_QOS_PUB_H
#define HCCL_INC_ADAPTER_QOS_PUB_H

#include "hccl_common.h"

constexpr u32 HCCL_STREAM_DEFAULT_GROUP_ID = 0xFFFFFFFFU;

HcclResult hrtGetQosConfig(const u32 groupId, u32 &qosCfg);

#endif  // HCCL_INC_ADAPTER_PROF_PUB_H