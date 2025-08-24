/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 * Description: comop 公共头文件
 */

#ifndef DTYPE_COMMON_H
#define DTYPE_COMMON_H

#include <unordered_map>
#include "hccl/base.h"

// 2 is sizeof(float16), 8 is sizeof(float64), 2 is sizeof(bfloat16)..
constexpr u32 SIZE_TABLE[HCCL_DATA_TYPE_RESERVED] = {sizeof(s8), sizeof(s16), sizeof(s32),
    2, sizeof(float), sizeof(s64), sizeof(u64), sizeof(u8), sizeof(u16), sizeof(u32), 8, 2, 16};

// 对内芯片类型
enum class DevType {
    DEV_TYPE_910 = 0,
    DEV_TYPE_310P3 = 1, // PG
    DEV_TYPE_910B = 2,
    DEV_TYPE_310P1 = 3, // AG
    DEV_TYPE_910_93 = 4,
    DEV_TYPE_NOSOC = 5,
    DEV_TYPE_COUNT = 6
};

#endif // DTYPE_COMMON_H