/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
* \file hccl_tilingdata.h
* \brief
*/
#ifndef LIB_HCCL_HCCL_TILINGDATA_H
#define LIB_HCCL_HCCL_TILINGDATA_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(Mc2ServerCfg) // server通用参数
TILING_DATA_FIELD_DEF(uint32_t, version);          // tiling结构体版本号
TILING_DATA_FIELD_DEF(uint8_t, debugMode);         // 调测模式
TILING_DATA_FIELD_DEF(uint8_t, sendArgIndex);      // 发送数据参数索引，对应算子原型的参数顺序
TILING_DATA_FIELD_DEF(uint8_t, recvArgIndex);      // 接收数据参数索引，对应算子原型的参数顺序
TILING_DATA_FIELD_DEF(uint8_t, commOutArgIndex);   // 通信输出参数索引，对应算子原型的参数顺序
TILING_DATA_FIELD_DEF_ARR(uint8_t, 8, reserved);   // 保留字段
END_TILING_DATA_DEF; // 16 bytes
REGISTER_TILING_DATA_CLASS(Mc2ServerCfgOpApi, Mc2ServerCfg)

BEGIN_TILING_DATA_DEF(Mc2HcommCfg) // 各通信域中的每个通信任务
// MC2特定优化tiling配置
TILING_DATA_FIELD_DEF(uint8_t, skipLocalRankCopy);    // 跳过本卡拷贝，在通信结果只需要给MC2内部计算使用或者本卡拷贝由aicore完成时,
                                                      // 可以跳过本卡数据send-recv搬运
TILING_DATA_FIELD_DEF(uint8_t, skipBufferWindowCopy); // 跳过hbm到window间搬运 0不跳过，1跳过snd-window, 2跳过window-rcv
TILING_DATA_FIELD_DEF(uint8_t, stepSize);             // 通信步长，粗粒度融合时填0,
                                                      // 细粒度融合时连续计算stepsize块数据再commit或wait通信
TILING_DATA_FIELD_DEF_ARR(char, 13, reserved);        // 保留字段
TILING_DATA_FIELD_DEF_ARR(char, 128, groupName);      // groupName
TILING_DATA_FIELD_DEF_ARR(char, 128, algConfig);      // 算法配置
TILING_DATA_FIELD_DEF(uint32_t, opType);              // tiling结构体版本号
TILING_DATA_FIELD_DEF(uint32_t, reduceType);          // reduce类型
END_TILING_DATA_DEF; // 280 bytes
REGISTER_TILING_DATA_CLASS(Mc2HcommCfgOpApi, Mc2HcommCfg)

BEGIN_TILING_DATA_DEF(Mc2InitTiling)
TILING_DATA_FIELD_DEF_ARR(uint8_t, 64, reserved);
END_TILING_DATA_DEF; // 64 bytes
REGISTER_TILING_DATA_CLASS(Mc2InitTilingOpApi, Mc2InitTiling)

BEGIN_TILING_DATA_DEF(Mc2CcTiling)
TILING_DATA_FIELD_DEF_ARR(uint8_t, 280, reserved);
END_TILING_DATA_DEF; // 280 bytes
REGISTER_TILING_DATA_CLASS(Mc2CcTilingOpApi, Mc2CcTiling)
} // namespace optiling
#endif