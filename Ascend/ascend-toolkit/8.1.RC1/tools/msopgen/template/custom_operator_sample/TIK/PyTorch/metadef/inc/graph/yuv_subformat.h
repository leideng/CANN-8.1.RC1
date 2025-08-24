/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef INC_YUV_SUBFORMAT_H_
#define INC_YUV_SUBFORMAT_H_
namespace ge {
enum YUVSubFormat {
  YUV420_SP = 1,
  YVU420_SP,
  YUV422_SP,
  YVU422_SP,
  YUV440_SP,
  YVU440_SP,
  YUV444_SP,
  YVU444_SP,
  YUYV422_PACKED,
  YVYU422_PACKED,
  YUV444_PACKED,
  YVU444_PACKED,
  YUV400
};
}  // namespace ge
#endif  // INC_YUV_SUBFORMAT_H_
