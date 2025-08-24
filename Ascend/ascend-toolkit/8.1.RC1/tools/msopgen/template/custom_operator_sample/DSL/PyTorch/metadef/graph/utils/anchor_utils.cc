/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#include "graph/utils/anchor_utils.h"
#include <algorithm>
#include "graph/debug/ge_util.h"
#include "common/ge_common/debug/ge_log.h"

namespace ge {
// Get anchor status
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY AnchorStatus AnchorUtils::GetStatus(const DataAnchorPtr &data_anchor) {
  if (data_anchor == nullptr) {
    REPORT_INNER_ERROR("E18888", "param data_anchor is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] The input data anchor is invalid.");
    return ANCHOR_RESERVED;
  }
  return data_anchor->status_;
}

// Set anchor status
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus AnchorUtils::SetStatus(const DataAnchorPtr &data_anchor,
                                                                                  const AnchorStatus anchor_status) {
  if ((data_anchor == nullptr) || (anchor_status == ANCHOR_RESERVED)) {
    REPORT_INNER_ERROR("E18888", "The input data anchor or input data format is invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] The input data anchor or input data format is invalid.");
    return GRAPH_FAILED;
  }
  data_anchor->status_ = anchor_status;
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY int32_t AnchorUtils::GetIdx(const AnchorPtr &anchor) {
  // Check if it can add edge between DataAnchor
  const auto data_anchor = Anchor::DynamicAnchorCast<DataAnchor>(anchor);
  if (data_anchor != nullptr) {
    return data_anchor->GetIdx();
  }
  // Check if it can add edge between ControlAnchor
  const auto ctrl_anchor = Anchor::DynamicAnchorCast<ControlAnchor>(anchor);
  if (ctrl_anchor != nullptr) {
    return ctrl_anchor->GetIdx();
  }
  return -1;
}
}  // namespace ge
