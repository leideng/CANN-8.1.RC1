/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#include "inc/graph/axis_type_info.h"

namespace ge {
void AxisTypeInfo::AddInputCutInfo(CutInfo &input_cut_info) {
  relate_inputs_.emplace_back(input_cut_info);
}

void AxisTypeInfo::AddOutputCutInfo(CutInfo &output_cut_info) {
  relate_outputs_.emplace_back(output_cut_info);
}

graphStatus AxisTypeInfo::GetInputCutInfo(const size_t index, CutInfo &input_cut_info) const {
  return DoGetCutInfo(relate_inputs_, index, input_cut_info);
}

graphStatus AxisTypeInfo::GetOutputCutInfo(const size_t index, CutInfo &output_cut_info) const {
  return DoGetCutInfo(relate_outputs_, index, output_cut_info);
}

void AxisTypeInfo::AddInputValueCutInfo(const CutInfo &cut_info) {
  relate_input_values_.emplace_back(cut_info);
}

void AxisTypeInfo::AddOutputValueCutInfo(const CutInfo &cut_info) {
  relate_output_values_.emplace_back(cut_info);
}

graphStatus AxisTypeInfo::GetInputValueCutInfo(const size_t index, CutInfo &cut_info) const {
  return DoGetCutInfo(relate_input_values_, index, cut_info);
}

graphStatus AxisTypeInfo::GetOutputValueCutInfo(const size_t index, CutInfo &cut_info) const {
  return DoGetCutInfo(relate_output_values_, index, cut_info);
}

graphStatus AxisTypeInfo::DoGetCutInfo(const std::vector<CutInfo> &cut_infos,
                                       const size_t index,
                                       CutInfo &cut_info) {
  if (cut_infos.size() <= index) {
    return GRAPH_FAILED;
  }
  cut_info = cut_infos[index];
  return GRAPH_SUCCESS;
}
}
