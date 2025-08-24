/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file getcdim.cc
 * \brief
 */

#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <bitset>
#include <iostream>
#include "graph/types.h"
#include "def_types.h"
#include "exe_graph/runtime/extended_kernel_context.h"
#include "exe_graph/runtime/shape.h"
#include "exe_graph/runtime/tiling_context.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "getcdim.h"

namespace ops {
  const int32_t AXIS_NCHW_DIM_C = 1;
  const int32_t AXIS_NHWC_DIM_C = 3;
  const int32_t AXIS_HWCN_DIM_C = 2;
  const int32_t AXIS_CHWN_DIM_C = 0;
  const int32_t NDHWC_DIM_C = 4;
  const int32_t NCDHW_DIM_C = 1;
  const int32_t DHWCN_DIM_C = 3;
  const int32_t DHWNC_DIM_C = 4;

  const std::map<ge::Format, const int32_t> CDIM_INDEX_OF_FORMAT {
    {ge::FORMAT_NCHW, AXIS_NCHW_DIM_C},
    {ge::FORMAT_HWCN, AXIS_HWCN_DIM_C},
    {ge::FORMAT_NHWC, AXIS_NHWC_DIM_C},
    {ge::FORMAT_CHWN, AXIS_CHWN_DIM_C},
    {ge::FORMAT_NDHWC, NDHWC_DIM_C},
    {ge::FORMAT_NCDHW, NCDHW_DIM_C},
    {ge::FORMAT_DHWCN, DHWCN_DIM_C},
    {ge::FORMAT_DHWNC, DHWNC_DIM_C}
  };

  int64_t GetCDim(gert::TilingContext * const context, const size_t index, const bool is_input) {
    if (context == nullptr) {
      return -1;
    }
    auto extend_context = ops::PtrToPtr<gert::TilingContext, gert::ExtendedKernelContext>(context);
    auto compute_node_info = extend_context->GetComputeNodeInfo();
    if (compute_node_info == nullptr) {
      return -1;
    }
    auto kernel_context = ops::PtrToPtr<gert::TilingContext, gert::KernelContext>(context);
    const gert::CompileTimeTensorDesc *td = nullptr;
    gert::StorageShape *storage_shape = nullptr;
    if (is_input) {
      td = compute_node_info->GetInputTdInfo(index);
      storage_shape = kernel_context->MutableInputPointer<gert::StorageShape>(index);
    } else {
      td = compute_node_info->GetOutputTdInfo(index);
      storage_shape = kernel_context->GetOutputPointer<gert::StorageShape>(index);
    }
    if ((td == nullptr) || (storage_shape == nullptr)) {
      return -1;
    }
    const auto original_format = td->GetOriginFormat();
    const auto iter = CDIM_INDEX_OF_FORMAT.find(original_format);
    if (iter == CDIM_INDEX_OF_FORMAT.cend()) {
      return -1;
    }
    gert::Shape &origin_shape = storage_shape->MutableOriginShape();
    const auto expend_dims = td->GetExpandDimsType();
    gert::Shape expand_shape;
    (void)expend_dims.Expand(origin_shape, expand_shape);

    if (static_cast<size_t>(iter->second) >= expand_shape.GetDimNum()) {
      return -1;
    }
    if (expand_shape.GetDimNum() == origin_shape.GetDimNum()) {
      return static_cast<int64_t>(origin_shape.GetDim(iter->second));
    } else {
      return static_cast<int64_t>(expand_shape.GetDim(iter->second));
    }
  }

  int64_t GetInputCDim(gert::TilingContext *kernel_context, const size_t index) {
    return GetCDim(kernel_context, index, true);
  }
  
  int64_t GetOutputCDim(gert::TilingContext *kernel_context, const size_t index) {
    return GetCDim(kernel_context, index, false);
  }
}  // namespace ops
