/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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
 * \file foreach_op_register.h
 * \brief
 */
#include "foreach_functors.h"
#include "multi_tensor_apply.h"
#include "foreach_op_def.h"

#ifndef __FOREACH_OP_REGISTER_H__
#define __FOREACH_OP_REGISTER_H__

using namespace AscendC;
using namespace foreachFunctors;
using namespace multiTensorApply;

namespace foreachOp {

#define IS_CHECK_ONE_DATA_TYPE(USER_DTYPE_INPUT0) (IsSameType<DTYPE_INPUT0, USER_DTYPE_INPUT0>::value)

#define IS_CHECK_TWO_DATA_TYPE(USER_DTYPE_INPUT0, USER_DTYPE_INPUT1) \
  (IsSameType<DTYPE_INPUT0, USER_DTYPE_INPUT0>::value && IsSameType<DTYPE_INPUT1, USER_DTYPE_INPUT1>::value)

#define IS_CHECK_THREE_DATA_TYPE(USER_DTYPE_INPUT0, USER_DTYPE_INPUT1, USER_DTYPE_INPUT2)                      \
  (IsSameType<DTYPE_INPUT0, USER_DTYPE_INPUT0>::value && IsSameType<DTYPE_INPUT1, USER_DTYPE_INPUT1>::value && \
   IsSameType<DTYPE_INPUT2, USER_DTYPE_INPUT2>::value)

#define IS_CHECK_SCALAR_TYPE(USER_DTYPE_SCALAR) (IsSameType<DTYPE_ALPHA, USER_DTYPE_SCALAR>::value)

#define IS_CHECK_SCALARLIST_TYPE(USER_DTYPE_SCALAR_LIST) (IsSameType<DTYPE_ALPHALIST, USER_DTYPE_SCALAR_LIST>::value)

// unary
#define TENSORLIST_FOREACH_UNARY_OP_CALL(KEY_ID, OP, INPUT0, OUTPUT, USER_DTYPE_INPUT0, WORKSPACE, TILINGDATA) \
  if (TILING_KEY_IS(KEY_ID) && IS_CHECK_ONE_DATA_TYPE(USER_DTYPE_INPUT0)) {                                    \
    GM_ADDR tensorList[2] = {INPUT0, OUTPUT};                                                                  \
    multiTensorApplyKernel<2, 1, 1, MultiTensorApplyTilingData, DTYPE_INPUT0, DTYPE_INPUT0>(                   \
        tensorList, WORKSPACE, &TILINGDATA, UnaryOpListFunctor<DTYPE_INPUT0, DTYPE_INPUT0, 2, 1>(),            \
        OP<DTYPE_INPUT0>);                                                                                     \
  }

#define TENSORLIST_FOREACH_UNARY_WITH_SCALAR_OP_CALL(KEY_ID, OP, INPUT0, OUTPUT, ALPHA, USER_DTYPE_INPUT0, \
                                                     USER_DTYPE_SCALAR, WORKSPACE, TILINGDATA)             \
  if (TILING_KEY_IS(KEY_ID) && IS_CHECK_ONE_DATA_TYPE(USER_DTYPE_INPUT0) &&                                \
      IS_CHECK_SCALAR_TYPE(USER_DTYPE_SCALAR)) {                                                           \
    GM_ADDR tensorList[2] = {INPUT0, OUTPUT};                                                              \
    multiTensorApplyKernel<2, 1, 1, MultiTensorApplyTilingData, DTYPE_INPUT0, DTYPE_ALPHA>(                \
        tensorList, WORKSPACE, &TILINGDATA, UnaryOpScalarFunctor<DTYPE_INPUT0, DTYPE_ALPHA, 2, 1>(),       \
        OP<DTYPE_INPUT0, DTYPE_ALPHA>, ALPHA);                                                             \
  }

#define TENSORLIST_FOREACH_UNARY_WITH_SCALARLIST_OP_CALL(KEY_ID, OP, INPUT0, OUTPUT, ALPHALIST, USER_DTYPE_INPUT0, \
                                                         USER_DTYPE_SCALARLIST, WORKSPACE, TILINGDATA)             \
  if (TILING_KEY_IS(KEY_ID) && IS_CHECK_ONE_DATA_TYPE(USER_DTYPE_INPUT0) &&                                        \
      IS_CHECK_SCALARLIST_TYPE(USER_DTYPE_SCALARLIST)) {                                                           \
    GM_ADDR tensorList[2] = {INPUT0, OUTPUT};                                                                      \
    multiTensorApplyKernel<2, 1, 1, MultiTensorApplyTilingData, DTYPE_INPUT0, DTYPE_ALPHA>(                        \
        tensorList, WORKSPACE, &TILINGDATA, UnaryOpScalarListFunctor<DTYPE_INPUT0, DTYPE_ALPHA, 2, 1>(),           \
        OP<DTYPE_INPUT0, DTYPE_ALPHA>, ALPHALIST);                                                                 \
  }

// binary
#define TENSORLIST_FOREACH_BINARY_OP_CALL(KEY_ID, OP, INPUT0, INPUT1, OUTPUT, USER_DTYPE_INPUT0, USER_DTYPE_INPUT1, \
                                          WORKSPACE, TILINGDATA)                                                    \
  if (TILING_KEY_IS(KEY_ID) && IS_CHECK_TWO_DATA_TYPE(USER_DTYPE_INPUT0, USER_DTYPE_INPUT1)) {                      \
    GM_ADDR tensorList[3] = {INPUT0, INPUT1, OUTPUT};                                                               \
    multiTensorApplyKernel<3, 2, 2, MultiTensorApplyTilingData, DTYPE_INPUT0, DTYPE_INPUT0>(                        \
        tensorList, WORKSPACE, &TILINGDATA, BinaryOpListFunctor<DTYPE_INPUT0, 3, 2>(), OP<DTYPE_INPUT0>);           \
  }

#define TENSORLIST_FOREACH_BINARY_WITH_SCALAR_OP_CALL(KEY_ID, OP, INPUT0, INPUT1, OUTPUT, ALPHA, USER_DTYPE_INPUT0, \
                                                      USER_DTYPE_INPUT1, USER_DTYPE_SCALAR, WORKSPACE, TILINGDATA)  \
  if (TILING_KEY_IS(KEY_ID) && IS_CHECK_TWO_DATA_TYPE(USER_DTYPE_INPUT0, USER_DTYPE_INPUT1) &&                      \
      IS_CHECK_SCALAR_TYPE(USER_DTYPE_SCALAR)) {                                                                    \
    GM_ADDR tensorList[3] = {INPUT0, INPUT1, OUTPUT};                                                               \
    multiTensorApplyKernel<3, 2, 2, MultiTensorApplyTilingData, DTYPE_INPUT0, DTYPE_ALPHA>(                         \
        tensorList, WORKSPACE, &TILINGDATA, BinaryOpScalarFunctor<DTYPE_INPUT0, DTYPE_ALPHA, 3, 2>(),               \
        OP<DTYPE_INPUT0, DTYPE_ALPHA>, ALPHA);                                                                      \
  }

#define TENSORLIST_FOREACH_BINARY_WITH_SCALARLIST_OP_CALL(KEY_ID, OP, INPUT0, INPUT1, OUTPUT, ALPHALIST,  \
                                                          USER_DTYPE_INPUT0, USER_DTYPE_INPUT1,           \
                                                          USER_DTYPE_SCALARLIST, WORKSPACE, TILINGDATA)   \
  if (TILING_KEY_IS(KEY_ID) && IS_CHECK_TWO_DATA_TYPE(USER_DTYPE_INPUT0, USER_DTYPE_INPUT1) &&            \
      IS_CHECK_SCALARLIST_TYPE(USER_DTYPE_SCALARLIST)) {                                                  \
    GM_ADDR tensorList[3] = {INPUT0, INPUT1, OUTPUT};                                                     \
    multiTensorApplyKernel<3, 2, 2, MultiTensorApplyTilingData, DTYPE_INPUT0, DTYPE_ALPHA>(               \
        tensorList, WORKSPACE, &TILINGDATA, BinaryOpScalarListFunctor<DTYPE_INPUT0, DTYPE_ALPHA, 3, 2>(), \
        OP<DTYPE_INPUT0, DTYPE_ALPHA>, ALPHALIST);                                                        \
  }

// ternary
#define TENSORLIST_FOREACH_TERNARY_OP_CALL(KEY_ID, OP, INPUT0, INPUT1, INPUT2, OUTPUT, USER_DTYPE_INPUT0,           \
                                           USER_DTYPE_INPUT1, USER_DTYPE_INPUT2, WORKSPACE, TILINGDATA)             \
  if (TILING_KEY_IS(KEY_ID) && IS_CHECK_THREE_DATA_TYPE(USER_DTYPE_INPUT0, USER_DTYPE_INPUT1, USER_DTYPE_INPUT2)) { \
    GM_ADDR tensorList[4] = {INPUT0, INPUT1, INPUT2, OUTPUT};                                                       \
    multiTensorApplyKernel<4, 3, 3, MultiTensorApplyTilingData, DTYPE_INPUT0, DTYPE_INPUT0>(                        \
        tensorList, WORKSPACE, &TILINGDATA, TernaryOpListFunctor<DTYPE_INPUT0, 4, 3>(), OP<DTYPE_INPUT0>);          \
  }

#define TENSORLIST_FOREACH_TERNARY_WITH_SCALAR_OP_CALL(KEY_ID, OP, INPUT0, INPUT1, INPUT2, OUTPUT, ALPHA,           \
                                                       USER_DTYPE_INPUT0, USER_DTYPE_INPUT1, USER_DTYPE_INPUT2,     \
                                                       USER_DTYPE_SCALAR, WORKSPACE, TILINGDATA)                    \
  if (TILING_KEY_IS(KEY_ID) && IS_CHECK_THREE_DATA_TYPE(USER_DTYPE_INPUT0, USER_DTYPE_INPUT1, USER_DTYPE_INPUT2) && \
      IS_CHECK_SCALAR_TYPE(USER_DTYPE_SCALAR)) {                                                                    \
    GM_ADDR tensorList[4] = {INPUT0, INPUT1, INPUT2, OUTPUT};                                                       \
    multiTensorApplyKernel<4, 3, 3, MultiTensorApplyTilingData, DTYPE_INPUT0, DTYPE_ALPHA>(                         \
        tensorList, WORKSPACE, &TILINGDATA, TernaryOpScalarFunctor<DTYPE_INPUT0, DTYPE_ALPHA, 4, 3>(),              \
        OP<DTYPE_INPUT0, DTYPE_ALPHA>, ALPHA);                                                                      \
  }

#define TENSORLIST_FOREACH_TERNARY_WITH_SCALARLIST_OP_CALL(KEY_ID, OP, INPUT0, INPUT1, INPUT2, OUTPUT, ALPHALIST,   \
                                                           USER_DTYPE_INPUT0, USER_DTYPE_INPUT1, USER_DTYPE_INPUT2, \
                                                           USER_DTYPE_SCALARLIST, WORKSPACE, TILINGDATA)            \
  if (TILING_KEY_IS(KEY_ID) && IS_CHECK_THREE_DATA_TYPE(USER_DTYPE_INPUT0, USER_DTYPE_INPUT1, USER_DTYPE_INPUT2) && \
      IS_CHECK_SCALARLIST_TYPE(USER_DTYPE_SCALARLIST)) {                                                            \
    GM_ADDR tensorList[4] = {INPUT0, INPUT1, INPUT2, OUTPUT};                                                       \
    multiTensorApplyKernel<4, 3, 3, MultiTensorApplyTilingData, DTYPE_INPUT0, DTYPE_ALPHA>(                         \
        tensorList, WORKSPACE, &TILINGDATA, TernaryOpScalarListFunctor<DTYPE_INPUT0, DTYPE_ALPHA, 4, 3>(),          \
        OP<DTYPE_INPUT0, DTYPE_ALPHA>, ALPHALIST);                                                                  \
  }

}  // namespace foreachOp

#endif