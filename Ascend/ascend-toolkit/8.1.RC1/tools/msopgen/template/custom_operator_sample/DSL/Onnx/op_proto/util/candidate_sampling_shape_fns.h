/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
 * \file candidate_sampling_shape_fns.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_UTIL_CANDIDATE_SAMPLING_SHAPE_FNS_H_
#define OPS_BUILT_IN_OP_PROTO_UTIL_CANDIDATE_SAMPLING_SHAPE_FNS_H_

#include "common_shape_fns.h"

namespace ge {
/**
 * Set output shape that as same as a input for candidate sampling op
 * @param op Operator
 * @return status whether Shape's condition Satisfied
 */
graphStatus CandidateSamplerShape(Operator& op);
}  // namespace ge
#endif  // OPS_BUILT_IN_OP_PROTO_UTIL_CANDIDATE_SAMPLING_SHAPE_FNS_H_
