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
 * \file array_ops_shape_fns.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_UTIL_ARRAY_OPS_SHAPE_FNS_H_
#define OPS_BUILT_IN_OP_PROTO_UTIL_ARRAY_OPS_SHAPE_FNS_H_

#include "graph/operator.h"

namespace ge {
/* *
 * infer pad grad op shape
 * @param op Operator which need to infershape
 * @return status whether infershape success
 */
graphStatus PadGradShapeFn(Operator& op);
}  // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_UTIL_ARRAY_OPS_SHAPE_FNS_H_
