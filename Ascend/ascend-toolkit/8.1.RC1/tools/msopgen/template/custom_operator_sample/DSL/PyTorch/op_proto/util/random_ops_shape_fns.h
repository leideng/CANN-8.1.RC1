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
 * \file random_ops_shape_fns.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_UTIL_RANDOM_OPS_SHAPE_FNS_H_
#define OPS_BUILT_IN_OP_PROTO_UTIL_RANDOM_OPS_SHAPE_FNS_H_

#include "common_shape_fns.h"

namespace ge {

/**
 * Set output shape that as same as a input for random op
 * @param op Operator
 * @param shape_name A input shape name
 * @param out_name Output name
 * @return status whether Shape's condition Satisfied
 */
graphStatus RandomShape(Operator& op, const std::string& shape_name, const std::string& out_name);

/**
 * Set output shape that as same as a input for random op
 * and set output data type
 * @param op Operator
 * @param shape_name A input shape name
 * @param date_type_attr_name Data type attr name associated to output
 * @param out_name Output name
 * @return status whether Shape's condition Satisfied
 */
graphStatus RandomShapeWithDataType(Operator& op, const std::string& shape_name, const std::string& date_type_attr_name,
                                    const std::string& out_name);
}  // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_UTIL_RANDOM_OPS_SHAPE_FNS_H_
