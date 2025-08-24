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
#ifndef OP_API_OP_API_COMMON_INC_FORMAT_UTILS_H
#define OP_API_OP_API_COMMON_INC_FORMAT_UTILS_H

#include <acl/acl_base.h>
#include "common_types.h"

namespace op {
constexpr uint32_t BIT_NUM_OF_ONE_BYTE = 8U;
constexpr uint32_t BIT_THREE_BYTES = 24U;

bool IsPrivateFormat(Format format);
Format ToFormat(const std::string &formatStr);
ge::AscendString ToString(Format format);

inline int32_t GetPrimaryFormat(int32_t format)
{
    return static_cast<int32_t>(static_cast<uint32_t>(format) & 0xffU);
}

inline Format GetPrimaryFormat(Format format)
{
    return static_cast<Format>(static_cast<uint32_t>(format) & 0xffU);
}

inline int32_t GetSubFormat(int32_t format)
{
    return static_cast<int32_t>((static_cast<uint32_t>(format) & 0xffff00U) >> BIT_NUM_OF_ONE_BYTE);
}

inline int32_t GetSubFormat(Format format)
{
    return static_cast<int32_t>((static_cast<uint32_t>(format) & 0xffff00U) >> BIT_NUM_OF_ONE_BYTE);
}

inline bool HasSubFormat(int32_t format)
{
    return GetSubFormat(format) > 0;
}

inline int64_t GetC0Format(int32_t format)
{
    return static_cast<int64_t>(1
        << (static_cast<int32_t>((static_cast<uint32_t>(format) & 0xf000000U) >> BIT_THREE_BYTES) - 1));
}

inline bool HasC0Format(int32_t format)
{
    return ((static_cast<uint32_t>(format) & 0xf000000U) >> BIT_THREE_BYTES) > 0;
}

inline Format GetFormatFromSub(int32_t primaryFormat, int32_t subFormat)
{
    return static_cast<Format>((static_cast<uint32_t>(primaryFormat) & 0xffU)
        | ((static_cast<uint32_t>(subFormat) & 0xffffU) << 8U));
}

inline Format ToOpFormat(aclFormat format)
{
    if (format != aclFormat::ACL_FORMAT_UNDEFINED) {
        return static_cast<Format>(format);
    }

    return Format::FORMAT_MAX;
}

inline aclFormat ToAclFormat(Format format)
{
    static const std::vector<Format> CAN_CONVERT_TO_ACL_FORMAT_LIST = {Format::FORMAT_NCHW, Format::FORMAT_NHWC,
                                                                       Format::FORMAT_ND, Format::FORMAT_NC1HWC0,
                                                                       Format::FORMAT_FRACTAL_Z,
                                                                       Format::FORMAT_NC1HWC0_C04, Format::FORMAT_HWCN,
                                                                       Format::FORMAT_NDHWC, Format::FORMAT_FRACTAL_NZ,
                                                                       Format::FORMAT_NCDHW, Format::FORMAT_NDC1HWC0,
                                                                       Format::FORMAT_FRACTAL_Z_3D, Format::FORMAT_NC,
                                                                       Format::FORMAT_NCL};
    auto iter = std::find(CAN_CONVERT_TO_ACL_FORMAT_LIST.begin(),
                          CAN_CONVERT_TO_ACL_FORMAT_LIST.end(),
                          format);
    if (iter == CAN_CONVERT_TO_ACL_FORMAT_LIST.end()) {
        return aclFormat::ACL_FORMAT_UNDEFINED;
    }

    return static_cast<aclFormat>(format);
}

} // namespace op

#endif // OP_API_OP_API_COMMON_INC_FORMAT_UTILS_H
