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
 * \file kernel_struct_transpose.h
 * \brief
 */
#ifndef ASCENDC_MODULE_STRUCT_TRANSPOSE_H
#define ASCENDC_MODULE_STRUCT_TRANSPOSE_H

namespace AscendC {
enum class TransposeType : uint8_t {
    // default value
    TRANSPOSE_TYPE_NONE,
    // { shape:[B, A1, A3 / 16, A2 / 16, 16, 16], format:"NZ"} -->{ shape:[B, A2, A1, A3], ori_shape:[B, A2, A1, A3],
    // format:"ND"}
    TRANSPOSE_NZ2ND_0213,
    // { shape:[B, A1, A3 / 16, A2 / 16, 16, 16], format:"NZ"}-->{ shape:[B, A2, A3 / 16, A1 / 16, 16, 16],
    // origin_shape:[B, A2, A1, A3], format:"NZ"}
    TRANSPOSE_NZ2NZ_0213,
    // { shape:[B, H / 16, S / 16, 16, 16], format:"NZ"}-->{ shape:[B, N, H/N/16, S / 16, 16, 16], ori_shape:[B, N, S,
    // H/N], format:"NZ"}
    TRANSPOSE_NZ2NZ_012_WITH_N,
    // { shape:[B, H / 16, S / 16, 16, 16], format:"NZ"}-->{ shape:[B, N, S, H/N], ori_shape:[B, N, S, H/N],
    // format:"ND"}
    TRANSPOSE_NZ2ND_012_WITH_N,
    // { shape:[B, N, H/N/16, S/16, 16, 16], format:"NZ"}-->{ shape:[B, S, H], ori_shape:[B, S, H], format:"ND"}
    TRANSPOSE_NZ2ND_012_WITHOUT_N,
    // { shape:[B, N, H/N/16, S/16, 16, 16], format:"NZ"}-->{ shape:[B, H/16, S/16, 16, 16], ori_shape:[B, S, H],
    // format:"NZ"}
    TRANSPOSE_NZ2NZ_012_WITHOUT_N,
    TRANSPOSE_ND2ND_ONLY,    // { shape:[H, W], format:"ND"} -->{ shape:[W, H], format:"ND"}
    TRANSPOSE_ND_UB_GM,      //  [B, N, S, H/N] -> [B, S, H]
    TRANSPOSE_GRAD_ND_UB_GM, //  [B, S, H] -> [B, N, S, H/N]
    TRANSPOSE_ND2ND_B16,     // { shape:[16, 16], format:"ND", dataType: B16} -->{ shape:[16, 16], format:"ND"}
    TRANSPOSE_NCHW2NHWC,     // [ N, C, H, W] -> [N, H, W, C]
    TRANSPOSE_NHWC2NCHW      // [ N, H, W, C] -> [N, C, H, W]
};

struct TransDataTo5HDParams {
    __aicore__ TransDataTo5HDParams() {}

    __aicore__ TransDataTo5HDParams(const bool dstHighHalfIn, const bool srcHighHalfIn, const uint8_t repeatTimesIn,
        const uint16_t dstRepStrideIn, const uint16_t srcRepStrideIn)
        : dstHighHalf(dstHighHalfIn),
          srcHighHalf(srcHighHalfIn),
          repeatTimes(repeatTimesIn),
          dstRepStride(dstRepStrideIn),
          srcRepStride(srcRepStrideIn)
    {}

    bool dstHighHalf = false;
    bool srcHighHalf = false;
    uint8_t repeatTimes = 1;
    uint16_t dstRepStride = 0;
    uint16_t srcRepStride = 0;
};

struct TransposeParamsExt {
    __aicore__ TransposeParamsExt() {}

    __aicore__ TransposeParamsExt(const uint16_t nSizeIn, const uint16_t cSizeIn, const uint16_t hSizeIn,
        const uint16_t wSizeIn, const TransposeType transposeTypeIn)
        : nSize(nSizeIn),
          cSize(cSizeIn),
          hSize(hSizeIn),
          wSize(wSizeIn),
          transposeType(transposeTypeIn)
    {}

    uint16_t nSize = 0;
    uint16_t cSize = 0;
    uint16_t hSize = 0;
    uint16_t wSize = 0;
    TransposeType transposeType = TransposeType::TRANSPOSE_ND2ND_B16;
};
} // namespace AscendC
#endif // ASCENDC_MODULE_STRUCT_TRANSPOSE_H
