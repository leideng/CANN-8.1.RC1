/* Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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
 * \file kernel_operator_mm_check.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_MM_CHECK_H
#define ASCENDC_MODULE_OPERATOR_MM_CHECK_H

#include "kernel_check.h"

namespace AscendC {

template <typename T>
__aicore__ static inline bool ChannelSizeRemainder(const uint16_t channelSize, uint16_t remainder[], uint16_t size)
{
    uint16_t oneBlkNum = ONE_BLK_SIZE / sizeof(T);
    if constexpr (IsSameType<T, int4b_t>::value) {
        oneBlkNum = 64;  // 1 block = 64 int4b_t
    }
    for (uint16_t i = 0; i < size; i++) {
        if (channelSize % oneBlkNum == remainder[i]) {
            return true;
        }
    }
    return false;
}
 //check fmLocal, filterLocal align
template <typename DstT, typename Src0T, typename Src1T>
__aicore__ static inline void CheckMmadAlign(const LocalTensor<DstT>& dstLocal, const LocalTensor<Src0T>& fmLocal,
    const LocalTensor<Src1T>& filterLocal) {
    constexpr uint64_t ALIGN_1024B = 1024;
    if constexpr ((IsSameType<PrimT<Src0T>, half>::value) && (IsSameType<PrimT<Src1T>, half>::value) &&
        (IsSameType<PrimT<DstT>, half>::value)) {
        CheckTensorAlign<DstT>(dstLocal, VALUE_512, "dstLocal", "Mmad");
    } else {
        CheckTensorAlign<DstT>(dstLocal, ALIGN_1024B, "dstLocal", "Mmad");
    }
    CheckTensorAlign<Src0T>(fmLocal, VALUE_512, "fmLocal", "Mmad");
    CheckTensorAlign<Src1T>(filterLocal, VALUE_512, "filterLocal", "Mmad");
}

// check LoadData2D datatype
template <typename T>
__aicore__ static inline void CheckLoadData2dDatatype()
{
#if __CCE_AICORE__ == 200
    ASCENDC_ASSERT((SupportType<PrimT<T>, uint8_t, int8_t, uint16_t, int16_t, half>()),
        {KERNEL_LOG(KERNEL_ERROR, "Failed to "
        "check dtype in LoadData with LoadData2DParams, current api support dtype combination is src and dst both: "
        "uint8_t / int8_t / uint16_t / int16_t / half.");});
#elif __CCE_AICORE__ == 220
    ASCENDC_ASSERT((SupportType<PrimT<T>, uint8_t, int8_t, uint16_t, int16_t, half, bfloat16_t, uint32_t, int32_t,
        float, int4b_t>()),
        {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in LoadData with LoadData2DParams, current api "
        "support dtype combination is src and dst both uint8_t / int8_t / uint16_t / int16_t / half / bfloat16_t / "
        "uint32_t / int32_t / float / int4b_t.");});
#elif defined(__DAV_M310__)
    ASCENDC_ASSERT((SupportType<PrimT<T>, uint8_t, int8_t, half, uint16_t, int16_t, int4b_t>()),
        {KERNEL_LOG(KERNEL_ERROR,
        "Failed to check dtype in LoadData with LoadData2DParamsV2, current api support dtype combination is src and "
        "dst both: uint8_t / int8_t / half / uint16_t / int16_t / int4b_t.");});
#endif
}

// check LoadData3D params
__aicore__ static inline void CheckLoadData3dParams(const uint16_t srcHeight, const uint16_t srcWeight,
    const uint8_t srcWStride, const uint8_t srcHStride)
{
    ASCENDC_CHECK_VALUE_RANGE(srcHeight, MIN_LOAD3D_L1, MAX_LOAD3D_L1, "l1H", "LoadData with LoadData3DParams");
    ASCENDC_CHECK_VALUE_RANGE(srcWeight, MIN_LOAD3D_L1, MAX_LOAD3D_L1, "l1W", "LoadData with LoadData3DParams");
    ASCENDC_CHECK_VALUE_RANGE(srcWStride, MIN_LOAD3D_STRIDE, MAX_LOAD3D_STRIDE, "strideW",
        "LoadData with LoadData3DParams");
    ASCENDC_CHECK_VALUE_RANGE(srcHStride, MIN_LOAD3D_STRIDE, MAX_LOAD3D_STRIDE, "strideH",
        "LoadData with LoadData3DParams");
}

// check Load3dv2 ChannelSize
template <typename T>
__aicore__ static inline void CheckLoadData3dv2ChannelSize(const uint16_t channelSize)
{
#if __CCE_AICORE__ == 200
    if constexpr (IsSameType<PrimT<T>, half>::value) {
        uint16_t remainderList[] = {4, 8};
        ASCENDC_ASSERT((ChannelSizeRemainder<PrimT<T>>(channelSize, remainderList, 2) || channelSize == 16),
            {KERNEL_LOG(KERNEL_ERROR, "Failed to check param channelSize value in LoadData with LoadData3DParamsV2 "
            "with dtype half, it should be: 16 or channelSize % 16 = 4 / 8, current value is %u", channelSize);});
    } else if constexpr(SupportType<PrimT<T>, int8_t, uint8_t>()) {
        uint16_t remainderList[] = {4, 8, 16};
        ASCENDC_ASSERT((ChannelSizeRemainder<PrimT<T>>(channelSize, remainderList, 3) || channelSize == 32),
            {KERNEL_LOG(KERNEL_ERROR, "Failed to check param channelSize value in LoadData with LoadData3DParamsV2 "
            "with dtype int8_t / uint8_t, it should be: 32 or channelSize % 32 = 4 / 8 / 16, current value is %u",
            channelSize);});
    } else if constexpr (IsSameType<PrimT<T>, int4b_t>::value) {
        uint16_t remainderList[] = {8, 16, 32};
        ASCENDC_ASSERT((ChannelSizeRemainder<PrimT<T>>(channelSize, remainderList, 3) || channelSize == 64),
            {KERNEL_LOG(KERNEL_ERROR, "Failed to check param channelSize value in LoadData with LoadData3DParamsV2 "
            "with dtype int4b_t, it should be: 64 or channelSize % 64 = 8 / 16 / 32, current value is %u",
            channelSize);});
    }
#elif __CCE_AICORE__ >= 220 || defined(__DAV_M310__)
#if defined(__DAV_M310__)
    if constexpr (IsSameType<PrimT<T>, half>::value) {
        uint16_t remainderList[] = {0, 4, 8};
        ASCENDC_ASSERT((ChannelSizeRemainder<PrimT<T>>(channelSize, remainderList, 3)),
            {KERNEL_LOG(KERNEL_ERROR, "Failed to "
            "check param channelSize value in LoadData with LoadData3DParamsV2 with dtype half, it should be: "
            "channelSize % 16 = 0 / 4 / 8, current value is %u", channelSize);});
    }
#else
    if constexpr (SupportType<PrimT<T>, half, bfloat16_t>()) {
        uint16_t remainderList[] = {0, 4, 8};
        ASCENDC_ASSERT((ChannelSizeRemainder<PrimT<T>>(channelSize, remainderList, 3)),
            {KERNEL_LOG(KERNEL_ERROR, "Failed to "
            "check param channelSize value in LoadData with LoadData3DParamsV2 with dtype half / bfloat16_t, it should "
            "be: channelSize % 16 = 0 / 4 / 8, current value is %u", channelSize);});
    }
#endif
    if constexpr (SupportType<PrimT<T>, float, int32_t, uint32_t>()) {
        uint16_t remainderList[] = {0, 4};
        ASCENDC_ASSERT((ChannelSizeRemainder<PrimT<T>>(channelSize, remainderList, 2)),
            {KERNEL_LOG(KERNEL_ERROR, "Failed to "
            "check param channelSize value in LoadData with LoadData3DParamsV2 with dtype float / int32_t / uint32_t, "
            "it should be: channelSize % 8 = 0 / 4, current value is %u", channelSize);});
    } else if constexpr (SupportType<PrimT<T>, int8_t, uint8_t>()) {
        uint16_t remainderList[] = {0, 4, 8, 16};
        ASCENDC_ASSERT((ChannelSizeRemainder<PrimT<T>>(channelSize, remainderList, 4)),
            {KERNEL_LOG(KERNEL_ERROR, "Failed to "
            "check param channelSize value in LoadData with LoadData3DParamsV2 with dtype int8_t / uint8_t, it should "
            "be: channelSize % 32 = 0 / 4 / 8 / 16, current value is %u", channelSize);});
    } else if constexpr (IsSameType<PrimT<T>, int4b_t>::value) {
        uint16_t remainderList[] = {0, 8, 16, 32};
        ASCENDC_ASSERT((ChannelSizeRemainder<PrimT<T>>(channelSize, remainderList, 4)),
            {KERNEL_LOG(KERNEL_ERROR, "Failed to "
            "check param channelSize value in LoadData with LoadData3DParamsV2 with dtype int4b_t, it should be: "
            "channelSize % 64 = 0 / 8 / 16 / 32, current value is %u", channelSize);});
    }
#endif
}

// check LoadData3dv2 matrix params
template <typename T>
__aicore__ static inline void CheckLoadData3dv2MatrixParams(const uint16_t kExtension, const uint16_t mExtension,
    const uint16_t kStartPt, const uint16_t mStartPt) {
    constexpr uint16_t base16 = 16;
    if constexpr (SupportType<PrimT<T>, half, int8_t, int4b_t>()) {
        ASCENDC_ASSERT((mExtension % base16 == 0), { KERNEL_LOG(KERNEL_ERROR, "Failed to check mExtension value in "
            "LoadData with LoadData3DParamsV2 when dtype is half / int8_t / int4b_t, it should be divisible by 16, "
            "current value is %u", mExtension);});
    }
    uint16_t kExtBase = (SupportType<PrimT<T>, int4b_t>()) ? 64 : ONE_BLK_SIZE / sizeof(PrimT<T>);
    if constexpr (SupportType<PrimT<T>, half, int8_t, int4b_t, int32_t, uint32_t, float>()) {
        ASCENDC_ASSERT((kExtension % kExtBase == 0), { KERNEL_LOG(KERNEL_ERROR, "Failed to check kExtension value in "
            "LoadData with LoadData3DParamsV2 when dtype is half / int8_t / int4b_t / int32_t / uint32_t / float, it "
            "should be divisible by %u, current value is %u", kExtBase, kExtension);});
        ASCENDC_ASSERT((kStartPt % kExtBase == 0), { KERNEL_LOG(KERNEL_ERROR, "Failed to check kStartPt value in "
            "LoadData with LoadData3DParamsV2 when dtype is half / int8_t / int4b_t / int32_t / uint32_t / float, it "
            "should be divisible by %u, current value is %u", kExtBase, kStartPt);});
    }
#if __CCE_AICORE__ == 200
    if constexpr (SupportType<PrimT<T>, half, int8_t, int4b_t>()) {
        ASCENDC_ASSERT((mStartPt % base16 == 0), { KERNEL_LOG(KERNEL_ERROR, "Failed to check mStartPt value in "
            "LoadData with LoadData3DParamsV2 when dtype is half / int8_t / int4b_t, it should be divisible by 16, "
            "current value is %u", mStartPt);});
    }
#elif __CCE_AICORE__ == 220
    ASCENDC_CHECK_VALUE_RANGE(mStartPt, 0, UINT15_MAX, "mStartPt", "LoadData with LoadData3DParamsV2");
#endif
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_MM_CHECK_H