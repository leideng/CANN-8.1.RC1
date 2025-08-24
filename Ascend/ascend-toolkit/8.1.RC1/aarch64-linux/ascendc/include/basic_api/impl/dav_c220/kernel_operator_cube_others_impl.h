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
 * \file kernel_operator_cube_others_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_CUBE_OTHERS_IMPL_H
#define ASCENDC_MODULE_OPERATOR_CUBE_OTHERS_IMPL_H

#ifndef ASCENDC_CPU_DEBUG
namespace AscendC {
__aicore__ inline void CopyCbufToBt(uint64_t dst, __cbuf__ void *src, uint64_t config)
{
    copy_cbuf_to_bt(dst, src, config);
}

__aicore__ inline void CopyCbufToBt(uint64_t dst, __cbuf__ void *src, uint16_t convControl, uint16_t nBurst,
    uint16_t lenBurst, uint16_t sourceGap, uint16_t dstGap)
{
    copy_cbuf_to_bt(dst, src, convControl, nBurst, lenBurst, sourceGap, dstGap);
}

__aicore__ inline void CopyCbufToFbuf(__fbuf__ void *dst, __cbuf__ void *src, uint64_t config)
{
    copy_cbuf_to_fbuf(dst, src, config);
}

__aicore__ inline void CopyCbufToFbuf(__fbuf__ void *dst, __cbuf__ void *src, uint16_t burstNum, uint16_t burstLen,
    uint16_t srcGapSize, uint16_t dstGapSize)
{
    copy_cbuf_to_fbuf(dst, src, burstNum, burstLen, srcGapSize, dstGapSize);
}

__aicore__ inline void CopyCbufToGm(__gm__ void *dst, __cbuf__ void *src, uint64_t config)
{
    copy_cbuf_to_gm(dst, src, config);
}

__aicore__ inline void CopyCbufToGm(__gm__ void *dst, __cbuf__ void *src, uint8_t sid, uint16_t nBurst,
    uint16_t lenBurst, uint16_t srcStride, uint16_t dstStride)
{
    copy_cbuf_to_gm(dst, src, sid, nBurst, lenBurst, srcStride, dstStride);
}

template <typename T>
__aicore__ inline void CopyGmToCbufMultiNd2nzB16(__cbuf__ T *dst, __gm__ T *src, uint64_t xm, uint64_t xt)
{
    copy_gm_to_cbuf_multi_nd2nz_b16(dst, src, xm, xt);
}

template <typename T>
__aicore__ inline void CopyGmToCbufMultiNd2nzB16(__cbuf__ T *dst, __gm__ T *src, uint8_t sid, uint16_t ndNum,
    uint16_t nValue, uint16_t dValue, uint16_t srcNdMatrixStride, uint16_t srcDValue, uint16_t dstNzC0Stride,
    uint16_t dstNzNStride, uint16_t dstNzMatrixStride)
{
    copy_gm_to_cbuf_multi_nd2nz_b16(dst, src, sid, ndNum, nValue, dValue, srcNdMatrixStride, srcDValue, dstNzC0Stride,
        dstNzNStride, dstNzMatrixStride);
}

template <typename T>
__aicore__ inline void CopyGmToCbufMultiNd2nzB32s(__cbuf__ T *dst, __gm__ T *src, uint64_t xm, uint64_t xt)
{
    copy_gm_to_cbuf_multi_nd2nz_b32s(dst, src, xm, xt);
}

template <typename T>
__aicore__ inline void CopyGmToCbufMultiNd2nzB32s(__cbuf__ T *dst, __gm__ T *src, uint8_t sid, uint16_t ndNum,
    uint16_t nValue, uint16_t dValue, uint16_t srcNdMatrixStride, uint16_t srcDValue, uint16_t dstNzC0Stride,
    uint16_t dstNzNStride, uint16_t dstNzMatrixStride)
{
    copy_gm_to_cbuf_multi_nd2nz_b32s(dst, src, sid, ndNum, nValue, dValue, srcNdMatrixStride, srcDValue, dstNzC0Stride,
        dstNzNStride, dstNzMatrixStride);
}

template <typename T>
__aicore__ inline void CopyGmToCbufMultiNd2nzB8(__cbuf__ T *dst, __gm__ T *src, uint64_t xm, uint64_t xt)
{
    copy_gm_to_cbuf_multi_nd2nz_b8(dst, src, xm, xt);
}

template <typename T>
__aicore__ inline void CopyGmToCbufMultiNd2nzB8(__cbuf__ T *dst, __gm__ T *src, uint8_t sid, uint16_t ndNum,
    uint16_t nValue, uint16_t dValue, uint16_t srcNdMatrixStride, uint16_t srcDValue, uint16_t dstNzC0Stride,
    uint16_t dstNzNStride, uint16_t dstNzMatrixStride)
{
    copy_gm_to_cbuf_multi_nd2nz_b8(dst, src, sid, ndNum, nValue, dValue, srcNdMatrixStride, srcDValue, dstNzC0Stride,
        dstNzNStride, dstNzMatrixStride);
}

template <typename T> __aicore__ inline void CopyMatrixCbufToCc(__cc__ T *dst, __cbuf__ T *src, uint64_t config)
{
    copy_matrix_cbuf_to_cc(dst, src, config);
}

template <typename T> __aicore__ inline void CopyMatrixCbufToCc(__cc__ T *dst, __cbuf__ float *src, uint64_t config)
{
    copy_matrix_cbuf_to_cc(dst, src, config);
}

template <typename T>
__aicore__ inline void CopyMatrixCbufToCc(__cc__ T *dst, __cbuf__ T *src, uint16_t nBurst, uint16_t lenBurst,
    uint16_t srcStride, uint16_t dstStride)
{
    copy_matrix_cbuf_to_cc(dst, src, nBurst, lenBurst, srcStride, dstStride);
}

template <typename T>
__aicore__ inline void CopyMatrixCbufToCc(__cc__ T *dst, __cbuf__ float *src, uint16_t nBurst, uint16_t lenBurst,
    uint16_t srcStride, uint16_t dstStride)
{
    copy_matrix_cbuf_to_cc(dst, src, nBurst, lenBurst, srcStride, dstStride);
}

template <typename T>
__aicore__ inline void CopyMatrixCcToCbuf(__cbuf__ T *dst, __cc__ float *src, uint64_t xm, uint64_t xt)
{
    copy_matrix_cc_to_cbuf(dst, src, xm, xt);
}

template <typename T>
__aicore__ inline void CopyMatrixCcToCbuf(__cbuf__ T *dst, __cc__ int32_t *src, uint64_t xm, uint64_t xt)
{
    copy_matrix_cc_to_cbuf(dst, src, xm, xt);
}

template <typename T>
__aicore__ inline void CopyMatrixCcToCbuf(__cbuf__ T *dst, __cc__ float *src, uint8_t sid, uint16_t NSize,
    uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride, uint8_t UnitFlagMode, QuantMode_t QuantPRE,
    uint8_t ReLUPRE, bool channelSplit, bool NZ2ND_EN)
{
    copy_matrix_cc_to_cbuf(dst, src, sid, NSize, MSize, dstStride_dst_D, srcStride, UnitFlagMode, QuantPRE, ReLUPRE,
        channelSplit, NZ2ND_EN);
}

template <typename T>
__aicore__ inline void CopyMatrixCcToCbuf(__cbuf__ T *dst, __cc__ int32_t *src, uint8_t sid, uint16_t NSize,
    uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride, uint8_t UnitFlagMode, QuantMode_t QuantPRE,
    uint8_t ReLUPRE, bool channelSplit, bool NZ2ND_EN)
{
    copy_matrix_cc_to_cbuf(dst, src, sid, NSize, MSize, dstStride_dst_D, srcStride, UnitFlagMode, QuantPRE, ReLUPRE,
        channelSplit, NZ2ND_EN);
}

template <typename T>
__aicore__ inline void CopyMatrixCcToGm(__gm__ T *dst, __cc__ float *src, uint64_t xm, uint64_t xt)
{
    copy_matrix_cc_to_gm(dst, src, xm, xt);
}

template <typename T>
__aicore__ inline void CopyMatrixCcToGm(__gm__ T *dst, __cc__ int32_t *src, uint64_t xm, uint64_t xt)
{
    copy_matrix_cc_to_gm(dst, src, xm, xt);
}

template <typename T>
__aicore__ inline void CopyMatrixCcToGm(__gm__ T *dst, __cc__ float *src, uint8_t sid, uint16_t NSize, uint16_t MSize,
    uint32_t dstStride_dst_D, uint16_t srcStride, uint8_t UnitFlagMode, QuantMode_t QuantPRE, uint8_t ReLUPRE,
    bool channelSplit, bool NZ2ND_EN)
{
    copy_matrix_cc_to_gm(dst, src, sid, NSize, MSize, dstStride_dst_D, srcStride, UnitFlagMode, QuantPRE, ReLUPRE,
        channelSplit, NZ2ND_EN);
}

template <typename T>
__aicore__ inline void CopyMatrixCcToGm(__gm__ T *dst, __cc__ int32_t *src, uint8_t sid, uint16_t NSize, uint16_t MSize,
    uint32_t dstStride_dst_D, uint16_t srcStride, uint8_t UnitFlagMode, QuantMode_t QuantPRE, uint8_t ReLUPRE,
    bool channelSplit, bool NZ2ND_EN)
{
    copy_matrix_cc_to_gm(dst, src, sid, NSize, MSize, dstStride_dst_D, srcStride, UnitFlagMode, QuantPRE, ReLUPRE,
        channelSplit, NZ2ND_EN);
}

template <typename T> __aicore__ inline void CreateCaMatrix(__ca__ T *dst, int64_t repeat, half value)
{
    create_ca_matrix(dst, repeat, value);
}

template <typename T> __aicore__ inline void CreateCaMatrix(__ca__ T *dst, int64_t repeat, uint32_t value)
{
    create_ca_matrix(dst, repeat, value);
}

template <typename T> __aicore__ inline void CreateCbMatrix(__cb__ T *dst, int64_t repeat, half value)
{
    create_cb_matrix(dst, repeat, value);
}

template <typename T> __aicore__ inline void CreateCbMatrix(__cb__ T *dst, int64_t repeat, uint32_t value)
{
    create_cb_matrix(dst, repeat, value);
}

template <typename T> __aicore__ inline void CreateCbufMatrix(__cbuf__ T *dst, int64_t repeat, half value)
{
    create_cbuf_matrix(dst, repeat, value);
}

template <typename T> __aicore__ inline void CreateCbufMatrix(__cbuf__ T *dst, int64_t repeat, uint32_t value)
{
    create_cbuf_matrix(dst, repeat, value);
}

template <typename T>
__aicore__ inline void Img2colv2CbufToCa(__ca__ T *dst, __cbuf__ T *src, uint16_t stepK, uint16_t stepM, uint16_t posK,
    uint16_t posM, uint8_t strideW, uint8_t strideH, uint8_t Wk, uint8_t Hk, uint8_t dilationW, uint8_t dilationH,
    bool filterW, bool filterH, bool transpose, bool fmatrixCtrl, uint16_t sizeChannel)
{
    img2colv2_cbuf_to_ca(dst, src, stepK, stepM, posK, posM, strideW, strideH, Wk, Hk, dilationW, dilationH, filterW,
        filterH, transpose, fmatrixCtrl, sizeChannel);
}

template <typename T>
__aicore__ inline void Img2colv2CbufToCa(__ca__ T *dst, __cbuf__ T *src, uint64_t config0, uint64_t config1)
{
    img2colv2_cbuf_to_ca(dst, src, config0, config1);
}

template <typename T>
__aicore__ inline void Img2colv2CbufToCb(__cb__ T *dst, __cbuf__ T *src, uint16_t stepK, uint16_t stepM, uint16_t posK,
    uint16_t posM, uint8_t strideW, uint8_t strideH, uint8_t Wk, uint8_t Hk, uint8_t dilationW, uint8_t dilationH,
    bool filterW, bool filterH, bool transpose, bool fmatrixCtrl, uint16_t sizeChannel)
{
    img2colv2_cbuf_to_cb(dst, src, stepK, stepM, posK, posM, strideW, strideH, Wk, Hk, dilationW, dilationH, filterW,
        filterH, transpose, fmatrixCtrl, sizeChannel);
}

template <typename T>
__aicore__ inline void Img2colv2CbufToCb(__cb__ T *dst, __cbuf__ T *src, uint64_t config0, uint64_t config1)
{
    img2colv2_cbuf_to_cb(dst, src, config0, config1);
}

template <typename T>
__aicore__ inline void Img2colv2CbufToUb(__ubuf__ T *dst, __cbuf__ T *src, uint16_t stepK, uint16_t stepM,
    uint16_t posK, uint16_t posM, uint8_t strideW, uint8_t strideH, uint8_t Wk, uint8_t Hk, uint8_t dilationW,
    uint8_t dilationH, bool filterW, bool filterH, bool transpose, bool fmatrixCtrl, uint16_t sizeChannel)
{
    img2colv2_cbuf_to_ub(dst, src, stepK, stepM, posK, posM, strideW, strideH, Wk, Hk, dilationW, dilationH, filterW,
        filterH, transpose, fmatrixCtrl, sizeChannel);
}

template <typename T>
__aicore__ inline void Img2colv2CbufToUb(__ubuf__ T *dst, __cbuf__ T *src, uint64_t config0, uint64_t config1)
{
    img2colv2_cbuf_to_ub(dst, src, config0, config1);
}

template <typename T>
__aicore__ inline void LoadCbufToCa(__ca__ T *dst, __cbuf__ T *src, uint64_t config, bool transpose)
{
    load_cbuf_to_ca(dst, src, config, transpose);
}

template <typename T>
__aicore__ inline void LoadCbufToCa(__ca__ T *dst, __cbuf__ T *src, uint64_t config, bool transpose,
    addr_cal_mode_t addr_cal_mode)
{
    load_cbuf_to_ca(dst, src, config, transpose, addr_cal_mode);
}

template <typename T>
__aicore__ inline void LoadCbufToCa(__ca__ T *dst, __cbuf__ T *src, uint16_t baseIdx, uint8_t repeat,
    uint16_t srcStride, uint16_t dstStride, uint8_t sid, bool transpose, addr_cal_mode_t addr_cal_mode)
{
    load_cbuf_to_ca(dst, src, baseIdx, repeat, srcStride, dstStride, sid, transpose, addr_cal_mode);
}

template <typename T>
__aicore__ inline void LoadCbufToCa(__ca__ T *dst, __cbuf__ T *src, uint16_t baseIdx, uint8_t repeat,
    uint16_t srcStride, uint16_t dstStride, uint8_t sid, bool hw_wait_ctrl, bool transpose,
    addr_cal_mode_t addr_cal_mode)
{
    load_cbuf_to_ca(dst, src, baseIdx, repeat, srcStride, dstStride, sid, hw_wait_ctrl, transpose, addr_cal_mode);
}

template <typename T>
__aicore__ inline void LoadCbufToCa(__ca__ T *dst, __cbuf__ T *src, uint16_t baseIdx, uint8_t repeat,
    uint16_t srcStride, uint8_t sid, bool transpose)
{
    load_cbuf_to_ca(dst, src, baseIdx, repeat, srcStride, sid, transpose);
}

__aicore__ inline void LoadCbufToCaS4(__ca__ void *dst, __cbuf__ void *src, uint64_t config, bool transpose,
    addr_cal_mode_t addr_cal_mode)
{
    load_cbuf_to_ca_s4(dst, src, config, transpose, addr_cal_mode);
}

__aicore__ inline void LoadCbufToCaS4(__ca__ void *dst, __cbuf__ void *src, uint16_t baseIdx, uint8_t repeat,
    uint16_t srcStride, uint16_t dstStride, uint8_t sid, bool transpose, addr_cal_mode_t addr_cal_mode)
{
    load_cbuf_to_ca_s4(dst, src, baseIdx, repeat, srcStride, dstStride, sid, transpose, addr_cal_mode);
}

__aicore__ inline void LoadCbufToCaS4(__ca__ void *dst, __cbuf__ void *src, uint16_t baseIdx, uint8_t repeat,
    uint16_t srcStride, uint16_t dstStride, uint8_t sid, bool hw_wait_ctrl, bool transpose,
    addr_cal_mode_t addr_cal_mode)
{
    load_cbuf_to_ca_s4(dst, src, baseIdx, repeat, srcStride, dstStride, sid, hw_wait_ctrl, transpose, addr_cal_mode);
}

template <typename T>
__aicore__ inline void LoadCbufToCaTranspose(__ca__ T *dst, __cbuf__ T *src, uint64_t config, uint64_t fracStride)
{
    load_cbuf_to_ca_transpose(dst, src, config, fracStride);
}

template <typename T>
__aicore__ inline void LoadCbufToCaTranspose(__ca__ T *dst, __cbuf__ T *src, uint16_t indexID, uint8_t repeat,
    uint16_t srcStride, uint16_t dstStride, bool addrmode, uint16_t dstFracStride)
{
    load_cbuf_to_ca_transpose(dst, src, indexID, repeat, srcStride, dstStride, addrmode, dstFracStride);
}

template <typename T>
__aicore__ inline void LoadCbufToCb(__cb__ T *dst, __cbuf__ T *src, uint64_t config, bool transpose)
{
    load_cbuf_to_cb(dst, src, config, transpose);
}

template <typename T>
__aicore__ inline void LoadCbufToCb(__cb__ T *dst, __cbuf__ T *src, uint64_t config, bool transpose,
    addr_cal_mode_t addr_cal_mode)
{
    load_cbuf_to_cb(dst, src, config, transpose, addr_cal_mode);
}

template <typename T>
__aicore__ inline void LoadCbufToCb(__cb__ T *dst, __cbuf__ T *src, uint16_t baseIdx, uint8_t repeat,
    uint16_t srcStride, uint16_t dstStride, uint8_t sid, bool transpose, addr_cal_mode_t addr_cal_mode)
{
    load_cbuf_to_cb(dst, src, baseIdx, repeat, srcStride, dstStride, sid, transpose, addr_cal_mode);
}

template <typename T>
__aicore__ inline void LoadCbufToCb(__cb__ T *dst, __cbuf__ T *src, uint16_t baseIdx, uint8_t repeat,
    uint16_t srcStride, uint16_t dstStride, uint8_t sid, bool hw_wait_ctrl, bool transpose,
    addr_cal_mode_t addr_cal_mode)
{
    load_cbuf_to_cb(dst, src, baseIdx, repeat, srcStride, dstStride, sid, hw_wait_ctrl, transpose, addr_cal_mode);
}

template <typename T>
__aicore__ inline void LoadCbufToCb(__cb__ T *dst, __cbuf__ T *src, uint16_t baseIdx, uint8_t repeat,
    uint16_t srcStride, uint8_t sid, bool transpose)
{
    load_cbuf_to_cb(dst, src, baseIdx, repeat, srcStride, sid, transpose);
}

__aicore__ inline void LoadCbufToCbS4(__cb__ void *dst, __cbuf__ void *src, uint64_t config, bool transpose,
    addr_cal_mode_t addr_cal_mode)
{
    load_cbuf_to_cb_s4(dst, src, config, transpose, addr_cal_mode);
}

__aicore__ inline void LoadCbufToCbS4(__cb__ void *dst, __cbuf__ void *src, uint16_t baseIdx, uint8_t repeat,
    uint16_t srcStride, uint16_t dstStride, uint8_t sid, bool transpose, addr_cal_mode_t addr_cal_mode)
{
    load_cbuf_to_cb_s4(dst, src, baseIdx, repeat, srcStride, dstStride, sid, transpose, addr_cal_mode);
}

__aicore__ inline void LoadCbufToCbS4(__cb__ void *dst, __cbuf__ void *src, uint16_t baseIdx, uint8_t repeat,
    uint16_t srcStride, uint16_t dstStride, uint8_t sid, bool hw_wait_ctrl, bool transpose,
    addr_cal_mode_t addr_cal_mode)
{
    load_cbuf_to_cb_s4(dst, src, baseIdx, repeat, srcStride, dstStride, sid, hw_wait_ctrl, transpose, addr_cal_mode);
}

template <typename T>
__aicore__ inline void LoadCbufToCbTranspose(__cb__ T *dst, __cbuf__ T *src, uint64_t config, uint64_t fracStride)
{
    load_cbuf_to_cb_transpose(dst, src, config, fracStride);
}

template <typename T>
__aicore__ inline void LoadCbufToCbTranspose(__cb__ T *dst, __cbuf__ T *src, uint16_t indexID, uint8_t repeat,
    uint16_t srcStride, uint16_t dstStride, bool addrmode, uint16_t dstFracStride)
{
    load_cbuf_to_cb_transpose(dst, src, indexID, repeat, srcStride, dstStride, addrmode, dstFracStride);
}

__aicore__ inline void LoadCbufToCbSp(__cb__ int8_t *dst, __cbuf__ int8_t *src, uint64_t config)
{
    load_cbuf_to_cb_sp(dst, src, config);
}

__aicore__ inline void LoadCbufToCbSp(__cb__ int8_t *dst, __cbuf__ int8_t *src, uint16_t startID, uint8_t repeatTime)
{
    load_cbuf_to_cb_sp(dst, src, startID, repeatTime);
}

template <typename T> __aicore__ inline void LoadGmToCa(__ca__ T *dst, __gm__ T *src, uint64_t config)
{
    load_gm_to_ca(dst, src, config);
}

template <typename T>
__aicore__ inline void LoadGmToCa(__ca__ T *dst, __gm__ T *src, uint64_t config, addr_cal_mode_t addr_cal_mode)
{
    load_gm_to_ca(dst, src, config, addr_cal_mode);
}

template <typename T>
__aicore__ inline void LoadGmToCa(__ca__ T *dst, __gm__ T *src, uint16_t baseIdx, uint8_t repeat, uint16_t srcStride,
    uint16_t dstStride, uint8_t sid, addr_cal_mode_t addr_cal_mode)
{
    load_gm_to_ca(dst, src, baseIdx, repeat, srcStride, dstStride, sid, addr_cal_mode);
}

template <typename T>
__aicore__ inline void LoadGmToCa(__ca__ T *dst, __gm__ T *src, uint16_t baseIdx, uint8_t repeat, uint16_t srcStride,
    uint16_t dstStride, uint8_t sid, bool hw_wait_ctrl, addr_cal_mode_t addr_cal_mode)
{
    load_gm_to_ca(dst, src, baseIdx, repeat, srcStride, dstStride, sid, hw_wait_ctrl, addr_cal_mode);
}

template <typename T>
__aicore__ inline void LoadGmToCa(__ca__ T *dst, __gm__ T *src, uint16_t baseIdx, uint8_t repeat, uint16_t srcStride,
    uint8_t sid)
{
    load_gm_to_ca(dst, src, baseIdx, repeat, srcStride, sid);
}

template <typename T> __aicore__ inline void LoadGmToCb(__cb__ T *dst, __gm__ T *src, uint64_t config)
{
    load_gm_to_cb(dst, src, config);
}

template <typename T>
__aicore__ inline void LoadGmToCb(__cb__ T *dst, __gm__ T *src, uint64_t config, addr_cal_mode_t addr_cal_mode)
{
    load_gm_to_cb(dst, src, config, addr_cal_mode);
}

template <typename T>
__aicore__ inline void LoadGmToCb(__cb__ T *dst, __gm__ T *src, uint16_t baseIdx, uint8_t repeat, uint16_t srcStride,
    uint16_t dstStride, uint8_t sid, addr_cal_mode_t addr_cal_mode)
{
    load_gm_to_cb(dst, src, baseIdx, repeat, srcStride, dstStride, sid, addr_cal_mode);
}

template <typename T>
__aicore__ inline void LoadGmToCb(__cb__ T *dst, __gm__ T *src, uint16_t baseIdx, uint8_t repeat, uint16_t srcStride,
    uint16_t dstStride, uint8_t sid, bool hw_wait_ctrl, addr_cal_mode_t addr_cal_mode)
{
    load_gm_to_cb(dst, src, baseIdx, repeat, srcStride, dstStride, sid, hw_wait_ctrl, addr_cal_mode);
}

template <typename T>
__aicore__ inline void LoadGmToCb(__cb__ T *dst, __gm__ T *src, uint16_t baseIdx, uint8_t repeat, uint16_t srcStride,
    uint8_t sid)
{
    load_gm_to_cb(dst, src, baseIdx, repeat, srcStride, sid);
}

// cbuf
template <typename T> __aicore__ inline void LoadGmToCbuf(__cbuf__ T *dst, __gm__ T *src, uint64_t config)
{
    load_gm_to_cbuf(dst, src, config);
}

template <typename T>
__aicore__ inline void LoadGmToCbuf(__cbuf__ T *dst, __gm__ T *src, uint64_t config, addr_cal_mode_t addr_cal_mode)
{
    load_gm_to_cbuf(dst, src, config, addr_cal_mode);
}

template <typename T>
__aicore__ inline void LoadGmToCbuf(__cbuf__ T *dst, __gm__ T *src, uint16_t baseIdx, uint8_t repeat,
    uint16_t srcStride, uint16_t dstStride, uint8_t sid, addr_cal_mode_t addr_cal_mode)
{
    load_gm_to_cbuf(dst, src, baseIdx, repeat, srcStride, dstStride, sid, addr_cal_mode);
}

template <typename T>
__aicore__ inline void LoadGmToCbuf(__cbuf__ T *dst, __gm__ T *src, uint16_t baseIdx, uint8_t repeat,
    uint16_t srcStride, uint16_t dstStride, uint8_t sid, bool hw_wait_ctrl, addr_cal_mode_t addr_cal_mode)
{
    load_gm_to_cbuf(dst, src, baseIdx, repeat, srcStride, dstStride, sid, hw_wait_ctrl, addr_cal_mode);
}

template <typename T>
__aicore__ inline void LoadGmToCbuf(__cbuf__ T *dst, __gm__ T *src, uint16_t baseIdx, uint8_t repeat,
    uint16_t srcStride, uint8_t sid)
{
    load_gm_to_cbuf(dst, src, baseIdx, repeat, srcStride, sid);
}

template <typename T> __aicore__ inline void LoadImageToCbuf(__cbuf__ T *dst, uint64_t xs, uint64_t xt)
{
    load_image_to_cbuf(dst, xs, xt);
}

template <typename T>
__aicore__ inline void LoadImageToCbuf(__cbuf__ T *dst, uint16_t horSize, uint16_t verSize, uint16_t horStartP,
    uint16_t verStartP, uint16_t sHorRes, uint8_t topPadSize, uint8_t botPadSize, uint16_t lPadSize, uint16_t rPadSize,
    uint8_t sid)
{
    load_image_to_cbuf(dst, horSize, verSize, horStartP, verStartP, sHorRes, topPadSize, botPadSize, lPadSize, rPadSize,
        sid);
}

template <typename T> __aicore__ inline void Mad(__cc__ float *c, __ca__ T *a, __cb__ T *b, uint64_t config)
{
    mad(c, a, b, config);
}

template <typename T>
__aicore__ inline void Mad(__cc__ float *c, __ca__ T *a, __cb__ T *b, uint16_t m, uint16_t k, uint16_t n,
    uint8_t unitFlag, bool kDirectionAlign, bool cmatrixSource, bool cmatrixInitVal)
{
    mad(c, a, b, m, k, n, unitFlag, kDirectionAlign, cmatrixSource, cmatrixInitVal);
}

template <typename T>
__aicore__ inline void Mad(__cc__ float *dst, __ca__ T *src0, __cb__ T *src1, uint16_t m, uint16_t k, uint16_t n,
    uint8_t featOffset, uint8_t smaskOffset, uint8_t unitFlag, bool kDirectionAlign, bool isWeightOffset,
    bool cmatrixSource, bool cmatrixInitVal)
{
    mad(dst, src0, src1, m, k, n, featOffset, smaskOffset, unitFlag, kDirectionAlign, isWeightOffset, cmatrixSource,
        cmatrixInitVal);
}

template <typename T> __aicore__ inline void Mad(__cc__ int32_t *c, __ca__ T *a, __cb__ T *b, uint64_t config)
{
    mad(c, a, b, config);
}

template <typename T>
__aicore__ inline void Mad(__cc__ int32_t *c, __ca__ T *a, __cb__ T *b, uint16_t m, uint16_t k, uint16_t n,
    uint8_t unitFlag, bool kDirectionAlign, bool cmatrixSource, bool cmatrixInitVal)
{
    mad(c, a, b, m, k, n, unitFlag, kDirectionAlign, cmatrixSource, cmatrixInitVal);
}

template <typename T>
__aicore__ inline void Mad(__cc__ int32_t *dst, __ca__ T *src0, __cb__ T *src1, uint16_t m, uint16_t k, uint16_t n,
    uint8_t featOffset, uint8_t smaskOffset, uint8_t unitFlag, bool kDirectionAlign, bool isWeightOffset,
    bool cmatrixSource, bool cmatrixInitVal)
{
    mad(dst, src0, src1, m, k, n, featOffset, smaskOffset, unitFlag, kDirectionAlign, isWeightOffset, cmatrixSource,
        cmatrixInitVal);
}

__aicore__ inline void SetFmatrixB(uint64_t config)
{
    set_fmatrix_b(config);
}

__aicore__ inline void SetL0SetValue(half config)
{
    set_l0_set_value_h(config);
}

__aicore__ inline void SetL0SetValue(__bf16 config)
{
    set_l0_set_value_bf16(config);
}

template <typename T> __aicore__ inline void SetL0A2D(__ca__ T *dst, int64_t config)
{
    set_l0a_2d(dst, config);
}

template <typename T> __aicore__ inline void SetL0B2D(__cb__ T *dst, int64_t config)
{
    set_l0b_2d(dst, config);
}

template <typename T> __aicore__ inline void SetL12D(__cbuf__ T *dst, int64_t config)
{
    set_l1_2d(dst, config);
}

__aicore__ inline void SetL3DRPT(uint64_t config)
{
    set_l3d_rpt(config);
}

template <typename T> __aicore__ inline void SetLreluALPHA(T config)
{
    set_lrelu_alpha(config);
}

__aicore__ inline void SetNdPARA(uint64_t config)
{
    set_nd_para(config);
}

__aicore__ inline void SetQuantPRE(uint64_t config)
{
    set_quant_pre(config);
}
} // namespace AscendC
#endif
#endif // ASCENDC_MODULE_OPERATOR_CUBE_OTHERS_IMPL_H