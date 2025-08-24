/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
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
 * \file swin_attention_score_split_b_n.h
 * \brief
 */
#ifndef __SWIN_ATTENTION_SCORE_SPLIT_B_N_H__
#define __SWIN_ATTENTION_SCORE_SPLIT_B_N_H__

#include "kernel_operator.h"
#include "swin_batchmatmul.h"

namespace swin {

using namespace AscendC;
using namespace matmul;

template<typename T, typename OT>
class SwinAttentionScore {
public:
    __aicore__ inline SwinAttentionScore() {};
    __aicore__ inline void Init(__gm__ uint8_t*  query, __gm__ uint8_t*  key, __gm__ uint8_t*  value, __gm__ uint8_t*  input_mask1,
        __gm__ uint8_t*  sm, __gm__ uint8_t*  attentionOut, __gm__ uint8_t*  workspace, const SwinAttentionScoreTilingData* tiling, TPipe* tPipe);
    __aicore__ inline void Process();

#if defined(__DAV_C220_CUBE__)
    // define qk_bmm
    typedef BatchMatmulType<TPosition::GM, CubeFormat::ND, T> qType;
    typedef BatchMatmulType<TPosition::GM, CubeFormat::ND, T> kType;
    typedef BatchMatmulType<TPosition::GM, CubeFormat::ND, T> qkType;
    BatchMatmulImpl<qType, kType, qkType, qkType> qk_bmm;

    // define sv_bmm
    typedef BatchMatmulType<TPosition::GM, CubeFormat::ND, T> sType;
    typedef BatchMatmulType<TPosition::GM, CubeFormat::ND, T> vType;
    typedef BatchMatmulType<TPosition::GM, CubeFormat::ND, T> scoreType;
    BatchMatmulImpl<sType, vType, scoreType, scoreType> sv_bmm;
#endif

#if defined(__DAV_C220_VEC__)
    TQue<QuePosition::VECIN, 1> qkResQueue;
    TQue<QuePosition::VECIN, 1> softmaxTmpQueue;
    TQue<QuePosition::VECIN, 1> biasInQueue;
    TBuf<> tmpSoftmaxUb;
    T scale_val;
#endif

protected:
    const SwinAttentionScoreTilingData* tilingData;
    TPipe* pipe;
    // define the que
    GlobalTensor<T> queryGm;
    GlobalTensor<T> keyGm;
    GlobalTensor<T> valueGm;
    GlobalTensor<T> attenMaskGm;
    GlobalTensor<T> scaleGm;
    GlobalTensor<T> attentionOutGm;
    GlobalTensor<T> workspaceGm;

    uint32_t core_idx;
    uint32_t aiv_idx;
    uint32_t this_block_idx;
    uint32_t this_head_idx;
    uint32_t ss_size;
    uint32_t sh_size;
    uint32_t nsh_size;
    uint32_t nss_size;

    uint64_t sconfigs[N_SYNC * 2];
    uint64_t sid[N_SYNC * 2];

    template <bool is_first = false>
    __aicore__ inline void Stage1BmmQK(int idx, int flag);

    template <bool is_first = false>
    __aicore__ inline void Stage2VecAddSoftmax(int idx, int flag,
        LocalTensor<T> &qkResUb, LocalTensor<T> &biasUb,
        LocalTensor<T> &qkResUb2, LocalTensor<T> &softmaxTmpUb, LocalTensor<T> &softmaxMaxUb);
    template <bool is_first = false, bool is_last = false>
    __aicore__ inline void Stage3BMMSV(int idx, int flag);
};

template<typename T, typename OT>
__aicore__ inline void SwinAttentionScore<T, OT>::Init(__gm__ uint8_t*  query, __gm__ uint8_t*  key, __gm__ uint8_t*  value, __gm__ uint8_t*  input_mask1,
        __gm__ uint8_t* sm, __gm__ uint8_t*  attentionOut, __gm__ uint8_t*  workspace, const SwinAttentionScoreTilingData* tiling, TPipe* tPipe) {
    core_idx = get_block_idx();
    aiv_idx = get_subblockid();

    // init global buffer
    queryGm.SetGlobalBuffer((__gm__ T*)query);
    keyGm.SetGlobalBuffer((__gm__ T*)key);
    valueGm.SetGlobalBuffer((__gm__ T*)value);
    attenMaskGm.SetGlobalBuffer((__gm__ OT*)input_mask1);
    scaleGm.SetGlobalBuffer((__gm__ T*)sm);

    workspaceGm.SetGlobalBuffer((__gm__ T*)workspace);
    attentionOutGm.SetGlobalBuffer((__gm__ T*)attentionOut);
    
    tilingData = tiling;
    pipe = tPipe;

    ss_size = tilingData->dimS * tilingData->dimS;
    sh_size = tilingData->dimS * tilingData->dimH;
    nsh_size = tilingData->totalHead * sh_size;
    nss_size = tilingData->totalHead * ss_size;

#ifdef __DAV_C220_CUBE__
    qk_bmm.Init(tilingData->dimS, tilingData->dimS, tilingData->dimH, tilingData->batchPerLoop, tPipe, EVENT_ID2, 2);
    sv_bmm.Init(tilingData->dimS, tilingData->dimH, tilingData->dimS, tilingData->batchPerLoop, tPipe, EVENT_ID3);
#endif

#ifdef __DAV_C220_VEC__
    pipe->InitBuffer(qkResQueue, 2, tilingData->vectorPerLoop * ss_size * sizeof(T));
    pipe->InitBuffer(softmaxTmpQueue, 1, tilingData->batchPerBrch * ss_size * sizeof(T));
    pipe->InitBuffer(biasInQueue, 1, tilingData->totalHead * ss_size * sizeof(T));
    pipe->InitBuffer(tmpSoftmaxUb, tilingData->batchPerLoop * tilingData->dimS * sizeof(T) * 2);
#endif

    uint64_t mode = 2; // inner-group aic/aiv sync
    for (uint64_t i = 0;i < N_SYNC * 2; i++) {
        sid[i] = i + 4;
        sconfigs[i] = 1 | (mode << 4) | (sid[i] << 8);
    }
}

// setup an automatic sync approach?
template<typename T, typename OT>
__aicore__ inline void SwinAttentionScore<T, OT>::Process() {

    int last_idxs = tilingData->totalBatch * tilingData->totalHead;
    int step = tilingData->batchPerLoop; // for schedule-2

#ifdef __CCE_KT_TEST__
    last_idxs = 16 * 8;
    if (aiv_idx == 1) return;
#endif
    // MAKESURE: last_idxs % step == 0
    int per_core_idxs = (last_idxs / step + (tilingData->usedCores - 1)) / tilingData->usedCores;
    int from_idx = per_core_idxs * (core_idx) * step;
    int to_idx = per_core_idxs * (core_idx + 1) * step;
    if (to_idx >= last_idxs) {
        to_idx = last_idxs;
    }

    // if this is not even at all
    if (from_idx > last_idxs) {
        return;
    }

    // repeat computation to fill the pipeline
    if (to_idx - from_idx == 2 * step) {
        from_idx -= step;
    } else if (to_idx - from_idx == step) {
        from_idx -= (2 * step);
    }

    uint32_t qkResShapeArray[] = {tilingData->vectorPerLoop * tilingData->dimS, tilingData->dimS};
    uint32_t biasShapeArray[] = {tilingData->totalHead * tilingData->dimS, tilingData->dimS};
    uint32_t tmpSoftmaxShapeArray[] = {tilingData->batchPerBrch * tilingData->dimS, tilingData->dimS};
    uint32_t tmpShapeArray[] = {tilingData->vectorPerLoop * tilingData->dimS, 1};

    LocalTensor<T> qkResUb[2];
    LocalTensor<T> biasUb;
    LocalTensor<T> softmaxMaxUb;
    LocalTensor<T> softmaxSumUb;
    LocalTensor<T> softmaxTmpUb;
    LocalTensor<T> softmaxReduceTmpUb;

#ifdef __DAV_C220_VEC__
        for (int b = 0; b < 2; b++) {
            qkResUb[b] = qkResQueue.AllocTensor<T>();
        }
        softmaxReduceTmpUb = softmaxTmpQueue.AllocTensor<T>();
        biasUb = biasInQueue.AllocTensor<T>();

        softmaxTmpUb = tmpSoftmaxUb.Get<T>(2 * tilingData->vectorPerLoop * tilingData->dimS * 2);
        softmaxMaxUb = softmaxTmpUb[2 * tilingData->vectorPerLoop * tilingData->dimS];

        DataCopy(biasUb, attenMaskGm, tilingData->totalHead * ss_size);

        // copy data to scale
        scale_val = scaleGm.GetValue(0);
#endif
    
    /*
     * Using cross overlapping vector-cube computation
     * For vector-time / cube-time = 2, the benefit over sequential computation is 1/2
     */
    to_idx = to_idx - tilingData->batchPerLoop * 2;
    int flag_idx = 0;

    // 1. warn up stages
    Stage1BmmQK<true>(from_idx, flag_idx);
#ifdef __DAV_C220_CUBE__
        set_flag(PIPE_FIX, PIPE_MTE2, EVENT_ID2);
        wait_flag(PIPE_FIX, PIPE_MTE2, EVENT_ID2);
#endif

    Stage1BmmQK(from_idx + tilingData->batchPerLoop, flag_idx + 1);

#ifdef __DAV_C220_VEC__
        Stage2VecAddSoftmax<true>(from_idx, flag_idx, qkResUb[1], biasUb, softmaxReduceTmpUb, softmaxTmpUb, softmaxMaxUb);
#endif

#ifdef __DAV_C220_CUBE__
        set_flag(PIPE_FIX, PIPE_MTE2, EVENT_ID2);
        wait_flag(PIPE_FIX, PIPE_MTE2, EVENT_ID2);
#endif

    Stage1BmmQK(from_idx + tilingData->batchPerLoop * 2, flag_idx + 2);
    Stage3BMMSV<true, false>(from_idx, flag_idx);

#ifdef __DAV_C220_VEC__
        Stage2VecAddSoftmax<true>(from_idx + tilingData->batchPerLoop, flag_idx + 1, qkResUb[(flag_idx % 2)], biasUb, softmaxReduceTmpUb, softmaxTmpUb, softmaxMaxUb);
#endif

    flag_idx += 1;

    // 2. stable stages
    for (int idx = from_idx + tilingData->batchPerLoop; idx < to_idx; idx += step, flag_idx += 1) {
        Stage1BmmQK(idx + tilingData->batchPerLoop * 2, flag_idx + 2);
        Stage3BMMSV(idx, flag_idx);
#ifdef __DAV_C220_VEC__
            Stage2VecAddSoftmax(idx + tilingData->batchPerLoop, flag_idx + 1, qkResUb[(flag_idx % 2)], biasUb, softmaxReduceTmpUb, softmaxTmpUb, softmaxMaxUb);
#endif
    }

    // 3. cool down stages
    Stage3BMMSV<false, true>(to_idx, flag_idx);
#ifdef __DAV_C220_VEC__
        Stage2VecAddSoftmax(to_idx + tilingData->batchPerLoop, flag_idx + 1, qkResUb[(flag_idx % 2)], biasUb, softmaxReduceTmpUb, softmaxTmpUb, softmaxMaxUb);
        wait_flag(PIPE_MTE3, PIPE_MTE2, 0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, 1);
#endif
    Stage3BMMSV<false, true>(to_idx + tilingData->batchPerLoop, flag_idx + 1);
}
template<typename T, typename OT>
template <bool is_first>
__aicore__ inline void SwinAttentionScore<T, OT>::Stage1BmmQK(int idx, int flag) {
#ifdef __DAV_C220_CUBE__
        // SPEEDUP: overlap QK loading, and SV loading
        auto cube_flag = (flag % 2);
        flag = (flag % N_SYNC);

        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);

        qk_bmm.CopyTensorA(queryGm[(idx) * sh_size], false, cube_flag);
        qk_bmm.CopyTensorB(keyGm[(idx) * sh_size], true, cube_flag);

        set_flag(PIPE_M, PIPE_MTE1, cube_flag);
        wait_flag(PIPE_M, PIPE_MTE1, cube_flag);
        set_flag(PIPE_MTE2, PIPE_MTE1, cube_flag);
        wait_flag(PIPE_MTE2, PIPE_MTE1, cube_flag);
        qk_bmm.template Iterate(false, cube_flag);

        auto work_idx = (core_idx * N_SYNC * 2 + flag) * tilingData->batchPerLoop * ss_size;
        qk_bmm.GetTensorC(workspaceGm[work_idx], false, true);
        qk_bmm.End();
        ffts_cross_core_sync(PIPE_FIX, sconfigs[flag * 2]);
#endif
}

template<typename T, typename OT>
template <bool is_first>
__aicore__ inline void SwinAttentionScore<T, OT>::Stage2VecAddSoftmax(int idx, int flag,
    LocalTensor<T> &qkResUb, LocalTensor<T> &biasUb,
    LocalTensor<T> &qkResUb2, LocalTensor<T> &softmaxTmpUb, LocalTensor<T> &softmaxMaxUb) {
    // wait for cube to finish
#if defined(__DAV_C220_VEC__)

    auto vec_flag = (flag % 2);
    flag = (flag % N_SYNC);

    // do sync
    wait_flag_dev(sid[flag * 2]);

    if (!is_first) {
        wait_flag(PIPE_MTE3, PIPE_MTE2, vec_flag);
    }

    auto work_idx = ((core_idx * N_SYNC * 2 + flag) * tilingData->batchPerLoop
        + aiv_idx * tilingData->vectorPerLoop) * ss_size;
    DataCopy(qkResUb, workspaceGm[work_idx], tilingData->vectorPerLoop * ss_size);

    set_flag(PIPE_MTE2, PIPE_V, vec_flag);
    wait_flag(PIPE_MTE2, PIPE_V, vec_flag);
    Muls(qkResUb, qkResUb, static_cast<T>(scale_val), tilingData->vectorPerLoop * ss_size);

    if (tilingData->headPerLoop > 0) {
    for (int i = 0;i < tilingData->headPerLoop;i++) {
        Add(qkResUb[tilingData->totalHead * ss_size * i], biasUb, qkResUb[tilingData->totalHead * ss_size * i], tilingData->totalHead * ss_size);
    }
    } else {
        auto head_idx = (idx % tilingData->totalHead);
        head_idx = (head_idx / tilingData->batchPerHead);
        Add(qkResUb, biasUb[(head_idx * tilingData->batchPerHead + aiv_idx * tilingData->vectorPerLoop) * ss_size], qkResUb, tilingData->vectorPerLoop * ss_size);
    }

    /*
     * Softmax impl
     */
    const int32_t MAX_REPEAT_TIMES = tilingData->reduceRepeat;
    uint32_t elementNumPerBlk = ONE_BLK_SIZE / sizeof(T);         // half=16 float=8
    uint32_t elementNumPerRep = ONE_REPEAT_BYTE_SIZE / sizeof(T); // fp16=128 , fp32=64

    auto mask = tilingData->dimS;
    auto range = tilingData->vectorPerLoop * tilingData->dimS / MAX_REPEAT_TIMES;
    auto tail = 0;

        for (uint32_t i = 0; i < range; i++) {
            // first save to tmptensor上 vcmax onlyvalue mode , dst repstride unit =2B/4B,
            WholeReduceMax(softmaxMaxUb[i * MAX_REPEAT_TIMES], qkResUb[i * MAX_REPEAT_TIMES * tilingData->dimS], mask,
                MAX_REPEAT_TIMES, 1, 1, tilingData->dimS / elementNumPerBlk, ReduceOrder::ORDER_ONLY_VALUE);
        }


    // tmptensor brcb to dstmax
    BrcbRepeatParams brcbParams;

    const int brcbSize = 32;
    const int brcbElements = brcbSize / sizeof(T);
    const int nBrcbs = tilingData->dimS * sizeof(T) / brcbSize;
    brcbParams.dstBlkStride = nBrcbs;
    brcbParams.dstRepStride = BRCB_BROADCAST_NUMBER * nBrcbs;
    auto batchPerBrcb = tilingData->batchPerBrch;
    auto bradcast_range = batchPerBrcb * tilingData->dimS / BRCB_BROADCAST_NUMBER;

    for (int s = 0; s < tilingData->brcbPerLoop; s++) {
        // loop over 64 / 16 = 4 times
        for (int i = 0;i < nBrcbs;i++)
            Brcb(qkResUb2[brcbElements * i], softmaxMaxUb[batchPerBrcb * tilingData->dimS * s], bradcast_range, brcbParams);

        Sub(qkResUb[batchPerBrcb * ss_size * s], qkResUb[batchPerBrcb * ss_size * s], qkResUb2, batchPerBrcb * ss_size);
    }

    Exp(qkResUb, qkResUb, tilingData->vectorPerLoop * ss_size); // exp(x-max)

    // sum{exp(x-max)}
    for (uint32_t i = 0; i < range; i++) {
        // first save to tmptensor上 vcmax onlyvalue mode , dst repstride unit =2B/4B,
        WholeReduceSum(softmaxTmpUb[i * MAX_REPEAT_TIMES], qkResUb[i * MAX_REPEAT_TIMES * tilingData->dimS], mask,
            MAX_REPEAT_TIMES, 1, 1, tilingData->dimS / elementNumPerBlk);
    }

    for (int s = 0; s < tilingData->brcbPerLoop; s++) {
        for (int i = 0;i < nBrcbs;i++)
            Brcb(qkResUb2[brcbElements * i], softmaxTmpUb[batchPerBrcb * tilingData->dimS * s], bradcast_range, brcbParams);

        // exp(x-max) / sum{exp(x-max)}
        Div(qkResUb[batchPerBrcb * ss_size * s], qkResUb[batchPerBrcb * ss_size * s], qkResUb2, batchPerBrcb * ss_size); // sum{exp(x)}
    }

    set_flag(PIPE_V, PIPE_MTE3, vec_flag);
    wait_flag(PIPE_V, PIPE_MTE3, vec_flag);

    DataCopy(workspaceGm[work_idx], qkResUb, tilingData->vectorPerLoop * ss_size);

    set_flag(PIPE_MTE3, PIPE_MTE2, vec_flag);
    ffts_cross_core_sync(PIPE_MTE3, sconfigs[flag * 2 + 1]);
#endif
}

template<typename T, typename OT>
template <bool is_first, bool is_last>
__aicore__ inline void SwinAttentionScore<T, OT>::Stage3BMMSV(int idx, int flag) {
    // copy data from GM to ub
#ifdef __DAV_C220_CUBE__
        // should wait previous MTE1, emmit it here
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID4);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID4);
        flag = (flag % N_SYNC);
        sv_bmm.CopyTensorB(valueGm[(idx) * sh_size]);
        // preload v into t0b
        // 提前v的preload,先注释掉效果看看如何，这里可能可以减少5-10us
        wait_flag_dev(sid[flag * 2 + 1]);

        auto work_idx = (core_idx * N_SYNC * 2 + flag) * tilingData->batchPerLoop * ss_size;
        sv_bmm.CopyTensorA(workspaceGm[work_idx]);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID7);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID7);
        sv_bmm.Iterate();
        sv_bmm.GetTensorC(attentionOutGm[(idx) * sh_size], false, true);
        sv_bmm.End();
#endif
}
} // namespace swin

#endif // __SWIN_ATTENTION_SCORE_SPLIT_B_N_H__