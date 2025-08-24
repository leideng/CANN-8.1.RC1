/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
 * \file vec.h
 * \brief
 */
#ifndef ATP_VEC_H_
#define ATP_VEC_H_
#include "elems.h"

#ifdef __CCE_AICORE__
#include "kernel_operator.h"
using namespace AscendC;
#else
// CPU编译过程中需要Mock掉Kernel侧的实现
#define __aicore__
#define __global__
#define __gm__
struct half {
  uint16_t value;
};
struct bfloat16_t {
  uint16_t value;
};
template <class T>
struct LocalTensor {};

template <class T>
struct GlobalTensor {
  void SetGlobalBuffer(T* y) {}
};
#endif
// 只是一个数据类型定义
struct uint1_t {};

namespace AscendC {
namespace Vec {

template <int outSize, int inSize, int tempSize = 0, int fixedBuf = 0, class... DataType_>
struct FunBase {
  constexpr static int OutputSize = outSize;
  constexpr static int InputSize = inSize;
  constexpr static int TempSize = tempSize;
  constexpr static int FixedSize = fixedBuf;

  // 数据类型列表
  using DataTypes = Elems<DataType_...>;

  // 输入的数据类型列表
  template <int offset>
  using FunRetArgType = typename DataTypes::template At<offset>;

  // 输出的数据类型列表
  template <int offset>
  using FunInArgType = typename DataTypes::template At<outSize + offset>;
};

// 单输入
template <class R, class In1, int tempBufSize = 0, int fixedBuf = 0>
struct ElemwiseUnaryOP : public FunBase<1, 1, tempBufSize, fixedBuf, R, In1> {};

// 双输入
template <class R, class In1, class In2, int tempBufSize = 0, int fixedBuf = 0>
struct ElemwiseBinaryOP : public FunBase<1, 2, tempBufSize, fixedBuf, R, In1, In2> {};

// 三输入
template <class R, class In1, class In2, class In3, int tempBufSize = 0, int fixedBuf = 0>
struct ElemwiseTernaryOP : public FunBase<1, 3, tempBufSize, fixedBuf, R, In1, In2, In3> {};

// 特殊场景，7OP和9OP
constexpr static int32_t OP_NUM_4T3S = 7;
constexpr static int32_t OP_NUM_5T4S = 9;

template <class R, class In1, class In2, class In3, class In4, class In5, class In6, class In7,
          int32_t tempBufSize = 0, int32_t fixedBuf = 0>
struct Elemwise7OP : public FunBase<1, OP_NUM_4T3S, tempBufSize, fixedBuf, R, In1, In2, In3, In4, In5, In6, In7> {};

template <class R, class In1, class In2, class In3, class In4, class In5, class In6, class In7, class In8, class In9,
          int32_t tempBufSize = 0, int32_t fixedBuf = 0>
struct Elemwise9OP : public FunBase<1, OP_NUM_5T4S, tempBufSize, fixedBuf, R, In1, In2, In3, In4, In5, In6, In7, In8, In9> {};

template <class T>
struct Duplicate : public ElemwiseUnaryOP<T, T> {
  __aicore__ inline Duplicate(const LocalTensor<T>& dstLocal, const T& scalar, const int32_t& count) {
#ifdef __CCE_AICORE__
    AscendC::Duplicate(dstLocal, scalar, count);
#endif
  }
};

template <class T>
struct Reciprocal : public ElemwiseUnaryOP<T, T> {
  __aicore__ inline Reciprocal(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src, const int32_t& count) {
#ifdef __CCE_AICORE__
    AscendC::Reciprocal(dstLocal, src, count);
#endif
  }
};

template <class T>
struct Log : public ElemwiseUnaryOP<T, T> {
  __aicore__ inline Log(LocalTensor<T>& dst, LocalTensor<T>& src, int count) {
#ifdef __CCE_AICORE__
    AscendC::Log(dst, src, count);
#endif
  }
};

template <class T>
struct Abs : public ElemwiseUnaryOP<T, T> {
  __aicore__ inline Abs(LocalTensor<T>& dst, LocalTensor<T>& src, int count) {
#ifdef __CCE_AICORE__
    AscendC::Abs(dst, src, count);
#endif
  }
};

template <class R, class T, int roundMode>
struct Cast : public ElemwiseUnaryOP<R, T> {
  __aicore__ inline Cast(LocalTensor<R>& dst, LocalTensor<T>& src, const uint32_t& count) {
#ifdef __CCE_AICORE__
    AscendC::Cast(dst, src, static_cast<AscendC::RoundMode>(roundMode), count);
#endif
  }
};

template <class T>
struct Exp : public ElemwiseUnaryOP<T, T> {
  __aicore__ inline Exp(LocalTensor<T>& dst, LocalTensor<T>& src, int count) {
#ifdef __CCE_AICORE__
    AscendC::Exp(dst, src, count);
#endif
  }
};

template <class T>
struct Sqrt : public ElemwiseUnaryOP<T, T> {
  __aicore__ inline Sqrt(LocalTensor<T>& dst, LocalTensor<T>& src, int count) {
#ifdef __CCE_AICORE__
    AscendC::Sqrt(dst, src, count);
#endif
  }
};

template <class T>
struct ReduceOp : public ElemwiseUnaryOP<T, T> {
  __aicore__ inline ReduceOp(LocalTensor<T>& dst, LocalTensor<T>& src, int count) {
#ifdef __CCE_AICORE__
    (void)dst;
    (void)src;
    (void)count;
#endif
  }
};

template <class T, int roundMode>
struct Truncate : public ElemwiseUnaryOP<T, T> {
  __aicore__ inline Truncate(LocalTensor<T>& dst, LocalTensor<T>& src, const uint32_t& count) {
#ifdef __CCE_AICORE__
    AscendC::Truncate<T, static_cast<AscendC::RoundMode>(roundMode)>(dst, src, count);
#endif
  }
};

template <class T>
struct Add : public ElemwiseBinaryOP<T, T, T> {
  __aicore__ inline Add(const LocalTensor<T>& dst, const LocalTensor<T>& src1, LocalTensor<T>& src2, int count) {
#ifdef __CCE_AICORE__
    AscendC::Add(dst, src1, src2, count);
#endif
  }
};

template <class T>
struct Adds : public ElemwiseBinaryOP<T, T, T> {
  __aicore__ inline Adds(LocalTensor<T>& dst, LocalTensor<T>& src, T& scalar, int count) {
#ifdef __CCE_AICORE__
    AscendC::Adds(dst, src, scalar, count);
#endif
  }
  __aicore__ inline Adds(LocalTensor<T>& dst, T& scalar, LocalTensor<T>& src, int count) {
#ifdef __CCE_AICORE__
    AscendC::Adds(dst, src, scalar, count);
#endif
  }
};

template <class T>
struct Relu : public ElemwiseBinaryOP<T, T, T> {
  __aicore__ inline Relu(const LocalTensor<T>& dst, const LocalTensor<T>& src, int32_t count) {
#ifdef __CCE_AICORE__
    AscendC::Relu(dst, src, count);
#endif
  }
};

template <class T>
struct Max : public ElemwiseBinaryOP<T, T, T> {
  __aicore__ inline Max(const LocalTensor<T>& dst, const LocalTensor<T>& src1, LocalTensor<T>& src2, int count) {
#ifdef __CCE_AICORE__
    AscendC::Max(dst, src1, src2, count);
#endif
  }
};

template <class T>
struct Maxs : public ElemwiseBinaryOP<T, T, T> {
  __aicore__ inline Maxs(LocalTensor<T>& dst, LocalTensor<T>& src, T& scalar, int count) {
#ifdef __CCE_AICORE__
    AscendC::Maxs(dst, src, scalar, count);
#endif
  }
  __aicore__ inline Maxs(LocalTensor<T>& dst, T& scalar, LocalTensor<T>& src, int count) {
#ifdef __CCE_AICORE__
    AscendC::Duplicate(dst, scalar, count);
    AscendC::Max(dst, dst, src, count);
#endif
  }
};

template <class T>
struct Min : public ElemwiseBinaryOP<T, T, T> {
  __aicore__ inline Min(const LocalTensor<T>& dst, const LocalTensor<T>& src1, LocalTensor<T>& src2, int count) {
#ifdef __CCE_AICORE__
    AscendC::Min(dst, src1, src2, count);
#endif
  }
};

template <class T>
struct Mins : public ElemwiseBinaryOP<T, T, T> {
  __aicore__ inline Mins(LocalTensor<T>& dst, LocalTensor<T>& src, T& scalar, int count) {
#ifdef __CCE_AICORE__
    AscendC::Mins(dst, src, scalar, count);
#endif
  }
  __aicore__ inline Mins(LocalTensor<T>& dst, T& scalar, LocalTensor<T>& src, int count) {
#ifdef __CCE_AICORE__
    AscendC::Mins(dst, src, scalar, count);
#endif
  }
};

template <class T>
struct Sub : public ElemwiseBinaryOP<T, T, T> {
  __aicore__ inline Sub(LocalTensor<T>& dst, LocalTensor<T>& src1, LocalTensor<T>& src2, int count) {
#ifdef __CCE_AICORE__
    AscendC::Sub(dst, src1, src2, count);
#endif
  }
};

template <class T>
struct Mul : public ElemwiseBinaryOP<T, T, T> {
  __aicore__ inline Mul(LocalTensor<T>& dst, LocalTensor<T>& src1, LocalTensor<T>& src2, int count) {
#ifdef __CCE_AICORE__
    AscendC::Mul(dst, src1, src2, count);
#endif
  }
};

template <class T>
struct Muls : public ElemwiseBinaryOP<T, T, T> {
  __aicore__ inline Muls(const LocalTensor<T>& dst, const T& scalar, const LocalTensor<T>& src, int count) {
#ifdef __CCE_AICORE__
    AscendC::Muls(dst, src, scalar, count);
#endif
  }
  __aicore__ inline Muls(const LocalTensor<T>& dst, const LocalTensor<T>& src, const T& scalar, int count) {
#ifdef __CCE_AICORE__
    AscendC::Muls(dst, src, scalar, count);
#endif
  }
};

template <class T>
struct Div : public ElemwiseBinaryOP<T, T, T> {
  __aicore__ inline Div(LocalTensor<T>& dst, LocalTensor<T>& src1, LocalTensor<T>& src2, int count) {
#ifdef __CCE_AICORE__
    AscendC::Div(dst, src1, src2, count);
#endif
  }
};

template <class T>
struct DivHighPrecision : public ElemwiseBinaryOP<T, T, T> {
  __aicore__ inline DivHighPrecision(LocalTensor<T>& dst, LocalTensor<T>& src1, LocalTensor<T>& src2, int count) {
#ifdef __CCE_AICORE__
    static constexpr DivConfig config = {DivAlgo::DIFF_COMPENSATION};
    AscendC::Div<T, config>(dst, src1, src2, count);
#endif
  }
};

template <class T>
struct Divs : public ElemwiseBinaryOP<T, T, T> {
  __aicore__ inline Divs(LocalTensor<T>& dst, LocalTensor<T>& src, T& scalar, int count) {
#ifdef __CCE_AICORE__
    AscendC::Duplicate(dst, scalar, count);
    AscendC::Div(dst, src, dst, count);
#endif
  }
  __aicore__ inline Divs(LocalTensor<T>& dst, T& scalar, LocalTensor<T>& src, int count) {
#ifdef __CCE_AICORE__
    AscendC::Duplicate(dst, scalar, count);
    AscendC::Div(dst, dst, src, count);
#endif
  }
};

template <class T>
struct Subs : public ElemwiseBinaryOP<T, T, T> {
  __aicore__ inline Subs(LocalTensor<T>& dst, LocalTensor<T>& src, T& scalar, int count) {
#ifdef __CCE_AICORE__
    AscendC::Duplicate(dst, scalar, count);
    AscendC::Sub(dst, src, dst, count);
#endif
  }
  __aicore__ inline Subs(LocalTensor<T>& dst, T& scalar, LocalTensor<T>& src, int count) {
#ifdef __CCE_AICORE__
    AscendC::Duplicate(dst, scalar, count);
    AscendC::Sub(dst, dst, src, count);
#endif
  }
};

template <class T>
struct LeakyRelu : public ElemwiseBinaryOP<T, T, T> {
  __aicore__ inline LeakyRelu(LocalTensor<T>& dst, LocalTensor<T>& src, T& scalarValue, int count) {
#ifdef __CCE_AICORE__
    AscendC::LeakyRelu(dst, src, scalarValue, count);
#endif
  }
};

template <class R, class T, int cmpMode>
struct Compare : public ElemwiseBinaryOP<R, T, T> {
  __aicore__ inline Compare(LocalTensor<R>& dstLocal, LocalTensor<T>& src0Local, LocalTensor<T>& src1Local,
                            uint32_t count) {
#ifdef __CCE_AICORE__
    AscendC::Compare(dstLocal, src0Local, src1Local, static_cast<AscendC::CMPMODE>(cmpMode), count);
#endif
  }

  __aicore__ inline Compare(LocalTensor<R>& dstLocal, LocalTensor<T>& src0Local, T src1Scalar, uint32_t count) {
#ifdef __CCE_AICORE__
    AscendC::CompareScalar(dstLocal, src0Local, src1Scalar, static_cast<AscendC::CMPMODE>(cmpMode), count);
#endif
  }
};

template <class T>
struct And : public ElemwiseBinaryOP<T, T, T> {
  __aicore__ inline And(LocalTensor<T>& dst, LocalTensor<T>& src1, LocalTensor<T>& src2, const int32_t& count) {
#ifdef __CCE_AICORE__
    AscendC::And(dst, src1, src2, count);
#endif
  }
};

template <class T>
struct Ands : public ElemwiseBinaryOP<T, T, T> {
  __aicore__ inline Ands(LocalTensor<T>& dst, LocalTensor<T>& src, T& scalar, const int32_t& count) {
#ifdef __CCE_AICORE__
    AscendC::Ands(dst, src, scalar, count);
#endif
  }
  __aicore__ inline Ands(LocalTensor<T>& dst, T& scalar, LocalTensor<T>& src, const int32_t& count) {
#ifdef __CCE_AICORE__
    AscendC::Ands(dst, scalar, src, count);
#endif
  }
};

template <class T>
struct Or : public ElemwiseBinaryOP<T, T, T> {
  __aicore__ inline Or(LocalTensor<T>& dst, LocalTensor<T>& src1, LocalTensor<T>& src2, const int32_t& count) {
#ifdef __CCE_AICORE__
    AscendC::Or(dst, src1, src2, count);
#endif
  }
};

template <class U, class T, int selMode>
struct Select : public ElemwiseTernaryOP<T, U, T, T> {
  __aicore__ inline Select(const LocalTensor<T>& dstLocal, const LocalTensor<U>& selMask,
                           const LocalTensor<T>& src0Local, const LocalTensor<T>& src1Local, uint32_t count) {
#ifdef __CCE_AICORE__
    AscendC::Select(dstLocal, selMask, src0Local, src1Local, static_cast<AscendC::SELMODE>(selMode), count);
#endif
  }

  __aicore__ inline Select(const LocalTensor<T>& dstLocal, const LocalTensor<U>& selMask,
                           const LocalTensor<T>& src0Local, T src1Scalar, uint32_t count) {
#ifdef __CCE_AICORE__
    AscendC::Select(dstLocal, selMask, src0Local, src1Scalar,
                    AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, count);
#endif
  }

   __aicore__ inline Select(const LocalTensor<T>& dstLocal, const LocalTensor<U>& selMask,
                            T src0Scalar, const LocalTensor<T>& src1Local, uint32_t count) {
#ifdef __CCE_AICORE__
    AscendC::Select(dstLocal, selMask, src0Scalar, src1Local,
                    AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, count);
#endif
  }
};

template <class T>
struct CopyIn : public ElemwiseUnaryOP<T, T> {
  __aicore__ inline CopyIn(const LocalTensor<T>& dst, const GlobalTensor<T>& src, uint32_t count) {
#ifdef __CCE_AICORE__
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    AscendC::DataCopyPad(dst, src, copyParams, padParams);
#endif
  }
};

static constexpr uint8_t NDDMA_DIM = 5;
template <class T, int group = 0>
struct CopyInBrc : public ElemwiseUnaryOP<T, T, 1>{
#ifdef __DAV_C310__
  __aicore__ inline CopyInBrc(const LocalTensor<T>& dst, const GlobalTensor<T>& src,
                              const MultiCopyParams<T, NDDMA_DIM>& params,
                              const MultiCopyConfig& config = kDefaultMultiCopyConfig)
  {
    AscendC::DataCopy<T, NDDMA_DIM, config>(dst, src, params);
  }
#endif
  __aicore__ inline CopyInBrc(const LocalTensor<T>& dst, const GlobalTensor<T>& src, uint32_t count)
  {
#ifdef __CCE_AICORE__
    AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    AscendC::DataCopyPad(dst, src, copyParams, padParams);
#endif
  }
};

template <class T, int group = 0, int constRank=-1>
struct Brc : public ElemwiseUnaryOP<T, T>{
  __aicore__ inline Brc(LocalTensor<T>& dst, LocalTensor<T>& src, int count)
  // __aicore__ inline Brc(LocalTensor<T>& dst, LocalTensor<T>& src, const uint32_t* dstShape, const uint32_t* srcShape, BroadcastTiling* tiling)
  {
#ifdef __CCE_AICORE__
    // AscendC::Broadcast<T, constRank>(dst, src, dstShape, srcShape, tiling);
#endif
  }
};

template <class T>
struct CopyOut : public ElemwiseUnaryOP<T, T> {
  __aicore__ inline CopyOut(GlobalTensor<T>& dst, LocalTensor<T>& src, int count) {
#ifdef __CCE_AICORE__
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
    AscendC::DataCopyPad(dst, src, copyParams);
#endif
  }
};

template <class T>
struct IsCastOp {
  constexpr static bool Value = false;
};

template <class U, class T, int mode>
struct IsCastOp<Cast<U, T, mode>> {
  constexpr static bool Value = true;
};

template <class T>
struct IsCastNoOp {
  constexpr static bool Value = false;
};

template <class U, class T, int mode>
struct IsCastNoOp<Cast<U, T, mode>> {
  constexpr static bool Value = std::is_same<U, T>::value;
};

template <class T>
struct IsCopyInBrcOp {
  constexpr static bool Value = false;
};

template <class T, int group>
struct IsCopyInBrcOp<CopyInBrc<T, group>> {
  constexpr static bool Value = true;
};

template <class... T>
struct IsCopyInOp {
  constexpr static bool Value = false;
};

template <class T>
struct IsCopyInOp<CopyIn<T>> {
  constexpr static bool Value = true;
};

template <class T, int group>
struct IsCopyInOp<CopyInBrc<T, group>> {
  constexpr static bool Value = true;
};

template <class T>
struct IsVecBrcOp {
  constexpr static bool Value = false;
};

template <class T, int group, int constRank>
struct IsVecBrcOp<Brc<T, group, constRank>> {
  constexpr static bool Value = true;
};

template <class... T>
struct IsCopyOutOp {
  constexpr static bool Value = false;
};

template <class T>
struct IsCopyOutOp<CopyOut<T>> {
  constexpr static bool Value = true;
};

template <class... T>
struct IsReduceOp {
  constexpr static bool Value = false;
};

template <class T>
struct IsReduceOp<ReduceOp<T>> {
  constexpr static bool Value = true;
};

template <class... T>
struct IsDuplicateOp {
  constexpr static bool Value = false;
};

template <class T>
struct IsDuplicateOp<Duplicate<T>> {
  constexpr static bool Value = true;
};

}  // namespace Vec
}  // namespace AscendC

#endif  // ATP_VEC_H_
