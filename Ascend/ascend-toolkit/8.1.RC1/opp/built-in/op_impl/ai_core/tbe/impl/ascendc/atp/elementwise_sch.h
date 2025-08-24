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
 * \file elementwise_sch.h
 * \brief
 */
#ifndef ATP_ELEMENTWISE_SCH_H_
#define ATP_ELEMENTWISE_SCH_H_
#include "dag.h"
#include "kernel_operator.h"
#include "sync.h"
#include "vec.h"

#pragma "lib"

namespace AscendC {

constexpr uint64_t BYTE_LENGTH = 8;

template <uint64_t schMode, class ElemDag, class SchOptCfg = void>
class ElementwiseSch {
 public:
  __aicore__ inline explicit ElementwiseSch(const EleBaseTilingData* baseTilingData, TPipe *pipe) {
    tilingData = baseTilingData;
    pipePtr = pipe;
  }

  /**
   * 初始化ElementwiseSch对象
   * @tparam Args 输入Args类型
   * @param args 输入args参数列表，需要匹配DAG图中PlaceHolder的顺序[In0, In1..., Out0, Out1...]
   */
  template <class... Args>
  __aicore__ inline void Init(Args... args) {
    static_assert(inputNums + outputNums == sizeof...(Args),
                  "ElementwiseSch.Init args num should match DAG holders.");
    InitInputArgs<0>(args...);  // 调用入参分析,input,output
    RUN_LOG("BufferNum: %d, Mte2Num: %d, Mte3Num: %d, BufLevel: %d",
            ElemDag::BufferNum, ElemDag::Mte2Num,
            ElemDag::Mte3Num, ElemDag::BufLevel);
    pipePtr->InitBuffer(buf, tilingData->ubFormer * ElemDag::MaxDtypeBytes * ElemDag::BufferNum);
    tensorPool = buf.Get<uint8_t>();
    blockLen = tilingData->ubFormer * ElemDag::MaxDtypeBytes;
  }

  /**
   * 执行ElementwiseSch
   */
  __aicore__ inline void Process() {
    uint64_t offset = 0;
    uint64_t loopNum =
        GetBlockIdx() == tilingData->blockNum - 1 ? tilingData->ubLoopOfTailBlock : tilingData->ubLoopOfFormerBlock;
    uint64_t tailNum =
        GetBlockIdx() == tilingData->blockNum - 1 ? tilingData->ubTailOfTailBlock : tilingData->ubTailOfFormerBlock;
    uint64_t i = 0;
    for (; i < loopNum - 1; i++) {
      // ub整循环处理的元素个数
      if (i & 1) {
          Run<0, 1>(offset, tilingData->ubFormer);
      } else {
          Run<0, 0>(offset, tilingData->ubFormer);
      }
       offset += tilingData->ubFormer;
    }

    // ub尾循环处理的元素个数
    if (i & 1) {
        Run<0, 1>(offset, tailNum);
    } else {
        Run<0, 0>(offset, tailNum);
    }
  }

  /**
   * 设置DAG图上Var类型PlaceHolder的值
   * @tparam U Var的数据类型
   * @tparam index Var的索引
   * @param value
   */
  template <typename U, int index>
  __aicore__ inline void SetVar(U value) {
    static_assert(index < ElemDag::Vars::Size,
                  "The index exceeds the number of Vars defined in DAG.");
    scalars.template Set<index>(value);
  }

 protected:
  /**
   * 当前核处理数据的起始偏移
   * @tparam DataType 当前输入的数据类型，当数据类型是1bit时，偏移需要除以8
   * @return
   */
  template <typename DataType>
  __aicore__ inline int64_t CalcBlockOffset() {
    if constexpr (std::is_same<DataType, uint1_t>::value) {
      return tilingData->blockFormer * GetBlockIdx() / BYTE_LENGTH;
    }
    return tilingData->blockFormer * GetBlockIdx() * sizeof(DataType);
  }

  template <int target, typename T>
  struct GetHolderByPos {
    using Type = void;
  };

  template <int target, template <typename...> typename Holders, typename Holder>
  struct GetHolderByPos<target, Holders<Holder>> {
    using Type = __aux::Condition<target == Holder::Pos, Holder, void>;
  };

  template <int target, template <typename...> typename Holders, typename Holder, typename... HolderTs>
  struct GetHolderByPos<target, Holders<Holder, HolderTs...>> {
    using Type =
        __aux::Condition<target == Holder::Pos, Holder, typename GetHolderByPos<target, Holders<HolderTs...>>::Type>;
  };

  // 初始化输出
  template <int start, class... Args>
  __aicore__ inline void InitOutputArgs(GM_ADDR y, Args... args) {
    if constexpr (start < outputNums) {
      using Holder = typename GetHolderByPos<start, typename ElemDag::OutHolders>::Type;
      using DataType = typename Holder::DType;
      outGm[start].SetGlobalBuffer((__gm__ uint8_t*)y + CalcBlockOffset<DataType>());
    }

    if constexpr (start + 1 < outputNums) {
      InitOutputArgs<start + 1>(args...);
    }
  }

  // 初始化输入
  template <int start, class... Args>
  __aicore__ inline void InitInputArgs(GM_ADDR x, Args... args) {
    if constexpr (start < inputNums) {
      using Holder = typename GetHolderByPos<start, typename ElemDag::InHolders>::Type;
      using DataType = typename Holder::DType;
      if constexpr (Placeholder::IsInScalar<Holder>::Value) {
        inGm[start].SetGlobalBuffer((__gm__ uint8_t*)x);
      } else {
        inGm[start].SetGlobalBuffer((__gm__ uint8_t*)x + CalcBlockOffset<DataType>());
      }
    }

    if constexpr (start + 1 <= inputNums) {
      InitInputArgs<start + 1>(args...);
    } else {
      // inputNums == 0时，x实际时第一个输出
      InitOutputArgs<0>(x, args...);
    }
  }

  /**
   * 搬入Op
   * @tparam Op CopyIn的Bind节点
   * @tparam pos CopyIn在FunList内的索引
   * @tparam pingPong DoubleBuffer中所处的ping/pong阶段
   * @param offset 当前处理的数据在GM上的偏移
   * @param tileLength 当前处理的数据块大小
   */
  template <typename Op, int pos, int32_t pingPong>
  __aicore__ inline void CopyIn(uint64_t offset, uint64_t tileLength) {
    static_assert(Op::InHolders::Size == 1, "CopyIn input inHolders num should be 1.");
    using input = typename Op::InHolders::template At<0>;
    using inputType = typename Op::template FunInArgType<0>;
    if constexpr (std::is_same<typename input::DType, uint1_t>::value) {
      static_assert(std::is_same<inputType, uint8_t>::value,
                    "CopyIn data type is inconsistent with in holder data type.");
      offset = offset / BYTE_LENGTH;
      tileLength = tileLength / BYTE_LENGTH;
    } else {
      static_assert(std::is_same<typename input::DType, inputType>::value,
                    "CopyIn data type is inconsistent with in holder data type.");
    }

    // Prepare input args
    constexpr uint8_t bufId = ElemDag::BufferIds[pingPong][pos];
    LocalTensor<inputType> inTensor = tensorPool[bufId * blockLen].template ReinterpretCast<inputType>();
#ifndef __CCE_KT_TEST__
    inTensor.SetBufferLen(blockLen / sizeof(inputType));
#endif
    GlobalTensor<inputType> globalTensor;
    globalTensor.SetGlobalBuffer(
        reinterpret_cast<__gm__ inputType*>(inGm[input::Pos].GetPhyAddr(offset * sizeof(inputType))));
    // Set getBuf
    GetTensor<TPosition::VECIN>(bufId);
    // Run copyIn
    Vec::CopyIn<inputType>(inTensor, globalTensor, tileLength);
    // Set rlsBuf
    ReleaseTensor<TPosition::VECIN>(bufId);
  }

  /**
   * 输出Op
   * @tparam Op CopyOut的Bind节点
   * @tparam pos CopyOut在FunList内的索引
   * @tparam pingPong DoubleBuffer中所处的ping/pong阶段
   * @param offset 当前处理的数据在GM上的偏移
   * @param tileLength 当前处理的数据块大小
   */
  template <typename Op, int pos, int32_t pingPong>
  __aicore__ inline void CopyOut(uint64_t offset, uint64_t tileLength) {
    static_assert(Op::Args::Size == 2, "Input args should be 2");
    using input = typename Op::Args::template At<1>;
    using output = typename Op::Args::template At<0>;
    using inputType = typename Op::template FunInArgType<0>;
    static_assert(Placeholder::IsOutHolder<output>::Value, "output args should be out holder");
    if constexpr (std::is_same<typename output::DType, uint1_t>::value) {
      static_assert(std::is_same<inputType, uint8_t>::value,
                    "CopyOut data type is inconsistent with out holder data type.");
      offset = offset / BYTE_LENGTH;
      tileLength = tileLength / BYTE_LENGTH;
    } else {
      static_assert(std::is_same<typename output::DType, inputType>::value,
                    "CopyOut data type is inconsistent with Op data type.");
    }

    // Prepare input args
    constexpr uint8_t bufId = ElemDag::BufferIds[pingPong][GetFunOutputPos<input>()];
    LocalTensor<inputType> localTensor = tensorPool[bufId * blockLen].template ReinterpretCast<inputType>();
#ifndef __CCE_KT_TEST__
    localTensor.SetBufferLen(blockLen / sizeof(inputType));
#endif
    static_assert(output::Pos < outputNums, "output Pos is not less than output number.");
    GlobalTensor<inputType> globalTensor;
    globalTensor.SetGlobalBuffer(
        reinterpret_cast<__gm__ inputType*>(outGm[output::Pos].GetPhyAddr(offset * sizeof(inputType))));
    // Set getBuf
    GetTensor<TPosition::VECOUT>(bufId);
    // Run func
    Vec::CopyOut<inputType>(globalTensor, localTensor, tileLength);
    // Set rlsBuf
    ReleaseTensor<TPosition::VECOUT>(bufId);
  }

  /**
   * 遍历DAG，计算Op的position
   * @tparam Op 待计算的Op, 需要是Bind类型
   * @tparam start 查询起始位置的Position，默认从FunList的第0个开始匹配
   * @return Op在FunList中的索引坐标
   */
  template <class Op, int start = 0>
  __aicore__ constexpr static inline int GetFunOutputPos() {
    if constexpr (std::is_same<typename ElemDag::FunList::template At<start>, Op>::value) {
      return start;
    } else if constexpr (start + 1 < ElemDag::FunList::Size) {
      return GetFunOutputPos<Op, start + 1>();
    }
    static_assert(start + 1 < ElemDag::FunList::Size, "The required output in FunList is not found.");
    return -1;
  }

  /**
   * 尝试获取指定Bind节点的输出Tensor
   * @tparam TensorType LocalTensor的数据类型
   * @tparam tensorOp 指定的Bind节点
   * @tparam pingPong DoubleBuffer中所处的ping/pong阶段
   * @tparam isGet 输入来源于同一个节点的时候，需要避免重复Get，第一个使用true, 后续使用false
   * @return 返回对应的LocalTensor对象
   */
  template <typename TensorType, typename tensorOp, int32_t pingPong, bool isGet = true>
  __aicore__ inline constexpr LocalTensor<TensorType> TryGetTensor() {
    // 如果数据来源于CopyIn, 需要使用GetTensor同步
    if constexpr (isGet) {
      GetTensor<TPosition::VECCALC>(ElemDag::BufferIds[pingPong][GetFunOutputPos<tensorOp>()]);
    }
    LocalTensor<TensorType> inputTensor =
        tensorPool[ElemDag::BufferIds[pingPong][GetFunOutputPos<tensorOp>()] * blockLen]
            .template ReinterpretCast<TensorType>();
#ifndef __CCE_KT_TEST__
    inputTensor.SetBufferLen(blockLen / sizeof(TensorType));
#endif

    return inputTensor;
  }

  /**
   * 尝试释放当前的Buffer
   * @tparam inputOp 待释放的Buffer所属的计算节点
   * @tparam pos 当前节点在FunList中的索引
   * @tparam pingPong DoubleBuffer中所处的ping/pong阶段
   * @tparam toPos 当前节点TPosition
   * @tparam isRelease 为避免重复释放
   * @return 避免重复Release
   */
  template <typename inputOp, int pos, int32_t pingPong, TPosition toPos = TPosition::VECCALC, bool isRelease = true>
  __aicore__ inline constexpr void TryReleaseTensor() {
    if constexpr (isRelease) {
      ReleaseTensor<toPos>(ElemDag::BufferIds[pingPong][GetFunOutputPos<inputOp>()]);
    }
  }

  /**
   * 根据DAG中Scalar描述获取Scalar的值
   * @tparam ScalarType Scalar的数据类型
   * @tparam scalarValue 存储Scalar值的Holder，主要有Var/InHolder/ConstValue三种类型的Scalar
   * @return 类型位ScalarType的Scalar值
   */
  template <typename ScalarType, typename scalarValue>
  __aicore__ inline constexpr ScalarType GetScalar() {
    static_assert(!(Placeholder::IsVar<scalarValue>::Value && Placeholder::IsInHolder<scalarValue>::Value &&
                    Placeholder::IsConstValue<scalarValue>::Value),
                  "The input parameter type is not FunBind, Var, Const or Holder.");
    if constexpr (Placeholder::IsVar<scalarValue>::Value) {
      ScalarType scalar = scalars.template Get<scalarValue::Pos>();
      return scalar;
    } else if constexpr (Placeholder::IsInHolder<scalarValue>::Value) {
      GlobalTensor<ScalarType> globalTensor;
      globalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ ScalarType*>(inGm[scalarValue::Pos].GetPhyAddr(0)));
      ScalarType scalar = globalTensor.GetValue(0);
      return scalar;
    } else if constexpr (Placeholder::IsConstValue<scalarValue>::Value) {
      ScalarType scalar = static_cast<ScalarType>(scalarValue::value);
      return scalar;
    }
  }

  /**
   * 单输入类型的操作， 只支持两种组合方式，F(Tensor), F(Scalar)
   * @tparam OutputType 输出数据类型
   * @tparam Op 当前计算节点，需要是Bind类型
   * @tparam pos 当前计算节点的位置
   * @tparam pingPong DoubleBuffer中所处的ping/pong阶段
   * @param outTensor 输出的LocalTensor对象
   * @param tileLength 单次处理的数据个数
   * @return
   */
  template <typename OutputType, class Op, int pos = 0, int32_t pingPong>
  __aicore__ inline constexpr void RunUnaryOp(LocalTensor<OutputType>& outTensor, uint64_t tileLength) {
    using Func = typename Op::Fun;
    using inputOp0 = typename Op::InArgs::template At<0>;
    using inputType0 = typename Op::template FunInArgType<0>;
    if constexpr (__aux::TypeIsFunBind<inputOp0>::Value) {
      LocalTensor<inputType0> inputTensor0 = TryGetTensor<inputType0, inputOp0, pingPong>();
      Func(outTensor, inputTensor0, tileLength);
      TryReleaseTensor<inputOp0, pos, pingPong>();
    } else {
      inputType0 scalar = GetScalar<inputType0, inputOp0>();
      Func(outTensor, scalar, tileLength);
    }
  }

  /**
   * 双输入类型的操作， 只支持三种组合方式，F(Tensor, Tensor), F(Tensor, Scalar), F(Scalar, Tensor)
   * @tparam OutputType 输出数据类型
   * @tparam Op 当前计算节点，需要是Bind类型
   * @tparam pos 当前计算节点的位置
   * @tparam pingPong DoubleBuffer中所处的ping/pong阶段
   * @param outTensor 输出的LocalTensor对象
   * @param tileLength 单次处理的数据个数
   * @return
   */
  template <typename OutputType, class Op, int pos = 0, int32_t pingPong>
  __aicore__ inline constexpr void RunBinaryOp(LocalTensor<OutputType>& outTensor, uint64_t tileLength) {
    // 至少有一个是Compute
    static_assert(Op::InFuns::Size > 0, "At least one compute input.");

    using Func = typename Op::Fun;
    // 输入Op
    using inputOp0 = typename Op::InArgs::template At<0>;
    using inputOp1 = typename Op::InArgs::template At<1>;
    // 输入数据类型
    using inputType0 = typename Op::template FunInArgType<0>;
    using inputType1 = typename Op::template FunInArgType<1>;

    if constexpr (__aux::TypeIsFunBind<inputOp0>::Value) {
      LocalTensor<inputType0> inputTensor0 = TryGetTensor<inputType0, inputOp0, pingPong>();
      if constexpr (__aux::TypeIsFunBind<inputOp1>::Value) {
        constexpr bool isSameWith01 = std::is_same<inputOp0, inputOp1>::value;
        // Func(Tensor, Tensor)
        LocalTensor<inputType1> inputTensor1 = TryGetTensor<inputType1, inputOp1, pingPong, !isSameWith01>();
        Func(outTensor, inputTensor0, inputTensor1, tileLength);
        TryReleaseTensor<inputOp1, pos, pingPong, TPosition::VECCALC, !isSameWith01>();
      } else {
        // Func(Tensor, Scalar)
        inputType1 scalar = GetScalar<inputType1, inputOp1>();
        Func(outTensor, inputTensor0, scalar, tileLength);
      }
      TryReleaseTensor<inputOp0, pos, pingPong>();
    } else if constexpr (__aux::TypeIsFunBind<inputOp1>::Value) {
      // Func(Scalar, Tensor)
      inputType0 scalar = GetScalar<inputType0, inputOp0>();
      LocalTensor<inputType1> inputTensor1 = TryGetTensor<inputType1, inputOp1, pingPong>();
      Func(outTensor, scalar, inputTensor1, tileLength);
      TryReleaseTensor<inputOp1, pos, pingPong>();
    } else {
      static_assert(!(__aux::TypeIsFunBind<inputOp0>::Value && __aux::TypeIsFunBind<inputOp1>::Value),
                    "The input parameter type does not include FunBind.");
    }
  }

  /**
   * 三输入类型的操作， 只支持四种组合方式，F(Tensor, Tensor，Tensor), F(Tensor, Tensor, Scalar), F(Scalar, Tensor)
   * @tparam OutputType 输出数据类型
   * @tparam Op 当前计算节点，需要是Bind类型
   * @tparam pos 当前计算节点的位置
   * @tparam pingPong DoubleBuffer中所处的ping/pong阶段
   * @param outTensor 输出的LocalTensor对象
   * @param tileLength 单次处理的数据个数
   * @return
   */
  template <typename OutputType, class Op, int pos = 0, int32_t pingPong>
  __aicore__ inline constexpr void RunTernaryOp(LocalTensor<OutputType> outTensor, uint64_t tileLength) {
    using Func = typename Op::Fun;
    // Prepare input args
    using inputOp0 = typename Op::InArgs::template At<0>;
    using inputOp1 = typename Op::InArgs::template At<1>;
    using inputOp2 = typename Op::InArgs::template At<2>;
    // 输入数据类型
    using inputType0 = typename Op::template FunInArgType<0>;
    using inputType1 = typename Op::template FunInArgType<1>;
    using inputType2 = typename Op::template FunInArgType<2>;

    // 至少有一个是Compute
    static_assert(Op::InFuns::Size > 0, "at least one compute input.");

    // 当前支支持前两个是compute,第三个可以是comput,也可以是scalar的
    static_assert(__aux::TypeIsFunBind<inputOp0>::Value, "input0 should be compute.");
    static_assert(__aux::TypeIsFunBind<inputOp1>::Value, "input1 should be compute.");

    // input0 is compute
    if constexpr (__aux::TypeIsFunBind<inputOp0>::Value) {
      LocalTensor<inputType0> inputTensor0 = TryGetTensor<inputType0, inputOp0, pingPong>();
      if constexpr (__aux::TypeIsFunBind<inputOp1>::Value) {
        constexpr bool isSameWith01 = std::is_same<inputOp0, inputOp1>::value;
        LocalTensor<inputType1> inputTensor1 = TryGetTensor<inputType1, inputOp1, pingPong, !isSameWith01>();
        if constexpr (__aux::TypeIsFunBind<inputOp2>::Value) {
          constexpr bool isSameWith012 =
              std::is_same<inputOp0, inputOp2>::value || std::is_same<inputOp1, inputOp2>::value;
          LocalTensor<inputType2> inputTensor2 = TryGetTensor<inputType2, inputOp2, pingPong, !isSameWith012>();
          Func(outTensor, inputTensor0, inputTensor1, inputTensor2, tileLength);
          TryReleaseTensor<inputOp2, pos, pingPong, TPosition::VECCALC, !isSameWith012>();
        } else {
          inputType2 scalar = GetScalar<inputType2, inputOp2>();
          Func(outTensor, inputTensor0, inputTensor1, scalar, tileLength);
        }
        TryReleaseTensor<inputOp1, pos, pingPong, TPosition::VECCALC, !isSameWith01>();
      } else {
        // 当前不支持其他场景
        static_assert(!__aux::TypeIsFunBind<inputOp1>::Value, "The input 1 parameter type does not include FunBind.");
      }
      TryReleaseTensor<inputOp0, pos, pingPong>();
    } else {
      // 当前不支持其他场景
      static_assert(!__aux::TypeIsFunBind<inputOp0>::Value, "The input 0 parameter type does not include FunBind.");
    }
  }

  /**
   * 7输入类型的操作， 只支持一种组合方式，F(Tensor, Tensor，Tensor, Scalar, Scalar, Scalar, Scalar)
   * @tparam OutputType 输出数据类型
   * @tparam Op 当前计算节点，需要是Bind类型
   * @tparam pos 当前计算节点的位置
   * @tparam pingPong DoubleBuffer中所处的ping/pong阶段
   * @param outTensor 输出的LocalTensor对象
   * @param tileLength 单次处理的数据个数
   * @return
   */
  template <typename OutputType, class Op, int pos = 0, int32_t pingPong>
  __aicore__ inline constexpr void RunOp7(LocalTensor<OutputType> outTensor, uint64_t tileLength) {
    using Func = typename Op::Fun;
    // Prepare input args
    using inputOp0 = typename Op::InArgs::template At<0>;
    using inputOp1 = typename Op::InArgs::template At<1>;
    using inputOp2 = typename Op::InArgs::template At<2>;
    using inputOp3 = typename Op::InArgs::template At<3>;
    using inputOp4 = typename Op::InArgs::template At<4>;
    using inputOp5 = typename Op::InArgs::template At<5>;
    using inputOp6 = typename Op::InArgs::template At<6>;

    // 输入数据类型
    using inputType0 = typename Op::template FunInArgType<0>;
    using inputType1 = typename Op::template FunInArgType<1>;
    using inputType2 = typename Op::template FunInArgType<2>;
    using inputType3 = typename Op::template FunInArgType<3>;
    using inputType4 = typename Op::template FunInArgType<4>;
    using inputType5 = typename Op::template FunInArgType<5>;
    using inputType6 = typename Op::template FunInArgType<6>;

    // 至少有一个是Compute
    static_assert(Op::InFuns::Size > 0, "at least one compute input.");

    // 当前支支持前两个是compute,第三个可以是comput,也可以是scalar的
    static_assert(__aux::TypeIsFunBind<inputOp0>::Value, "input0 should be compute.");
    static_assert(__aux::TypeIsFunBind<inputOp1>::Value, "input1 should be compute.");
    static_assert(__aux::TypeIsFunBind<inputOp2>::Value, "input1 should be compute.");

    // input0 is compute
    if constexpr (__aux::TypeIsFunBind<inputOp0>::Value) {
      LocalTensor<inputType0> inputTensor0 = TryGetTensor<inputType0, inputOp0, pingPong>();
      if constexpr (__aux::TypeIsFunBind<inputOp1>::Value) {
        constexpr bool isSameWith01 = std::is_same<inputOp0, inputOp1>::value;
        LocalTensor<inputType1> inputTensor1 = TryGetTensor<inputType1, inputOp1, pingPong, !isSameWith01>();
        if constexpr (__aux::TypeIsFunBind<inputOp2>::Value) {
          constexpr bool isSameWith012 =
              std::is_same<inputOp0, inputOp2>::value || std::is_same<inputOp1, inputOp2>::value;
          LocalTensor<inputType2> inputTensor2 = TryGetTensor<inputType2, inputOp2, pingPong, !isSameWith012>();

          inputType3 scalar3 = GetScalar<inputType3, inputOp3>();
          inputType4 scalar4 = GetScalar<inputType4, inputOp4>();
          inputType5 scalar5 = GetScalar<inputType5, inputOp5>();
          inputType6 scalar6 = GetScalar<inputType6, inputOp6>();

          Func(outTensor, inputTensor0, inputTensor1, inputTensor2, scalar3, scalar4, scalar5, scalar6, tileLength);
          TryReleaseTensor<inputOp2, pos, pingPong, TPosition::VECCALC, !isSameWith012>();
        } else {
          // 当前不支持其他场景
          static_assert(!__aux::TypeIsFunBind<inputOp2>::Value, "The input 2 parameter type does not include FunBind.");
        }
        TryReleaseTensor<inputOp1, pos, pingPong, TPosition::VECCALC, !isSameWith01>();
      } else {
        // 当前不支持其他场景
        static_assert(!__aux::TypeIsFunBind<inputOp1>::Value, "The input 1 parameter type does not include FunBind.");
      }
      // Free Buffer
      TryReleaseTensor<inputOp0, pos, pingPong>();
    } else {
      // 当前不支持其他场景
      static_assert(!__aux::TypeIsFunBind<inputOp0>::Value, "The input 0 parameter type does not include FunBind.");
    }
  }

  /**
   * 9输入类型的操作， 只支持一种组合方式，F(Tensor, Tensor，Tensor, Tensor, Scalar, Scalar, Scalar, Scalar, Scalar)
   * @tparam OutputType 输出数据类型
   * @tparam Op 当前计算节点，需要是Bind类型
   * @tparam pos 当前计算节点的位置
   * @tparam pingPong DoubleBuffer中所处的ping/pong阶段
   * @param outTensor 输出的LocalTensor对象
   * @param tileLength 单次处理的数据个数
   * @return
   */
  template <typename OutputType, class Op, int pos = 0, int32_t pingPong>
  __aicore__ inline constexpr void RunOp9(LocalTensor<OutputType> outTensor, uint64_t tileLength) {
    using Func = typename Op::Fun;
    // Prepare input args
    using inputOp0 = typename Op::InArgs::template At<0>;
    using inputOp1 = typename Op::InArgs::template At<1>;
    using inputOp2 = typename Op::InArgs::template At<2>;
    using inputOp3 = typename Op::InArgs::template At<3>;
    using inputOp4 = typename Op::InArgs::template At<4>;
    using inputOp5 = typename Op::InArgs::template At<5>;
    using inputOp6 = typename Op::InArgs::template At<6>;
    using inputOp7 = typename Op::InArgs::template At<7>;
    using inputOp8 = typename Op::InArgs::template At<8>;

    // 输入数据类型
    using inputType0 = typename Op::template FunInArgType<0>;
    using inputType1 = typename Op::template FunInArgType<1>;
    using inputType2 = typename Op::template FunInArgType<2>;
    using inputType3 = typename Op::template FunInArgType<3>;
    using inputType4 = typename Op::template FunInArgType<4>;
    using inputType5 = typename Op::template FunInArgType<5>;
    using inputType6 = typename Op::template FunInArgType<6>;
    using inputType7 = typename Op::template FunInArgType<7>;
    using inputType8 = typename Op::template FunInArgType<8>;

    // 至少有一个是Compute
    static_assert(Op::InFuns::Size > 0, "at least one compute input.");

    // 当前支支持前两个是compute,第三个可以是comput,也可以是scalar的
    static_assert(__aux::TypeIsFunBind<inputOp0>::Value, "input0 should be compute.");
    static_assert(__aux::TypeIsFunBind<inputOp1>::Value, "input1 should be compute.");
    static_assert(__aux::TypeIsFunBind<inputOp2>::Value, "input2 should be compute.");
    static_assert(__aux::TypeIsFunBind<inputOp3>::Value, "input2 should be compute.");

    // input0 is compute
    if constexpr (__aux::TypeIsFunBind<inputOp0>::Value) {
      LocalTensor<inputType0> inputTensor0 = TryGetTensor<inputType0, inputOp0, pingPong>();
      if constexpr (__aux::TypeIsFunBind<inputOp1>::Value) {
        constexpr bool isSameWith01 = std::is_same<inputOp0, inputOp1>::value;
        LocalTensor<inputType1> inputTensor1 = TryGetTensor<inputType1, inputOp1, pingPong, !isSameWith01>();
        if constexpr (__aux::TypeIsFunBind<inputOp2>::Value) {
          constexpr bool isSameWith012 =
              std::is_same<inputOp0, inputOp2>::value || std::is_same<inputOp1, inputOp2>::value;
          LocalTensor<inputType2> inputTensor2 = TryGetTensor<inputType2, inputOp2, pingPong, !isSameWith012>();

          if constexpr (__aux::TypeIsFunBind<inputOp3>::Value) {
            constexpr bool isSameWith0123 =
                std::is_same<inputOp0, inputOp3>::value ||
                std::is_same<inputOp1, inputOp3>::value ||
                std::is_same<inputOp2, inputOp3>::value;
            LocalTensor<inputType3> inputTensor3 = TryGetTensor<inputType3, inputOp3, pingPong, !isSameWith0123>();

            inputType4 scalar4 = GetScalar<inputType4, inputOp4>();
            inputType5 scalar5 = GetScalar<inputType5, inputOp5>();
            inputType6 scalar6 = GetScalar<inputType6, inputOp6>();
            inputType7 scalar7 = GetScalar<inputType7, inputOp7>();
            inputType8 scalar8 = GetScalar<inputType8, inputOp8>();

            Func(outTensor, inputTensor0, inputTensor1, inputTensor2, inputTensor3,
                 scalar4, scalar5, scalar6, scalar7, scalar8, tileLength);
            TryReleaseTensor<inputOp3, pos, pingPong, TPosition::VECCALC, !isSameWith0123>();
          } else {
            // 当前不支持其他场景
            static_assert(!__aux::TypeIsFunBind<inputOp3>::Value, "The input 3 parameter type does not include FunBind.");
          }
          TryReleaseTensor<inputOp2, pos, pingPong, TPosition::VECCALC, !isSameWith012>();
        } else {
          // 当前不支持其他场景
          static_assert(!__aux::TypeIsFunBind<inputOp2>::Value, "The input 2 parameter type does not include FunBind.");
        }
        TryReleaseTensor<inputOp1, pos, pingPong, TPosition::VECCALC, !isSameWith01>();
      } else {
        // 当前不支持其他场景
        static_assert(!__aux::TypeIsFunBind<inputOp1>::Value, "The input 1 parameter type does not include FunBind.");
      }
      // Free Buffer
      TryReleaseTensor<inputOp0, pos, pingPong>();
    } else {
      // 当前不支持其他场景
      static_assert(!__aux::TypeIsFunBind<inputOp0>::Value, "The input 0 parameter type does not include FunBind.");
    }
  }

  /**
   * 执行常规计算节点
   * @tparam Op 除CopyIn/CopyOut之外的常规计算节点，Bind类型
   * @tparam pos 当前计算节点在FunList中的索引
   * @tparam pingPong DoubleBuffer中所处的ping/pong阶段
   * @param tileLength 当前计算的数据块大小
   * @return
   */
  template <class Op, int pos = 0, int32_t pingPong>
  __aicore__ inline constexpr void RunNormalOp(uint64_t tileLength) {
    // Prepare output args
    using outputType = typename Op::template FunRetArgType<0>;
    constexpr uint8_t bufID = ElemDag::BufferIds[pingPong][pos];
    LocalTensor<outputType> outTensor = tensorPool[bufID * blockLen].template ReinterpretCast<outputType>();
#ifndef __CCE_KT_TEST__
    outTensor.SetBufferLen(blockLen / sizeof(outputType));
#endif
    // Set getBuf
    GetTensor<TPosition::VECCALC>(bufID);
    // Run Op
    if constexpr (Op::InArgs::Size == 1) {
      RunUnaryOp<outputType, Op, pos, pingPong>(outTensor, tileLength);
    } else if constexpr (Op::InArgs::Size == 2) {
      RunBinaryOp<outputType, Op, pos, pingPong>(outTensor, tileLength);
    } else if constexpr (Op::InArgs::Size == 3) {
      RunTernaryOp<outputType, Op, pos, pingPong>(outTensor, tileLength);
    } else if constexpr (Op::InArgs::Size == 7) {
      RunOp7<outputType, Op, pos, pingPong>(outTensor, tileLength);
    } else if constexpr (Op::InArgs::Size == 9) {
      RunOp9<outputType, Op, pos, pingPong>(outTensor, tileLength);
    } else {
      static_assert(Op::InArgs::Size > 3, "Unsupported compute with InArgs more then 3.");
    }
    // Set rlsBuf
    ReleaseTensor<TPosition::VECCALC>(bufID);
  }

  /**
   * 处理不需要实际执行的计算节点
   * @tparam Op NoOp计算节点
   * @tparam pos NoOp节点在FunList中的索引
   * @tparam pingPong DoubleBuffer中所处的ping/pong阶段
   * @return
   */
  template <class Op, int pos = 0, int32_t pingPong>
  __aicore__ inline constexpr void RunNoOp() {
    static_assert(Op::InFuns::Size == 1, "No op need one compute input.");
    using inputOp0 = typename Op::InArgs::template At<0>;
    ElemDag::BufferIds[pingPong][pos] = ElemDag::BufferIds[pingPong][GetFunOutputPos<inputOp0>()];
  }

  // 遍历执行图
  template <int pos = 0, int32_t pingPong>
  __aicore__ inline void Run(uint64_t offset, uint64_t tileLength) {
    // Run current func
    using Op = typename ElemDag::FunList::template At<pos>;
    using Func = typename Op::Fun;
    RUN_LOG("RUN.Func[%s]: ArgsSize: %ld, PingPong:%ld, GmOffset:%ld, TileLength:%ld\n", PRINT_TYPE(Func),
            Op::Args::Size, pingPong, offset, tileLength);
    if constexpr (__aux::IsSameTemplateType<Func, Vec::CopyIn>::Value) {
      CopyIn<Op, pos, pingPong>(offset, tileLength);
    } else if constexpr (__aux::IsSameTemplateType<Func, Vec::CopyOut>::Value) {
      CopyOut<Op, pos, pingPong>(offset, tileLength);
    } else if constexpr (Vec::IsCastOp<Func>::Value && std::is_same<typename Op::template FunRetArgType<1>,
                                                                    typename Op::template FunRetArgType<0>>::value) {
      RUN_LOG("Cast with same src and dst type, skip cast.\n");
      RunNoOp<Op, pos, pingPong>();
    } else {
      RunNormalOp<Op, pos, pingPong>(tileLength);
    }

    // Run next func
    if constexpr (pos + 1 < ElemDag::FunList::Size) {
      Run<pos + 1, pingPong>(offset, tileLength);
    }
  }

  private:
  constexpr static int inputNums = ElemDag::InputSize;
  constexpr static int outputNums = ElemDag::OutputSize;
  GlobalTensor<uint8_t> inGm[inputNums];
  GlobalTensor<uint8_t> outGm[outputNums];
  TPipe *pipePtr;
  TBuf<TPosition::VECCALC> buf;
  LocalTensor<uint8_t> tensorPool;
  int blockLen = 0;

  const EleBaseTilingData* tilingData;
  typename ElemDag::VarType scalars;
};

}  // namespace AscendC

#endif  // ATP_ELEMENTWISE_SCH_H_
