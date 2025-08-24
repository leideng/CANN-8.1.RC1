//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef __CLANG_CCE_SIMT_H__
#define __CLANG_CCE_SIMT_H__

#include <type_traits>

namespace cce {

#define CCE_FUNC __attribute__((cce_builtin_api, always_inline))[aicore]
#define LAUNCH_BOUND(N) __attribute__((cce_launch_bounds(N)))

// forward declaration
template <int Dims> class item;

namespace detail {

// Type traits identical to those in std in newer versions. Can be removed when
// SYCL requires a newer version of the C++ standard.
// C++14
template <bool B, class T = void>
using enable_if_t = typename std::enable_if<B, T>::type;

/// Builder
class Builder {
public:
  Builder() = delete;

  template <int Dims>
  static CCE_FUNC item<Dims> createItem(unsigned long long Info, int G0, int G1,
                                        int G2) {
    return item<Dims>(Info, G0, G1, G2);
  }
};
} // namespace detail

template <int dimensions = 1> class range {
public:
  template <int N = dimensions>
  CCE_FUNC range(typename detail::enable_if_t<(N == 1), int> dim0 = 0)
      : common_array{dim0} {}
  template <int N = dimensions>
  CCE_FUNC range(typename detail::enable_if_t<(N == 2), int> dim0, int dim1)
      : common_array{dim0, dim1} {}

  template <int N = dimensions>
  CCE_FUNC range(typename detail::enable_if_t<(N == 3), int> dim0, int dim1,
                 int dim2)
      : common_array{dim0, dim1, dim2} {}

  CCE_FUNC range(const range<dimensions> &rhs) = default;
  CCE_FUNC range(range<dimensions> &&rhs) = default;
  CCE_FUNC range<dimensions> &operator=(const range<dimensions> &rhs) = default;
  CCE_FUNC range<dimensions> &operator=(range<dimensions> &&rhs) = default;
  CCE_FUNC range() = delete;

  /* -- common interface members -- */

  CCE_FUNC int get(int dimension) const { return common_array[dimension]; }
  CCE_FUNC int &operator[](int dimension) { return common_array[dimension]; }
  CCE_FUNC int operator[](int dimension) const {
    return common_array[dimension];
  }

private:
  int common_array[dimensions];
};

#ifdef __pp_deduction_guides
// Deduction guides
range(int)->range<1>;
range(int, int)->range<2>;
range(int, int, int)->range<3>;
#endif

/// helper  __get_local_id
// primary template,DO NOT define
template <int dim> static CCE_FUNC int __get_local_id(int dimension);

template <> static CCE_FUNC int __get_local_id<1>(int dimension) {
  // Currently, the hardware does not have NumDim configuration and always 3d
  // axis enabled, so we do the consistant reverse order mapping
  // Read Davinci SIMT programming proposal section 5.1.1. The implicit VF
  // object and SIMT_VF ABI

#if !defined(CCE_SIMT_HAVE_NUM_DIM_ENABLE_BITS)
  // how to assert dimension == 0?
  return __cce_simt_get_TID_Z();
#endif
}

template <> static CCE_FUNC int __get_local_id<2>(int dimension) {
#if !defined(CCE_SIMT_HAVE_NUM_DIM_ENABLE_BITS)
  // TODO: how to assert dimension == 0 | 1?
  if (dimension == 0)
    return __cce_simt_get_TID_Z();
  if (dimension == 1)
    return __cce_simt_get_TID_Y();
#endif
}

template <> static CCE_FUNC int __get_local_id<3>(int dimension) {
#if !defined(CCE_SIMT_HAVE_NUM_DIM_ENABLE_BITS)
  // how to assert dimension == 0 | 1?
  if (dimension == 0)
    return __cce_simt_get_TID_Z();
  if (dimension == 1)
    return __cce_simt_get_TID_Y();
  if (dimension == 2)
    return __cce_simt_get_TID_X();
#endif
}

/// helper  __get_local_range
// primary template,DO NOT define
template <int dim> static CCE_FUNC int __get_local_range(int dimension);

template <> static CCE_FUNC int __get_local_range<1>(int dimension) {
#if !defined(CCE_SIMT_HAVE_NUM_DIM_ENABLE_BITS)
  // how to assert dimension == 0?
  return __cce_simt_get_BLOCK_DIM_Z();
#endif
}

template <> static CCE_FUNC int __get_local_range<2>(int dimension) {
#if !defined(CCE_SIMT_HAVE_NUM_DIM_ENABLE_BITS)
  // how to assert dimension == 0 | 1?
  if (dimension == 0)
    return __cce_simt_get_BLOCK_DIM_Z();
  if (dimension == 1)
    return __cce_simt_get_BLOCK_DIM_Y();
#endif
}

template <> static CCE_FUNC int __get_local_range<3>(int dimension) {
#if !defined(CCE_SIMT_HAVE_NUM_DIM_ENABLE_BITS)
  // how to assert dimension == 0 | 1?
  if (dimension == 0)
    return __cce_simt_get_BLOCK_DIM_Z();
  if (dimension == 1)
    return __cce_simt_get_BLOCK_DIM_Y();
  if (dimension == 2)
    return __cce_simt_get_BLOCK_DIM_X();
#endif
}

/// helper  __get_local_linear_id
// primary template,DO NOT define
template <int dim> static CCE_FUNC int __get_local_linear_id();

template <> static CCE_FUNC int __get_local_linear_id<1>() {
  return __get_local_id<1>(0);
}

template <> static CCE_FUNC int __get_local_linear_id<2>() {
  return __get_local_id<2>(0) * __get_local_range<2>(1) + __get_local_id<2>(1);
}

template <> static CCE_FUNC int __get_local_linear_id<3>() {
  return __get_local_id<3>(0) * __get_local_range<3>(1) *
             __get_local_range<3>(2) +
         __get_local_id<3>(1) * __get_local_range<3>(2) + __get_local_id<3>(2);
}

// cce::item is equal to sycl::h_item
template <int dim> class item {
public:
  // item is not user constructable
  item() = delete;

  item(const item &hi) = default;

  item &operator=(const item &hi) = default;
  /* -- common interface members -- */
  CCE_FUNC int get_group_range(int dimension) const { return GSize[dimension]; }
  CCE_FUNC int get_group_id(int dimension) const { return GId[dimension]; }
  // SYCL style dimension order is reversed from opencl,
  // i.e., for range<3>: dim(2) is the continuous elements, then dim(1)
  CCE_FUNC int get_group_linear_id() const {
    if (dim == 1)
      return GId[0];

    if (dim == 2)
      return GId[0] * GSize[1] + GId[1];

    if (dim == 3)
      return (GId[0] * GSize[1] * GSize[2]) + (GId[1] * GSize[2]) + GId[2];
  }

  CCE_FUNC int get_global_range(int dimension) const {
    return get_global_range<dim>(dimension);
  }
  CCE_FUNC int get_global_id(int dimension) const {
    return get_global_id<dim>(dimension);
  }
  CCE_FUNC int get_global_linear_id() const {
    return get_global_linear_id<dim>();
  }

  CCE_FUNC int get_local_range(int dimension) const {
    return __get_local_range<dim>(dimension);
  }
  // local id has real intrinsics
  CCE_FUNC int get_local_id(int dimension) const {
    return __get_local_id<dim>(dimension);
  }
  CCE_FUNC int get_local_linear_id() const {
    return __get_local_linear_id<dim>();
  }

private:
  template <int N = dim> CCE_FUNC int get_global_linear_id() const {
    if (N == 1)
      return get_global_id(0);

    if (N == 2)
      return get_global_id(0) * get_global_range(1) + get_global_id(1);

    if (N == 3)
      return get_global_id(0) * get_global_range(1) * get_global_range(2) +
             get_global_id(1) * get_global_range(2) + get_global_id(2);
  }

  template <int N = dim> CCE_FUNC int get_global_id(int dimension) const {
    return get_group_id(dimension) * __get_local_range<N>(dimension) +
           __get_local_id<N>(dimension);
  }

  template <int N = dim> CCE_FUNC int get_global_range(int dimension) const {
    return __get_local_range<N>(dimension) * get_group_range(dimension);
  }

protected:
  friend class detail::Builder;
  // TODO: should do 3d dimension restoring from 1d blockId
  CCE_FUNC item(unsigned long long Info, int G0, int G1, int G2)
      : InvokeInfo(Info) {
    GSize[0] = G0;
    if (dim > 1)
      GSize[1] = G1;
    if (dim > 2)
      GSize[2] = G2;
  }

public:
  int GSize[dim]; // The GroupSize
  int GId[dim];   // The GroupId for each dimension
  unsigned long long InvokeInfo;
};

class HandlerVF {
public:
  // FIXME:: Not user constructable, HandlerVF() = delete;
  CCE_FUNC HandlerVF() { PassGSizeFromHost = false; };

  template <int dimensions = 1, typename SimtFuncTy>
  CCE_FUNC void parallel_for_work_item(range<dimensions> NumWorkItems,
                                       SimtFuncTy KernelObject) {
    cce::item<dimensions> Item = configSimtInvocation<dimensions>(NumWorkItems);
    // TODO: SIMT_VF invoke ABI lowering need to pass Info.Data as Xm parameter
    KernelObject(Item);
  }

  // Another version of PFWI, for current workround of lacking upack lambda
  // cpatured parameters, we should not expose this to user later
  template <int dimensions = 1, typename SimtFuncTy, typename... KArgsTy>
  CCE_FUNC void parallel_for_work_item(range<dimensions> NumWorkItems,
                                       SimtFuncTy KernelObject,
                                       KArgsTy... Args) {
    cce::item<dimensions> Item = configSimtInvocation<dimensions>(NumWorkItems);
    // TODO: SIMT_VF invoke ABI lowering need to pass Info.Data as Xm parameter
    KernelObject(Args..., Item);
  }

  // TODO: remove this later, tmp publish this function for manually pass GSize
  // from host
  CCE_FUNC void setGSizeFrom(int G0) { setGSize(G0, 1, 1); }
  CCE_FUNC void setGSizeFrom(int G0, int G1) { setGSize(G0, G1, 1); }
  CCE_FUNC void setGSizeFrom(int G0, int G1, int G2) { setGSize(G0, G1, G2); }

private:
  template <int dimensions = 1>
  CCE_FUNC cce::item<dimensions>
  configSimtInvocation(range<dimensions> NumWorkItems) {
    // Passing LocalSize
    setInvokeInfo<dimensions>(NumWorkItems);
    // Info.V.NumRegs not assigned here, it will be set in lowering call

    // HandlerVF accepts Gsize tuple from host for ND-Range compatibale
    // execution mode, for Hierarchical Parallelism mode it is (block_num, 1, 1)
    // and does not need pass from host.
    //   1) The main difference is that, for Hierarchical Mode, the <<<dim>>>
    //   follows legacy 1 dimensional design, so we don' t need to pass GSize
    //   from host part, and the orginal blockIdx, blockDim builtin variable is
    //   accessable inside kernel function. 2) For ND-range mode, it should pass
    //   from host side.
    // For more detail please refer to design doc:
    //   Section 3.1 of Davinci_SIMT_Programming_Guide.pdf

    if (!PassGSizeFromHost) {
      GSize[0] = block_num;
      // For SIMT Hierarchical Parallelism Execution Mode, no higher dim size.
      GSize[1] = 1;
      GSize[2] = 1;
    }
    cce::item<dimensions> Item = detail::Builder::createItem<dimensions>(
        Info, GSize[0], GSize[1], GSize[2]);

    // Restore GlobalSize & GlobalId from block_num and block_idx;
    restoreMultipleDimFromFlatten<dimensions>(Item);

    return Item;
  }

  template <int dim = 1>
  CCE_FUNC void restoreMultipleDimFromFlatten(item<dim> &Item) {
    int G0 = Item.get_group_range(0);
    int G1 = dim > 1 ? Item.get_group_range(1) : 1;
    int G2 = dim > 2 ? Item.get_group_range(2) : 1;
    Item.GId[0] = block_idx / (G1 * G2); // G1*G2 is plane Size
    int LinearIdInPlane = block_idx % (G1 * G2);
    if (dim > 1)
      Item.GId[1] = LinearIdInPlane / G2;
    if (dim > 2)
      Item.GId[2] = LinearIdInPlane % G2;
  }

  template <int dim = 1> CCE_FUNC void setInvokeInfo(range<dim> &NumWorkItems) {
    Info.Data = 0;
    Info.V.LS0 = NumWorkItems[0];               // LS0
    Info.V.LS1 = dim > 1 ? NumWorkItems[1] : 1; // LS1
    Info.V.LS2 = dim > 2 ? NumWorkItems[2] : 1; // LS2
  }

  // For calling after implicit WG creation for ND-Range compatible execution
  // mode
  CCE_FUNC void setGSize(int G0, int G1, int G2) {
    PassGSizeFromHost = true;
    GSize[0] = G0;
    GSize[1] = G1;
    GSize[2] = G2;
  }

private:
  // TODO: Hardware ISA does not give the bitfields order now.
  union SIMTVFInvokeInfo {
    struct {
      unsigned long long LS2 : 12;    // LocalSize2 - BLOCK_DIM.X
      unsigned long long RSVD0 : 4;   // Reserved
      unsigned long long LS1 : 12;    // LocalSize1 - BLOCK_DIM.Y
      unsigned long long RSVD1 : 4;   // Reserved
      unsigned long long LS0 : 12;    // LocalSIze0 - BLOCK_DIM.Z
      unsigned long long RSVD2 : 4;   // Reserved
      unsigned long long NumRegs : 8; // Number of Registers Per Thread
      unsigned long long RSVD3 : 8;   // Reserved
    } V;
    unsigned long long Data;

    CCE_FUNC operator unsigned long long() { return Data; }
  };

  SIMTVFInvokeInfo Info;
  int GSize[3]; // Passed from Host to WG by kernel implicit args
  bool PassGSizeFromHost{false};
};

} // namespace cce
#endif // __CLANG_CCE_SIMT_H__
