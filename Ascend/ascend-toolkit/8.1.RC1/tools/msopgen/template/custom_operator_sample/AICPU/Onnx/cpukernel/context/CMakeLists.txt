cmake_minimum_required(VERSION 3.14.1)
project(cpu_kernels_context)
set(BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(proto_src_files
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_proto/proto/cpu_attr.proto
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_proto/proto/cpu_node_def.proto
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_proto/proto/cpu_tensor.proto
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_proto/proto/cpu_tensor_shape.proto
)
protobuf_generate(aicpu_proto PROTO_SRCS PROTO_HDRS ${proto_src_files} TARGET)

set (local_context_src_files
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_proto/node_def.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_proto/node_def_impl.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_proto/tensor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_proto/tensor_impl.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_proto/tensor_shape.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_proto/tensor_shape_impl.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_proto/attr_value.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_proto/attr_value_impl.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/common/device.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/common/context.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/common/device_cpu_kernel.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/common/cpu_kernel_register.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/common/cpu_kernel_utils.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/common/host_sharder.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/common/device_sharder.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/common/eigen_threadpool.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/common/eigen_threadpool_embedding.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/common/cpu_kernel_cache.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/common/async_event_util.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/common/async_cpu_kernel.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/cust_op/cust_op_log.cc
    ${PROTO_SRCS}
)

set (aicpu_cust_src_files
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_proto/node_def.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_proto/node_def_impl.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_proto/tensor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_proto/tensor_impl.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_proto/tensor_shape.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_proto/tensor_shape_impl.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_proto/attr_value.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_proto/attr_value_impl.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/common/host_sharder.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/common/device_sharder.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/common/eigen_threadpool.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/common/device.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/common/context.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/common/cpu_kernel_register.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/common/cpu_kernel_utils.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/common/cpu_kernel_cache.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/common/async_event_util.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/common/async_cpu_kernel.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/cust_op/cust_op_log.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/cust_op/cust_dlog_record.cc
    ${PROTO_SRCS}
)

set(local_context_stub_files
   ${CMAKE_CURRENT_SOURCE_DIR}/stub/aicpu_sharder.cc
)

if(BUILD_OPEN_PROJECT)
  set(_proto_include "${CMAKE_BINARY_DIR}/proto/aicpu_proto")
  set(local_context_inc_path
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_proto
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/stub
    ${AICPU_OPP_ENV}/inc
    ${OPP_INC_DIR}/include/aicpu
    ${_proto_include}
    ${Protobuf_INCLUDE}
    ${FWKACLLIB_INCLUDE}
    ${FWKACLLIB_INCLUDE}/aicpu/common
    ${CANN_ROOT}/third_party/air/base/exec_runtime
    ${C_SEC_INCLUDE}
    ${OPENSDK_DIR}/grpc/include/
  )

  add_library(cpu_kernels_context_static STATIC
    ${local_context_src_files}
    $<$<NOT:$<STREQUAL:${PRODUCT_SIDE},device>>:${local_context_stub_files}>
  )

  target_include_directories(cpu_kernels_context_static PRIVATE
    ${local_context_inc_path}
  )

  target_compile_definitions(cpu_kernels_context_static PRIVATE
    _FORTIFY_SOURCE=2
    google=ascend_private
    $<$<NOT:$<STREQUAL:${PRODUCT_SIDE},device>>:LOG_CPP>
  )

  target_compile_options(cpu_kernels_context_static PRIVATE
    -O2
    -std=c++11
    -ftrapv
    -fstack-protector-all
    -fPIC
    -Werror=sign-compare
    -Werror=unused-function
    -Werror=unused-variable
    -Wno-address
    $<$<STREQUAL:${PRODUCT_SIDE},device>:-fvisibility-inlines-hidden>
    $<$<STREQUAL:${PRODUCT_SIDE},device>:-fvisibility=hidden>
    -DEIGEN_MPL2_ONLY
    -DEigen=ascend_Eigen
  )

  target_link_libraries(cpu_kernels_context_static PRIVATE
    $<BUILD_INTERFACE:intf_pub_aicpu>
    Eigen3::Eigen
  )

  set(cpu_kernels_context_static ${CMAKE_CURRENT_BINARY_DIR}/libcpu_kernels_context_static.a)

  set_target_properties(cpu_kernels_context_static
      PROPERTIES
      OUTPUT_NAME cpu_kernels_context
  )

  add_library(aicpu_nodedef_builder STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/common/node_def_builder.cpp
  )

  add_dependencies(aicpu_nodedef_builder cpu_kernels_context_static)

  target_include_directories(aicpu_nodedef_builder PRIVATE
     ${local_context_inc_path}
  )

  target_compile_definitions(aicpu_nodedef_builder PRIVATE
    _FORTIFY_SOURCE=2
    google=ascend_private
  )

  target_compile_options(aicpu_nodedef_builder PRIVATE
    -O2
    -std=c++11
    -ftrapv
    -fstack-protector-all
    -fPIC
    -Werror
    -DEIGEN_MPL2_ONLY
    -DEigen=ascend_Eigen
  )

  if("x${PRODUCT_SIDE}" STREQUAL "xdevice")
    if (MINRC)
      set(CMAKE_CXX_COMPILER /usr/bin/aarch64-linux-gnu-g++)
      set(CMAKE_C_COMPILER /usr/bin/aarch64-linux-gnu-gcc)
    else()
      set(CMAKE_CXX_COMPILER ${TOOLCHAIN_DIR}/bin/aarch64-target-linux-gnu-g++)
      set(CMAKE_C_COMPILER ${TOOLCHAIN_DIR}/bin/aarch64-target-linux-gnu-gcc)
    endif()
    add_subdirectory(stub)

    set(OPS_AICPU_PATH "${INSTALL_PATH}/aicpu")
    cann_install(
      TARGET      cpu_kernels_context_static
      FILES       $<TARGET_FILE:cpu_kernels_context_static>
      DESTINATION "${OPS_AICPU_PATH}"
    )

    cann_install(
      TARGET      aicpu_nodedef_builder
      FILES       $<TARGET_FILE:aicpu_nodedef_builder>
      DESTINATION "${OPS_AICPU_PATH}"
    )
  endif()

else()
set(local_context_inc_path
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_proto
    ${TOP_DIR}/inc
    ${TOP_DIR}/inc/aicpu
    ${TOP_DIR}/inc/aicpu/common
    ${TOP_DIR}/inc/aicpu/cpu_kernels
    ${TOP_DIR}/inc/aicpu/aicpu_schedule/aicpu_sharder
    ${TOP_DIR}/inc/external/aicpu
    ${TOP_DIR}/libc_sec/include
    ${TOP_DIR}/abl/libc_sec/include
    ${TOP_DIR}/abl/mmpa/inc
    ${TOP_DIR}/abl/mmpa/inc/mmpa
    ${TOP_DIR}/open_source/eigen
    ${TOP_DIR}/asl/ops/cann/third_party/air/base/exec_runtime
    ${TOP_DIR}/out/${product}
    ${CMAKE_BINARY_DIR}/proto/aicpu_proto/proto/cpu_proto/proto
    ${CMAKE_BINARY_DIR}/proto/aicpu_proto/proto
    ${CMAKE_BINARY_DIR}/proto/aicpu_proto
)

set(aicpu_cust_inc_path
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_proto
    ${TOP_DIR}/inc
    ${TOP_DIR}/inc/aicpu
    ${TOP_DIR}/inc/aicpu/common
    ${TOP_DIR}/inc/aicpu/cpu_kernels
    ${TOP_DIR}/inc/aicpu/aicpu_schedule/aicpu_sharder
    ${TOP_DIR}/inc/external/aicpu
    ${TOP_DIR}/libc_sec/include
    ${TOP_DIR}/abl/libc_sec/include
    ${TOP_DIR}/abl/mmpa/inc
    ${TOP_DIR}/abl/mmpa/inc/mmpa
    ${TOP_DIR}/open_source/eigen
    ${TOP_DIR}/asl/ops/cann/third_party/air/base/exec_runtime
    ${TOP_DIR}/out/${product}
    ${CMAKE_BINARY_DIR}/proto/aicpu_proto/proto/cpu_proto/proto
    ${CMAKE_BINARY_DIR}/proto/aicpu_proto/proto
    ${CMAKE_BINARY_DIR}/proto/aicpu_proto
)

add_library(cpu_kernels_context SHARED
    ${local_context_src_files}
    ${local_context_stub_files}
)

add_library(aicpu_cust_log_static STATIC
    ${aicpu_cust_src_files}
)

add_dependencies(aicpu_cust_log_static aicpu_proto)

target_include_directories(aicpu_cust_log_static PRIVATE
     ${aicpu_cust_inc_path}
     ${local_context_stub_files}
)

target_link_libraries(aicpu_cust_log_static PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    $<BUILD_INTERFACE:slog_headers>
    $<BUILD_INTERFACE:cce_headers>
    ascend_protobuf_static
    -Wl,-z,relro,-z,now
    $<$<NOT:$<STREQUAL:${RESERVED_SYMBOL},true>>:-s>
    -ldl
    -shared
    $<$<STREQUAL:${x86_aarch64_host},x86_or_aarch64_on_host>:alog>
)

target_compile_definitions(aicpu_cust_log_static PRIVATE
    _FORTIFY_SOURCE=2
    google=ascend_private
    $<$<NOT:$<STREQUAL:${PRODUCT_SIDE},device>>:LOG_CPP>
)

target_compile_options(aicpu_cust_log_static PRIVATE
    -O2
    -std=c++11
    -ftrapv
    -fstack-protector-all
    -Werror=sign-compare
    -Werror=unused-function
    -Werror=unused-variable
    -Wno-address
    -fvisibility-inlines-hidden
    -fvisibility=hidden
)

add_dependencies(cpu_kernels_context aicpu_proto)

add_library(cpu_kernels_context_static STATIC
    ${local_context_src_files}
    ${local_context_stub_files}
)

add_dependencies(cpu_kernels_context_static aicpu_proto)

target_link_libraries(cpu_kernels_context PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    $<BUILD_INTERFACE:slog_headers>
    $<BUILD_INTERFACE:cce_headers>
    slog
    PUBLIC c_sec
    ascend_protobuf
    mmpa
    -ldl
)


##cpu_kernels_context
target_include_directories(cpu_kernels_context PRIVATE
     ${local_context_inc_path}
)

target_compile_definitions(cpu_kernels_context PRIVATE
    _FORTIFY_SOURCE=2
    $<$<STREQUAL:${PRODUCT_SIDE},host>:VISIBILITY>
    google=ascend_private
)

target_compile_options(cpu_kernels_context PRIVATE
    -O2
    -std=c++11
    -ftrapv
    -Werror=sign-compare
    -Werror=unused-function
    -Werror=unused-variable
    -Wno-address
    $<$<STREQUAL:${PRODUCT_SIDE},host>:-fvisibility-inlines-hidden>
    $<$<STREQUAL:${PRODUCT_SIDE},host>:-fvisibility=hidden>
    -DEIGEN_MPL2_ONLY
    -DEigen=ascend_Eigen
)

target_link_options(cpu_kernels_context PRIVATE
    -Wl,-z,relro,-z,now
    $<$<NOT:$<STREQUAL:${RESERVED_SYMBOL},true>>:-s>
    $<$<STREQUAL:${PRODUCT_SIDE},host>:-Wl,-Bsymbolic -Wl,--exclude-libs,ALL>
)

target_include_directories(cpu_kernels_context_static PRIVATE
     ${local_context_inc_path}
)

target_compile_definitions(cpu_kernels_context_static PRIVATE
    _FORTIFY_SOURCE=2
    google=ascend_private
    $<$<NOT:$<STREQUAL:${PRODUCT_SIDE},device>>:LOG_CPP>
)

target_compile_options(cpu_kernels_context_static PRIVATE
    -O2
    -std=c++11
    -ftrapv
    -fstack-protector-all
    -Werror=sign-compare
    -Werror=unused-function
    -Werror=unused-variable
    -Wno-address
    $<$<STREQUAL:${PRODUCT_SIDE},device>:-fvisibility-inlines-hidden>
    $<$<STREQUAL:${PRODUCT_SIDE},device>:-fvisibility=hidden>
    -DEIGEN_MPL2_ONLY
    -DEigen=ascend_Eigen
)

set(x86_aarch64_host not_x86_or_aarch64_on_host)
if((${PRODUCT_SIDE} STREQUAL host) AND (${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL x86_64 OR ${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL aarch64))
  set(x86_aarch64_host x86_or_aarch64_on_host)
endif()

target_link_libraries(cpu_kernels_context_static PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    $<BUILD_INTERFACE:slog_headers>
    $<BUILD_INTERFACE:cce_headers>
    ascend_protobuf_static
    -Wl,-z,relro,-z,now
    $<$<NOT:$<STREQUAL:${RESERVED_SYMBOL},true>>:-s>
    -ldl
    -shared
    $<$<STREQUAL:${x86_aarch64_host},x86_or_aarch64_on_host>:alog>
)

add_library(aicpu_nodedef_builder STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/common/node_def_builder.cpp
)

add_dependencies(aicpu_nodedef_builder cpu_kernels_context_static)

target_include_directories(aicpu_nodedef_builder PRIVATE
    ${local_context_inc_path}
)

target_compile_definitions(aicpu_nodedef_builder PRIVATE
    _FORTIFY_SOURCE=2
    google=ascend_private
)

target_compile_options(aicpu_nodedef_builder PRIVATE
    -O2
    -std=c++11
    -ftrapv
    -fstack-protector-all
    -fPIC
    -Werror
    -Wno-address
    -DEIGEN_MPL2_ONLY
    -DEigen=ascend_Eigen
)

if("x${PRODUCT}" STREQUAL "xascend310" OR "x${PRODUCT}" STREQUAL "xascend910" OR "x${PRODUCT}" STREQUAL "xascend310p" OR "x${PRODUCT}" STREQUAL "xascend310rc")
  set(INSTALL_SHARED_LIBRARY_DIR lib/${PRODUCT})
  set(INSTALL_STATIC_LIBRARY_DIR ${INSTALL_SHARED_LIBRARY_DIR})
elseif("${PRODUCT_SIDE}" STREQUAL "device" AND "x${PRODUCT}" STREQUAL "xascend")
  set(INSTALL_SHARED_LIBRARY_DIR lib/ascend/)
  set(INSTALL_STATIC_LIBRARY_DIR ${INSTALL_SHARED_LIBRARY_DIR})
elseif("${PRODUCT_SIDE}" STREQUAL "device" AND (("x${PRODUCT}" STREQUAL "xascend610") OR ("x${PRODUCT}" STREQUAL "xascend610Lite") OR ("x${PRODUCT}" STREQUAL "xbs9sx2a") OR ("x${PRODUCT}" STREQUAL "xmc61am21a") OR ("x${PRODUCT}" STREQUAL "xas31xm1") OR ("x${PRODUCT}" STREQUAL "xmc62cm12a")))
  # libcpu_kernel_context.a is directly obtained from opp package as ascend
  set(INSTALL_SHARED_LIBRARY_DIR lib/)
  set(INSTALL_STATIC_LIBRARY_DIR lib/ascend/)
else()
  set(INSTALL_SHARED_LIBRARY_DIR lib/)
  set(INSTALL_STATIC_LIBRARY_DIR ${INSTALL_SHARED_LIBRARY_DIR})
endif()

install(TARGETS cpu_kernels_context_static OPTIONAL
    EXPORT cpu_kernels_context_static-targets
    ARCHIVE DESTINATION ${INSTALL_STATIC_LIBRARY_DIR}
)

install(TARGETS aicpu_cust_log_static OPTIONAL
    EXPORT aicpu_cust_log_static-targets
    LIBRARY DESTINATION ${INSTALL_SHARED_LIBRARY_DIR}
)

install(TARGETS cpu_kernels_context OPTIONAL
    EXPORT cpu_kernels_context-targets
    LIBRARY DESTINATION ${INSTALL_SHARED_LIBRARY_DIR}
)

install(TARGETS aicpu_nodedef_builder OPTIONAL
    EXPORT aicpu_nodedef_builder-targets
    LIBRARY DESTINATION ${INSTALL_SHARED_LIBRARY_DIR}
)

set(cpu_kernels_context_static ${CMAKE_CURRENT_BINARY_DIR}/libcpu_kernels_context_static.a)

set(aicpu_cust_log_static ${CMAKE_CURRENT_BINARY_DIR}/libaicpu_cust_log_static.a)

set_target_properties(cpu_kernels_context_static
    PROPERTIES
    OUTPUT_NAME cpu_kernels_context
)

set_target_properties(aicpu_cust_log_static
    PROPERTIES
    OUTPUT_NAME aicpu_cust_log
)
endif()
