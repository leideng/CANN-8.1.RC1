cmake_minimum_required(VERSION 3.14)
project(cpu_kernels_llt)

set(CMAKE_CXX_COMPILER g++)
set(CMAKE_C_COMPILER gcc)

if ("${ASCEND_CUSTOM_PATH}" STREQUAL "")
    message(WARNING "ASCEND_CUSTOM_PATH was not set, use env var ASCEND_AICPU_PATH instead.")
    if ("$ENV{ASCEND_AICPU_PATH}" STREQUAL "")
        message(FATAL_ERROR "ASCEND_AICPU_PATH was not set, compile failed.")
        return()
    endif()
    set(ASCEND_CUSTOM_PATH $ENV{ASCEND_AICPU_PATH})
endif()

if (NOT "${IMPL}" STREQUAL "FALSE")
    add_custom_target(third_kernel)
    add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
    set(CMAKE_CXX_STANDARD 11)
    set(AICPU_OPP_ENV ${ASCEND_CUSTOM_PATH}/opp/built-in/op_impl/aicpu/aicpu_kernel)
    set(PROJECT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../../)
    set(OP_PATH ${PROJECT_PATH}/cpukernel/impl/)

    add_library(gtest SHARED IMPORTED)
    set_target_properties(gtest PROPERTIES IMPORTED_LOCATION ${PROJECT_PATH}/testcases/libs/gtest/libgtest.a)
    include(${PROJECT_PATH}/third_party/eigen.cmake)
    add_dependencies(third_kernel eigen_headers)
    include(${PROJECT_PATH}/third_party/secure_c_kernel.cmake)
    add_dependencies(third_kernel securec_kernel_headers)

    add_library(gtest_main SHARED IMPORTED)
    set_target_properties(gtest_main PROPERTIES IMPORTED_LOCATION ${PROJECT_PATH}/testcases/libs/gtest/libgtest_main.a)

	set(GTEST_INCLUDE  ${PROJECT_PATH}/testcases/libs/gtest/include)

	set(EIGEN_INCLUDE  ${PROJECT_PATH}/testcases/ut/third_party/src/eigen)
	set(SECURE_C_KERNEL_INCLUDE  ${PROJECT_PATH}/testcases/ut/third_party/src/secure_c_kernel/include)
    file(GLOB OP_IMPL_FILE ${OP_PATH}/**.cc)
    message(STATUS "OP_IMPL_FILE=${OP_IMPL_FILE}")
    file(GLOB TEST_FILE_CPP ${CMAKE_CURRENT_SOURCE_DIR}/**/**_impl.cpp)
    file(GLOB TEST_FILE_CC ${CMAKE_CURRENT_SOURCE_DIR}/**/**_impl.cc)
    file(GLOB OP_UTIL_CC ${OP_PATH}/utils/kernel_util.cc ${OP_PATH}/utils/bcast.cc ${OP_PATH}/utils/sparse_tensor.cc ${OP_PATH}/utils/eigen_tensor.cc)
    file(GLOB SECURE_C_KERNEL ${PROJECT_PATH}/testcases/ut/third_party/src/secure_c_kernel/src/**.c)
    set(_cpu_kernels_llt_files
      ${TEST_FILE_CPP}
      ${TEST_FILE_CC}
      ${OP_IMPL_FILE}
      ${OP_UTIL_CC}
      ${SECURE_C_KERNEL}
    )

    link_directories(${AICPU_OPP_ENV}/lib/aarch64
                     ${AICPU_OPP_ENV}/lib/x86
                     ${ASCEND_CUSTOM_PATH}/compiler/lib64)

    add_executable(cpu_kernels_llt
      ${_cpu_kernels_llt_files}
    )

    set(PROJECT_DIR "$ENV{PROJECT_PATH}")
    set(OP_PROTO_SRC_DIR ${PROJECT_PATH}/op_proto)
    include_directories(
            "${PROJECT_DIR}/fwkacllib/inc"
            "${PROJECT_DIR}/metadef/inc/external"
            "${OP_PROTO_SRC_DIR}/util"
            )
    target_include_directories(cpu_kernels_llt PRIVATE
      ${GTEST_INCLUDE}
      ${ASCEND_CUSTOM_PATH}/opp/built-in/op_impl/aicpu/aicpu_kernel/inc/
      ${OP_PATH}
      ${EIGEN_INCLUDE}
      ${PROJECT_PATH}/cpukernel/context/inc
      ${PROJECT_PATH}/cpukernel/impl/utils
      ${SECURE_C_KERNEL_INCLUDE}
      ${PROJECT_PATH}/cpukernel/context/inc
      ${PROJECT_PATH}/cpukernel/impl/utils
    )

    add_dependencies(cpu_kernels_llt third_kernel)
    target_link_libraries(cpu_kernels_llt
      gtest
      gtest_main
      gcov
      pthread
      cpu_kernels_context
      aicpu_nodedef_builder
      ascend_protobuf
      c_sec
      ascendalog
      -ldl
    )

    target_compile_options(cpu_kernels_llt PUBLIC
      -g
      -O0
      --coverage
      -fprofile-arcs
      -ftest-coverage
      -w
      -Dgoogle=ascend_private
      $<$<COMPILE_LANGUAGE:CXX>:-std=c++11>
      $<$<STREQUAL:${ENABLE_ASAN},true>:-fsanitize=address -fno-omit-frame-pointer -static-libasan -fsanitize=undefined -static-libubsan>
      -fPIC
	  -DEIGEN_MPL2_ONLY
    )

    set(AICPU_UTEST "${PROJECT_PATH}/out/bin/aicpu_all_ut_test")
    set_target_properties(cpu_kernels_llt PROPERTIES OUTPUT_NAME "aicpu_all_ut" RUNTIME_OUTPUT_DIRECTORY  ${AICPU_UTEST})
endif()
if(NOT "${PROTO}" STREQUAL "FALSE")
    add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

    set(PROJECT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../../)
    set(GTEST_DIR ${PROJECT_PATH}/testcases/libs/gtest)
    set(ATC_DIR ${ASCEND_CUSTOM_PATH}/compiler)
    set(OP_PROTO_SRC_DIR ${PROJECT_PATH}/op_proto)

    message(STATUS "ATC_DIR=${ATC_DIR}")

    enable_testing()

    set(PROJECT_DIR "$ENV{PROJECT_PATH}")
    include_directories(
            "${GTEST_DIR}/include"
            "${ATC_DIR}/include"
            "${OP_PROTO_SRC_DIR}"
            "${OP_PROTO_SRC_DIR}/util"
            "${PROJECT_DIR}/fwkacllib/inc"
            "${PROJECT_DIR}/metadef/inc/external"
            )

    aux_source_directory(${OP_PROTO_SRC_DIR} OP_PROTO_SOURCE_SRCS)
    file(GLOB OP_PROTO_TEST_FILES_CPP **/**proto.cpp)
    file(GLOB OP_PROTO_TEST_FILES_CC **/**proto.cc)

    link_directories(
            "${ATC_DIR}/lib64"
            "${GTEST_DIR}"
    )
	include_directories(${PROJECT_DIR}/fwkacllib/inc)
    include_directories(${PROJECT_DIR}/metadef/inc/external)
    include_directories(${PROJECT_DIR}/third_party/src/secure_c_proto/include)
    include_directories(${PROJECT_DIR})
    include_directories(${PROJECT_DIR}/metadef/)
    include_directories(${PROJECT_DIR}/metadef/inc)
    include_directories(${OP_PROTO_SRC_DIR}/util)
    include_directories(${PROJECT_DIR}/log)
    include_directories(${PROJECT_DIR}/metadef/inc/external/graph)

    aux_source_directory(${OP_PROTO_SRC_DIR} OP_PROTO_SOURCE_SRCS)
    file(GLOB OP_PROTO_TEST_FILES_CPP **proto.cpp)
    file(GLOB OP_PROTO_TEST_FILES_CC **proto.cc)

    link_directories(
            "${ATC_DIR}/lib64"
            "${GTEST_DIR}"
    )

    set(CUSTOM_OBJECT_NAME "aicpu_proto_test")
    add_executable(${CUSTOM_OBJECT_NAME}
            ${PROJECT_PATH}/testcases/ut/aicpu_test/test_main.cc
            ${OP_PROTO_SOURCE_SRCS}
            ${OP_PROTO_TEST_FILES_CPP}
            ${OP_PROTO_TEST_FILES_CC})

    target_link_libraries(${CUSTOM_OBJECT_NAME} gtest c_sec ascendalog pthread error_manager graph register cpu_context)
endif()
