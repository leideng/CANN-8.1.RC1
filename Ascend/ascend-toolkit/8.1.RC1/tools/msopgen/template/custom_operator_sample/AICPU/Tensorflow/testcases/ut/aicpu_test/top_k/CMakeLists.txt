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
    set(PROJECT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../../../)
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
	set(SECURE_C_KERNEL_INCLUDE  ${PROJECT_PATH}/testcases/ut/aicpu_test/third_party/src/secure_c_kernel/include)

    if(EXISTS "${OP_PATH}/top_k_kernels.cc")
        set(OP_IMPL_FILE ${OP_PATH}/top_k_kernels.cc)
    elseif(EXISTS "${OP_PATH}/top_k_kernel.cc")
        message("top_k_kernels.cc was not found, top_k_kernel.cc was found as the impl file")
        set(OP_IMPL_FILE ${OP_PATH}/top_k_kernel.cc)
    elseif(EXISTS "${OP_PATH}/top_k.cc")
        message("top_k_kernels.cc and top_k_kernel.cc were not found, "
                "top_k.cc was found as the impl file")
        set(OP_IMPL_FILE ${OP_PATH}/top_k.cc)
    else()
        message("top_k_kernels.cc, top_k.cc and top_k_kernel.cc "
                "were not found, will found the impl file whose name matches top_k_**.cc")
        file(GLOB OP_IMPL_FILE ${OP_PATH}/top_k_**.cc)
    endif()

    file(GLOB TEST_FILE_CPP ${CMAKE_CURRENT_SOURCE_DIR}/**_impl.cpp)
    file(GLOB TEST_FILE_CC ${CMAKE_CURRENT_SOURCE_DIR}/**_impl.cc)
    file(GLOB SECURE_C_KERNEL ${PROJECT_PATH}/testcases/ut/aicpu_test/third_party/src/secure_c_kernel/src/**.c)
    file(GLOB OP_UTILS ${OP_PATH}/utils/kernel_util.cc ${OP_PATH}/utils/kernel_util.cc)
    set(_cpu_kernels_llt_files
      ${TEST_FILE_CPP}
      ${TEST_FILE_CC}
      ${OP_IMPL_FILE}
      ${SECURE_C_KERNEL}
      ${OP_UTILS}
    )

    link_directories(${AICPU_OPP_ENV}/lib/aarch64
                     ${AICPU_OPP_ENV}/lib/x86
                     ${ASCEND_CUSTOM_PATH}/compiler/lib64)

    add_executable(cpu_kernels_llt
      ${_cpu_kernels_llt_files}
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

    set(AICPU_UTEST "${PROJECT_PATH}/out/bin/aicpu_top_k_ut_test")
    set_target_properties(cpu_kernels_llt PROPERTIES OUTPUT_NAME "aicpu_top_k_ut" RUNTIME_OUTPUT_DIRECTORY  ${AICPU_UTEST})
endif()
if(NOT "${PROTO}" STREQUAL "FALSE")
    add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

    set(PROJECT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../../../)
    set(GTEST_DIR ${PROJECT_PATH}/testcases/libs/gtest)
    set(ATC_DIR ${ASCEND_CUSTOM_PATH}/compiler)
    set(OP_PROTO_SRC_DIR ${PROJECT_PATH}/op_proto)

    message(STATUS "ATC_DIR=${ATC_DIR}")

    enable_testing()

    include_directories(
            "${GTEST_DIR}/include"
            "${ATC_DIR}/include"
            "${OP_PROTO_SRC_DIR}"
            )

    aux_source_directory(${OP_PROTO_SRC_DIR} OP_PROTO_SOURCE_SRCS)
    file(GLOB OP_PROTO_TEST_FILES_CPP **proto.cpp)
    file(GLOB OP_PROTO_TEST_FILES_CC **proto.cc)

    link_directories(
            "${ATC_DIR}/lib64"
            "${GTEST_DIR}"
    )

    set(CUSTOM_OBJECT_NAME "top_k_proto_test")
    add_executable(${CUSTOM_OBJECT_NAME}
            ${PROJECT_PATH}/testcases/ut/aicpu_test/test_main.cc
            ${OP_PROTO_SOURCE_SRCS}
            ${OP_PROTO_TEST_FILES_CPP}
            ${OP_PROTO_TEST_FILES_CC})

    target_link_libraries(${CUSTOM_OBJECT_NAME} gtest c_sec ascendalog pthread error_manager graph register)
endif()
