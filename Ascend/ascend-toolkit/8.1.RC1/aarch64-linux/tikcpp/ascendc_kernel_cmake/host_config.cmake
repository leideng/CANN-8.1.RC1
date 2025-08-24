set(ascend910b_list ascend910b1 ascend910b2 ascend910b2c ascend910b3 ascend910b4 ascend910b4-1 ascend910_9391 ascend910_9381 ascend910_9372 ascend910_9392 ascend910_9382 ascend910_9362)
set(ascend910_list  ascend910a ascend910proa ascend910b ascend910prob ascend910premiuma)
set(ascend310p_list ascend310p1 ascend310p3 ascend310p5 ascend310p7 ascend310p3vir01 ascend310p3vir02 ascend310p3vir04 ascend310p3vir08)
set(ascend310b_list ascend310b1 ascend310b2 ascend310b3 ascend310b4)
set(all_product ${ascend910b_list} ${ascend910_list} ${ascend310p_list})

if(NOT DEFINED SOC_VERSION)
    message(FATAL_ERROR "SOC_VERSION value not set.")
endif()

string(TOLOWER "${SOC_VERSION}" _LOWER_SOC_VERSION)

if(_LOWER_SOC_VERSION IN_LIST ascend910b_list)
    set(DYNAMIC_MODE ON)
    set(BUILD_MODE   aiv)
elseif(_LOWER_SOC_VERSION IN_LIST ascend910_list)
    set(BUILD_MODE   c100)
elseif(_LOWER_SOC_VERSION IN_LIST ascend310p_list)
    set(BUILD_MODE   m200)
elseif(_LOWER_SOC_VERSION IN_LIST ascend310b_list)
    set(BUILD_MODE   m300)
else()
    message(FATAL_ERROR "SOC_VERSION ${SOC_VERSION} does not support, the support list is ${all_product}")
endif()

if(NOT DEFINED RUN_MODE)
    set(RUN_MODE "npu")
endif()

if(NOT DEFINED ASCEND_KERNEL_LAUNCH_ONLY)
    set(ASCEND_KERNEL_LAUNCH_ONLY OFF)
endif()

if (NOT EXISTS "${ASCEND_CANN_PACKAGE_PATH}")
    message(FATAL_ERROR "${ASCEND_CANN_PACKAGE_PATH} does not exist, please check the setting of ASCEND_CANN_PACKAGE_PATH.")
endif()

set(ASCEND_PYTHON_EXECUTABLE "python3" CACHE STRING "python executable program")

if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/tools/ccec_compiler)
    set(ASCENDC_DEVKIT_PATH ${ASCEND_CANN_PACKAGE_PATH}/tools)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/compiler/ccec_compiler)
    set(ASCENDC_DEVKIT_PATH ${ASCEND_CANN_PACKAGE_PATH}/compiler)
else()
    set(ASCENDC_DEVKIT_PATH ${ASCEND_CANN_PACKAGE_PATH}/ascendc_devkit)
endif()

set(CCEC_PATH           ${ASCENDC_DEVKIT_PATH}/ccec_compiler/bin)
set(CCEC_LINKER        "${CCEC_PATH}/ld.lld")

set(ASCENDC_RUNTIME_OBJ_TARGET       ascendc_runtime_obj)
set(ASCENDC_RUNTIME_STATIC_TARGET    ascendc_runtime_static)
set(ASCENDC_RUNTIME_CONFIG           ascendc_runtime.cmake)
set(ASCENDC_PACK_KERNEL              ${ASCEND_CANN_PACKAGE_PATH}/bin/ascendc_pack_kernel)
set(ASCENDC_RUNTIME                  ${ASCEND_CANN_PACKAGE_PATH}/lib64/libascendc_runtime.a)

set(CMAKE_SKIP_RPATH TRUE)
include(ExternalProject)
