get_filename_component(ASCENDC_KERNEL_CMAKE_DIR "${CMAKE_CURRENT_LIST_DIR}" ABSOLUTE)

include(${ASCENDC_KERNEL_CMAKE_DIR}/host_config.cmake)
include(${ASCENDC_KERNEL_CMAKE_DIR}/host_intf.cmake)
include(${ASCENDC_KERNEL_CMAKE_DIR}/function.cmake)



