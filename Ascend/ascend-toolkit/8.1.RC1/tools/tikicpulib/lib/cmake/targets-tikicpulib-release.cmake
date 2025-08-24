#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "tikicpulib_stubreg" for configuration "Release"
set_property(TARGET tikicpulib_stubreg APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(tikicpulib_stubreg PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "c_sec"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libtikicpulib_stubreg.so"
  IMPORTED_SONAME_RELEASE "libtikicpulib_stubreg.so"
  )

list(APPEND _cmake_import_check_targets tikicpulib_stubreg )
list(APPEND _cmake_import_check_files_for_tikicpulib_stubreg "${_IMPORT_PREFIX}/lib/libtikicpulib_stubreg.so" )

# Import target "tikicpulib_cceprint" for configuration "Release"
set_property(TARGET tikicpulib_cceprint APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(tikicpulib_cceprint PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "tikicpulib_stubreg"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libtikicpulib_cceprint.so"
  IMPORTED_SONAME_RELEASE "libtikicpulib_cceprint.so"
  )

list(APPEND _cmake_import_check_targets tikicpulib_cceprint )
list(APPEND _cmake_import_check_files_for_tikicpulib_cceprint "${_IMPORT_PREFIX}/lib/libtikicpulib_cceprint.so" )

# Import target "tikicpulib_npuchk" for configuration "Release"
set_property(TARGET tikicpulib_npuchk APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(tikicpulib_npuchk PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "tikicpulib_stubreg;c_sec"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libtikicpulib_npuchk.so"
  IMPORTED_SONAME_RELEASE "libtikicpulib_npuchk.so"
  )

list(APPEND _cmake_import_check_targets tikicpulib_npuchk )
list(APPEND _cmake_import_check_files_for_tikicpulib_npuchk "${_IMPORT_PREFIX}/lib/libtikicpulib_npuchk.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
