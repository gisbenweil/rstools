#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "jpeg-static" for configuration "Release"
set_property(TARGET jpeg-static APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(jpeg-static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "C"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libjpeg.lib"
  )

list(APPEND _IMPORT_CHECK_TARGETS jpeg-static )
list(APPEND _IMPORT_CHECK_FILES_FOR_jpeg-static "${_IMPORT_PREFIX}/lib/libjpeg.lib" )

# Import target "jpeg-shared" for configuration "Release"
set_property(TARGET jpeg-shared APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(jpeg-shared PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/jpeg.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/jpeg.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS jpeg-shared )
list(APPEND _IMPORT_CHECK_FILES_FOR_jpeg-shared "${_IMPORT_PREFIX}/lib/jpeg.lib" "${_IMPORT_PREFIX}/bin/jpeg.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
