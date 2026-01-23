#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "hdf4::xdr-static" for configuration "Release"
set_property(TARGET hdf4::xdr-static APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(hdf4::xdr-static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "C"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libxdr.lib"
  )

list(APPEND _IMPORT_CHECK_TARGETS hdf4::xdr-static )
list(APPEND _IMPORT_CHECK_FILES_FOR_hdf4::xdr-static "${_IMPORT_PREFIX}/lib/libxdr.lib" )

# Import target "hdf4::xdr-shared" for configuration "Release"
set_property(TARGET hdf4::xdr-shared APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(hdf4::xdr-shared PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/xdr.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/xdr.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS hdf4::xdr-shared )
list(APPEND _IMPORT_CHECK_FILES_FOR_hdf4::xdr-shared "${_IMPORT_PREFIX}/lib/xdr.lib" "${_IMPORT_PREFIX}/bin/xdr.dll" )

# Import target "hdf4::hdf-static" for configuration "Release"
set_property(TARGET hdf4::hdf-static APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(hdf4::hdf-static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "C"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libhdf.lib"
  )

list(APPEND _IMPORT_CHECK_TARGETS hdf4::hdf-static )
list(APPEND _IMPORT_CHECK_FILES_FOR_hdf4::hdf-static "${_IMPORT_PREFIX}/lib/libhdf.lib" )

# Import target "hdf4::hdf-shared" for configuration "Release"
set_property(TARGET hdf4::hdf-shared APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(hdf4::hdf-shared PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/hdf.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/hdf.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS hdf4::hdf-shared )
list(APPEND _IMPORT_CHECK_FILES_FOR_hdf4::hdf-shared "${_IMPORT_PREFIX}/lib/hdf.lib" "${_IMPORT_PREFIX}/bin/hdf.dll" )

# Import target "hdf4::mfhdf-static" for configuration "Release"
set_property(TARGET hdf4::mfhdf-static APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(hdf4::mfhdf-static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "C"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libmfhdf.lib"
  )

list(APPEND _IMPORT_CHECK_TARGETS hdf4::mfhdf-static )
list(APPEND _IMPORT_CHECK_FILES_FOR_hdf4::mfhdf-static "${_IMPORT_PREFIX}/lib/libmfhdf.lib" )

# Import target "hdf4::mfhdf-shared" for configuration "Release"
set_property(TARGET hdf4::mfhdf-shared APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(hdf4::mfhdf-shared PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/mfhdf.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/mfhdf.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS hdf4::mfhdf-shared )
list(APPEND _IMPORT_CHECK_FILES_FOR_hdf4::mfhdf-shared "${_IMPORT_PREFIX}/lib/mfhdf.lib" "${_IMPORT_PREFIX}/bin/mfhdf.dll" )

# Import target "hdf4::mfhdf_fcstub-static" for configuration "Release"
set_property(TARGET hdf4::mfhdf_fcstub-static APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(hdf4::mfhdf_fcstub-static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "C"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libmfhdf_fcstub.lib"
  )

list(APPEND _IMPORT_CHECK_TARGETS hdf4::mfhdf_fcstub-static )
list(APPEND _IMPORT_CHECK_FILES_FOR_hdf4::mfhdf_fcstub-static "${_IMPORT_PREFIX}/lib/libmfhdf_fcstub.lib" )

# Import target "hdf4::mfhdf_fcstub-shared" for configuration "Release"
set_property(TARGET hdf4::mfhdf_fcstub-shared APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(hdf4::mfhdf_fcstub-shared PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/mfhdf_fcstub.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/mfhdf_fcstub.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS hdf4::mfhdf_fcstub-shared )
list(APPEND _IMPORT_CHECK_FILES_FOR_hdf4::mfhdf_fcstub-shared "${_IMPORT_PREFIX}/lib/mfhdf_fcstub.lib" "${_IMPORT_PREFIX}/bin/mfhdf_fcstub.dll" )

# Import target "hdf4::mfhdf_fortran-static" for configuration "Release"
set_property(TARGET hdf4::mfhdf_fortran-static APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(hdf4::mfhdf_fortran-static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "Fortran"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libmfhdf_fortran.lib"
  )

list(APPEND _IMPORT_CHECK_TARGETS hdf4::mfhdf_fortran-static )
list(APPEND _IMPORT_CHECK_FILES_FOR_hdf4::mfhdf_fortran-static "${_IMPORT_PREFIX}/lib/libmfhdf_fortran.lib" )

# Import target "hdf4::mfhdf_fortran-shared" for configuration "Release"
set_property(TARGET hdf4::mfhdf_fortran-shared APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(hdf4::mfhdf_fortran-shared PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/mfhdf_fortran.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/mfhdf_fortran.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS hdf4::mfhdf_fortran-shared )
list(APPEND _IMPORT_CHECK_FILES_FOR_hdf4::mfhdf_fortran-shared "${_IMPORT_PREFIX}/lib/mfhdf_fortran.lib" "${_IMPORT_PREFIX}/bin/mfhdf_fortran.dll" )

# Import target "hdf4::hdf_fcstub-static" for configuration "Release"
set_property(TARGET hdf4::hdf_fcstub-static APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(hdf4::hdf_fcstub-static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "C"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libhdf_fcstub.lib"
  )

list(APPEND _IMPORT_CHECK_TARGETS hdf4::hdf_fcstub-static )
list(APPEND _IMPORT_CHECK_FILES_FOR_hdf4::hdf_fcstub-static "${_IMPORT_PREFIX}/lib/libhdf_fcstub.lib" )

# Import target "hdf4::hdf_fcstub-shared" for configuration "Release"
set_property(TARGET hdf4::hdf_fcstub-shared APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(hdf4::hdf_fcstub-shared PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/hdf_fcstub.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/hdf_fcstub.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS hdf4::hdf_fcstub-shared )
list(APPEND _IMPORT_CHECK_FILES_FOR_hdf4::hdf_fcstub-shared "${_IMPORT_PREFIX}/lib/hdf_fcstub.lib" "${_IMPORT_PREFIX}/bin/hdf_fcstub.dll" )

# Import target "hdf4::hdf_fortran-static" for configuration "Release"
set_property(TARGET hdf4::hdf_fortran-static APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(hdf4::hdf_fortran-static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "Fortran"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libhdf_fortran.lib"
  )

list(APPEND _IMPORT_CHECK_TARGETS hdf4::hdf_fortran-static )
list(APPEND _IMPORT_CHECK_FILES_FOR_hdf4::hdf_fortran-static "${_IMPORT_PREFIX}/lib/libhdf_fortran.lib" )

# Import target "hdf4::hdf_fortran-shared" for configuration "Release"
set_property(TARGET hdf4::hdf_fortran-shared APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(hdf4::hdf_fortran-shared PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/hdf_fortran.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/hdf_fortran.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS hdf4::hdf_fortran-shared )
list(APPEND _IMPORT_CHECK_FILES_FOR_hdf4::hdf_fortran-shared "${_IMPORT_PREFIX}/lib/hdf_fortran.lib" "${_IMPORT_PREFIX}/bin/hdf_fortran.dll" )

# Import target "hdf4::hdp" for configuration "Release"
set_property(TARGET hdf4::hdp APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(hdf4::hdp PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/hdp.exe"
  )

list(APPEND _IMPORT_CHECK_TARGETS hdf4::hdp )
list(APPEND _IMPORT_CHECK_FILES_FOR_hdf4::hdp "${_IMPORT_PREFIX}/bin/hdp.exe" )

# Import target "hdf4::hdfimport" for configuration "Release"
set_property(TARGET hdf4::hdfimport APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(hdf4::hdfimport PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/hdfimport.exe"
  )

list(APPEND _IMPORT_CHECK_TARGETS hdf4::hdfimport )
list(APPEND _IMPORT_CHECK_FILES_FOR_hdf4::hdfimport "${_IMPORT_PREFIX}/bin/hdfimport.exe" )

# Import target "hdf4::hdiff" for configuration "Release"
set_property(TARGET hdf4::hdiff APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(hdf4::hdiff PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/hdiff.exe"
  )

list(APPEND _IMPORT_CHECK_TARGETS hdf4::hdiff )
list(APPEND _IMPORT_CHECK_FILES_FOR_hdf4::hdiff "${_IMPORT_PREFIX}/bin/hdiff.exe" )

# Import target "hdf4::hrepack" for configuration "Release"
set_property(TARGET hdf4::hrepack APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(hdf4::hrepack PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/hrepack.exe"
  )

list(APPEND _IMPORT_CHECK_TARGETS hdf4::hrepack )
list(APPEND _IMPORT_CHECK_FILES_FOR_hdf4::hrepack "${_IMPORT_PREFIX}/bin/hrepack.exe" )

# Import target "hdf4::hrepack_check" for configuration "Release"
set_property(TARGET hdf4::hrepack_check APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(hdf4::hrepack_check PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/hrepack_check.exe"
  )

list(APPEND _IMPORT_CHECK_TARGETS hdf4::hrepack_check )
list(APPEND _IMPORT_CHECK_FILES_FOR_hdf4::hrepack_check "${_IMPORT_PREFIX}/bin/hrepack_check.exe" )

# Import target "hdf4::hdf_java" for configuration "Release"
set_property(TARGET hdf4::hdf_java APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(hdf4::hdf_java PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/hdf_java.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/hdf_java.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS hdf4::hdf_java )
list(APPEND _IMPORT_CHECK_FILES_FOR_hdf4::hdf_java "${_IMPORT_PREFIX}/lib/hdf_java.lib" "${_IMPORT_PREFIX}/lib/hdf_java.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
