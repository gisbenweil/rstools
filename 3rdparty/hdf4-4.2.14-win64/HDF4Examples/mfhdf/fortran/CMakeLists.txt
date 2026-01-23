cmake_minimum_required (VERSION 3.2.2)
# --------------------------------------------------------------------
# Notes: When creating examples they should be prefixed
# with "f_ex_". This allows for easier filtering of the examples.
# --------------------------------------------------------------------
project (HDF4Examples_MFHDF_FORTRAN C CXX Fortran)

#-----------------------------------------------------------------------------
# Setup include Directories
#-----------------------------------------------------------------------------
INCLUDE_DIRECTORIES (
    ${CMAKE_Fortran_MODULE_DIRECTORY}
    ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
)

#-----------------------------------------------------------------------------
# Define Sources
#-----------------------------------------------------------------------------
include (Fortran_sourcefiles.cmake)

if (BUILD_TESTING)
  # Remove any output file left over from previous test run
  add_test (
      NAME MFHDF_FORTRAN_EXAMPLES-clearall-objects
      COMMAND    ${CMAKE_COMMAND}
          -E remove
          SDS.hdf
          SDSchunked.hdf
          SDScompressed.hdf
          SDSUNLIMITED.hdf
          SLABS.hdf
  )
  if (NOT "${last_test}" STREQUAL "")
    set_tests_properties (MFHDF_FORTRAN_EXAMPLES-clearall-objects PROPERTIES DEPENDS ${last_test} LABELS ${PROJECT_NAME})
  else ()
    set_tests_properties (MFHDF_FORTRAN_EXAMPLES-clearall-objects PROPERTIES LABELS ${PROJECT_NAME})
  endif ()
  set (last_test "MFHDF_FORTRAN_EXAMPLES-clearall-objects")
endif ()

foreach (example ${examples})
  get_filename_component(example_name ${example} NAME_WE)
  add_executable (f_exmf_${example_name} ${PROJECT_SOURCE_DIR}/${example})
  TARGET_FORTRAN_WIN_PROPERTIES (f_exmf_${example_name} ${LIB_TYPE} "" "")
  set_target_properties (f_exmf_${example_name} PROPERTIES LINKER_LANGUAGE Fortran)
  target_link_libraries (f_exmf_${example_name} ${LINK_LIBS})
  if (BUILD_TESTING)
    add_test (
        NAME f_exmftest_${example_name}
        COMMAND "${CMAKE_COMMAND}"
            -D "TEST_PROGRAM=$<TARGET_FILE:f_exmf_${example_name}>"
            -D "TEST_ARGS:STRING="
            -D "TEST_FOLDER=${PROJECT_BINARY_DIR}"
            -D "TEST_EXPECT=0"
            -D "TEST_SKIP_COMPARE=TRUE"
            -D "TEST_OUTPUT=${testname}.out"
            -D "TEST_LIBRARY_DIRECTORY=${CMAKE_TEST_LIB_DIRECTORY}"
            -P "${HDF4EX_RESOURCES_DIR}/runTest.cmake"
    )
    if (NOT "${last_test}" STREQUAL "")
      set_tests_properties (f_exmftest_${example_name} PROPERTIES DEPENDS ${last_test} LABELS ${PROJECT_NAME})
    else ()
      set_tests_properties (f_exmftest_${example_name} PROPERTIES LABELS ${PROJECT_NAME})
    endif ()
    set (last_test "f_exmftest_${example_name}")
  endif ()
endforeach ()

foreach (example ${skip_examples})
  get_filename_component(example_name ${example} NAME_WE)
  add_executable (f_exmf_${example_name} ${PROJECT_SOURCE_DIR}/${example})
  TARGET_FORTRAN_WIN_PROPERTIES (f_exmf_${example_name} ${LIB_TYPE} "" "")
  set_target_properties (f_exmf_${example_name} PROPERTIES LINKER_LANGUAGE Fortran)
  target_link_libraries (f_exmf_${example_name} ${LINK_LIBS})
  if (BUILD_TESTING)
    add_test (
        NAME f_exmftest_${example_name}
        COMMAND ${CMAKE_COMMAND} -E echo "SKIP f_exmftest_${example_name}"
    )
  endif ()
endforeach ()
