cmake_minimum_required (VERSION 3.2.2)
project (HDF4Examples_JAVA_GR Java)

set (CMAKE_VERBOSE_MAKEFILE 1)

INCLUDE_DIRECTORIES (
    ${HDFJAVA_LIB_DIR}
    ${JAVA_INCLUDE_PATH}
    ${JAVA_INCLUDE_PATH2}
)

#-----------------------------------------------------------------------------
# Define Sources, one file per application
#-----------------------------------------------------------------------------
include (Java_sourcefiles.cmake)

if (WIN32 AND NOT CYGWIN)
  set (CMAKE_JAVA_INCLUDE_FLAG_SEP ";")
else ()
  set (CMAKE_JAVA_INCLUDE_FLAG_SEP ":")
endif ()

set (CMAKE_JAVA_CLASSPATH ".")
foreach (CMAKE_INCLUDE_PATH ${CMAKE_JAVA_INCLUDE_PATH})
  set (CMAKE_JAVA_CLASSPATH "${CMAKE_JAVA_CLASSPATH}${CMAKE_JAVA_INCLUDE_FLAG_SEP}${CMAKE_INCLUDE_PATH}")
endforeach ()

foreach (HCP_JAR ${CMAKE_JAVA_INCLUDE_PATH})
  get_filename_component (_HCP_FILE ${HCP_JAR} NAME)
  set (HDFJAVA_CLASSJARS "${_HCP_FILE} ${HDFJAVA_CLASSJARS}")
endforeach ()

foreach (example ${HDF_JAVA_EXAMPLES})
  get_filename_component(example_name ${example} NAME_WE)
  file (WRITE ${PROJECT_BINARY_DIR}/Manifest.txt
  "Main-Class: exAN.${example_name}
Class-Path: ${HDFJAVA_CLASSJARS}
"
  )
  ADD_JAR (${example_name} SOURCES ${example} MANIFEST ${PROJECT_BINARY_DIR}/Manifest.txt)
  get_target_property (${example_name}_JAR_FILE ${example_name} JAR_FILE)
endforeach ()

if (CMAKE_BUILD_TYPE MATCHES Debug)
  set (CMD_ARGS "-Dhdf.hdflib.HDFLibrary.loadLibraryName=hdf_java_debug;")
endif()

if (BUILD_TESTING)
  macro (ADD_H4_TEST resultfile resultcode)
    add_test (
        NAME jnative-h4-${resultfile}
        COMMAND "${CMAKE_COMMAND}"
            -D "TEST_TESTER=${CMAKE_Java_RUNTIME}"
            -D "TEST_PROGRAM=exGR.${resultfile}"
            -D "TEST_ARGS:STRING=${CMD_ARGS}${ARGN}"
            -D "TEST_CLASSPATH:STRING=${CMAKE_JAVA_CLASSPATH}${CMAKE_JAVA_INCLUDE_FLAG_SEP}${${resultfile}_JAR_FILE}"
            -D "TEST_LIBRARY_DIRECTORY=${HDF4_JAVA_LIBRARIES}"
            -D "TEST_FOLDER=${HDF4Examples_JAVA_GR_BINARY_DIR}"
            -D "TEST_OUTPUT=${HDF4Examples_JAVA_GR_BINARY_DIR}/${resultfile}.out"
            -D "TEST_REFERENCE=${resultfile}.txt"
            -D "TEST_EXPECT=${resultcode}"
            -D "TEST_SKIP_COMPARE=TRUE"
            -P "${HDF4EX_RESOURCES_DIR}/jrunTest.cmake"
    )
    if (NOT "${last_test}" STREQUAL "")
      set_tests_properties (jnative-h4-${resultfile} PROPERTIES DEPENDS ${last_test})
    endif ()
    set (last_test "jnative-h5-${resultfile}")
  endmacro ()

  foreach (example ${HDF_JAVA_EXAMPLES})
    get_filename_component(example_name ${example} NAME_WE)
    add_test (
        NAME jnative-h4-${example_name}-clearall-objects
        COMMAND    ${CMAKE_COMMAND}
            -E remove
            ${HDF4Examples_JAVA_GR_DIR}/${example_name}.hdf
            ${example_name}.out
            ${example_name}.out.err
    )
    if (NOT "${last_test}" STREQUAL "")
      set_tests_properties (jnative-h4-${example_name}-clearall-objects PROPERTIES DEPENDS ${last_test})
    endif ()
    add_test (
        NAME jnative-h4-${example_name}-copy-objects
        COMMAND    ${CMAKE_COMMAND}
            -E copy_if_different
            ${HDF4Examples_JAVA_GR_SOURCE_DIR}/testfiles/${example_name}.txt
            ${HDF4Examples_JAVA_GR_BINARY_DIR}/${example_name}.txt
    )
    set_tests_properties (jnative-h4-${example_name}-copy-objects PROPERTIES DEPENDS jnative-h4-${example_name}-clearall-objects)
    set (last_test "jnative-h4-${example_name}-copy-objects")
    ADD_H4_TEST (${example_name} 0)
  endforeach ()

endif ()
