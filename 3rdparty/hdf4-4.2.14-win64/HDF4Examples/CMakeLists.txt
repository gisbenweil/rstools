cmake_minimum_required (VERSION 3.2.2)
project (HDF4Examples C CXX)

#-----------------------------------------------------------------------------
# Instructions for use : Normal Build
#
# For standard build of HDF4 examples.
#-----------------------------------------------------------------------------
set (CMAKE_NO_SYSTEM_FROM_IMPORTED 1)

#-----------------------------------------------------------------------------
# Allow Visual Studio solution directories
#-----------------------------------------------------------------------------
# Provide a way for Visual Studio Express users to turn OFF the new FOLDER
# organization feature. Default to ON for non-Express users. Express users must
# explicitly turn off this option to build HDF4 in the Express IDE...
#
option (HDF4_USE_FOLDERS "Enable folder grouping of projects in IDEs." ON)
mark_as_advanced (HDF4_USE_FOLDERS)
if (HDF4_USE_FOLDERS)
  set_property (GLOBAL PROPERTY USE_FOLDERS ON)
endif ()

#-----------------------------------------------------------------------------
# Define some CMake variables for use later in the project
#-----------------------------------------------------------------------------
set (HDF4EX_RESOURCES_DIR       ${PROJECT_SOURCE_DIR}/config/cmake)
set (HDF4EX_HDFSRC_DIR          ${PROJECT_SOURCE_DIR}/hdf)
set (HDF4EX_HDF_FORTRAN_DIR     ${PROJECT_SOURCE_DIR}/hdf/fortran)
set (HDF4EX_MFHDFSRC_DIR        ${PROJECT_SOURCE_DIR}/mfhdf)
set (HDF4EX_MFHDF_FORTRAN_DIR   ${PROJECT_SOURCE_DIR}/mfhdf/fortran)
set (HDF4EX_JAVA_DIR            ${PROJECT_SOURCE_DIR}/JAVA)

if (HDF_BUILD_JAVA)
  find_package (Java)
endif ()

#-----------------------------------------------------------------------------
# Include some macros for reusable code
#-----------------------------------------------------------------------------
include (${HDF4EX_RESOURCES_DIR}/HDFMacros.cmake)
include (${HDF4EX_RESOURCES_DIR}/HDF4Macros.cmake)

SET_HDF_BUILD_TYPE()

#-----------------------------------------------------------------------------
# Setup output Directories
#-----------------------------------------------------------------------------
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY
    ${PROJECT_BINARY_DIR}/bin CACHE PATH "Single Directory for all Executables."
)
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY
    ${PROJECT_BINARY_DIR}/bin CACHE PATH "Single Directory for all Libraries"
)
set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY
    ${PROJECT_BINARY_DIR}/bin CACHE PATH "Single Directory for all static libraries."
)
set (CMAKE_Fortran_MODULE_DIRECTORY
    ${PROJECT_BINARY_DIR}/bin CACHE PATH "Single Directory for all fortran modules."
)
set (CMAKE_JAVA_TARGET_OUTPUT_DIR
    ${PROJECT_BINARY_DIR}/bin CACHE PATH "Single Directory for all java jars."
)
set (CMAKE_PDB_OUTPUT_DIRECTORY
    ${PROJECT_BINARY_DIR}/bin CACHE PATH "Single Directory for all pdb files."
)

#-----------------------------------------------------------------------------
# When building utility executables that generate other (source) files :
# we make use of the following variables defined in the root CMakeLists.
# Certain systems may add /Debug or /Release to output paths
# and we need to call the executable from inside the CMake configuration
#-----------------------------------------------------------------------------
set (EXE_EXT "")
if (WIN32)
  set (EXE_EXT ".exe")
  add_definitions (-D_BIND_TO_CURRENT_VCLIBS_VERSION=1)
  add_definitions (-D_CRT_SECURE_NO_WARNINGS)
  add_definitions (-D_CONSOLE)
endif ()

if (MSVC)
  set (CMAKE_MFC_FLAG 0)
endif ()

set (MAKE_SYSTEM)
if (CMAKE_BUILD_TOOL MATCHES "make")
  set (MAKE_SYSTEM 1)
endif ()

set (CFG_INIT "/${CMAKE_CFG_INTDIR}")
if (MAKE_SYSTEM)
  set (CFG_INIT "")
endif ()


if (HDF_BUILD_JAVA)
  INCLUDE_DIRECTORIES (
      ${JAVA_INCLUDE_PATH}
      ${JAVA_INCLUDE_PATH2}
  )
  set (CMAKE_JAVA_INCLUDE_PATH "")
endif ()

#-----------------------------------------------------------------------------
# Compiler specific flags : Shouldn't there be compiler tests for these
#-----------------------------------------------------------------------------
if (CMAKE_COMPILER_IS_GNUCC)
  set (CMAKE_C_FLAGS "${CMAKE_ANSI_CFLAGS} ${CMAKE_C_FLAGS} -std=c99 -fomit-frame-pointer -finline-functions -fno-common")
endif ()
if (CMAKE_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_LOADED)
  set (CMAKE_CXX_FLAGS "${CMAKE_ANSI_CFLAGS} ${CMAKE_CXX_FLAGS} -fomit-frame-pointer -finline-functions -fno-common")
endif ()

#-----------------------------------------------------------------------------
# HDF4 support
#-----------------------------------------------------------------------------
set (CMAKE_MODULE_PATH ${HDF4EX_RESOURCES_DIR} ${CMAKE_MODULE_PATH})

if (NOT HDF4_HDF4_HEADER)
  if (BUILD_SHARED_LIBS)
    set (FIND_HDF_COMPONENTS C shared)
  else ()
    set (FIND_HDF_COMPONENTS C static)
  endif ()
  if (HDF_BUILD_FORTRAN)
    set (FIND_HDF_COMPONENTS ${FIND_HDF_COMPONENTS} Fortran)
  endif ()
  if (HDF_BUILD_JAVA)
    set (FIND_HDF_COMPONENTS ${FIND_HDF_COMPONENTS} Java)
  endif ()
  message (STATUS "HDF4 find comps: ${FIND_HDF_COMPONENTS}")
  set (SEARCH_PACKAGE_NAME ${HDF4_PACKAGE_NAME})

  find_package (HDF4 NAMES ${SEARCH_PACKAGE_NAME} COMPONENTS ${FIND_HDF_COMPONENTS})
  message (STATUS "HDF4 C libs:${HDF4_FOUND} static:${HDF4_static_C_FOUND} and shared:${HDF4_shared_C_FOUND}")
  message (STATUS "HDF4 Fortran libs: static:${HDF4_static_Fortran_FOUND} and shared:${HDF4_shared_Fortran_FOUND}")
  message (STATUS "HDF4 Java libs: static:${HDF4_static_Java_FOUND} and shared:${HDF4_shared_Java_FOUND}")
  if (HDF4_FOUND)
    if (NOT TARGET hdf4::hdp)
      add_executable (hdf4::hdp IMPORTED)
    endif()
    if (NOT HDF4_static_C_FOUND AND NOT HDF4_shared_C_FOUND)
      set (FIND_HDF_COMPONENTS C)
      if (HDF_BUILD_FORTRAN)
        set (FIND_HDF_COMPONENTS ${FIND_HDF_COMPONENTS} Fortran)
      endif ()
      if (HDF_BUILD_JAVA)
        set (FIND_HDF_COMPONENTS ${FIND_HDF_COMPONENTS} Java)
      endif ()
      message (STATUS "HDF4 find comps: ${FIND_HDF_COMPONENTS}")

      find_package (HDF4 NAMES ${SEARCH_PACKAGE_NAME} COMPONENTS ${FIND_HDF_COMPONENTS})
      message (STATUS "HDF4 libs:${HDF4_FOUND} C:${HDF4_C_FOUND} Fortran:${HDF4_Fortran_FOUND}")
      set (LINK_LIBS ${LINK_LIBS} ${HDF4_LIBRARIES})
      if (HDF4_BUILD_SHARED_LIBS)
        add_definitions (-DH4_BUILT_AS_DYNAMIC_LIB)
      else ()
        add_definitions (-DH4_BUILT_AS_STATIC_LIB)
      endif ()
      if (BUILD_SHARED_LIBS AND WIN32)
        set_property (TARGET hdf4::hdp PROPERTY IMPORTED_LOCATION "${HDF4_TOOLS_DIR}/hdpdll")
      else ()
        set_property (TARGET hdf4::hdp PROPERTY IMPORTED_LOCATION "${HDF4_TOOLS_DIR}/hdp")
      endif ()
      if (HDF_BUILD_JAVA)
        set (CMAKE_JAVA_INCLUDE_PATH "${CMAKE_JAVA_INCLUDE_PATH};${HDF4_JAVA_INCLUDE_DIRS}")
        message (STATUS "HDF4 jars:${HDF4_JAVA_INCLUDE_DIRS}}")
      endif ()
    else ()
      if (BUILD_SHARED_LIBS AND HDF4_shared_C_FOUND)
        set (LINK_LIBS ${LINK_LIBS} ${HDF4_C_SHARED_LIBRARY})
      else ()
        set (LINK_LIBS ${LINK_LIBS} ${HDF4_C_STATIC_LIBRARY})
      endif ()
      if (HDF_BUILD_FORTRAN)
        if (BUILD_SHARED_LIBS AND HDF4_shared_Fortran_FOUND)
          set (LINK_LIBS ${LINK_LIBS} ${HDF4_FORTRAN_SHARED_LIBRARY})
        else ()
          set (LINK_LIBS ${LINK_LIBS} ${HDF4_FORTRAN_STATIC_LIBRARY})
        endif ()
      endif ()
      if (HDF_BUILD_JAVA)
        set (CMAKE_JAVA_INCLUDE_PATH "${CMAKE_JAVA_INCLUDE_PATH};${HDF4_JAVA_INCLUDE_DIRS}")
        message (STATUS "HDF4 jars:${HDF4_JAVA_INCLUDE_DIRS}}")
      endif ()
    endif ()

    set (HDF4_HAVE_HDF_H 1)
    set (HDF4_HAVE_HDF4 1)
    set (HDF4_HDF4_HEADER "hdf.h")
    set (HDF4_INCLUDE_DIR_GEN ${HDF4_INCLUDE_DIR})
    message (STATUS "HDF4-${HDF4_VERSION_STRING} found: INC=${HDF4_INCLUDE_DIR} TOOLS=${HDF4_TOOLS_DIR}")
  else ()
    message (FATAL_ERROR " HDF4 is Required for HDF4 Examples")
  endif ()
  set (HDF4_PACKAGE_NAME ${SEARCH_PACKAGE_NAME})
else ()
  # This project is being called from within another and HDF4 is already configured
  set (HDF4_HAVE_HDF_H 1)
  set (HDF4_HAVE_HDF4 1)
endif ()

INCLUDE_DIRECTORIES (${HDF4_INCLUDE_DIR})
if (HDF_BUILD_FORTRAN)
  INCLUDE_DIRECTORIES (${HDF4_INCLUDE_DIR_FORTRAN})
endif ()
message (STATUS "HDF4 link libs: ${LINK_LIBS}")
if (WIN32)
  set(CMAKE_TEST_LIB_DIRECTORY "${HDF4_TOOLS_DIR}")
else ()
  set(CMAKE_TEST_LIB_DIRECTORY "${HDF4_LIBRARY_PATH}")
endif ()

#-----------------------------------------------------------------------------
# Option to use Shared/Static libs, default is static
#-----------------------------------------------------------------------------
set (LIB_TYPE STATIC)
if (BUILD_SHARED_LIBS)
  set (LIB_TYPE SHARED)
endif ()
set (CMAKE_POSITION_INDEPENDENT_CODE ON)

#-----------------------------------------------------------------------------
# Option to allow the user to disable compiler warnings
#-----------------------------------------------------------------------------
option (HDF_DISABLE_COMPILER_WARNINGS "Disable compiler warnings" OFF)
if (HDF_DISABLE_COMPILER_WARNINGS)
  # MSVC uses /w to suppress warnings.  It also complains if another
  # warning level is given, so remove it.
  if (MSVC)
    set (HDF_WARNINGS_BLOCKED 1)
    string (REGEX REPLACE "(^| )([/-])W[0-9]( |$)" " " CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /W0")
    string (REGEX REPLACE "(^| )([/-])W[0-9]( |$)" " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W0")
  endif ()
  if (WIN32)
    add_definitions (-D_CRT_SECURE_NO_WARNINGS)
  endif ()
  # Borland uses -w- to suppress warnings.
  if (BORLAND)
    set (HDF_WARNINGS_BLOCKED 1)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -w-")
  endif ()

  # Most compilers use -w to suppress warnings.
  if (NOT HDF_WARNINGS_BLOCKED)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -w")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w")
  endif ()
endif ()

#-----------------------------------------------------------------------------
# This is in here to help some of the GCC based IDES like Eclipse
# and code blocks parse the compiler errors and warnings better.
#-----------------------------------------------------------------------------
if (CMAKE_COMPILER_IS_GNUCC)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fmessage-length=0")
endif ()
if (CMAKE_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_LOADED)
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fmessage-length=0")
endif ()

#-----------------------------------------------------------------------------
# All examples need the main include directories
#-----------------------------------------------------------------------------
INCLUDE_DIRECTORIES (${PROJECT_BINARY_DIR})

#-----------------------------------------------------------------------------
# Dashboard and Testing Settings
#-----------------------------------------------------------------------------
option (BUILD_TESTING "Build HDF4 Example Testing" OFF)
if (BUILD_TESTING)
  set (DART_TESTING_TIMEOUT 1200
      CACHE INTEGER
      "Timeout in seconds for each test (default 1200=20minutes)"
  )
  enable_testing ()
  include (CTest)
  include (${PROJECT_SOURCE_DIR}/CTestConfig.cmake)
  configure_file (${HDF4EX_RESOURCES_DIR}/CTestCustom.cmake ${PROJECT_BINARY_DIR}/CTestCustom.ctest @ONLY)
endif ()

#-----------------------------------------------------------------------------
# Build examples
#-----------------------------------------------------------------------------
if (HDF_BUILD_C)
  add_subdirectory (hdf/examples)
  add_subdirectory (mfhdf/examples)
endif ()

#-----------------------------------------------------------------------------
# Option to build Fortran examples
# Make sure this appears before the CONFIGURE_FILE step
# so that fortran name mangling is detected before writing H4config.h
#-----------------------------------------------------------------------------
# Set default name mangling : overridden by Fortran detection in fortran dir
set (H4_F77_FUNC "H4_F77_FUNC(name,NAME) name ## _")
set (H4_F77_FUNC_ "H4_F77_FUNC_(name,NAME) name ## __")
if (EXISTS "${PROJECT_SOURCE_DIR}/mfhdf/fortran" AND IS_DIRECTORY "${PROJECT_SOURCE_DIR}/mfhdf/fortran")
  option (HDF_BUILD_FORTRAN "Build FORTRAN support" OFF)
  if (HDF_BUILD_FORTRAN AND HDF4_BUILD_FORTRAN)
    if (WIN32 AND NOT CYGWIN)
      set (H4_F77_FUNC "H4_F77_FUNC(name,NAME) NAME")
      set (H4_F77_FUNC_ "H4_F77_FUNC_(name,NAME) NAME")
    endif ()
    include (${HDF4EX_RESOURCES_DIR}/HDF4UseFortran.cmake)
    add_subdirectory (mfhdf/fortran)
    add_subdirectory (hdf/fortran)
  endif ()
endif ()

#-----------------------------------------------------------------------------
# Option to build JAVA examples
#-----------------------------------------------------------------------------
if (EXISTS "${PROJECT_SOURCE_DIR}/JAVA" AND IS_DIRECTORY "${PROJECT_SOURCE_DIR}/JAVA")
  option (HDF_BUILD_JAVA "Build JAVA support" OFF)
  if (HDF_BUILD_JAVA AND HDF4_BUILD_JAVA)
    include (${HDF4EX_RESOURCES_DIR}/UseJava.cmake)
    add_subdirectory (JAVA)
  endif ()
endif ()

#-----------------------------------------------------------------------------
# Configure the README.txt file for the binary package
#-----------------------------------------------------------------------------
HDF_README_PROPERTIES(HDF_BUILD_FORTRAN)

#-----------------------------------------------------------------------------
# Source Packaging Option
#-----------------------------------------------------------------------------
option (PACK_EXAMPLE_SOURCE "Package HDF4 Example Source File" OFF)
if (PACK_EXAMPLE_SOURCE)
  set(CPACK_SOURCE_IGNORE_FILES
      "^${PROJECT_SOURCE_DIR}/\\\\.git*"
      "^${PROJECT_SOURCE_DIR}/\\\\.project"
  )
  set (CPACK_PACKAGE_VENDOR "HDF_Group")
  set (CPACK_PACKAGE_NAME "HDF4Examples")
  set (CPACK_GENERATOR "TGZ")
  set (CPACK_SOURCE_GENERATOR "TGZ")
  if (WIN32)
    set (CPACK_GENERATOR "ZIP")
    set (CPACK_SOURCE_GENERATOR "ZIP")
  endif()
  set (CPACK_TOPLEVEL_TAG "HDF4Examples-${CPACK_PACKAGE_VERSION}-Source")
  set (CPACK_SOURCE_TOPLEVEL_TAG "HDF4Examples-${CPACK_PACKAGE_VERSION}-Source")
  set (CPACK_SOURCE_PACKAGE_FILE_NAME "HDF4Examples-Source")
  INCLUDE(CPack)
  # Installation types
  cpack_add_install_type(Full DISPLAY_NAME "Everything")

  # Component groups
  cpack_add_component_group(Development
      EXPANDED
      DESCRIPTION "All of the examples to test HDF library"
  )
  cpack_add_component(source
      DISPLAY_NAME "Source Files"
      DESCRIPTION "Source files of HDF Library Examples"
      GROUP Development
      INSTALL_TYPES Full
  )
endif ()
