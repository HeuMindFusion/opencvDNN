cmake_minimum_required(VERSION 3.5)

project(testdnn)

set(CMAKE_CXX_STANDARD 14)

set(WINDOWS TRUE CACHE INTERNAL "")
set(PLATFORM_PREFIX "win64" CACHE INTERNAL "")
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
if (CMAKE_HOST_WIN32)
    set(HOME_DIR $ENV{HOMEDRIVE}$ENV{HOMEPATH} CACHE INTERNAL "")
else()
    set(HOME_DIR $ENV{HOME} CACHE INTERNAL "")
endif()
if ("TEST$ENV{MAKE_DEPS_WORKING_DIR}" STREQUAL "TEST")
    set(MAKE_DEPS_WORKING_DIR ${HOME_DIR}/makedeps)
else()    
    set(MAKE_DEPS_WORKING_DIR $ENV{MAKE_DEPS_WORKING_DIR})	
endif()

if (NOT ROOT_DIR_DEFINED)
    add_definitions(-DROOT_DIR="${PROJECT_SOURCE_DIR}")
    set(ROOT_DIR_DEFINED ON)
endif()

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

include(${MAKE_DEPS_WORKING_DIR}/${PLATFORM_PREFIX}/deps.cmake)


find_package(OpenCV 4.0 REQUIRED)
#find_package(OpenCV CONFIG PATHS ${OpenCV_DIR} NO_DEFAULT_PATH)

include_directories(${OpenCV_INCLUDE_DIRS})


find_package(Boost REQUIRED)

find_package(Eigen3 REQUIRED)


include_directories(${OpenCV_INCLUDE_DIRS})



include_directories(${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR})

project(camera_capture)

if (NOT ROOT_DIR_DEFINED)
    add_definitions(-DROOT_DIR="${PROJECT_SOURCE_DIR}")
    set(ROOT_DIR_DEFINED ON)
endif()

list(APPEND SAMPLE_CODE main.cpp
config.hpp)


add_executable(testdnn ${SAMPLE_CODE})
target_link_libraries(testdnn ${OpenCV_LIBRARIES} ${OpenCV_LIBS})

