cmake_minimum_required(VERSION 3.3)
find_package(OpenCV REQUIRED)

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

project(CV)
SET_PROPERTY(GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS TRUE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
include_directories(${OpenCV_INCLUDE_DIRS})
set(SOURCE_FILES dominant_plane_tracking.cpp)
add_executable(dom ${SOURCE_FILES})
target_link_libraries(dom ${OpenCV_LIBS})
