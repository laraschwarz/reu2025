cmake_minimum_required(VERSION 3.0.0)
project(VA)

# ...existing code...
set(OpenCV_DIR "$ENV{CONDA_PREFIX}/lib/cmake/opencv4")
include_directories("$ENV{CONDA_PREFIX}/include/opencv4")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories("include/")

add_executable(VA main.cpp detector.cpp HungarianAlg.cpp 
Kalman.cpp Multitracker.cpp yolo_utils.cpp Track.cpp Display.cpp SpeedEstimator.cpp
Intersection.cpp MultiCams.cpp)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(ONNXRUNTIME_ROOT_PATH /home/iaitech/onnxruntime)
set(ONNXRUNTIME_INCLUDE_DIRS ${ONNXRUNTIME_ROOT_PATH}/include/onnxruntime
                             ${ONNXRUNTIME_ROOT_PATH}/onnxruntime
                             ${ONNXRUNTIME_ROOT_PATH}/include/onnxruntime/core/session/)
set(ONNXRUNTIME_LIB ${ONNXRUNTIME_ROOT_PATH}/build/Linux/Release/libonnxruntime.so)
include_directories(${ONNXRUNTIME_INCLUDE_DIRS})
target_link_libraries(VA ${ONNXRUNTIME_LIB})

target_compile_features(VA PRIVATE cxx_std_14)
target_link_libraries(VA ${OpenCV_LIBS})

