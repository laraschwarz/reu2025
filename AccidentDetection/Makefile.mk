# simple Makefile for AccidentDetection

CXX        = g++
CXXFLAGS   = -std=c++17 \
             -I. \
             -Ionnxruntime-linux-x64-1.22.0/include \
             $(shell pkg-config --cflags opencv4)

LDFLAGS    = -Lonnxruntime-linux-x64-1.22.0/lib \
             -lonnxruntime \
             -Wl,-rpath,$$PWD/onnxruntime-linux-x64-1.22.0/lib \
             $(shell pkg-config --libs opencv4)

SRCS       = main.cpp \
            yolo_utils.cpp \
            detector.cpp \
            MultiCams.cpp \
            Intersection.cpp \
            Multitracker.cpp \
            SpeedEstimator.cpp \
            Track.cpp \
            HungarianAlg.cpp \
            Kalman.cpp \

OBJS       = $(SRCS:.cpp=.o)
TARGET     = accident_detector

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(OBJS) $(LDFLAGS) -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)
