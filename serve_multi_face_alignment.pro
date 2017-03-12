QT += core
QT -= gui

CONFIG += c++11

TARGET = multi_face_alignment
CONFIG += console
CONFIG -= app_bundle
DEFINES += USE_CNNFEATURE
TEMPLATE = app

QMAKE_CXXFLAGS += -fopenmp

SOURCES += serve_main.cpp \
    cnndensefeature.cpp \
    io_utils.cpp \
    face_imgs.cpp \
    face_img.cpp \
    part_3dmm_face.cpp \
    train.cpp \
    train2.cpp \
    train3.cpp \
    divide_orientation_space.cpp \
    random_tool.cpp \
    siftdectector.cpp

HEADERS += \
    cnndensefeature.h \
    io_utils.h \
    face_imgs.h \
    face_img.h \
    part_3dmm_face.h \
    tri_mesh.h \
    train.h \
    train2.h \
    train3.h \
    divide_orientation_space.h \
    random_num_generator.h \
    random_tool.h \
    supplement_gpu_math_functions.hpp \
    siftdectector.h

INCLUDEPATH += /home/jby/caffe/include \
                            /home/jby/caffe/build/include \
                            /home/jby/eigen3 \
                            /usr/local/cuda-7.5/include \
                            /usr/include \
                            /usr/local/include \
			    /home/jby/vlfeat-0.9.20/vl \

LIBS += -L/home/jby/caffe/build/lib \
            -L/usr/local/lib \
	    -L/usr/lib64/ \
	-L/usr/local/cuda-7.5/lib64 \
	-L/home/jby/vlfeat-0.9.20/bin/glnxa64 \

LIBS += -lcaffe -lglog -lboost_system -lprotobuf -lgflags \
        -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs\
        -lOpenMeshCore -lOpenMeshTools \
	-lgomp -lpthread \
        -lcuda -lcudart -lcublas\
	-lvl \

