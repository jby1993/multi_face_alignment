QT += core
QT -= gui

CONFIG += c++11

TARGET = multi_face_alignment
CONFIG += console
CONFIG -= app_bundle

TEMPLATE = app

SOURCES += serve_main.cpp \
    cnndensefeature.cpp \
    io_utils.cpp \
    face_imgs.cpp \
    part_3dmm_face.cpp \
    train.cpp \
    divide_orientation_space.cpp \
    random_tool.cpp

HEADERS += \
    cnndensefeature.h \
    io_utils.h \
    face_imgs.h \
    part_3dmm_face.h \
    tri_mesh.h \
    train.h \
    divide_orientation_space.h \
    random_num_generator.h \
    random_tool.h

INCLUDEPATH += /home/jby/caffe/include \
                            /home/jby/caffe/build/include \
                            /usr/include/eigen3 \
                            /usr/local/cuda-7.5/include \
                            /usr/include \
                            /usr/local/include \

LIBS += -L/home/jby/caffe/build/lib \
            -L/usr/lib/x86_64-linux-gnu \
            -L/usr/local/lib/OpenMesh \

LIBS += -lcaffe -lglog -lboost_system -lprotobuf \
        -lopencv_core -lopencv_highgui -lopencv_imgproc \
        -lOpenMeshCore -lOpenMeshTools \

