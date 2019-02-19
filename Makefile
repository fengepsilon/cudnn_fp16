CXX := g++
NVCC = /usr/local/cuda/bin/nvcc
BIN_DIR := ./
OBJ_DIR := ./obj



INC := -I./include  -I/usr/local/cuda/include
LIB := -L/usr/local/lib   -L./libs -L/usr/local/cuda/lib64
LIB +=    -lcudart -lcublas  -lcudnn -lopencv_world


SRC := main.cpp #$(shell find . -name "*.cpp")

OBJ := $(SRC:%.cpp=%.o)

CFLAGS := -DLINUX -DNDEBUG -O2 -Wno-sign-compare -fPIC -std=c++11  -DUSE_CUDNN

TARGET := $(BIN_DIR)/xxx.exe

.PHONY:all
all:$(TARGET) $(OBJ)

#build middle obj
%.o:%.cpp
	$(CXX) $(CFLAGS) $(INC) -o $@ -c $<

$(TARGET):$(OBJ)
	$(CXX) $(CFLAGS) -o $@ $(OBJ) $(LIB)

.PHONY:clean
clean:
	rm -f $(TARGET) $(OBJ)

.PHONY:love
love:clean all
