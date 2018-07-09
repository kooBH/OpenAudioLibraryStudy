SRC =
OBJ = $(SRC:.cu=.o) 
TARGET = $(OBJ:.o=)
FLAG = -Wno-deprecated-gpu-targets

#컴파일 옵션
COPITON=
#링크 옵션
LOPTION=

CC = nvcc

ifeq ($(SRC), 2_matrixMul.cu)
COPTION += -Xcompiler -fopenmp 
LOPTION += -lgomp
endif
ifeq ($(SRC), 5_cublas.cu)
LOPTION += -lcublas
endif
ifeq ($(SRC), 5_cublas_v2.cu)
LOPTION += -lcublas
endif
ifeq ($(SRC), 7_batchedCublas.cu)
LOPTION += -lcublas
endif
ifeq ($(SRC), 7_batchedCublas_stream.cu)
LOPTION += -lcublas
endif
ifeq ($(SRC), temp.cu)
LOPTION += -lcublas
endif

$(TARGET) : $(OBJ)
	$(CC)  $(FLAG) $(LOPTION) -o  out_$@ $^
	rm *.o

$(OBJ) : $(SRC)
	$(CC) $(FLAG) $(COPTION) -c $^

