#!/bin/sh

#gcc -DKML_ILP64 -m64 mkl-lab-solution.c -L${MKLROOT}/ -L${MKLROOT}/lib/intel64 -I${MKLROOT}/include -lmkl_intel_ilp64 -lm -lmkl_intel_thread -lmkl_core -lpthread -liomp5 -static 

#되는 코드
#gcc -m64 mkl-lab-solution.o -DMKL_ILP64 -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_sequential.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm -ldl -I${MKLROOT}/include -lm

TARGET=gemm
#TARGET=mkl-lab-solution.c

gcc -o $TARGET  -DMKL_ILP64 -m64 -I${MKLROOT}/include  $TARGET.c    -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_sequential.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm -ldl  -lm
