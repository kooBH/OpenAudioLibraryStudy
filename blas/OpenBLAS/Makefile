CC = gcc
CCFLAGS = -L. -lopenblas -lpthread


gemm : gemm.o
	$(CC) -o $@ $^ $(CCFLAGS)

gemm.o : gemm.c
	$(CC) -c $^

clean : 
	rm -f *.o
