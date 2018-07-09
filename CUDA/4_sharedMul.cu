#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MATRIX_TYPE int
#define BLOCK_SIZE 8
#define THREAD_SIZE 64
#define TILE 64

void stopwatch(int);
//CUDA 배열 곱 
__global__ void cuda_mul(MATRIX_TYPE*,MATRIX_TYPE*,MATRIX_TYPE*,int);

__global__ void shared_mul(MATRIX_TYPE*,MATRIX_TYPE*,MATRIX_TYPE*,int);

__global__ void exam_mul(MATRIX_TYPE*,MATRIX_TYPE*,MATRIX_TYPE*,int);

int main()
{

	//1024 by 1024 행렬
	const int width = 1024;
	const int height = width;
	const int matrix_size = width*height;
	const int buffer_size = matrix_size*sizeof(MATRIX_TYPE);

	MATRIX_TYPE *host_A,*host_B,*host_C;
	
	host_A = (MATRIX_TYPE*)malloc(buffer_size);
	host_B = (MATRIX_TYPE*)malloc(buffer_size);
	host_C = (MATRIX_TYPE*)malloc(buffer_size);
	
	for(int i=0;i<matrix_size;i++)
	{
	host_A[i] = i;
	host_B[i] = i;
	host_C[i] =0;
	}
	

	printf("Multiply matrix (%dX%d ) * (%dX%d)\n",width,width,width,width);

	MATRIX_TYPE *device_A,*device_B,*device_C;

	dim3 Db(1024,1024,1);


	cudaMalloc((void**)&device_A,buffer_size  );
	cudaMalloc((void**)&device_B,buffer_size  );
	cudaMalloc((void**)&device_C,buffer_size  );

			
	printf("cuda_mul\n");
	stopwatch(0);
	cudaMemcpy(device_A,host_A,buffer_size,cudaMemcpyHostToDevice);
	cudaMemcpy(device_B,host_B,buffer_size,cudaMemcpyHostToDevice);

	cuda_mul<<<1,Db>>>(device_A,device_B,device_C,width);
	cudaMemcpy(host_C,device_C,buffer_size,cudaMemcpyDeviceToHost);
	stopwatch(1);

	for(int i=0;i<matrix_size;i++)
	{
	host_A[i] = i;
	host_B[i] = i;
	host_C[i] =0;
	}

	dim3 Sg(BLOCK_SIZE,BLOCK_SIZE,1);
	dim3 Sb(THREAD_SIZE,THREAD_SIZE,1);
	
	printf("shared_mul\n");
	stopwatch(0);
	cudaMemcpy(device_A,host_A,buffer_size,cudaMemcpyHostToDevice);
	cudaMemcpy(device_B,host_B,buffer_size,cudaMemcpyHostToDevice);
	exam_mul<<<1,Db>>>(device_A,device_B,device_C,width);
	cudaMemcpy(host_C,device_C,buffer_size,cudaMemcpyDeviceToHost);
	stopwatch(1);
	
	
	
	cudaFree(device_A);
	cudaFree(device_B);
	cudaFree(device_C);

	free(host_A);	
	free(host_B);	
	free(host_C);	

	
	return 0;
}

__global__ void cuda_mul(MATRIX_TYPE* A, MATRIX_TYPE* B, MATRIX_TYPE* C, int w)
{

	MATRIX_TYPE v;
	v = 0;

	for(int i =0;i<w;i++)
	{
		v += A[threadIdx.y*w + i] * B[threadIdx.x *w + i];
	}

	C[threadIdx.x *w + threadIdx.y] = v;

}


__global__ void shared_mul(MATRIX_TYPE*A,MATRIX_TYPE*B,MATRIX_TYPE*C,int w)
{

/*
	Dg(16,16,1)
	Db(64,64,1)

  0,0    1,0
	---------
	|
0,1	|    1,1  
	|

		1 0 2 4
	 1	
     0
     2
     4

*/
	__shared__ MATRIX_TYPE SA[THREAD_SIZE][THREAD_SIZE];
	__shared__ MATRIX_TYPE SB[THREAD_SIZE][THREAD_SIZE];

	MATRIX_TYPE v;

	SA[threadIdx.x][threadIdx.y] = A[blockIdx.y *w +blockIdx.x];
	SB[threadIdx.x][threadIdx.y] = B[blockIdx.x *w +blockIdx.y];

	v = 0;


	/*
	A 의 한 타일을 사용하는 모든 B의 타일들을 연산
		
	
	O O O O    X X X X 
	O O O O    O O O O  
	X O O O    O O O O
    O O O O    O O O O 
	*/

}	

void stopwatch(int flag)
{
	const long long NANOS = 1000000000LL;
	static struct timespec startTS,endTS;
	static long long Diff = 0;
	
	//start
	if(flag == 0)
	{
		Diff = 0;
		if(-1 == clock_gettime(CLOCK_MONOTONIC,&startTS))
			printf("Failed to call clock_gettime\n");
	}
	//end
	else if(flag == 1)
	{
		
		if(-1 == clock_gettime(CLOCK_MONOTONIC,&endTS))
			printf("Failed to call clock_gettime\n");
		Diff = NANOS * (endTS.tv_sec - startTS.tv_sec) + (endTS.tv_nsec - startTS.tv_nsec);
		
		printf("elapsed time : % lld micros\n",Diff/1000);
	}
	else
	{
		printf("wrong flag | 0 : start, 1 : end\n");
	}

}
__global__ void exam_mul(MATRIX_TYPE*A,MATRIX_TYPE*B,MATRIX_TYPE*C,int w)
{
	int bx = blockIdx.x;
	int by = blockIdx.y;

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	
	int aBegin = w * TILE * by;
	
	int aEnd = aBegin +w -1;
	
	int aStep = TILE;

	int bBegin = TILE *bx;
	
	int bStep = TILE * w;

	MATRIX_TYPE Csub = 0;

	for(int a = aBegin, b = bBegin;
			a <= aEnd;
			a += aStep, b+= bStep)
	{
		__shared__ MATRIX_TYPE As[TILE][TILE];
		__shared__ MATRIX_TYPE Bs[TILE][TILE];

		As[ty][tx] = A[a + w * ty + tx];
		Bs[ty][tx] = B[b + w * ty + tx];

		__syncthreads();


		for(int k=0;k<TILE;k++)
			Csub += As[ty][k] * Bs[k][tx];
	
		__syncthreads();	
	
	}	

	int c = w * TILE * by + TILE * bx;
	C[c + w * ty + ty] = Csub;

}

