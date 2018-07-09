#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "cublas_v2.h"

#define MAT_TYPE double
#define MAT_SIZE 1024
#define N MAT_SIZE
#define N2 MAT_SIZE*MAT_SIZE

#define BLOCK 256
#define THREAD 512

void stopwatch(int);
__global__ void cuda_mul(MAT_TYPE*,MAT_TYPE*,MAT_TYPE*,int);
int main()
{
	MAT_TYPE* host_A;
	MAT_TYPE* host_B;
	MAT_TYPE* host_C;

	MAT_TYPE* dev_A;
	MAT_TYPE* dev_B;
	MAT_TYPE* dev_C;
	
	double alpha = 1.0;
	double beta = 0.0;
	
	host_A = (MAT_TYPE*)malloc(sizeof(MAT_TYPE)*N2);
	host_B = (MAT_TYPE*)malloc(sizeof(MAT_TYPE)*N2);
	host_C = (MAT_TYPE*)malloc(sizeof(MAT_TYPE)*N2);

	for (int i=0;i < N2; i++)
	{
		host_A[i] = rand()/(MAT_TYPE)RAND_MAX;
		host_B[i] = rand()/(MAT_TYPE)RAND_MAX;
		host_C[i] = rand()/(MAT_TYPE)RAND_MAX;
	}

	cudaMalloc((void**)&dev_A,N2 * sizeof(MAT_TYPE));
	cudaMalloc((void**)&dev_B,N2 * sizeof(MAT_TYPE));
	cudaMalloc((void**)&dev_C,N2 * sizeof(MAT_TYPE));

	//cublas 초기화
	cublasHandle_t handle;
    cublasCreate(&handle);

	cublasSetVector(N2,sizeof(MAT_TYPE),host_A,1,dev_A,1);	
	cublasSetVector(N2,sizeof(MAT_TYPE),host_B,1,dev_B,1);	
	cublasSetVector(N2,sizeof(MAT_TYPE),host_C,1,dev_C,1);	

	printf("(1024 X 1024)  * (1024 X 1024)\n");
	
	printf("cublas dgemm : ");	
	stopwatch(0);
	cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,N,N,N,&alpha,dev_A,N,dev_B,N,&beta,dev_C,N);
	stopwatch(1);
	cublasGetVector(N2,sizeof(MAT_TYPE),dev_C,1,host_C,1);

	cublasDestroy(handle);

	cudaThreadSynchronize();	

	dim3 Dg(BLOCK,BLOCK,1);
	dim3 Db(THREAD,THREAD,1);

	printf("cuda matrix multiplication ");
	stopwatch(0);
    cuda_mul<<<Dg,Db>>>(dev_A,dev_B,dev_C,N);	
	stopwatch(1);

	free(host_A);
	free(host_B);
	free(host_C);

	cudaFree(dev_A);
	cudaFree(dev_B);
	cudaFree(dev_C);

	return 0;

}


__global__ void cuda_mul(MAT_TYPE* A,MAT_TYPE* B,MAT_TYPE* C,int w)
{
	int tid,tx,ty;

	tx = blockDim.x * blockIdx.x + threadIdx.x;
	ty = blockDim.y * blockIdx.y + threadIdx.y;
	tid = w*ty + tx;

	MAT_TYPE v = 0;
	MAT_TYPE a = 0;
	MAT_TYPE b = 0;

	for(int i=0;i< w;i++)
	{
		a = A[ty * w + i];
		b = B[i * w + tx];
		v += a+b;
	}

	C[tid]= v;
}




void stopwatch(int flag)
{
	const long long NANOS = 1000000000LL;
	static struct timespec startTS,endTS;
	static long long diff = 0;
	
	//start
	if(flag == 0)
	{
		diff = 0;
		if(-1 == clock_gettime(CLOCK_MONOTONIC,&startTS))
			printf("Failed to call clock_gettime\n");
	}
	//end
	else if(flag == 1)
	{		
		if(-1 == clock_gettime(CLOCK_MONOTONIC,&endTS))
			printf("Failed to call clock_gettime\n");
		diff = NANOS * (endTS.tv_sec - startTS.tv_sec) + (endTS.tv_nsec - startTS.tv_nsec);
		
		printf("elapsed time : % lld microsec\n",diff/1000);
	}
	else
	{
		printf("wrong flag | 0 : start, 1 : end\n");
	}

}
 
