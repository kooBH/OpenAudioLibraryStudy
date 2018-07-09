#include <stdio.h>
#include <time.h>
#include <omp.h>

/*소요 시간 측정 함수
 0 : 측정 시작 시간
 1 : 측정 종료 시간 -> 걸린 시간을 출력
*/
void stopwatch(int);
//CUDA 배열 곱 
__global__ void cuda_mul(int*,int*,int*,int);

//omp 배열 곱, omp 변수에 따라 편차가 크다
void omp_mul(int*,int*,int*,int);
//일반적인 배열 곱
void c_mul(int*,int*,int*,int);
int main()
{

	//1024 by 1024 행렬
	const int width = 1024;
	const int height = width;
	const int matrix_size = width*height;
	const int buffer_size = matrix_size*sizeof(int);

	int *host_A,*host_B,*host_C,*C_C,*omp_C;
	
	host_A = (int*)malloc(buffer_size);
	host_B = (int*)malloc(buffer_size);
	host_C = (int*)malloc(buffer_size);
	C_C = (int*)malloc(buffer_size);
	omp_C = (int*)malloc(buffer_size);
	
	for(int i=0;i<matrix_size;i++)
	{
	host_A[i] = i;
	host_B[i] = i;
	host_C[i] =0;
	C_C[i] = 0;	
	omp_C[i] = 0;
	}
	

	printf("Multiply matrix (%dX%d ) * (%dX%d)\n",width,width,width,width);

	printf("c_mul\n");
	stopwatch(0);	
	c_mul(host_A,host_B,C_C,width);
	stopwatch(1);

	printf("omp_mul\n");

	double omp_start,omp_end;
	
	omp_start = omp_get_wtime();
	omp_mul(host_A,host_B,omp_C,width);
	omp_end=omp_get_wtime();
	
	printf("elased time : %f  ms\n",(omp_end-omp_start)*1000);
	
	int*device_A,*device_B,*device_C;

	/*	
		CUDA 곱 함수에서
		블록의 가로의길이 * 블록의 최대 idx + 쓰레드 가로 최대 idx
		= 배열의 행이나 열의 최대 idx
		가 되어야 연산이 되기 때문에

		1024 X 1024 연산을 위해
	
		Block 256 X 256
		THREAD 512 X 512 

		256 * 2 + 512 = 1024 
		로 하였다

		배열 크기에 따라 블록과 쓰레드의 수를 조절해야한다		
	*/

	dim3 Dg(256,256,1);
	dim3 Db(512,512,1);


	cudaMalloc((void**)&device_A,buffer_size  );
	cudaMalloc((void**)&device_B,buffer_size  );
	cudaMalloc((void**)&device_C,buffer_size  );

			
	printf("cuda_mul\n");
	stopwatch(0);
	cudaMemcpy(device_A,host_A,buffer_size,cudaMemcpyHostToDevice);
	cudaMemcpy(device_B,host_B,buffer_size,cudaMemcpyHostToDevice);

	cuda_mul<<<Dg,Db>>>(device_A,device_B,device_C,width);
	cudaMemcpy(host_C,device_C,buffer_size,cudaMemcpyDeviceToHost);
	stopwatch(1);
	
	
	cudaFree(device_A);
	cudaFree(device_B);
	cudaFree(device_C);

	free(host_A);	
	free(host_B);	
	free(host_C);	
	free(C_C);	

	
	return 0;
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

/*
	정방행렬을 곱하는
	C 코드와 CUDA코드 

*/

__global__ void cuda_mul(int* A, int* B, int* C, int w)
{
	int tid,tx,ty;

	//range of tx,ty 0 ~ w
	tx = blockDim.x * blockIdx.x + threadIdx.x;
	ty = blockDim.y * blockIdx.y + threadIdx.y;
	tid = w*ty + tx;

	int v = 0;
	int a = 0;
	int b = 0;


	/*
	oooo    oxo
	xxxx  X oxo 
	oooo    oxo
	        oxo
	*/

	for(int i=0;i< w;i++)
	{
		a = A[ty * w + i];
		b = B[i * w + tx];
		v += a+b;
	}

	C[tid]= v;
}


void omp_mul(int *A,int *B, int *C, int w)
{
	int col=0;
	int raw =0;
	int idx = 0;
	int dest=0;
	
	int const chunk = 1024;
#pragma omp parallel shared(col) private(raw,idx)
{
	#pragma omp for nowait
 	for ( col=0;col < w;col++)
	{
		for(raw =0;raw<w;raw++)
		{
			dest = col*w + raw;
			for( idx=0;idx<w;idx++)
			{
				C[dest] += A[col*w + idx] * B[idx*w + raw];
			}		
		}
	}
}
}

void c_mul(int *A,int *B, int *C, int w)
{
	int col=0;
	int raw =0;
	int idx = 0;
	int dest=0;

	for ( col=0;col < w;col++)
	{
		for(raw =0;raw<w;raw++)
		{
			dest = col*w + raw;
			for( idx=0;idx<w;idx++)
			{
				C[dest] += A[col*w + idx] * B[idx*w + raw];
			}		
		}
	}
}


