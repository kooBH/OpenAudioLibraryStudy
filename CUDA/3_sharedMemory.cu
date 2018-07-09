#include <stdio.h>
#include <stdlib.h>
#include <time.h>


#define ITER 340000
#define SIZE 500
#define BLOCK 1
#define THREAD 1


void stopwatch(int);
__global__ void manymanyLocal();
__global__ void manymanyGlobal(int*,int* );
__global__ void manymanyShared();

int main()
{

	printf("SIZE : %d\nBLOCK : %d\nTHREAD : %d\nITER : %d\n",SIZE,BLOCK,THREAD,ITER);
/*
	커널 함수는 비동기 함수라서 쓰레드만 돌려놓고 호스트는 그래도 진행한다  
	cudaTHreadSynchronize() 는 이전에 생성된 쓰레드들이 끝날때 까지 기다린다  
*/
/*
	stopwatch(0);
	printf("Local : ");
	for(int i=0; i< ITER;i++)
	{	
	manymanyLocal<<<BLOCK,THREAD>>>();
	cudaThreadSynchronize();
	}	
	stopwatch(1);
*/

	int *a;
	int *b;
	int t;		
	cudaMalloc((void**)&a,sizeof(int)*SIZE );
	cudaMalloc((void**)&b,sizeof(int)*SIZE);


	stopwatch(0);
	printf("Global : ");
	manymanyGlobal<<<BLOCK,THREAD>>>(a,b);
	cudaMemcpy(&t,a,sizeof(int),cudaMemcpyDeviceToHost);
//	cudaThreadSynchronize();
	stopwatch(1);

	cudaFree(b);
	
	stopwatch(0);
	printf("Shared : ");
	manymanyShared<<<BLOCK,THREAD>>>();
	cudaMemcpy(&t,a,sizeof(int),cudaMemcpyDeviceToHost);
	cudaFree(a);
//	cudaThreadSynchronize();
	stopwatch(1);



}

__global__ void manymanyLocal()
{
	int a[SIZE];
	int b[SIZE];

for(int j=0; j < ITER; j++)
	for(int i=threadIdx.x;i<SIZE;i+=THREAD)
	{
		a[i]=0;
		b[i]=0;
	}
 
}



__global__ void manymanyGlobal(int* a,int* b)
{
	
for(int j=0; j < ITER; j++)
	for(int i=threadIdx.x;i<SIZE;i+=THREAD)
	{
		a[i]=0;
		b[i]=0;
	}	
}

__global__ void manymanyShared()
{
	__shared__ int a[SIZE];
	__shared__ int b[SIZE];

	__syncthreads();	
		
for(int j=0; j < ITER; j++)
	for(int i=threadIdx.x;i<SIZE;i+=THREAD)
	{
		a[i]=0;
		b[i]=0;
	}	

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
		
		printf("elapsed time : % lld micros\n",diff/1000);
	}
	else
	{
		printf("wrong flag | 0 : start, 1 : end\n");
	}

}


