#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define BLOCK 30000
#define THREAD 1000

#define CHECK 1

void stopwatch(int);

//그냥 글로벌 메모리 사용
__global__ void count( int* cnt)
{
	(*cnt)++; 
}
//Atomic 함수 사용
__global__ void atomic_count( int* cnt)
{
	//Atomic 함수, 더하는 대상을 포인터로 지정해야한다
	//한번에 하나의 쓰레드만 작업한다
	atomicAdd(cnt,1);
}
//Atonic 함수와 Shared Memory 사용
__global__ void atomic_with_shared_count( int* cnt)
{
	__shared__ int s_cnt;
	
	//하나의 쓰레드만 초기화 시켜주면 된다
	if(threadIdx.x==0)
		s_cnt = 0;
	//블록단위 동기화
	__syncthreads();

		atomicAdd(&s_cnt,1);
	//블록단위 동기화
	__syncthreads();
	
	//하나의 쓰레드만 글로벌 변수에 더해주면 된다
	if(threadIdx.x==0)
		atomicAdd(cnt,s_cnt);

}
int main()
{
	int * host_cnt;
	int * dev_cnt;
	
	dim3 Dg(BLOCK,1,1);
	dim3 Db(THREAD,1,1);


	printf("BLOCK : %d\nTHREAD : %d\n",BLOCK,THREAD);

	host_cnt= (int*)malloc(sizeof(int));
	cudaMalloc((void**)&dev_cnt,sizeof(int));


	printf("Just cnt++ : ");
	stopwatch(0);
	cudaMemcpy(dev_cnt, host_cnt,sizeof(int), cudaMemcpyHostToDevice);	
	count<<<Dg,Db >>>(dev_cnt);
	cudaMemcpy(host_cnt,dev_cnt,sizeof(int),cudaMemcpyDeviceToHost);
	stopwatch(1);
#if CHECK
	printf("cnt : %d\n",*host_cnt);
#endif
	(*host_cnt)=0;
	
	printf("AtomicAdd : ");
	stopwatch(0);
	cudaMemcpy(dev_cnt, host_cnt,sizeof(int), cudaMemcpyHostToDevice);	
	atomic_count<<<Dg,Db >>>(dev_cnt);
	cudaMemcpy(host_cnt,dev_cnt,sizeof(int),cudaMemcpyDeviceToHost);
	stopwatch(1);
#if CHECK
	printf("cnt : %d\n",*host_cnt);
#endif
	(*host_cnt)=0;

	printf("AtomicAdd with Shared Memory : ");
	stopwatch(0);
	cudaMemcpy(dev_cnt, host_cnt,sizeof(int), cudaMemcpyHostToDevice);	
	atomic_with_shared_count<<<Dg,Db >>>(dev_cnt);
	cudaMemcpy(host_cnt,dev_cnt,sizeof(int),cudaMemcpyDeviceToHost);
	stopwatch(1);
#if CHECK
	printf("cnt : %d\n",*host_cnt);
#endif

	cudaFree(dev_cnt);
	free(host_cnt);
	
	return 0;
}


void stopwatch(int flag)
{
	enum clock_unit{nano = 0, micro , milli, sec} unit;
	
	const long long NANOS = 1000000000LL;
	static struct timespec startTS,endTS;
	static long long diff = 0;

	/*
		여기서 단위 조정
		nano, micro, milli, sec
	*/
	unit = micro;

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

		switch(unit)		
		{
			case nano :
				printf("% lld nano sec\n",diff);
			break;
			case micro :
				printf("%lld micro sec\n",diff/1000);
			break;
			case sec :
				printf("% lld sec\n",diff/1000000000);
			break;
			default :
				printf("% lld milli sec\n",diff/100000);
			break;	

		}
	}
	else
	{
		printf("wrong flag | 0 : start, 1 : end\n");
	}

}
