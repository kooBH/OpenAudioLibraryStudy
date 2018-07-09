#include <stdlib.h>
#include <stdio.h>


#define SIZE 10000
#define THREAD 1024
#define BLOCK 1024

void stopwatch(int);

__global__ void convert(double* A,double* C)
{
int idx = BLOCK*blockIdx.x + threadIdx.x;
int i;
int stride = BLOCK * THREAD;
      
for(i=idx;i<SIZE;i+=stride)
	 A[i] = C[SIZE-i-1]; 
    
}

int main()
{
 double* A,*C;
 double* cu_A,*cu_C;
 int i;

 printf("SIZE :  %d\nBLOCK : %d\nTHREAD : %d\n",SIZE,BLOCK,THREAD);
 
 A = (double*)malloc(sizeof(double) * SIZE);
 for(i=0;i<SIZE;i++)
  A[i] = i;

 cudaMalloc((void**)&cu_A,sizeof(double)*SIZE);

/******************************************************/ 
 printf("HOST - HOST - DEVICE   : ");
 stopwatch(0);  
 C = (double*)malloc(sizeof(double) * SIZE);
 for(i=0;i<SIZE;i++)
  C[i] = A[SIZE-i-1];
 cudaMemcpy(cu_A,C,sizeof(double)*SIZE,cudaMemcpyHostToDevice); 
 stopwatch(1);


/*****************************************************/
 printf("HOST - DEVICE - DEVICE : ");
 stopwatch(0);  
 cudaMalloc((void**)&cu_C,sizeof(double)*SIZE);
 cudaMemcpy(cu_C,A,sizeof(double)*SIZE,cudaMemcpyHostToDevice); 
 convert<<<BLOCK,THREAD>>>(cu_A,cu_C);
 cudaThreadSynchronize();
 stopwatch(1);
/************************************/

 free(A);
 free(C);
 cudaFree(cu_A);
 cudaFree(cu_C);


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
				printf("elapsed time : % lld nano sec\n",diff);
			break;
			case micro :
				printf("% lld micro sec\n",diff/1000);
			break;
			case sec :
				printf("elapsed time : % lld sec\n",diff/1000000000);
			break;
			default :
				printf("elapsed time : % lld milli sec\n",diff/100000);
			break;	

		}
	}
	else
	{
		printf("wrong flag | 0 : start, 1 : end\n");
	}

}

