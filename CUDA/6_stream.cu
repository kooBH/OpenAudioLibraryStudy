#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NUM_STREAM 100
#define NUM_BLOCK 1
#define NUM_THREAD 512

#define NUM_DATA 2000000
#define TYPE_DATA double

#define CHECK 0

void stopwatch(int);
void pp(int);


//a 에서  b 로 l 만큼
__global__ void data_trans(TYPE_DATA* a,TYPE_DATA* b,int l);

int main()
{
	cudaStream_t stream_array[NUM_STREAM];

	TYPE_DATA* host_a,*host_b;
	TYPE_DATA* dev_a ,*dev_b;

	
	cudaMallocHost((void**)&host_a,sizeof(TYPE_DATA)*NUM_DATA);
	cudaMallocHost((void**)&host_b,sizeof(TYPE_DATA)*NUM_DATA);

	cudaMalloc((void**)&dev_a,sizeof(TYPE_DATA)* NUM_DATA );
	cudaMalloc((void**)&dev_b,sizeof(TYPE_DATA)* NUM_DATA );

	printf("number of stream : %d\nnumber of data : %d\nnumber of block : %d\nnumber of thread : %d\n",NUM_STREAM,NUM_DATA,NUM_BLOCK,NUM_THREAD);

	srand(time(NULL));

	for(int i=0;i<NUM_DATA;i++)
		{
			host_a[i] = rand()/(TYPE_DATA)RAND_MAX;
		}

	printf("Creating Stream[%d]   : ",NUM_STREAM);
	stopwatch(0);
	for(int i=0;i<NUM_STREAM; i++)
		cudaStreamCreate(&(stream_array[i]));
	stopwatch(1);


	int offset[NUM_STREAM];
	for(int i=0;i<NUM_STREAM;i++)
{
	offset[i] = i * NUM_DATA/NUM_STREAM;

#if CHECK
	printf("offset[%d] : %d\n",i,offset[i]);
#endif
}
	/************************Streaming**********************************/


	printf("Reading & Processing with    Stream : ");
	stopwatch(0);
	//READ
	for(int i=0;i<NUM_STREAM;i++)
	{
		cudaMemcpyAsync(dev_a+offset[i],host_a+offset[i],sizeof(TYPE_DATA)*NUM_DATA/NUM_STREAM,cudaMemcpyHostToDevice,stream_array[i]);
	}
	//TRANS
	for(int j=0;j<NUM_STREAM;j++)
	{
		data_trans<<<NUM_BLOCK,NUM_THREAD,0,stream_array[j]>>>(dev_a,dev_b,NUM_DATA/NUM_STREAM);
	}
	//GET
	for(int i=0;i<NUM_STREAM;i++)
	{
		cudaMemcpyAsync(host_b+offset[i],dev_b+offset[i],sizeof(TYPE_DATA)*NUM_DATA/NUM_STREAM,cudaMemcpyDeviceToHost,stream_array[i]);
	}
	
	for(int i=0; i<NUM_STREAM;i++)
		cudaStreamSynchronize(stream_array[i]);

	stopwatch(1);

#if CHECK
	printf("CHECK 0-10, %d-%d\n",NUM_DATA-10,NUM_DATA-1);
	for(int i=0;i<10;i++)
		printf("%.4lf ",host_b[i]);
	printf("\n");
	for(int i=NUM_DATA-10;i<NUM_DATA;i++)
		printf("%.4lf ",host_b[i]);
	printf("\n");
#endif	
	

/************************No Streaming**************************************/

	printf("Reading & Processing without Stream : ");	
	stopwatch(0);

	cudaMemcpy(dev_a,host_a,sizeof(TYPE_DATA)*NUM_DATA,cudaMemcpyHostToDevice);

	data_trans<<<NUM_BLOCK,NUM_THREAD>>>(dev_a,dev_b,NUM_DATA);

	cudaMemcpy(host_b,dev_b,sizeof(TYPE_DATA)*NUM_DATA,cudaMemcpyDeviceToHost);
	
	stopwatch(1);

#if CHECK
	printf("CHECK 0-10, %d-%d\n",NUM_DATA-10,NUM_DATA-1);
	for(int i=0;i<10;i++)
		printf("%.4lf ",host_b[i]);
	printf("\n");
	for(int i=NUM_DATA-10;i<NUM_DATA;i++)
		printf("%.4lf ",host_b[i]);
	printf("\n");
	
#endif



	printf("Destroying Stream[%d] : ",NUM_STREAM);
	stopwatch(0);
	for(int i=0;i<NUM_STREAM; i++)
		cudaStreamDestroy(stream_array[i] );
	stopwatch(1);


	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(host_a);
	cudaFree(host_b);

	return 0;
}

__global__ void data_trans(TYPE_DATA* a,TYPE_DATA* b,int l)
{
	for(int i=threadIdx.x;i<l;i+=NUM_THREAD)
		b[i]=a[i];

}

void pp(int num)
{
	printf("%d\n",num);
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
				printf("elapsed time : % lld micros\n",diff);
			break;
			case micro :
				printf("elapsed time : % lld micros\n",diff/1000);
			break;
			case sec :
				printf("elapsed time : % lld micros\n",diff/1000000000);
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
 
