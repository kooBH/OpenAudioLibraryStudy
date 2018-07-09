#include "cublas_v2.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define CHECK 0
#define RESULT 0

#define DATA_TYPE double
#define BLOCK_NUM 1
#define THREAD_NUM 1
#define MATRIX_NUM 10
#define MATRIX_SIZE 10
#define ITER int
void stopwatch(int);

void cublas_error();
void pp(int p)
{printf("------------ %d-------------\n",p);}

void mat_out(DATA_TYPE*);

int main()
{
	printf("BLOCK_NUM :%d\nTHREAD_NUM : %d\nMATRIX_NUM : %d\nMATRIX_SIZE : (%d)X(%d)\n",BLOCK_NUM,THREAD_NUM,MATRIX_NUM,MATRIX_SIZE,MATRIX_SIZE);

	//host matrix array
	DATA_TYPE *h_a,*h_b,*h_c;
	//device matrix array
	DATA_TYPE *d_a,*d_b,*d_c;
	
	//blas parameters
	DATA_TYPE alpha=1,beta=0;
	int m,n,k,lda,ldb,ldc;
	cublasOperation_t transa,transb;
	
	long long stridea,strideb,stridec;

	//matrix sizes
	long long s2;
	long long s3;
//	cublasHandle_t handle;
	
	int offset[MATRIX_NUM];

	cublasHandle_t handle;

	cublasHandle_t handle_s[MATRIX_NUM];
	cudaStream_t stream[MATRIX_NUM];

	//디버그
	cublasStatus_t cublas_stat;
	cudaError_t cuda_stat;

/************************Initialization******************************************/	

	m=MATRIX_SIZE,n=MATRIX_SIZE,k=MATRIX_SIZE,lda=MATRIX_SIZE,ldb=MATRIX_SIZE,ldc=MATRIX_SIZE;
	s2 = MATRIX_SIZE * MATRIX_SIZE;
	s3 = MATRIX_SIZE * MATRIX_SIZE * MATRIX_SIZE;	

	transa = CUBLAS_OP_N;
	transb = CUBLAS_OP_N;	

	stridea = s2;
	strideb = s2;
	stridec = s2;
	
	cuda_stat = cudaMallocHost((void**)&h_a,sizeof(DATA_TYPE) * MATRIX_NUM*s3);

#if CHECK
	printf("cudaMallocHost : %d\n",cuda_stat);
#endif	
	cuda_stat=	cudaMallocHost((void**)&h_b,sizeof(DATA_TYPE) * MATRIX_NUM*s3);
#if CHECK
	printf("cudaMallocHost : %d\n",cuda_stat);
#endif
	cuda_stat=	cudaMallocHost((void**)&h_c,sizeof(DATA_TYPE) * MATRIX_NUM*s3);

	cudaMalloc((void**)&d_a,sizeof(DATA_TYPE)*MATRIX_NUM*s3);
	cudaMalloc((void**)&d_b,sizeof(DATA_TYPE)*MATRIX_NUM*s3);
	cudaMalloc((void**)&d_c,sizeof(DATA_TYPE)*MATRIX_NUM*s3);


	srand(time(NULL));

	cudaDeviceSynchronize();
	

	for(long long j=0;j<s3;j++)h_a[j]=rand()/(DATA_TYPE)RAND_MAX;
    for(long long j=0;j<s3;j++)h_b[j]=rand()/(DATA_TYPE)RAND_MAX;
	for(long long j=0;j<s3;j++)h_c[j]=0;

	cublasCreate(&handle);


	for(int i=0;i<MATRIX_NUM;i++)
		cublasCreate(&(handle_s[i]));
	for(int i=0;i<MATRIX_NUM;i++)
		cudaStreamCreate(&(stream[i]));
	for(int i=0;i<MATRIX_NUM;i++)
		cublasSetStream(handle_s[i],stream[i]);



/****************** GEMM  한번 ********************/
/*
	printf("a GEMM : \n");
	stopwatch(0);

	cudaMemcpy(d_a,h_a,sizeof(DATA_TYPE)*s2,cudaMemcpyHostToDevice);
	cudaMemcpy(d_b,h_b,sizeof(DATA_TYPE)*s2,cudaMemcpyHostToDevice);

	cublasDgemm(handle,transa,transb,m,n,k,&alpha,d_a,lda,d_b,ldb,&beta,d_c,ldc);	
	
	cudaMemcpy(h_c,d_c,sizeof(DATA_TYPE)*s2,cudaMemcpyDeviceToHost);

	stopwatch(1);


*/
/******************그냥 GEMM  ********************/
	
	printf("GEMMs : \n");
	stopwatch(0);

	cudaMemcpy(d_a,h_a,sizeof(DATA_TYPE)*s3,cudaMemcpyHostToDevice);
	cudaMemcpy(d_b,h_b,sizeof(DATA_TYPE)*s3,cudaMemcpyHostToDevice);

	for(int i=0;i<MATRIX_NUM;i++)
		offset[i] = i * s2;

	for(int i=0;i<MATRIX_NUM;i++)
{
	cublas_stat =	cublasDgemm(handle,transa,transb,m,n,k,&alpha,&d_a[offset[i]],lda,&d_b[offset[i]],ldb,&beta,&d_c[offset[i]],ldc);	

#if CHECK
	printf("DGEMM[%d] : %d\n",offset[i],cublas_stat);
#endif

}	
		cudaMemcpy(h_c,d_c,sizeof(DATA_TYPE)*s3,cudaMemcpyDeviceToHost);

	stopwatch(1);
#if RESULT
	mat_out(h_c);
#endif

	for(int i=0;i<MATRIX_NUM;i++)
		for(int j=0;j<MATRIX_SIZE;j++)
			for(int k=0;k<MATRIX_SIZE;k++)
				h_c[i*s2 + j*MATRIX_SIZE + k] = 0 ;



/******************BATCHED STRIDE GEMM  ********************/

	printf("BATCHED GEMM : \n");
	stopwatch(0);

	cudaMemcpy(d_a,h_a,sizeof(DATA_TYPE)*s3,cudaMemcpyHostToDevice);
	cudaMemcpy(d_b,h_b,sizeof(DATA_TYPE)*s3,cudaMemcpyHostToDevice);

cublas_stat=	cublasDgemmStridedBatched(handle,transa,transb,m,n,k,&alpha,d_a,lda,stridea,d_b,ldb,strideb,&beta,d_c,ldc,stridec,MATRIX_NUM);	
#if CHECK	
	printf("Dgemm Strided Bached : %d\n",cublas_stat);
#endif
	cudaMemcpy(h_c,d_c,sizeof(DATA_TYPE)*s3,cudaMemcpyDeviceToHost);

	stopwatch(1);
#if RESULT
	mat_out(h_c);
#endif

	for(int i=0;i<MATRIX_NUM;i++)
		for(int j=0;j<MATRIX_SIZE;j++)
			for(int k=0;k<MATRIX_SIZE;k++)
				h_c[i*s2 + j*MATRIX_SIZE + k] = 0 ;





/******************BATCHED GEMM  ********************/
/*
	printf("BATCH : \n");
	stopwatch(0);

	for(int i=0;i<MATRIX_NUM;i++)
		cudaMemcpy(d_a[i],h_a[i],sizeof(DATA_TYPE)*s2,cudaMemcpyHostToDevice);
	for(int i=0;i<MATRIX_NUM;i++)
		cudaMemcpy(d_b[i],h_b[i],sizeof(DATA_TYPE)*s2,cudaMemcpyHostToDevice);

	cublas_stat = cublasDgemmBatched(handle,transa,transb,m,n,k,&alpha,(const DATA_TYPE**)d_a,lda,(const DATA_TYPE**)d_b,ldb,&beta,d_c,ldc,MATRIX_NUM);	
#if CHECK
	printf("cublasDgemmBatched : %d\n",cublas_stat);
#endif

	
	for(int i=0;i<MATRIX_NUM;i++)
		cudaMemcpy(h_c[i],d_c[i],sizeof(DATA_TYPE)*s2,cudaMemcpyDeviceToHost);

	stopwatch(1);
#if CHECK
	mat_out(h_c);
#endif

*/
//	cublasDestroy(handle);





/******************STREAMED GEMM  ********************/
	cudaDeviceSynchronize();

	
	printf("STREAM : \n");
	stopwatch(0);


	for(int i=0;i<MATRIX_NUM;i++)
{	 cuda_stat = cudaMemcpyAsync(&d_a[offset[i]],&h_a[offset[i]],sizeof(DATA_TYPE)*s2,cudaMemcpyHostToDevice,stream[i]);

#if CHECK
	printf("cudaMemcpyAsync[%d] : %d\n",i,cuda_stat);
#endif
}	
	for(int i=0;i<MATRIX_NUM;i++)
		cudaMemcpyAsync(&d_b[offset[i]],&h_b[offset[i]],sizeof(DATA_TYPE)*s2,cudaMemcpyHostToDevice,stream[i]);

	for(int i=0;i<MATRIX_NUM;i++)
{		cublas_stat =	cublasDgemm(handle_s[i],transa,transb,m,n,k,&alpha,&d_a[offset[i]],lda,&d_b[offset[i]],ldb,&beta,&d_c[offset[i]],ldc);	
	
#if CHECK
	printf("cublasDgemm : %d\n",cublas_stat);

#endif
}

	
	for(int i=0;i<MATRIX_NUM;i++)
		cudaMemcpyAsync(&h_c[offset[i]],&d_c[offset[i]],sizeof(DATA_TYPE)*s2,cudaMemcpyDeviceToHost,stream[i]);

	for(int i=0;i<MATRIX_NUM;i++)
		cudaStreamSynchronize(stream[i]);
	stopwatch(1);


#if RESULT
	mat_out(h_c);
#endif


/***********DeAllocation**********************/
	
	cudaFree(h_a);
	cudaFree(h_b);
	cudaFree(h_c);
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	cublasDestroy(handle);

	
	for(int i=0;i<MATRIX_NUM;i++)
		cublasDestroy(handle_s[i]);
	for(int i=0;i<MATRIX_NUM;i++)
		cudaStreamDestroy(stream[i]);		

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
				printf("elapsed time : % lld micro sec\n",diff/1000);
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


void mat_out(DATA_TYPE*a)
{
	for(int i=0;i<MATRIX_NUM;i++)
	{
		printf("--- %d ---\n",i);
		
		for(int j=0;j<MATRIX_SIZE;j++)
		{
			for(int k=0;k<MATRIX_SIZE;k++)
			{
				printf("%.3lf ",a[i*MATRIX_SIZE*MATRIX_SIZE + j*MATRIX_SIZE + k]);
			}
			printf("\n");
		
		}





	}
	

}
