
# CUDA<a name ="TOP"></a> 
#### 1. [knowledge](#knowledge)
#### 2. [shared memory](#shared)
#### 1. [extention](#extention)
#### 2. [function](#function)
  + memory
#### 3. [example](#example)
  + [matrix multiplication](#matmul)
  

nvcc --version <- CUDA compiler version check  
nvidia-smi <- GPU 사용량 

## [knowledge](#TOP)<a name = "knowledge"></a>
+ GPU 는 SM(streaming multi-processor) + a 
+ SM은 SP(streaming processor)로 되어있으며 하나의 SP는 4개의 쓰레드를 수행할 수 있다
+ 코어수보다 쓰레드가 많으면 스위치하는 것이 아니라 대기를 시키기 때문에 쓰레드가 넘쳐도 무방
+ 쓰레드가 적으면 쉬는 코어가 생기기 떄문에 문제
+ 하나의 device는 하나의 grid 수행, grid는 block 으로 moudle은 thread로 구성
+ grid와 block 은 dim3로 3차원으로 이루어져 있다, 좌표별로 1024,1024,64 까지 가능하다 [참고](https://devtalk.nvidia.com/default/topic/978550/cuda-programming-and-performance/maximum-number-of-threads-on-thread-block/)
+ 연산식을 길게 풀어쓰는 것보다 짧게 여러개 만드는 것이 레지스터를 적게써서 코드 효율이 높아진다
```
sum = a1*b1 + a2*b2 + a3*b3 +a4*b4
보다는
sum  = a1*b1 
sum += a2*b2 
sum += a3*b3 
sum += a4*b4
```

## [SHARED MEMORY](#TOP)<a name = "shared"></a>
공유 메모리는 같은 블록내의 쓰레드끼리만 공유하는 메모리로 **캐시와 동등한 속도**로 사용할 수 있다  
하지만 크기가 작기 - 16KB 정도- 때문에 큰 자료의 연산을 위해선 분할이 필요하다  

```C++
__shared__ int ARRAY[512];  //정적할당 : 커널 함수에서 해야하며, 초기화는 할 수 없다.

extern __shared__ float host_delcared[]; //동적할당,
```

<details></summary>3_sharedMemory.cu</summary>

```C++
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define ITER 1000000
#define BLOCK 512
#define THREAD 64 

void stopwatch(int);
__global__ void manymanyglobal();
__global__ void manymanyshared();

int main()
{
	printf("iter : %d * %d\nBLOCK : %d\nTHREAD : %d\n",ITER,ITER,BLOCK,THREAD);
	
	printf("manymany global\n");
	stopwatch(0);
	manymanyglobal<<<BLOCK,THREAD>>>();
	stopwatch(1);

	printf("manymany shared\n");
	stopwatch(0);
	manymanyshared<<<BLOCK,THREAD>>>();
	stopwatch(1);

	return 0;
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
		printf("elapsed time : % lld ms\n",diff/1000000);
	}
	else
	{
		printf("wrong flag | 0 : start, 1 : end\n");
	}

}
__global__ void manymanyglobal()
{
	double a = 111.111;
	double b = 111.111;
	double c = 0;

	for(int i=0;i<ITER;i++)
		for(int j=0;j<ITER;j++)
			{
			c += a * b;
		    c -= a * b;	
			}
}
__global__ void manymanyshared()
{
	__shared__ double a; 
	__shared__ double b;
	__shared__ double c;
	a= 111.111;
	b= 111.111;
	c= 0;
	for(int i=0;i<ITER;i++)
		for(int j=0;j<ITER;j++)
			{
			c += a * b;
		    c -= a * b;	
			}
}

```

</details>

```
iter : 1000000 * 1000000
BLOCK : 512
THREAD : 64
manymany global
elapsed time :  201 ms
manymany shared
elapsed time :  0 ms
```


## [extention](#TOP)<a name = "extention"></a>

### Function
+ 리턴은 void 

+ \_\_global\_\_
  * device(GPU)에서 실행  
  * host(CPU)에서 호출
  * \_\_global\_\_ functuon<<<number of block, thread per block >>>(args)    
  * 재귀 불가능
+ \_\_device\_\_
  * device에서 실행
  * device에서 호출
+ \_\_host\_\_
  * 기본값 : host 실행, host 호출 
  
### Variable

## [function](#TOP)<a name="function"></a>

### Memory

cudaMalloc((void**)&대상포인터, 할당범위 )

```C++
int* device_pointer;
int array_size = 10;
cudaMalloc( (void**)&device_pointer,sizeof(int)*array_size);
```

cudaMemcpy(대상, 원본 , 크기 , 종류 )

종류 :   

enum | kind
---|---
cudaMemcpyHostToHost|Host -> Host  
cudaMemcpyHostToDevice|	Host -> Device  
cudaMemcpyDeviceToHost  |	Device -> Host  
cudaMemcpyDeviceToDevice| Device -> Device   

```C++
int host_array[10] = {1,2,3,4,5,6,7,8,9,10};
int host_empty_array[10]={0,};

//host_array의 내용을 device_pointer로 
cudaMemcpy(device_pointer,host_array, sizeof(int) * array_size, cudaMemcpyHostToDevice);

//device_pointer의 내용을 host_empty_array로 
cudaMemcpy(host_empty_array,device_pointer, sizeof(int) * array_size, cudaMemcpyDeviceToHost);

// --> host_empty_array : {1,2,3,4,5,6,7,8,9,10}

```

cudaFree(대상포인터)

```C++
cudaFree(device_pointer)
```

## [EXAMPLE](#TOP)<a name ="example"></a>

### matrix multiplication<a name ="matmul"></a>

비교를 위해 openMP사용  
+ 컴파일 옵션 : -Xcompiler -fopenmp
+ 링크 옵션 : -lgomp


<details><summary>2_matrixMul.cu</summary>

```C++
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
		
		printf("elapsed time : % lld ms\n",Diff/1000000);
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




```

</details>

결과

```bash
Multiply matrix (1024X1024 ) * (1024X1024)
c_mul
elapsed time :  6581 ms
omp_mul
elased time : 3365.798956  ms
cuda_mul
elapsed time :  3 ms

```






