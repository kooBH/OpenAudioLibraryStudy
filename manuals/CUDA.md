
# CUDA<a name ="TOP"></a> 
#### 1. [knowledge](#knowledge)
#### 1. [syncronise](#sync)
#### 2. [shared memory](#shared)
#### 1. [extention](#extention)
#### 2. [function](#function)
  + memory
#### 3. [example](#example)
  + [matrix multiplication](#matmul)
  

nvcc --version <- CUDA compiler version check  
nvidia-smi <- GPU 사용량 

## [knowledge](#TOP)<a name = "knowledge"></a>
+ 커널 함수는 CPU 관점에서는 [비동기](https://stackoverflow.com/questions/8473617/are-cuda-kernel-calls-synchronous-or-asynchronous)적이다 
+ 코어수보다 쓰레드가 많으면 스위치하는 것이 아니라 대기를 시키기 때문에 쓰레드가 넘쳐도 무방
+ 쓰레드가 적으면 쉬는 코어가 생기기 떄문에 문제
+ 하나의 device는 하나의 grid 수행, grid는 block 으로 block은 thread로 구성
+ grid와 block 은 dim3로 3차원으로 이루어져 있다, 좌표별로 1024,1024,64 까지 가능하다 [참고](https://devtalk.nvidia.com/default/topic/978550/cuda-programming-and-performance/maximum-number-of-threads-on-thread-block/)
+ 연산식을 길게 풀어쓰는 것보다 짧게 여러개 만드는 것이 레지스터를 적게써서 코드 효율이 높아진다 (레지스터를 초과하면 로컬 메모리를 사용한다
```
sum = a1*b1 + a2*b2 + a3*b3 +a4*b4
보다는
sum  = a1*b1 
sum += a2*b2 
sum += a3*b3 
sum += a4*b4
```
+ GPU 는 SM(streaming multi-processor) + a 
+ SM은 SP(streaming processor)로 되어있으며 하나의 SP는 4개의 쓰레드를 수행할 수 있다

## [syncronise](#TOP)<a name ="sync"></a>
커널 함수는 CPU 관점에서는 [비동기](https://stackoverflow.com/questions/8473617/are-cuda-kernel-calls-synchronous-or-asynchronous)적이다 
적이다 
GPU 관점에서는 한 커널 함수의 쓰레드들이 다 돌아야 다음 커널 함수를 호출하지만, 호스트에서는 아랑곳하지 않고 그냥 진행한다

```C++
호스트함수 A();
Kernel함수B <<<>>>();
호스트함수 C();
return 
```

이렇게 했을 때, 실행시 콘솔에서는 A,C 가 호출되어서 바로 return 이 될 것같지만, GPU 에서는 B가 돌고있기 때문에 GPU에서의 작업이 끝나야 return이 된다

```C++

//HOST 에서 (CPU관점) kernel 함수들을 동기화 시키려면
cudaThreadSynchronize()

//DEVICE 에서 (GPU관점) 자기자신 kernel 함수의 쓰레드들끼리 동기화 시키려면
__syncthreads();

```

또는 **cudaMemcpy**를 사용할 때도 기다린다


## [SHARED MEMORY](#TOP)<a name = "shared"></a>
공유 메모리는 같은 블록내의 쓰레드끼리만 공유하는 메모리로 **캐시와 동등한 속도**로 사용할 수 있다  
하지만 크기가 작기 - 16KB 정도- 때문에 큰 자료의 연산을 위해선 분할이 필요하다  
또한 호스트에서 값을 사용하기위해서 매번 글로벌 메모리에서 받고 보내야하기 때문에
**같은 값을 여러번** 사용하지 않는 다면 이점이 없다

```C++
__shared__ int ARRAY[512];  //정적할당 : 커널 함수에서 해야하며, 초기화는 할 수 없다.

extern __shared__ float host_delcared[]; //동적할당, 
kernel함수<<<그리드,블록, 여기에 shared메모리들의 크기를 넣으면된다(같은 자리부터 할당하기때문에)>>>(인자);
```

<details><summary>3_sharedMemory.cu</summary>

```C++
#include <stdio.h>
#include <stdlib.h>
#include <time.h>


#define ITER 40000
#define SIZE 500
#define BLOCK 1
#define THREAD 64


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
		
	cudaMalloc((void**)&a,sizeof(int)*SIZE );
	cudaMalloc((void**)&b,sizeof(int)*SIZE);


	stopwatch(0);
	printf("Global : ");
	manymanyGlobal<<<BLOCK,THREAD>>>(a,b);
	cudaThreadSynchronize();
	stopwatch(1);

	cudaFree(a);
	cudaFree(b);
	
	stopwatch(0);
	printf("Shared : ");
	manymanyShared<<<BLOCK,THREAD>>>();
	cudaThreadSynchronize();
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
```

</details>

```
SIZE : 500
BLOCK : 1
THREAD : 1
ITER : 40000
Global : elapsed time :  275768 micros
Shared : elapsed time :  212388 micros
```
```
SIZE : 500
BLOCK : 1
THREAD : 64
ITER : 40000
Global : elapsed time :  14307 micros
Shared : elapsed time :  11950 micros


```

<details>><summary>4_sharedMul.cu</summary>
	
```C++
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



```
	
</details>

```
Multiply matrix (1024X1024 ) * (1024X1024)
cuda_mul
elapsed time :  3657 micros
shared_mul
elapsed time :  3291 micros
```
큰 차이는 아니지만 빠르긴하다 
공유메모리를 제대로 활용할면 GPU별로 맞춰서 변수를 설정해야한다  


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

+ [\_\_shared\_\_](#shared)

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

### [syncronise](#sync)

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






