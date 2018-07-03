
# CUDA<a name ="TOP"></a> 
#### 1. [BASE](#knowledge)
#### 2. [SYNCHRONIZE](#sync)
#### 3. [SHARED MEMORY](#shared)
#### 4. [ATOMIC](#atomic)
#### 5. [STREAM](#stream)
#### 6. [CUBLAS](#blas)
#### 7. [FFT-WIP](#fft)
#### 8. [EXAMPLE](#example)
  + [matrix multiplication](#matmul)
  + [Gemm batchedGemm StreamedGemm](#7)
  
---
nvcc 로 컴파일
파일확장자는 .cu 헤더는 편한대로

```bash
$ ls
source.cu
$ nvcc -c source.cu
$ nvcc -o a.out source.o
```
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

---

## [extention](#TOP)<a name = "extention"></a>

### Function
+ 리턴은 void 

+ \_\_global\_\_
  * device(GPU)에서 실행  
  * host(CPU)에서 호출
  * \_\_global\_\_ functuon<<<number of block, thread per block >>>(args)    
  * 재귀 불가능

```C++
__global__ void kernel_func (int a*){<...>}

int main(){
<...>
kernel_func<<<16,64>>>(some_device_array);
<...>
```  
+ \_\_device\_\_
  * device에서 실행
  * device에서 호출
+ \_\_host\_\_
  * 기본값 : host 실행, host 호출 
  * 보통 함수 쓰듯이 시용
  
### Variable

+ [\_\_shared\_\_](#shared)

---

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

### [synchronise](#sync)

---

## [synchronise](#TOP)<a name ="sync"></a>
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
//단 블록단위 동기화이기 때문에 블록끼리는 안된다

```

또는 **cudaMemcpy**를 사용할 때도 기다린다

---

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

---

## [ATOMIC](#TOP)<a name = "atomic"></a>

여러 커널 함수들이 같은 글로벌 변수를 사용하면, 경쟁상태가 되어 제대로된 연산을 할 수없으며 느리기까지하다.    
Atomic 함수를 사용하면 한번에 하나의 쓰레드만 해당 변수를 사용하기 때문에 안전하며, 빠르기까지하다.   
또한 Block 단위로 공유하는 shared memory를 사용하면 더 좋은 성능을 낼 수 있다

[FUNCTION LIST](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions)

### 예시 

<details><summary>8_atomic.cu</summary>

```C++
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

```
</details>

&nbsp;  
#### 결과

```
BLOCK : 300
THREAD : 100
Just cnt++ : 48 micro sec
cnt : 2
AtomicAdd : 20 micro sec
cnt : 30000
AtomicAdd with Shared Memory : 21 micro sec
cnt : 30000

BLOCK : 3000
THREAD : 1000
Just cnt++ : 1329 micro sec
cnt : 89
AtomicAdd : 141 micro sec
cnt : 3000000
AtomicAdd with Shared Memory : 69 micro sec
cnt : 3000000

BLOCK : 30000
THREAD : 1000
Just cnt++ : 13924 micro sec
cnt : 844
AtomicAdd : 1113 micro sec
cnt : 30000000
AtomicAdd with Shared Memory : 421 micro sec
cnt : 30000000


```


---


## [STREAM](#TOP)<a name="stream"></a>

일반적으로 serial 하게 작업을 수행하면 Data를 다 넣고 연산을 수행하게된다.  하지만 Stream을 나눠서 수행하게되면, Data IO와 Process는 동시에 수행할 수 있기 때문에 병렬 작업이 가능하다.   
&nbsp;  
하지만 스트림 생성과 제거의 오버헤드가 있기 때문에, 자료의 양이 많아야 하며, IO와 Process의 비율도 생각해야한다  

![alt text](./CudaStream_2.png "CudaStream")

```C++
cudaStream_t 스트림
/*
스트림
*/

cudaMallocHost((void**)&포인터,  크기)
/*
비동기적으로 호스트메모리를 사용하려면 cudaMallocHost()로 할당해야한다 
CudaMalloc과 사용법은 같지만, 디바이스 메모리가 아닌 호스트 메모리를 할당한다
*/

cudaStreamCreate(&(cudaStream_t 스트림))
/*
스트림을 생성한다
*/

cudaMemcpyAsync(dest,src,size,op,cudaStream_t 스트림)
/*
cudaMemcpy에 cudaStream_t 인자가 추가되었다
비동기적으로 Memcpy를 수행한다 스트림별로 수행한다.
*/

kernel_func<<<block,thread,shared memory, cudaStream_t 스트림>>>(args...);
/*
커널 함수 호출시 <<< >>>의 4번째 인자에 스트림을 넣어주면 해당 스트림에서 비동기적으로 수행한다 
*/

cudaStreamSynchronize(cudaStream_t 스트림);
/*
스트림의 작업이 다 끝날때 까지 기다린다
*/

cudaStreamDestroy(cudaStream_t 스트림)
/*
생성된 스트림을 제거한다
*/

```

### 예시

&nbsp;  
<details><summary>6_stream.cu</summary>

```C++

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NUM_STREAM 20
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
	
	cudaThreadSynchronize();

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


```

</details>
&nbsp;  
<details><summary>결과 비교</summary>

```
number of stream : 10
number of data : 2000000
number of block : 1
number of thread : 512
Creating Stream[10]   : elapsed time :  604 micros
Reading & Processing with    Stream : elapsed time :  3501 micros
Reading & Processing without Stream : elapsed time :  4196 micros
Destroying Stream[10] : elapsed time :  16 micros

number of stream : 15
Creating Stream[15]   : elapsed time :  645 micros
Reading & Processing with    Stream : elapsed time :  3371 micros 
Reading & Processing without Stream : elapsed time :  4186 micros
Destroying Stream[15] : elapsed time :  22 micros

Creating Stream[20]   : elapsed time :  652 micros
Reading & Processing with    Stream : elapsed time :  3510 micros
Reading & Processing without Stream : elapsed time :  4197 micros
Destroying Stream[20] : elapsed time :  27 micros

number of stream : 15
number of data : 8000000
number of block : 1
number of thread : 512
Creating Stream[15]   : elapsed time :  581 micros
Reading & Processing with    Stream : elapsed time :  11797 micros
Reading & Processing without Stream : elapsed time :  16619 micros
Destroying Stream[15] : elapsed time :  19 micros

```
</details>

---


## [CUBLAS](#TOP)<a name = "blas"></a>

[CUBLAS DOCUMENT](https://docs.nvidia.com/cuda/cublas/index.html)

+ #include "cubals_v2.h"
+ 링크 옵션 : nvcc - lcublas

### 초기화
cublasCreate(cublasHandle_t*) & cublasDestory(cubalsHandle_t)  
cublasHandle_t 는 cublas context를 가지는 포인터    
cublasCreate()로 초기화하고    
cublasDestroy()로 해제해야한다  

```C++
cublasHandle_t handle;
cublasCreate(&handle); 

<cublas 사용 >

cublasDestory(handle);
```
### 리턴값
모든 cublas 함수는 cublasStatus_t을 반환한다 

Value | Meaning
--- | ---
CUBLAS_STATUS_SUCCESS| 성공
CUBLAS_STATUS_NOT_INITIALIZED |초기화 되지 않음, cublasCreate()를 먼저 해줘야한다
CUBLAS_STATUS_ALLOC_FAILED | 할당 실패, cudaMalloc()이 제대로 되지 않았다. 메모리 해제 요망
CUBLAS_STATUS_INVALID_VALUE |함수에 유효한 인자가 전달되지 않았다. 인자의 타입을 확인 요망
CUBLAS_STATUS_ARCH_MISMATCH | 현재 장치에선 지원해지 않는 기능사용, 보통 double precision에서 발생
CUBLAS_STATUS_MAPPING_ERROR |GPU메모리 접근실패. texture 메모리 해제 요망
CUBLAS_STATUS_EXECUTION_FAILED |커널 함수 호출 실패. 드라이버 버전이나 라이브러리 확인 요망
CUBLAS_STATUS_INTERNAL_ERROR | 내부 cublas 실패. 드라이버 버전이나 하드웨어 또는 할당해제된 변수에 접근하지는 확인 바람
CUBLAS_STATUS_NOT_SUPPORTED |지원하지 않음
CUBLAS_STATUS_LICENSE_ERROR |The functionnality requested requires some license and an error was detected when trying to check the current licensing. This error can happen if the license is not present or is expired or if the environment variable NVIDIA_LICENSE_FILE is not set properly. 


+ cublasSetVector(num, sizeof(type), X, incX , Y ,incY)
	기본적으로는  
	cublasSetVector(num,sizeof(type),X,1,Y,1)
	는
	cudaMemcpy(Y,X,num * sizeof(type),cudaMemcpyHostToDevice) 
	와 같다

	하지만
	길이가 n 인 배열 X와  
	m by n 행렬 Y 가 있을 때  

	```c++
	cublasSetVector(num, sizeof(type), X, 1 , Y ,n)
	```
	는 X를 Y의 1 번째 **열**에 넣는다  
	즉  

	```C++
	for(int i=0;i<num;i++)
		Y[incY * i ] = X[incX * i]
	```
	이런 느낌으로 넣는다는 것  
+ cublasGetVector(num, sizeof(type), X, incX , Y ,incY)
cublasSetVector의 반대, cudaMemcpyDeviceToHost라 보면된다  

### cublas\<T\>gemm()  
다른 BLAS와 비슷하나   
**column major**만 가능하다

#### cublasOperation_t 
Value |	Meaning
--- | ---
CUBLAS_OP_N | the non-transpose operation is selected
CUBLAS_OP_T | the transpose operation is selected
CUBLAS_OP_C | the conjugate transpose operation is selected

#### 함수원형과 사용
<details><summary>cublas<t>gemm()</summary>

```
cublasStatus_t cublasSgemm(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const float           *alpha,
                           const float           *A, int lda,
                           const float           *B, int ldb,
                           const float           *beta,
                           float           *C, int ldc)
cublasStatus_t cublasDgemm(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const double          *alpha,
                           const double          *A, int lda,
                           const double          *B, int ldb,
                           const double          *beta,
                           double          *C, int ldc)
cublasStatus_t cublasCgemm(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const cuComplex       *alpha,
                           const cuComplex       *A, int lda,
                           const cuComplex       *B, int ldb,
                           const cuComplex       *beta,
                           cuComplex       *C, int ldc)
cublasStatus_t cublasZgemm(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const cuDoubleComplex *alpha,
                           const cuDoubleComplex *A, int lda,
                           const cuDoubleComplex *B, int ldb,
                           const cuDoubleComplex *beta,
                           cuDoubleComplex *C, int ldc)
cublasStatus_t cublasHgemm(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const __half *alpha,
                           const __half *A, int lda,
                           const __half *B, int ldb,
                           const __half *beta,
                           __half *C, int ldc)

/*
 *  cbulas_?gemm(handler,transA,transB,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc)
 *
 *
 *   
 *   C := alpha * op(A)*op(B) + beta*C
 *
 *     op(X) =    i) X      when transX = CblasNoTrans
 *
 *     		 	 ii) X**T     ''        = CblasTrans
 *
 *     			iii) X**H     ''        = CblasConjTrans
 *
 *      m  = the number of rows of op(A)
 *      n  = the number of columns of op(B) and C 
 *      k  = the number of columns of op(A) and rows of op(B)
 *
 *
 *      lda : the first dimension of A
 *      ldb : 			''			 B
 *      ldc :			''			 C
 *
 * 		-the first dimension : the number of columns when CblasRowMajor
 *							   		''       rows    when CblasColMajor
 *
 * */


```
</details>

#### CUDA 4.0 부터 cublas.h 에서 cublas_v2.h 바뀌었다    
<details><summary>5_cubals_legacy.cu</summary>
	
+ cublasInit();
현재 host에 할당된 CPU 리소스를 cublas가 사용가능하게 할당하는 함수    
cublas 사용 전에 호출되어야한다  
+ cublasShutdown();
cublasInit()으로 할당된 GPU 리소스를 해제하는 함수  
cublas 사용 후에 호출되어야한다    
+ cublasAlloc( num, sizeof(type) , (void**)&target )
cudaMalloc의 wrapper 함수이기 떄문에 cudaMalloc((void**)&target, num * sizeof(type)) 기능상 차이가 없다  
섞어써도 무방  
+ cublasFree(target)
	
	
```C++

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "cublas.h"

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

	cublasAlloc(N2,sizeof(MAT_TYPE),(void**)&dev_A);
	cublasAlloc(N2,sizeof(MAT_TYPE),(void**)&dev_B);
	cublasAlloc(N2,sizeof(MAT_TYPE),(void**)&dev_C);

	cublasSetVector(N2,sizeof(MAT_TYPE),host_A,1,dev_A,1);	
	cublasSetVector(N2,sizeof(MAT_TYPE),host_B,1,dev_B,1);	
	cublasSetVector(N2,sizeof(MAT_TYPE),host_C,1,dev_C,1);	

	//cublas 초기화
	cublasInit();
	
	printf("(1024 X 1024)  * (1024 X 1024)\n");
	
	printf("cublas dgemm : ");	
	stopwatch(0);
	cublasDgemm('n','n',N,N,N,alpha,dev_A,N,dev_B,N,beta,dev_C,N);
	stopwatch(1);
	cublasGetVector(N2,sizeof(MAT_TYPE),dev_C,1,host_C,1);
	cublasShutdown();

	cudaThreadSynchronize();

	printf("cuda matrix multiplication ");
	stopwatch(0);
    cuda_mul<<<BLOCK,THREAD>>>(dev_A,dev_B,dev_C,N);	
	stopwatch(1);

	free(host_A);
	free(host_B);
	free(host_C);

	cublasFree(dev_A);
	cublasFree(dev_B);
	cublasFree(dev_C);



/*
	for(int i=0;i<N ; i++)
	{
		for(int j=0;j<N;j++)
			printf("%.3lf ",host_C[N*i + j]);
		printf("\n");
	}
*/
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


```

</details>

cublas.h 에서 cublas_v2.h 로 변경한 코드  

<details><summary>5_cubals_v2.cu</summary>

```C++

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

	printf("cuda matrix multiplication ");
	stopwatch(0);
    cuda_mul<<<BLOCK,THREAD>>>(dev_A,dev_B,dev_C,N);	
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

```

</details>


```
(1024 X 1024)  * (1024 X 1024)
cublas dgemm : elapsed time :  24 microsec
BOCK 256,256
THREAD 512,512
cuda matrix multiplication elapsed time :  4 microsec

```

#### [gemmBatch & gemm with Stream](#7)

---

## [FFT](#TOP)<a name = "fft"></a>
[Document](https://docs.nvidia.com/cuda/cufft/index.html#introduction)


---

## [EXAMPLE](#TOP)<a name ="example"></a>

### [matrix multiplication](#TOP)<a name ="matmul"></a>

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

### [Gemm batchedGemm StreamedGemm](#TOP)<a name="7"></a>

MATRIX_SIZE x MATRIX_SZIE 행렬이 MATRIX_NUM 만큼있을때, 행렬곱을 하는 코드  
1. dgemm을 MATRIX_NUM만큼  
2. dgemmStridedBatched로 한번에  
3. 스트림을 써서 입출력과 연산을 병렬적으로 dgemm을 MATRIX_NUM만큼  

<details><summary>7_batchedCublas.cu</summary>

```C++
#include "cublas_v2.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define CHECK 0
#define RESULT 0

#define DATA_TYPE double
#define MATRIX_NUM 10
#define MATRIX_SIZE 10
void stopwatch(int);
void pp(int p)
{printf("------------ %d-------------\n",p);}

void mat_out(DATA_TYPE*);

int main()
{
	printf("MATRIX_NUM : %d\nMATRIX_SIZE : (%d)X(%d)\n",MATRIX_NUM,MATRIX_SIZE,MATRIX_SIZE);

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
	s3 = MATRIX_SIZE * MATRIX_SIZE * MATRIX_NUM;	

	transa = CUBLAS_OP_N;
	transb = CUBLAS_OP_N;	

	stridea = s2;
	strideb = s2;
	stridec = s2;
	
	cuda_stat = cudaMallocHost((void**)&h_a,sizeof(DATA_TYPE) *s3);

#if CHECK
	printf("cudaMallocHost : %d\n",cuda_stat);
#endif	
	cuda_stat=	cudaMallocHost((void**)&h_b,sizeof(DATA_TYPE) *s3);
#if CHECK
	printf("cudaMallocHost : %d\n",cuda_stat);
#endif
	cuda_stat=	cudaMallocHost((void**)&h_c,sizeof(DATA_TYPE) *s3);

	cudaMalloc((void**)&d_a,sizeof(DATA_TYPE)*s3);
	cudaMalloc((void**)&d_b,sizeof(DATA_TYPE)*s3);
	cudaMalloc((void**)&d_c,sizeof(DATA_TYPE)*s3);


	srand(time(NULL));

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
	
	printf("GEMMs : ");
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

	printf("BATCHED GEMM : ");
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


/******************STREAMED GEMM  ********************/
	cudaDeviceSynchronize();

	
	printf("STREAM : ");
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
				printf("% lld nano sec\n",diff);
			break;
			case micro :
				printf("% lld micro sec\n",diff/1000);
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

```

</details>

&nbsp;  
#### 결과

```
MATRIX_NUM : 10
MATRIX_SIZE : (10)X(10)
GEMMs :  132 micro sec
BATCHED GEMM :  90 micro sec
STREAM :  152 micro sec

MATRIX_NUM : 100
MATRIX_SIZE : (500)X(500)
GEMMs :  218199 micro sec
BATCHED GEMM :  176977 micro sec
STREAM :  157057 micro sec

MATRIX_NUM : 10
MATRIX_SIZE : (1000)X(1000)
GEMMs :  153022 micro sec
BATCHED GEMM :  130539 micro sec
STREAM :  113286 micro sec

MATRIX_NUM : 100
MATRIX_SIZE : (1000)X(1000)
GEMMs :  1340405 micro sec
BATCHED GEMM :  1217660 micro sec
STREAM :  1100716 micro sec

MATRIX_NUM : 10
MATRIX_SIZE : (5000)X(5000)
GEMMs :  13155520 micro sec
BATCHED GEMM :  13146888 micro sec
STREAM :  12906191 micro sec

```





