# [OpenMP](../README.md)<a name = "TOP"></a>
+ [OpenMP](#OpenMP)
+ [MKL](#MKL)

1. OpenMP<a name="OpenMP"></a> 
+ 사용하기
OpenMP는 컴파일러에 포함 
	* Linux    
	gcc 컴파일 옵션으로 -fopenmp
	 * Windows  
		  Visual Studio 개발 환경에서 하려면
		  1. 프로젝트의 속성
		  2. 구성 속성 노드를 확장
		  3. C/C++ 노드를 확장
		  4. 언어 속성
		  5. OpenMP 지원 속성을 수정  => (컴파일 옵션에) /openmp 추가  

+ 예제 1
  hello.c
```c++
#include <stdio.h>
int main(){
	#pragma omp parallel
	{
	printf("hello\n");
	}
return 0;}
```
기본값으로 쓰레드는 동적으로 생성되게 되어있다 (코어수에 맞춰서 생성)  
이때 생성된 쓰레드들을 team으로 부른다  
첫 #pragma 구문은 openmp로 쓰레드작업을 처리하는 구역이며 중괄호로 구분한다.   
마지막 } 에는 join 구문이 암시적으로 적용된다.  
마지막 } 는 master 쓰레드만 통과할 수 있다
```bash
$ gcc -c hello.c -fopenmp
$ gcc -o hello hello.o -fopenmp
$ ./hello
hello
hello
hello
hello
hello
hello
hello
hello
```
+ 구조
#pragma opm directive-name [clause, ...] { ... }
	+ 예제 2
```c++
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define N 20000000
#define CHUNKSIZE 20000

int main()
{
double start,end,gap;		
float *a,*b,*c ;
long  i,chunk;

a=(float*)malloc(sizeof(float)*N);
b=(float*)malloc(sizeof(float)*N);
c=(float*)malloc(sizeof(float)*N);

for(i=0;i<N ; i++)
	a[i]=b[i] = i*1.0;
chunk = CHUNKSIZE;
	
start=omp_get_wtime();
#pragma omp parallel shared(a,b,c,chunk) private(i,tid)
	{	#pragma omp for schedule (dynamic,chunk) nowait
		for (i=0;i<N;i++)
			c[i] = a[i] + b[i];
	}

end=omp_get_wtime();
gap = end-start;
printf("gap 1 : %lf\n",gap);
	
start=omp_get_wtime();
for (i=0;i<N;i++)
	c[i] = a[i] + b[i];
end=omp_get_wtime();
gap = end-start;
printf("gap 2 : %lf\n",gap);
free(a);free(b);free(c);
return 0;
} 
```
```bash
$ make TARGET=a1
$ ./a1
gap 1 : 0.027515
gat 2 : 0.049899
```


1. directive-name    
  + parallel   
    여러 쓰레드를 통해 수행되는 구역, 쓰레드 team을 만든다 openMP사용에 기반이되는 구조  
  + for  
    바로 뒤에 따라오는 for문을 쓰레드 팀으로 병렬수행한다  
 2. clause    
  + schedule  
    반복문을 각 쓰레드에게 어느정도 할당할 건지 정하는 절  
  + shared     
    전 쓰레드가 공유하는 변수 지정   
  + private    	  
    각 쓰레드가 개인적으로 가질 변수 지정  
  
2. OpenBLAS      
사용시 문제가 생긴다면    
make 옵션으로 USE_OPENMP=1 을 주면된다  


3. Intel MKL  
 컴파일 옵션 알아보기에 옵션 존재  
 
 
 + 참고
 [tutorial](https://computing.llnl.gov/tutorials/openMP/)
 
 2. [MKL](#TOP)  
 MKL에서 OpenMP 를 사용하려면 [Intel® Math Kernel Library Link Line Advisor](https://software.intel.com/en-us/articles/intel-mkl-link-line-advisor) 에서  threading layer를 OpenMP로 하면된다. 컴파일 옵션외에는 추가적으로 코드를 입력할 필요가 없다. gemm함수 내에서 자체적으로 쓰레딩을 한다. OpenMP library도 GNU나 Intel 이나 기능상의 차이는 크게 없다. 아니면 그냥 OpenMP를 컴파일 옵션으로 주고 #pragma omp로 할 수도 있을 것이다. 추가적인 조작을 원한다면 직접 OpenMP 를 사용하는 것이 좋다.
 
 |   | 500X500 dgemm 3000 times   | 500X500 dgemm 20000 times  |
|---|---|---|
| Sequential   | 27  | x  |
| OpenMP   |  8 | 52  |
| GNU  | 8  | 52  |
| Intel  | 8  | 52  |
 
