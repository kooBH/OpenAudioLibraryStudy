# [OpenMP](../README.md)

1. OpenMP  
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
  + 

+ 

2. OpenBLAS    
사용시 문제가 생긴다면  
make 옵션으로 USE_OPENMP=1 을 주면된다


3. Intel MKL  
 컴파일 옵션 알아보기에 옵션 존재  
