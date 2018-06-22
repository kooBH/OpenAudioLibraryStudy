
# [Makefile](../README.md)<a name ="TOP"></a>
1. [gcc](#Makefile-gcc)
2. [library](#Makefile-library)
3. [Makefile by examples](#Makefile-Makefile)
	+ [basic](#basic)
	+ [macro](#macro)
	+ [library](#library)
	+ [directory](#directory)
	+ [wildcard](#wildcard)
	+ ----WIP---
	+ [sub-makefile](#sub-makefile)
	+ [suffix](#suffix)
	+ [overall](#overall)
	


아래의 코드가 있다고 하자

+ main.c
```C++
#include "hello.h"
int main()
{
	hello();
	return 0;
}
```

+ hello.h
```C++
#include <stdio.h>

void hello();

```

+ hello.c
```C++
#inclide "hello.h"

void hello()
{
	printf("hello world\n");
}

```

이 코드들을 빌드하려면
## [gcc](#TOP)<a name="Makefile-gcc"></a>

```bash
gcc -c main.c					//main.c 를 main.o 로 변환
gcc -c hello.c					//hello.c 를 hello.o 로 변환
gcc -o hello main.o hello.o 	//목적파일들을 엮어서 hello 빌드
```
하면 된다

gcc 의 기초적인 형태는
```bash
$ gcc -c 코드 (옵션)
$ gcc -o 실행파일명 목적파일(들) (옵션)	
```

또한 특정라이브러를 사용할 경우 
/usr/lib에 있는 표준라이브러리라도 
옵션을 사용해야한다

<pthread.h>를 사용할 경우
```bash
gcc -o threading threading.o -pthread
```
해야 한다


### [library](#TOP) <a name="Makefile-library"></a>

+ Archive | Static library

정적 라이브러리는 object의 결합체이다
링크할 때 통째로 같이 되기 때문에 코드를 빌드하는 것과
기능상의 차이는 없다. 하지만 여러 코드들을 하나의 .a로 컴파일하면
되기 때문에 편의상의 이점이 있다

```bash
gcc -c hello.c
ar cr libhello.a hello.o (+추가적인 코드)  # libhello.a 가 생성된다
gcc -o hello main.o -L. -lhello (-static)
```  
-L(.a파일 경로)  
-l(.a 이름, 앞의 lib 과 확장자.a를 뺀 이름만 넣는다)  
-static : 같은 이름의 .so가 있을 경우 .so를 우선적으로 빌드하기 때문에
-static을 붙이면 .a를 우선적으로 빌드한다

+ Shared Object | Dynamic library

공유 객체, 동적 라이브러리  
실행시에 파일과 별개로 외부의 라이브러리와 링크되며, 코드를 수정해야할 경우 .so만 교체하면 되기 때문에 유지보수의 이점이 있다

```bash
gcc -c -fPIC 							#목적파일을 만들때에도 옵션을 줘야한다
gcc -shared -fPIC -o libhello.so hello.o #libhello.so 생성
gcc -o hello main.o -L. -lhello	#연결
export LD_LIBRARY_PATH+=:libhello.so의 경로 #명령 사용시 주의!!
```
-fPIC : Position-Indepent Code  

전반적으로 .a와 용법은 같으나 실행시에 .so를 찾아야한다  
빌드할때의 -L 의 경로는 빌드 때만의 경로이며 빌드후에는 더이상 이용되지 않는다
실행 시에는  

1. /usr/lib
2. LD_LIBRARY_PATH

에서 찾게 되는데, 컴파일한 .so를 /usr/lib에 넣어주거나    
환경변수 LD_LIBRARY_PATH 에  
export LD_LIBRARY_PATH+=:(.so의 경로)로 추가해주면 된다  
환경변수는 잘못입력하면 골치아플 수 있으니 주의를 요구한다    
설정후
```bash
echo $LD_LIBRARY_PATH
```
로  잘 입력됐는지 확인하자

export 된 환경변수는 종료시에 사라지기 때문에   
~/.bashrc(터미널을 열때마다 실행 ) 이나  
~/.profile(부팅 후 유저 로그인 시 실행) 에 export 명령을 추가시키면 된다  

## [Makefile](#TOP)<a name ="Makefile-Makefile"></a>

Makefile은 gcc를 편리하게 이용하게 해준다
Makefie( 확장자 없음)을 작성한 뒤에 make 를 명령하면 Makefile에
짜여진 명령들이 실행된다

[참고](https://wiki.kldp.org/KoreanDoc/html/GNU-Make/GNU-Make.html#toc2)

기본적인 구성은

목표 : 종속성
(반드시 TAB)명령어

종속성이 충족되면(되는지 확인하고) 목표를 위한 명령어를 수행한다

### [basic](#TOP)<a name ="basic"></a>

<details><summary>Makefile/1_basic</summary>

```Makefile
#기본 타겟(가장 위에 있기 때문에) hello :  조건은 hello 와 main이 충족되어야한다
#조건이 맞다면 gcc -o <실행파일> <목적파일1> <목적파일2> 을 한다
exec : hello main
	gcc -o hello main.o  hello.o 

#main.o 를 만든다. main.c 가 있어야한다
main : main.c
	gcc -c main.c

#hello.o 를 만든다. hello.h 와 hello.c가 있어야한다
hello : hello.h hello.c
	gcc -c hello.c

#clean 타겟, make clean시 호출된다. 사용된 목적파일을 지운다
clean :
rm *.o	
```

</details>

### [macro](#TOP)<a name ="macro"></a>

<details><summary>Makefile/2_macro</summary>

```Makefile
#미리 지정된 매크로 'CC' : .c 파일의 컴파일러 
CC=gcc
#OBJ 매크로 지정
OBJS = main.o hello.o


#매크로 호출은 $(매크로명)
#
#$@ 현재 타겟
#$^ 현재 타겟의 종속항목
hello : $(OBJS)
	$(CC) -o $@ $^	
#각 OBJS 에 대해 .o 파일을 만드는  명령은 없지만
#make 에 그정도의 기능은 내장되어있다


clean : 
	rm *.o

```

</details>

### [library](#TOP)<a name ="library"></a>

다음과 같은 파일 3개가 있을 때  

<details><summary>main.c</summary>
	
```C++
#include "my_lib.h"
#include <math.h>
#include <stdio.h>

int main()
{
	printf("1 + 2 = %d\n",add(1,2));
	printf("2.2^10 = %lf\n",pow(2.2,10));
	return 0;
}
```
</details>

<details><summary>my_lib.h</summary>
	
```C++
int add(int,int);
```
</details>

<details><summary>my_lib.c</summary>
	
```C++
int add(int x,int y)
{
return x+y;
}
```
</details>

라이브러리를 만들어서 활용하고 싶다면

<details><summary>Makefile/3_library</summary>

```Makefile
CC=gcc
OBJS = main
#라이브러리로 만들 파일
#.c 와 .h 둘 다 사용할 것이기에 my_lib을 매크로로 해서
# $(LIBS).c  $(LIBS).h 로 사용
LIBS = my_lib

# <math.h> 를 사용하기 위한 옵션
FLAG = -lm

TARGET=hello

#라이브러리 사용 방식을 받을 매크로
#make 시 
#make LIB_OPTION=<옵션> 으로 해야한다
LIB_OPTION=


# 빌드는는
# make static : 정적
# make shared : 동적
# 으로 먼저 라이브러리를 생성하고 해야한다

default:
#옵션으로 SHARED 를 받았을 때
ifeq ($(LIB_OPTION), SHARED)
	@echo "SHARED"	
	$(CC) -c $(OBJS).c
	$(CC) -o $(TARGET) $(OBJS).o -L. -l$(LIBS)
	@echo "You need to export PATH to library"

```

</details>

### [directory](#TOP)<a name ="directory"></a>

+ src/hello.c
+ include/hello.h
<details><summary>maic.c</summary>

```C++
#include "include/hello.h"

int main()
{
	hello();
	return 0;
}	
	
```	
</details>


이렇게 파일의 티렉토리가 다를 경우에

<details><summary>Makefile/4_directory</summary>

```Makefile
CC=gcc
SRC =hello

#헤더 폴더를 받는 매크로
# -I<경로>  를 하면 해당 경로에서 헤더를 찾는다
DIR=-Iinclude

hello : $(SRC).o main.o
	$(CC) -o $@ $^	

#src/ 폴더에있는 파일들을 컴파일한다
$(SRC).o :  
	$(CC)  $(DIR) -c src/$(SRC).c

clean : 
rm *.o
```
</details>


