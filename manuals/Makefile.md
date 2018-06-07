
# Makefile
* [gcc](#Makefile-gcc)
* [library](#Makefile-library)
* [Makefile](#Makefile-Makefile)


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
## gcc<a name="Makefile-gcc"></a>

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


### library <a name="Makefile-library"></a>

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
~/.config 에 export 명령을 추가시키면 된다  


## Makefile<a name ="Makefile-Makefile"></a>

Makefile은 gcc를 편리하게 이용하게 해준다
Makefie( 확장자 없음)을 작성한 뒤에 make 를 명령하면 Makefile에
짜여진 명령들이 실행된다

[참고](https://wiki.kldp.org/KoreanDoc/html/GNU-Make/GNU-Make.html#toc2)

기본적인 구성은

목표 : 종속성
(반드시 TAB)명령어

종속성이 충족되면(되는지 확인하고) 목표를 위한 명령어를 수행한다

### 예제 1-1
```Makefile
hello : main.o hello.o
	gcc -o hello main.o hello.o 

main.o : hello.h main.c
	gcc -c main.c

hello.o : hello.h hello.c
	gcc -c hello.c

```

이 경우 처음에는 main.o 와 hello.o 가 없기에 hello를 위한 명령은 나중에 수행된다  
끝까지 종속성이 충족되지 않은 경우에는 수행되지 않는다

위의 예제와 같은 기능을 하는 예제

### 예제 1-2

```Makefile

#define customized suffixes
.SUFFIXES : .c .o

#Macro for objects
OBJ =  main.o hello.o 

#output file
TARGET = hello 

#make method
CC = gcc

#flag for g++ #use CFLAGS for gcc
CCFLAGS = -c 

$(TARGET) : $(OBJ)
	$(CC) -o $(TARGET) $(OBJ)

#remove used object files
clean :
	rm -rf $(OBJ) $(TARGET) core


```

주석은 # 을 사용한다  
Makefile 의 앞 부분에는 매크로를 지정해 줄 수 가 있다. 이는 Makefile 작성을 용이 하게 해준다

clean :  
이 부분은 make clean 을 할때 실행된다
