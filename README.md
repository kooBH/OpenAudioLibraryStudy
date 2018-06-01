# manuals
a few manuals for a few things

**INDEX**<a name="index"></a>

### 1.  [Makefile](#Makefile)
	[gcc](#1_1)
	[library](#1_2)
	[Makefile](#1_3)
### 2. [cmake](#cmake)

### 3. [RtAudio](#RtAudio) 

### 4. [BLAS](#BLAS)

---

# Makefile<a name="Makefile"></a>

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
## gcc<a name="1-1"></a>

```bash
gcc -c main.c					//main.c 를 main.o 로 변환
gcc -c hello.c					//hello.c 를 hello.o 로 변환
gcc -o hello main.o hello.o 	//목적파일들을 엮어서 hello 빌드
```
하면 된다

gcc 의 기초적인 형태는
```bash
gcc -o 실행파일명 코드(들) (옵션)
```

또한 특정라이브러를 사용할 경우 
/usr/lib에 있는 표준라이브러리라도 
옵션을 사용해야한다

<pthread.h>를 사용할 경우
```bash
gcc -o threading threading.o -pthread
```
해야 한다


### library <a name="1-2"></a>

+ Archive | Static library

정적 라이브러리는 object의 결합체이다
빌드할 때 같이 빌드되기 때문에 코드를 컴파일하고 빌드하는 것과
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
실행시에 링크되며, 코드를 수정해야할 경우 .so만 교체하면 되기 때문에 유지보수의 이점이 있다

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


## Makefile<a name ="1-3"></a>

Makefile은 gcc를 편리하게 이용하게 해준다
Makefie( 확장자 없음)을 작성한 뒤에 make 를 명령하면 Makefile에
짜여진 명령들이 실행된다

[참고](https://wiki.kldp.org/KoreanDoc/html/GNU-Make/GNU-Make.html#toc2)

### 예제 1-1
```Makefile
hello : main.o hello.o
	gcc -o hello main.o hello.o 

main.o : hello.h main.c
	gcc -c main.c

hello.o : hello.h hello.c
	gcc -c hello.c

```

위의 예제와 같은 기능을 한다

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

clean :  
이 부분은 make clean 을 할때 실행된다

---


# [CMAKE](#index)<a name="cmake"></a>

cmake 는 linux환경에서는 Makefile을 Windows환경에서는 비주얼 스튜디오 프로젝트를 만든다.


## 설치

+ linux
```bash
    $ sudo apt-get install cmake       
```
+  windows 


## 사용

1. 빌드할 프로젝트가 있는 폴더에
 CMakeLists.txt 를 만든다.

2.    CmakeLists.txt 를 작성한뒤
3.    $ cmake 를 해주면 Makefile 이 생성된다.

 ※builld 폴더 같은 걸 따로 만들어준 다음  
 $ cmake .. 으로 결과물을 따로 보관하는 것이 좋다.
             Makefile 을 작성해주는 것이아니라 Makefile 이 추가적으로 cmake가 만든 파일들을 이용하게 하는 것에 가깝디.



## CMakeLists.txt 작성

필수  :

+ cmake_minimum_required(VERSION 내.cmake의.버전)  
```CMake
 cmake_minimum_required(VERSION 3.5.1)  
```
cmake 최소 버전 요구사항 설정, 버전이 다르면 설정된 값이나 명령어 사용이 다를수 있다.
cmake 의 버전은  
```bash
 cmake -version 으로 알 수 있다
```  
기본 명령 :   

 +   set (변수명 들어갈값 )   
```CMake
set(SOURCES src1.c src2.c src3.c)   	
```
  변수를 불러올 때에는 ${변수명} 으로 불러온다

+ add_executable(파일명 들어갈코드 )  
```CMake
add_executable(hello hello.c)
#			or  
add_executable(programm SOURCES)  
```

파일명에 해당하는 실행파일을 뒤의 인자로 들어가는 코드를로 빌드하게 한다  

---
## 예시 2-1

/CMAKE

hello world 를 출력하는 hello.c가 있으면

CMakeLists.txt
```CMake
cmake_minimum_required(VERSION 3.5.1)
add_execuable(hello hello.c)
```

```bash
$ mkdir build    
$ cd build    
$ cmake ..    
$ make    
$ ./hello    
hello world    
```

---


다른 명령들 :  

+ prjoect(프로젝트명)  
 프로젝트의 이름을 정한다

+ message (STATUS "메세지")
cmake 도중에 메세지를 출력한다. 변수들의 값을 확인할 수 있다
```CMake
messgae (STATUS "src :  ${SOURCES}")	
```
+ include_directories()
헤더 폴더를 추가한다
```CMake
include_directories(header_folder)
```
headder_folder애서 헤더파일을 찾는다
		
+ find_package()
시스템에 있는 모듈을 찾는다  
cmake에 관련 값들은 cmake에서 설정해두었다  
```CMake
 find_package( Threads)
```
는	현재 OS의 쓰레드 관련 라이브러리를 찾는다  
			이 명령을 통해  
			MATH_THREAD_LIBS_INIT  
			CMAKE_USE_SPROC_INIT  
			CMAKE_USE_WIN32_THREAD_INIT  
			CMAKE_USE_PTHREADS_INIT  
			같은 변수들을 사용할 수 있게된다  
```CMake
 find_package(ALSA)  
```
는		ALSA_FOUND  
		ALSA_LIBRARIES  
		ALSA_INCLUDE_DIRS  
		ALSA_LIBRARY  
		ALSA_VERSION_STRING   
등을 사용할 수 있게된다			 


참고  
[modules](https://cmake.org/cmake/help/v3.0/manual/cmake-modules.7.html)
	- 패키지 목록
[find package](https://cmake.org/cmake/help/v3.0/command/find_package.html?highlight=find_package)



+	list(APPEND 값들)  
		- 변수에 값을 추가한다
		 set은 값을 덮어씌우는데  
		  list(APPEND  )를 쓰면 뒤에 이어 붙인다

+ add_definitions(-D변수 )  
		- 변수를 define 한다
		  코드 내에서 define 한것과 같은 효과를 가진다
		ex)
src.c
```C
if(_IS_DEFINE_)
	{
	 //do something
	}

```


이라는 코드가 있을때   
CMakeLists.txt에    
```CMake
add_definitions(-D_IS_DEFINE_)     
```
을 하면 코드에서 define 하지 않아도    
정의 된다    

+ if()  endif()
	- 조건문  

```CMake		
 if(LINUX)  
	... 리눅스 일 떄 
elseif(WIN32)  
	... 윈도우 일 때
endif()  
```			

OS별로 다르게 빌드 할 수 있다
			
	
	add_library(라이브러리명 옵션 파일)    
		- 라이브러리를 만든다  
		ex)add_library(hello SHARED ${SOURCES})  
			${SOURCES}에 있는 파일로 libhello.so 를 만든다  

		   add_library(bye STATIC ${CODES})  
			${COES}에 있는 파일로 bye.a를 만든다  

	target_link_libraries(TARGET LIBS)  
			- TARGET에 LIBS을 링크한다  
			  동적라이브러리를 실행파일에 링크할 수 있다  


```CMake
add_executable(out ${SOURCES})  
add_library(hello SHARED ${LIBSRC})  
target_link_libraries(out hello)  
```
SOURCES로 빌드한 out에
LIBSRC로 만든 libhello.so 라는 동적 라이브러리를
연결시켜준다

---


미리 설정된 변수 :  

	UNIX    : OS 가 UNIX 면 true 아니면 false
	APPLE   : OS 가 APPLE 이면 true 아니면 false, UNIX여도 true
	WIN32   : OS 가 WINDOWS 면 true 아니면 false,64bit이어도 true

참고  
 [variables](https://gitlab.kitware.com/cmake/community/wikis/doc/cmake/Useful-Variables)



## 예시 2-2

/RtAudio
```CMake
 cmake_minimum_required(VERSION 3.5.1)
 project(custom_RtAudio)
 
 #----Release, Debug ...
set(CMAKE_BUILD_TYPE Release)

 #set(LIST_DIRECTORIES true)
 #set(CMAKE_INCLUDE_CURRENT_DIR ON)

 #----directory for headers, cmake automatically sets dependency
 #include_directories(dic_header_folder)

 #set (SOURCES src/xx.c src/yy.c) : add src in var SOURCES
 
 #----can add multiple sources with wildcard by 'GLOB'
 #----not recommended, can be defined multiple times
 #file(GLOB SOURCES src\*.c) 
 #----src files
 #set(SOURCES RtAudio.cpp record.cpp)
 set(SOURCES record.cpp)
 
 #----set LINKLIBS for libs to be linked with the executable
 set(LINKLIBS)

 #----Check System OS
 #----There are Variables such as "UNIX","WIN32","APPLE",
 #----which are set by TRUE when that is target system's OS
 #----In APPLE OS, APPLE and UNIX set TRUE
 #----We set unix->alsa/windows->asio
 #----or if(CMAKE_SYSTEM MATCHES Linux)
 if(UNIX AND NOT APPLE)
    message (STATUS "Target System is UNIX")

    #----find_package(package) : find moudle and define following vars

    #----FindThreads( = find_package(Threads)) set variables
    #----MAKE_THREAD_LIBS_INIT     - the thread library
    #----CMAKE_USE_SPROC_INIT       - are we using sproc?
    #----CMAKE_USE_WIN32_THREADS_INIT - using WIN32 threads?
    #----CMAKE_USE_PTHREADS_INIT    - are we using pthreads
    #----CMAKE_HP_PTHREADS_INIT     - are we using hp pthreads
        

    #----find_package for ALSA
    #----ALSA_FOUND       - True if ALSA_INCLUDE_DIR & ALSA_LIBRARY are found
    #----ALSA_LIBRARIES   - Set when ALSA_LIBRARY is found
    #----ALSA_INCLUDE_DIRS - Set when ALSA_INCLUDE_DIR is found
    #----ALSA_INCLUDE_DIR - where to find asoundlib.h, etc.
    #----ALSA_LIBRARY     - the asound library
    #----ALSA_VERSION_STRING - the version of alsa found (since CMake 2.8.8)
    #find_package(ALSA REQUIRED)
    find_package(ALSA)    

    #----REQUIRED : if packaged is not found stops with an error message
    #----CMAKE_THREAD_PREFER_PTHRREAD : if there are multiple lib, use pthread
    find_package(Threads REQUIRED CMAKE_THREAD_PREFER_PTHREAD)


    include_directories(${ALSA_INCLUDE_DIR})
    #append libs to LINKLIBS
    list(APPEND LINKLIBS ${ALSA_LIBRARY} ${CMAKE_THREAD_LIBS_INIT})

    #----define __LINUX_ALSA__ for RtAudio.cpp
    #----considering as LINUX uses ALSA in default
    add_definitions(-D__LINUX_ALSA__)
#----= if(CMAKE_SYSTEM MATCHES Windows)
#----also true in win64
elseif(WIN32)
    message (STATUS "Target System is Windows")
    
    include_directories(include)
    list(APPEND LINKLIBS winmm ole32)

    #----add aditional srcs for windows
    list(APPEND SOURCES
        include/asio.cpp
        include/asiodrivers.cpp
        include/asiolist.cpp
        include/iasiothiscallresolver.cpp)
    add_definitions(-D__WINDOWS_ASIO__)

endif()

#----create shared library
#----Generate the shared library from the sources
#add_library(hello SHARED ${SOURCES})
add_library(rtaudio SHARED RtAudio.cpp rtaudio_c.cpp)
message (STATUS rtaudio)
#list(APPEND LINKLIBS rtaudio)

#create executable file with SOURCES
add_executable(record ${SOURCES})
#add_executable(record record.cpp)

#----For Windows env, but global data must be handled manually.
#----see [ https://blog.kitware.com/create-dlls-on-windows-without-declspec-using-new-cmake-export-all-feature/ ]
if(WIN32)
    #!!!! Didn't check whether works or not,yet
    set(BUILD_SHARED_LIBS ON)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()
message (STATUS ${LINKLIBS})
#----link external libraries with executable
#----order is quiet important

#It is RtAudio.cpp which actually uses LINKLIBS so link LINKLIBS with librtauido.so
target_link_libraries(rtaudio ${LINKLIBS})
target_link_libraries(record rtaudio)

add_executable(monitor monitor.c)

```

+ 참고사항


윈도우에서 DLL 사용시 아래의 ENTRY를 추가(bool,true)  
CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS  
BUILD_SHARED_LIBS  

단. global data value는 따로 처리해야한다.   
참고 :  https://blog.kitware.com/create-dlls-on-windows-without-declspec-using-new-cmake-export-all-feature/

---

# [RtAudio](#index)<a name="RtAudio"></a>

---

# [BLAS](#index)<a name="BLAS"></a>

