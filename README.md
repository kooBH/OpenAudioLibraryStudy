# manuals
a little manual for a few things

[A](./manuals/a.md)

**INDEX**<a name="index"></a>

### 1.  [Makefile](#Makefile)
* [gcc](#Makefile-gcc)
* [library](#Makefile-library)
* [Makefile](#Makefile-Makefile)
### 2. [cmake](#cmake)
* [설치](#cmake-setup)
* [사용](#cmake-execution)
* [CMakeLists.txt 작성](#cmake-cmakelists)
	1. [예제 1](#cmake-ex1)
	2. [예제 2](#cmake-ex2)

### 3. [RtAudio](#RtAudio) 
* [설치](#RtAudio-setup)
* [사용](#RtAudio-execution)
* [커스텀](#RtAudio-custom)
### 4. [CBLAS](#CBLAS)
* [OpenBLAS](#OpenBLAS)
* [MKL](#MKL)
* [예시](#cblas_ex)

### 5. [OpenMP](#OpenMP)
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

---


# [CMAKE](#index)<a name="cmake"></a>

cmake 는 linux환경에서는 Makefile을 Windows환경에서는 비주얼 스튜디오 프로젝트를 만든다.


## 설치<a name="cmake-setup"></a>

+ linux
```bash
    $ sudo apt-get install cmake       
```
+  windows 
  https://cmake.org/download/
  

## 사용<a name="cmake-execution"><a/>

1. 빌드할 프로젝트가 있는 폴더에
 CMakeLists.txt 를 만든다.

2.    CmakeLists.txt 를 작성한뒤
3.    $ cmake 를 해주면 Makefile 이 생성된다.

 ※builld 폴더 같은 걸 따로 만들어준 다음  
 $ cmake .. 으로 결과물을 따로 보관하는 것이 좋다.
             Makefile 을 작성해주는 것이아니라 Makefile 이 추가적으로 cmake가 만든 파일들을 이용하게 하는 것에 가깝디.



## CMakeLists.txt 작성<a name="cmake-cmakelists"></a>

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
## 예시 2-1<a name="cmake-ex1"></a>

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



## 예시 2-2<a name="cmake-ex2"></a>

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
## 설치<a name ="RtAudio-setup"></a>

```bash
sudo apt-get install libasound2-dev
sudo apt-get install git-core
git clone https://github.com/thestk/rtaudio.git\
```
libasound2-dev 는 ubuntu 의 기본사운드 API인 ALSA의 dev lib이며 빌드에 필수적이다
그리고 git에서 RtAudio를 가져온다

```bash
mkdir _build_
cd _build_
cmake ..
make
```

cmake 시에 부가적인 파일이 많이 생성되므로 _build_ 폴더를 만들라고 install.txt에 는 적혀있지만
다른 이름을 사용해도 무방하다
폴더를 만든 다음엔 빌드 폴더에서 cmake .. 으로 Makefile을 만들고 make를 하면 tests 폴더가 생성되고
거기에 RtAudio의 기본 프로그램들이 생성된다

## 사용<a name = "RtAudio-execution"></a>
* 장치표시 프로그램 | audioprobe    
ex) $ ./audioprobe

<pre>
RtAudio Version 5.0.0

Compiled APIs:
  Linux ALSA

Current API: Linux ALSA

Found 6 device(s) ...

Device Name = hw:HDA Intel PCH,0
Probe Status = Successful
Output Channels = 6
Input Channels = 2
Duplex Channels = 2
This is the default output device.
This is the default input device.
Natively supported data formats:
  16-bit int
  32-bit int
Supported sample rates = 44100 48000 96000 192000 

Device Name = hw:HDA Intel PCH,1
Probe Status = Successful
Output Channels = 2
Input Channels = 0
Duplex Channels = 0
This is NOT the default output device.
This is NOT the default input device.
Natively supported data formats:
  16-bit int
  32-bit int
Supported sample rates = 32000 44100 48000 88200 96000 192000 

Device Name = hw:HDA Intel PCH,2
Probe Status = Successful
Output Channels = 0
Input Channels = 2
Duplex Channels = 0
This is NOT the default output device.
This is NOT the default input device.
Natively supported data formats:
  16-bit int
  32-bit int
Supported sample rates = 44100 48000 96000 192000 

Device Name = hw:HDA Intel PCH,3
Probe Status = Successful
Output Channels = 8
Input Channels = 0
Duplex Channels = 0
This is NOT the default output device.
This is NOT the default input device.
Natively supported data formats:
  16-bit int
  32-bit int
Supported sample rates = 32000 44100 48000 88200 96000 176400 192000 

Device Name = hw:HDA Intel PCH,7
Probe Status = Successful
Output Channels = 8
Input Channels = 0
Duplex Channels = 0
This is NOT the default output device.
This is NOT the default input device.
Natively supported data formats:
  16-bit int
  32-bit int
Supported sample rates = 32000 44100 48000 88200 96000 176400 192000 

Device Name = default
Probe Status = Successful
Output Channels = 32
Input Channels = 32
Duplex Channels = 32
This is NOT the default output device.
This is NOT the default input device.
Natively supported data formats:
  16-bit int
  24-bit int
  32-bit int
  32-bit float
Supported sample rates = 4000 5512 8000 9600 11025 16000 22050 32000 44100 48000 88200 96000 176400 192000 
</pre>

장치 번호는 위에서 부터 0번이다 
채널 수, 지원하는 레이트 등을 보여준다  


* 녹음 프로그램 | record    
useage: record N fs <duration> <device> <channelOffset>  
    where N = number of channels,  
    fs = the sample rate,  
    duration = optional time in seconds to record (default = 2.0),  
    device = optional device to use (default = 0),  
    and channelOffset = an optional channel offset on the device (default = 0).  

ex)
```bash
$ record 9 48000 60 5
```
실행 폴더에 record.raw로 저장된다

## 커스텀<a name = "RtAudio-custom"></a>

RtAudio는 대부분의 환경에서 동작하는 라이브러리를 지향하기 때문에, 특정 환경에서만 사용할 경우  
쓰지 않는 요소들이 많다. 하지만
[The RtAudio Home Page](https://www.music.mcgill.ca/~gary/rtaudio/)
에 따르면 RtAudio.h RtAudio.cpp 만 있으면 되기 때문에 개별적으로 사용하기 
용이하다. 사이트에 rtaudio/tests에 있는 파일들을 활용하는 방법이 설명되어있기 때문에
둘러보는 것을 추천한다

이 gitbub 에 있는 RtAudio는
ubuntu 16.04 환경하의 record와 audioprobe만 build하며  [CMakeLists.txt](#cmake-ex2)  
record는 경로를 받아서 그 경로에 .wav 형식의 파일을 무한히 녹음하도록 되어있다

---

# [CBLAS](#index)<a name="CBLAS"></a>
+ OpenBLAS<a name="OpenBLAS"></a>  
	1. 설치
	```bash
	$ sudo apt-get install openblas-base 
	#/usr/lib/openblas-base/에 .a와 .so만 받는다 
	    
	$ git clone https://github.com/xianyi/OpenBLAS.git	
	#openblas project를 받는다  
	 ```
	 apt로 package를 받았을 경우 바로 사용하면된다  
	 git으로 받았을 경우에는  
	 make를 하면 CPU에 맞게 빌드해 준다  
	 또는 make TARGET=(CPU이름) 으로 지정해 줄 수도 있다
	   지원하는 CPU는 TargetList.txt에 있다  
	  	
	 2. 컴파일
	   + package를 받았을 경우   
	   -lopenblas  
	   만 해도 링크가 된다  
	   + 프로젝트를 받았을 경우
	    make 했을 때, libopenblas_CPU이름-r0.3.0.dev  .a 와 .so 가 생성된다  
	    -lopenblas_CPU이름-r0.3.0.dev 해주거나 라이브러리 파일의 이름을 바꿔줘서 옵션으로 받아주면 된다  
	    같은 이름의 라이브러리가 2개 나오기 때문에 -static 이나 -shared 로 명시를 해줘야 한다  
	    
	    또한 Thread를 포함하기 때문에  
	    -lpthread  
	    를 해주어야 한다
	    
	 3. 사용 
	   #include "cbals.h"
	   
	   
+ <a name="MKL">Intel MKL</a>
	 1. 설치
	https://software.seek.intel.com/performance-libraries
	에서 Submit 하고 파일 받아서  
	Sudo tar -xzvf 파일명  
	하면 나오는 install.sh 를 실행  
	 또는 install_GUI.sh 를 써도 된다
	
	2. 환경 변수
	(설치폴더)/compilers_and_libraries_2018/linux/mkl/bin/mklvars.sh  
	는 환경 변수를 설정해주는 스크립트	
	$ source (mkvars경로)/mklvars.sh (arch) 
	로 적용  
	(arch) 는 32bit 면 ia32 64bit면 intel64  
	  스크립트로 export한 환경변수는 터미널이 닫히면 지속되지 않으므로  
	  ~/.bashrc(터미널을 열때마다 실행 )  이나  
	  ~/.profile(부팅 후 유저 로그인 시 실행)  에
	  source (mkvars경로)/mklvars.sh (arch) 를 추가해주면 된다  
	  
	3. 컴파일  
	[예제 파일](http://software.intel.com/sites/default/files/article/171460/mkl-lab-solution.c)       
	[컴파일 옵션 알아보기](https://software.intel.com/en-us/articles/intel-mkl-link-line-advisor/)   
	컴파일 옵션 알아보는 사이트에서 자신의 조건에 맞는 컴파일 옵션을 찾는다  
		예 )
		+ link line  
		-Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_sequential.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm -ldl
		+ compile option  
		-DMKL_ILP64 -m64 -I${MKLROOT}/include
		+ 실제 명령  

		```bash 
		gcc  -DMKL_ILP64 -m64 -I${MKLROOT}/include  mkl-lab-solution.o  -Wl,--start-    group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/lib    mkl_sequential.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lpthr    ead -lm -ldl  -lm
		```  

		 옵션의 순서가 중요하다. 순서가 다르면 빌드 되지 않는다  
		[Guide](https://software.intel.com/en-us/articles/intel-math-kernel-library-intel-mkl-2018-getting-started)
	4. 사용  
	  #include "mkl.h"
	  
	  
+예시<a name="cblas_ex"></a>
```C++
#include "cblas.h"
#include <stdio.h>

int main(){
/*
 *  cblas_?gemm(layout,transA,transB,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc)
 *
 *    layout :   i) --->CblasRowMajor
 *    			   [0][1]  =  {0,1,2,3}
 *                	   [2][3]
 *
 *             ii)  |  [0][2] = {0,1,2,3}
 *                  |  [1][3]
 *                 \_/ CblasColMajor
 *
 *   
 *   C := alpha * op(A)*op(B) + beta*C
 *
 *     op(X) =  	  i) X      when transX = CblasNoTrans
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
 *      ldb : 		''	     B
 *      ldc :		''	     C
 *
 * 		-the first dimension : the number of columns when CblasRowMajor
 *		                 	   	''    rows   when CblasColMajor
 *
 * */
/* ex1
 * */	int i,j;
	
	float a1[]={1,1,1,1};
	float b1[]={1,2,3,4,5,6};
	float c1[6];

	int m1 = 2;
	int k1 = 2;
	int n1 = 3; 

	int lda1=k1;
	int ldb1=n1;
	int ldc1=n1;

	int alpha1 = 1;
	int beta1 = 0;
	
	cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,m1,n1,k1,alpha1,a1,k1,b1,n1,beta1,c1,n1);

	for(i=0;i< m1; i++)
	{
		for(j=0;j<n1; j++)
			printf("%4.2f ",c1[i*n1 + j]);
		printf("\n");
	}
	printf("\n");

/* ex2
 * ---->--->--->RowMajor
 *a2 | 0.1 0.4 |
 *   | 0.2 0.3 |  lda = 2 -> CblasTrans->  | 0.1 0.2 0.3 0.4 |  m = 2
 *   | 0.3 0.2 |                           | 0.4 0.3 0.2 0.1 |  k = 4      
 *   | 0.4 0.1 |
 *
 *
 *b2 | 10 |   k=4
 *   | 10 |   n=1
 *   | 10 |   ldb = 1
 *   | 10 |
 *
 *c2 | -110 |  m=2
 *   |   90 |  n=1
 *             ldc=1
 * */

	double a2[8]={0.1, 0.4, 0.2, 0.3, 0.3, 0.2, 0.4, 0.1};
	double b2[4]={10,10,10,10};
	double c2[2]={-100,100};
	
	int m2 = 2;
	int k2 = 4;
	int n2 = 1;

	int alpha2 = -1;
	int beta2 = 1;

	int lda2=m2;
	int ldb2=n2;
	int ldc2=n2;
cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,m2,n2,k2,alpha2,a2,lda2,b2,ldb2,beta2,c2,ldc2);
	for(i=0;i< m2; i++)
	{
		for(j=0;j<n2; j++)
			printf("%4.2f ",c2[i*n2 + j]);
		printf("\n");
	}	printf("\n");
/* ex3 = ex4
  *
  *a3 | 1-1i  2-2i  3-3i |
  *
  *b3 |  1+1i   4+4i |
  *   |  2+2i   5+5i |
  *   |  3+3i   6+6i |
  *
  *c3 | 28+28i  64+64i| 
  *
  *
  * */

	typedef struct fx{float r;float i;}fx;

	fx a3[3]={1,-1,2,-2,3,-3};
	fx b3[6]={1,1,2,2,3,3,4,4,5,5,6,6};
	fx c3[2]={0,0,0,0};
	
	int m3 = 1;
	int k3 = 3;
	int n3 = 2;

	fx alpha3 ={1,1};
	fx beta3 = {0,0};

	int lda3=1;
	int ldb3=3;
	int ldc3=1;
/*ex4
 * a3 | 1-1i |  ->CblasTransA | 1-1i  2-2i  3-3i |
 *    | 2-2i |
 *    | 3-3i |
 *
 *b3 |1+1i 2+2i 3+3i|    ->CblasTransB |  1+1i   4+4i |
 *   |4+4i 5+5i 6+6i|                  |  2+2i   5+5i |
 *                                     |  3+3i   6+6i |
 *
 *c3 | 28+28i  64+64i| 
 * */
cblas_cgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,m3,n3,k3,&alpha3,a3,lda3,b3,ldb3,&beta3,c3,ldc3);

	for(i=0;i< m3; i++)
	{	for(j=0;j<n3; j++)
			printf("%4.2f %+4.2fi ",c3[i*n3 + j].r,c3[i*n3+j].i);
		printf("\n");
	}	printf("\n");
	
	double a4[6] = {1,-1,2,-2,3,-3};
	double b4[12] = {1,1,4,4,2,2,5,5,3,3,6,6};
	double c4[4] = {0,0,0,0};

	int m4 = 1;
	int k4 = 3;
	int n4 = 2;

	double alpha4[2] = {1,1};
	double beta4[2] = {0,0};

	int lda4 = 3;
	int ldb4 = 2;
	int ldc4 = 1;
cblas_zgemm(CblasColMajor,CblasTrans,CblasTrans,m4,n4,k4,&alpha4,a4,lda4,b4,ldb4,&beta4,c4,ldc4);
	for(i=0;i< m4; i++)
	{	for(j=0;j<(n4*2); j+=2)
			printf("%2.2lf %+2.2lfi ",c4[i*n4 + j],c4[i*n4 + j+1]);
		printf("\n");
	}	printf("\n");
	return 0;
}
```

# [OpenMP](#index)<a name="OpenMP"></a>
1. OpenMP  
+ 사용하기
OpenMP는 컴파일러에 포함 
gcc 컴파일 옵션으로 -fopenmp

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
 
