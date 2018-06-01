# manuals
a few manuals for a few things

**INDEX**

1. a
2. [cmake](#cmake)
3. c
4. d



## CMAKE<a name="cmake"></a>

cmake 는 linux환경에서는 Makefile을 Windows환경에서는 비주얼 스튜디오 프로젝트를 만든다.


## 설치

linux
    $ sudo apt-get install cmake 로 설치한다       
      2.   windows 


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

cmake_minimum_required(VERSION 내.cmake의.버전)  
        ex) cmake_minimum_required(VERSION 3.5.1)  
    - cmake 최소 버전 요구사항 설정, 버전이 다르면 설정된 값이나 명령어 사용이 다를수 있다.

	- cmake 의 버전은  
		$ cmake -version 으로 알 수 있다
         
기본 명령 :   
    set (변수명 들어갈값 )   
        ex)set(SOURCES src1.c src2.c src3.c)   
	
		- 변수를 불러올 때에는 ${변수명} 으로 불러온다

	add_executable(파일명 들어갈코드 )  
		ex)add_executable(hello hello.c)
			or  
			add_executable(programm SOURCES)  
		- 파일명에 해당하는 실행파일을 뒤의 인자로 들어가는 코드를로 빌드하게 한다  

---
## 예시 2-1

/CMAKE

hello world 를 출력하는 hello.c가 있으면

CMakeLists.txt
<pre>
cmake_minimum_required(VERSION 3.5.1)
add_execuable(hello hello.c)
</pre>

$ mkdir build
$ cd build
$ cmake ..
$ make
$ ./hello
hello world

---
	
	prjoect(프로젝트명)  
		- 프로젝트의 이름을 정한다

	message (STATUS "메세지")
		- cmake 도중에 메세지를 출력한다. 변수들의 값을 확인할 수 있다
	
	include_directories()

	find_package()
	
	list()
	list(APPEND)

	add_definitions

	if()
	endif()

	add_library(SHARED)
	
	target_link_libraries()


---

변수 :  

## 예시 2-2

/RtAudio
<pre>
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

</pre>

참고사항


윈도우에서 DLL 사용시 아래의 ENTRY를 추가(bool,true)
CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS
BUILD_SHARED_LIBS

단. global data value는 따로 처리해야한다. 
참고 :  https://blog.kitware.com/create-dlls-on-windows-without-declspec-using-new-cmake-export-all-feature/

[variables](https://gitlab.kitware.com/cmake/community/wikis/doc/cmake/Useful-Variables)

[modules](https://cmake.org/cmake/help/v3.0/manual/cmake-modules.7.html)
alsa 나 thread같은 lib들을 찾아준다

[find package](https://cmake.org/cmake/help/v3.0/command/find_package.html?highlight=find_package)
모듈을 활용


