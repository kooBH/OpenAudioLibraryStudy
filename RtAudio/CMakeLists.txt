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
#file(GLOB SOURCES src/*.c)

#----src files
#set(SOURCES RtAudio.cpp record.cpp)
set(SOURCES record.cpp RtAudio.cpp) 

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


	add_executable(monitor monitor.c)

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
	include_directories(include)
	list(APPEND LINKLIBS winmm ole32)
	add_definitions(-D__WINDOWS_WASAPI__)
	message(STATUS "using Windows WASAPI")
	list(APPEND LINKLIBS uuid ksuser)
endif()	

#Windows 7 uses WASAPI
#	message (STATUS "Target System is Windows")	
#	list(APPEND LINKLIBS winmm ole32)
#	list(APPEND SOURCES
#		include/asio.cpp
#		include/asiodrivers.cpp
#		include/asiolist.cpp
#		include/iasiothiscallresolver.cpp)
#	add_definitions(-D__WINDOWS_ASIO__)

#----create shared library
#----Generate the shared library from the sources
#add_library(hello SHARED ${SOURCES})
#add_library(rtaudio SHARED RtAudio.cpp)
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

cmake_policy(set CMP0042 OLD)

target_link_libraries(record ${LINKLIBS})

#It is RtAudio.cpp which actually uses LINKLIBS so link LINKLIBS with librtauido.so
#target_link_libraries(rtaudio ${LINKLIBS})
#target_link_libraries(record rtaudio)

add_executable(audioprobe audioprobe.cpp RtAudio.cpp)
target_link_libraries(audioprobe ${LINKLIBS})

