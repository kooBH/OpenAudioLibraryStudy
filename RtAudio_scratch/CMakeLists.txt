cmake_minimum_required(VERSION 3.5.1)
project(custom_RtAudio)

#----Release, Debug ...
set(CMAKE_BUILD_TYPE Release)

#----src files
#set(SOURCES RtAudio.cpp record.cpp)
set(SOURCES RtAudio.cpp) 

#----set LINKLIBS for libs to be linked with the executable
set(LINKLIBS)

if(UNIX AND NOT APPLE) 
	message (STATUS "Target System is UNIX")

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
endif()

list(APPEND LINKLIBS RtAudio.cpp)

add_library(rtaudio STATIC ${LINKLIBS} )


