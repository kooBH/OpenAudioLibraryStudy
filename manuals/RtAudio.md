
# [RtAudio](../README.md)
1. [설치](#RtAudio-setup)
2. [사용](#RtAudio-execution)
3. [커스텀](#RtAudio-custom)

## 설치<a name ="RtAudio-setup"></a>
1. Linux  
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

2. Windows  
윈도우에서는 cmake를 통해 빌드하면된다. 단 RtAudio는 C++11의 기능을 사용하므로 VS2013 이상의 버전을 사용해야한다.  
(또는 VS2012에 c++11 컴파일 기능을 추가하거나)

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
