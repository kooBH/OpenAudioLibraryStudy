
# <a name ="TOP">[Open Audio Libs](../README.md)</a>

1. ### [SUMMARY](#summary)  
2. ### [CIGLET](#CIGLET)
4. ### [CMUSphinx](#CMUSphinx)
5. ### [openSMILE](#openSMILE)
---  

List  
1. fft, stft, istft 와 함수 형태
2. string 처리 함수 종류
3. 언어
4. Matrix, vector, Scalar 표현, complex 표현
5. 함수 인자 전달방식
6. 데이터 시각화 방식. plot or print
7. 데이터 I/O 어떻게 하는지. 데이터 포맷, 방식
8. OpenMP 지원 / CUDA 지원?

&nbsp;|Ciglet|HTK|CMUSphinx|openSMILE
---|---|---|---|---
 언어|C |C |C |C++
 string| X| | |
 matrix| 1D |2D | |
 vector| 1D | 1D| |
 complex|struct |struct | |
 fft| O | O| X |
 stft| O |X | X|
 blas| O | X | O|
 parameter| direct | | |
 visualizaion| customed GNUplot| | |
 data I/O|external library using simple iostream | | |
 openMP| O | X | |
 CUDA| X | alpha 3.5 | |
 Detail| [SEE](#1) |[SEE](#2) | [SEE](#3)|[SEE](#4)

 
## SUMMARY<a name = "summary"></a>  
 
### [CIGLET](#TOP)<a name="CIGLET"></a>
+ [Link](https://github.com/Sleepwalking/ciglet)
+ [MyDocumnet](./Ciglet.md)
+ lightweight C library for digital signal processing
+ C-written sIGnal codeLETs
+ simple and compact code
+ linux/windows
+ Matlab  to C conversion of frequently used ones 
---
+ fft 
```C++
void cig_fft(FP_TYPE* xr, FP_TYPE* xi, FP_TYPE* yr, FP_TYPE* yi, int n, FP_TYPE* buffer, FP_TYPE mode)  
```
+ stft 
```C++
void cig_stft_forward(FP_TYPE* x, int nx, int* center, int* nwin, int nfrm, int nfft, char* window, int subt_mean, int optlv,FP_TYPE* norm_factor, FP_TYPE* weight_factor, FP_TYPE** Xmagn, FP_TYPE** Xphse)  
```


### [Hidden Markov Model Toolkit (HTK)](#TOP)<a name="HTK"></a>
+ [Link](http://htk.eng.cam.ac.uk/)
+ C source form
+ well organized code
+ linux/windows
+ a set of library modules and tools  
+ http://htk.eng.cam.ac.uk/prot-docs/htkbook.pdf  
+ HCUDA: [CUDA based math kernel functions](http://htk.eng.cam.ac.uk/pdf/woodland_htk35_uea.pdf)

---

+ Data Type    
```C++
typedef unsigned long           Ulong;  
typedef unsigned short          Ushort;  
typedef signed char         Schar;  
typedef unsigned char           Uchar;  
typedef struct {double  real, imag;}    DoubleComplex;  
typedef struct {float   real, imag;}    FloatComplex;  
typedef struct {long    real, imag;}    LongComplex;  
typedef struct {short   real, imag;}    ShortComplex;  
typedef struct {Schar   real, imag;}    ScharComplex;  
typedef unsigned char           Bool;  
typedef unsigned short          Wchar;  e
```


### [CMUSphinx](#TOP)<a name="CMUSphinx"></a>
+ [Link](https://cmusphinx.github.io/)
+ [MyDocumnet](./CMUSphinx.md)
+ language model
+ not intented to mathmatical problems
+ Pocketsphinx — lightweight recognizer library written in C.
+ Sphinxbase — support library required by Pocketsphinx
+ Sphinx4 — adjustable, modifiable recognizer written in Java
+ Sphinxtrain — acoustic model training tools
+ linux/windows
---
### kaldi
+ [Link](https://github.com/kaldi-asr/kaldi)
+ C++
+ http://kaldi-asr.org/doc/
+ compile against the OpenFst toolkit (using it as a library)
+ include a matrix library that wraps standard BLAS and LAPACK routines
+ licensed under Apache 2.0, which is one of the least restrictive licenses available

### [openSMILE](#TOP)<a name="openSMILE"></a>
+ [Link](https://audeering.com/technology/opensmile/)
+ [MyDocumnet](./openSMILE.md)
+ c++ API
+ using PortAudio
+ General audio signal processing
+ Extraction of speech-related features
+ Statistical functionals
+ Multi-threading support for parallel feature extraction
+ Designed for real-time on-line processing, but also useful for off-line batch 

### [BeamformIt](https://github.com/xanguera/BeamformIt)
+ acoustic beamforming tool

### Common Traits
+ Don't use primitive data type. use defined data type throughout all source. for easy modification
+ If import another library, using functions with other name, with more consistency. Rename by inline or other method
