

# <a name ="TOP">[Open Audio Libs](../README.md)</a>

1. ### [SUMMARY](#summary)  
2. ### [CIGLET](#CIGLET)
3. ### [HTK](#HTK)
4. ### [CMUSphinx](#CMUSphinx)
5. ### [openSMILE](#openSMILE)
6. ### [Kaldi](#Kaldi)
---  

&nbsp;|Ciglet|HTK|CMUSphinx|openSMILE|Kaldi
---|---|---|---|---|---
 언어|C |C |C |C++ | C++
 string| X| X | O|    | X 
 matrix| 1D(num),2D(audio) |2D | sym_2D|   |1D
 vector| 1D | 1D| 1D |   | 1D
 complex|struct |struct | struct |   | 2_Vars(\_re,\_im) 
 fft| O | O| O | O|  O 
 MFCC| O |O | O| O | O
 blas| O | HMath | O|   | O
 argumnet| var | var | var | cofig   | var
 Usage | source   | bin  |  source   |  bin | bin   |
 Data I/O|wav | HWav, HAudio |mono, little-endian, raw 16-bit signed PCM audio,16k Hz|     |IOstream : bin or txt
 visualizaion| indirect GNUplot|HGraf| X|  | X
 openMP| O | X | X |   | X
 CUDA| X | HCUDA(alpha 3.5) | X |   | O
 Detail |[SEE](./CIGLET.md) | [SEE](./HTK.md)|[SEE](./CMUSphinx.md)| [SEE](./openSMILE.md)| [SEE](./Kaldi.md)

 
## SUMMARY<a name = "summary"></a>  
 
# [CIGLET](#TOP)<a name="CIGLET"></a>
+ [Link](https://github.com/Sleepwalking/ciglet)
+ [MyDocumnet](./CIGLET.md)
+ lightweight C library for digital signal processing
+ C-written sIGnal codeLETs
+ simple and compact code
+ linux/windows
+ Matlab  to C conversion of frequently used ones 

---


# [Hidden Markov Model Toolkit (HTK)](#TOP)<a name="HTK"></a>
+ [Link](http://htk.eng.cam.ac.uk/)
+ [MyDocumnet](./HTK.md)
+ C source form
+ well organized code
+ linux/windows
+ a set of library modules and tools  
+ http://htk.eng.cam.ac.uk/prot-docs/htkbook.pdf  

+ HCUDA: [CUDA based math kernel functions](http://htk.eng.cam.ac.uk/pdf/woodland_htk35_uea.pdf)

---

# [CMUSphinx](#TOP)<a name="CMUSphinx"></a>
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


# [Kaldi](#TOP)<a name = "Kaldi"></a>
+ [Link](https://github.com/kaldi-asr/kaldi)
+ [MyDocument](./Kaldi.md)
+ C++
+ http://kaldi-asr.org/doc/
+ compile against the OpenFst toolkit (using it as a library)
+ include a matrix library that wraps standard BLAS and LAPACK routines
+ licensed under Apache 2.0, which is one of the least restrictive licenses available

# [openSMILE](#TOP)<a name="openSMILE"></a>
+ [Link](https://audeering.com/technology/opensmile/)
+ [MyDocumnet](./openSMILE.md)
+ c++ API
+ using PortAudio
+ General audio signal processing
+ Extraction of speech-related features
+ Statistical functionals
+ Multi-threading support for parallel feature extraction
+ Designed for real-time on-line processing, but also useful for off-line batch 

# [BeamformIt](https://github.com/xanguera/BeamformIt)
+ acoustic beamforming tool

## Common Traits
+ Don't use primitive data type. use defined data type throughout all source. for easy modification
+ If import another library, using functions with other name, with more consistency. Rename by inline or other method
+ In terms of nameing, there is tendency. C : snake_case, C++ : camelCase. in C++ lib, procedural function and variables about it uses snake_case. But it is up to oneself  
