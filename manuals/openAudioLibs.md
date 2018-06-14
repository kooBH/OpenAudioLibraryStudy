
# <a name ="TOP">[Open Audio Libs](#../README.md)</a>

1. ### [SUMMARY](#summary)  
2. ### [CIGLET](./CIGLET.md#CIGLET)
4. ### [CMUSphinx](./CMUSphinx.md#CMUSphinx)
5. ### [openSMILE](./openSMILE.md#openSMILE)
---  

## SUMMARY<a name = "summary"></a>  
 
### [Ciglet](https://github.com/Sleepwalking/ciglet)
+ lightweight C library for digital signal processing
+ C-written sIGnal codeLETs
+ simple and compact code
+ linux/windows
+ Matlab  to C conversion of frequently used ones 

### [Hidden Markov Model Toolkit (HTK)](http://htk.eng.cam.ac.uk/)
+ C source form
+ well organized code
+ linux/windows
+ a set of library modules and tools  
+ http://htk.eng.cam.ac.uk/prot-docs/htkbook.pdf  


### [CMUSphinx](https://cmusphinx.github.io/)
+ language model
+ not intented to mathmatical problems
+ Pocketsphinx — lightweight recognizer library written in C.
+ Sphinxbase — support library required by Pocketsphinx
+ Sphinx4 — adjustable, modifiable recognizer written in Java
+ Sphinxtrain — acoustic model training tools
+ linux/windows

### [kaldi](https://github.com/kaldi-asr/kaldi)
+ C++
+ http://kaldi-asr.org/doc/
+ compile against the OpenFst toolkit (using it as a library)
+ include a matrix library that wraps standard BLAS and LAPACK routines
+ licensed under Apache 2.0, which is one of the least restrictive licenses available

### [openSMILE](https://audeering.com/technology/opensmile/)
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
