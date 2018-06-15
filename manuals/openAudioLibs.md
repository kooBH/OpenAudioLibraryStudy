
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
 string| X|  | O|
 matrix| 1D |2D | sym_2D|
 vector| 1D | 1D| 1D |
 complex|struct |struct | struct |
 fft| O | O| X |
 stft| O |X | X|
 blas| O | X | O|
 parameter| direct | | |
 Usage | source   |   |  bin   |  bin
 Data I/O|.wav only| | |
 visualizaion| indirect GNUplot| | |
 openMP| O | X | X |
 CUDA| X | alpha 3.5 | [?]((https://cmusphinx.github.io/page23/)
) |
 Detail| [SEE](#CIGLET) |[SEE](#HTK) | [SEE](#CMUSphinx)|[SEE](#openSMILE)

 
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
+ Data I/O
external library using simple iostream 

### [Hidden Markov Model Toolkit (HTK)](#TOP)<a name="HTK"></a>
+ [Link](http://htk.eng.cam.ac.uk/)
+ C source form
+ well organized code
+ linux/windows
+ a set of library modules and tools  
+ http://htk.eng.cam.ac.uk/prot-docs/htkbook.pdf  
---
+ HCUDA: [CUDA based math kernel functions](http://htk.eng.cam.ac.uk/pdf/woodland_htk35_uea.pdf)

---

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
+ String Modification  :  
  case.h  
  ```C++
  void ucase(char *str); //all upercase
  void lcase(char* str); //all lowercase
  int32 strcmp_nocase(const char *str1, const char *str2); //WIP, case insensitive string compare
  int32 strncmp_nocase(const char *str1, cons char *str2, size_t len); // strcmp_nocase + n
  ```  
  strfuncs.h  
  
```C++

#include <stdarg.h>  //stdarg.h is a header in the C standard library of the C programming language that allows functions to accept an indefinite number of arguments

/**
 * Concatenate a NULL-terminated argument list of strings, returning a
 * newly allocated string.
 **/
SPHINXBASE_EXPORT
char *string_join(const char *base, ...);  //variadic functions(strarg.h) : can have infinity number of arguments ( ... )

/**
 * Which end of a string to operate on for string_trim().
 */
enum string_edge_e {
    STRING_START,	/**< Beginning of string. */
    STRING_END,		/**< End of string. */
    STRING_BOTH		/**< Both ends of string. */
};

/**
 * Remove whitespace from a string, modifying it in-place.
 *
 * @param string string to trim, contents will be modified.
 * @param which one of STRING_START, STRING_END, or STRING_BOTH.
 */
SPHINXBASE_EXPORT
char *string_trim(char *string, enum string_edge_e which);

/**
 * Locale independent version of atof().
 *
 * This function behaves like atof() in the "C" locale.  Switching
 * locale in a threaded program is extremely uncool, therefore we need
 * this since we pass floats as strings in 1000 different places.
 */
SPHINXBASE_EXPORT
double atof_c(char const *str);

/* FIXME: Both of these string splitting functions basically suck.  I
 have attempted to fix them as best I can.  (dhuggins@cs, 20070808) */

/** 
 * Convert a line to an array of "words", based on whitespace separators.  A word
 * is a string with no whitespace chars in it.
 * Note that the string line is modified as a result: NULL chars are placed after
 * every word in the line.
 * Return value: No. of words found; -1 if no. of words in line exceeds n_wptr.
 */
SPHINXBASE_EXPORT
int32 str2words (char *line,	/**< In/Out: line to be parsed.  This
				   string will be modified! (NUL
				   characters inserted at word
				   boundaries) */
		 char **wptr,	/**< In/Out: Array of pointers to
				   words found in line.  The array
				   must be allocated by the caller.
				   It may be NULL in which case the
				   number of words will be counted.
				   This allows you to allcate it to
				   the proper size, e.g.:
				   
				   n = str2words(line, NULL, 0);
				   wptr = ckd_calloc(n, sizeof(*wptr));
				   str2words(line, wptr, n);
				*/
		 int32 n_wptr	/**< In: Size of wptr array, ignored
				   if wptr == NULL */
	);

/**
 * Yet another attempt at a clean "next-word-in-string" function.  See arguments below.
 * @return Length of word returned, or -1 if nothing found.
 * This allows you to scan through a line:
 *
 * <pre>
 * while ((n = nextword(line, delim, &word, &delimfound)) >= 0) {
 *     ... do something with word ..
 *     word[n] = delimfound;
 *     line = word + n;
 * }
 * </pre>
 */
SPHINXBASE_EXPORT
int32 nextword (char *line, /**< Input: String being searched for next word.
			       Will be modified by this function (NUL characters inserted) */
		const char *delim, /**< Input: A word, if found, must be delimited at either
			         end by a character from this string (or at the end
			         by the NULL char) */
		char **word,/**< Output: *word = ptr within line to beginning of first
			         word, if found.  Delimiter at the end of word replaced
			         with the NULL char. */
		char *delimfound /**< Output: *delimfound = original delimiter found at the end
				    of the word.  (This way, the caller can restore the
				    delimiter, preserving the original string.) */
	);

  ```
  
  
 + CUDA  
 [???](https://cmusphinx.github.io/page23/)


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
