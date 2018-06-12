### [CMUSphinx](./openAudioLibs.md#TOP)<a name = "CMUSphinx"></a>
+ [funtion list](#CMUSphinx_list)
+ [function prototype](#CMUSphinx_proto)
+ [license](#CMUSphinx_license)
---

https://cmusphinx.github.io/doc/pocketsphinx/index.html

 ### Unix System
+ Dependency :  
    1. gcc
    2. automake
    3. autoconf
    4. libtool
    5. bison
    6. swig
    7. python development package
    8. pulseaudio development

+ Respective Configuration Option 
  1. --without (ex)--without-swig-python
  2. --enable  (ex)--enable-fixed

---




## [FUNCTION LIST](#CMUSphinx)<a name="CMUSphinx_list"></a>

+ ### sphinxbase/include/sphinxbase/

Header|Discription
---|---
ad.h|brief generic live audio interface for recording and playback
agc.h|  Various forms of automatic gain control (AGC)
bio.h| Sphinx-3 binary file I/O functions
bitarr.h | brief An implementation bit array - memory efficient storage for digit int and float data
bitvec.h | brief An implementation of bit vectors
byteorder.h | Byte swapping ordering macros
case.h |  Upper/lower case conversion routines
ckd_alloc.h | Memory allocation package
clapack_lite.h | (GUESS)minimum lapack function
cmd_ln.h| Command line argument parsing
cmn.h| Various forms of cepstral mean normalization
err.h|brief Implementation of logging routines
f2c.h| Standard Fortran to C header file
feat.h|  Cepstral features computation
fe.h| (GUESS) about float-end
filename.h | File and path name operations
fixpoint.h| Fixed-point arithmetic macros
fsg_model.h| Word-level finite state graph
genrand.h| a portable random generator
glist.h| Module for maintaining a generic, linear linked-list structure
hash_table.h| Hash table module
heap.h| Generic heap structure for inserting in any and popping in sorted order
jsgf.h| JSGF grammar compiler
listelem_alloc.h  | brief Fast memory allocator for uniformly sized objects
logmath.h| brief Fast integer logarithmic addition operations
matrix.h|Matrix and linear algebra functions
mmio.h| brief Memory-mapped I/O wrappers for files
ngram_model.h| brief N-Gram language models
pio.h| Packaged I/O routines
prim_type.h| Primitive types; more machine-independent
priority_queue.h| Priority queue
profile.h| For timing and event counting
sbthread.h| brief Simple portable thread functions
sphinxbase_export.h| (GUESS)About Shared Object
strfuncs.h| brief Miscellaneous useful string functions
yin.h| brief Implementation of pitch estimation

+ ### sphinxbase/src/libsphinxbase/fe
HEADER|Discription
---|---
fe_internal.h |
fe_noise.h |
fe_prespch_buf.h |
fe_type.h | 
fe_warp_affine.h |
fe_warp.h |
fe_warp_inverse_linear.h|
fe_warp+piecewise_linear.h |

+ ### sphinxbase/src/libsphinxbase/lm
HEADER|Discription
---|---
jsgf_internal.h|
jsgf_parse.h
jsgf_sacnner.h|
lm_trie.h|
lm_trie_quant.h|
ngram_model_internal.h|
ngram_model_trie.h|
ngram_raw.h

+ sphinxbase/include/sphinxbase/clapack_lite.h
  sgemm_,sgemv_,ssymm_... parameter 도 gemm parm
  
+ sphinxbase/include/sphinxbase/cmn.h
//brief Apply Cepstral Mean Normaliztion (CMN) to the set of input mfc frames...

### sphinxbase/~/fe.h
typedef float32/fixed32 mfcc_t

### sphinxbase/~/cmn.h
<pre>
cmn_t* cmn_init(int32 veclen);
void cmn (cmn_t *cmn, mfcc_t **mfc,);
void cmn_live(cmn_t *cmn,mfcc_t ,nt32 varnorm,int32 nfr );
void cmn_live_update(cmn_t *cmn);
void cmn_live_set(cmn_t *cmn, mfcc_t const *vec);
void cmn_live_get(cmn_t *cmn, mfcc_t *vec);
void cmn_free (cmn_t *cmn);
</pre>
---

## [FUNCTION PROTOTYPE](#CMUSphinx)<a name = "CMUSphinx_proto"></a>

### sphinxbase/~/cmn.h

```C++
/** \file cmn.h
 * \brief Apply Cepstral Mean Normalization (CMN) to the set of input mfc frames.
 *
 * By subtractingthe mean of the input from each frame.  C0 is also included in this process.
 * This function operates on an entire utterance at a time.  Hence, the entire utterance
 * must be available beforehand (batchmode).
 */

/**
 * Types of cepstral mean normalization to apply to the features.
 */
typedef enum cmn_type_e {
    CMN_NONE = 0,
    CMN_BATCH,
    CMN_LIVE
} cmn_type_t;

/** String representations of cmn_type_t values. */
SPHINXBASE_EXPORT
extern const char *cmn_type_str[];

/** Convert string representation (from command-line) to cmn_type_t */
SPHINXBASE_EXPORT
cmn_type_t cmn_type_from_str(const char *str);

/** \struct cmn_t
 *  \brief wrapper of operation of the cepstral mean normalization. 
 */

typedef struct {
    mfcc_t *cmn_mean;   /**< Temporary variable: current means */
    mfcc_t *cmn_var;    /**< Temporary variables: stored the cmn variance */
    mfcc_t *sum;        /**< The sum of the cmn frames */
    int32 nframe;	/**< Number of frames */
    int32 veclen;	/**< Length of cepstral vector */
} cmn_t;

SPHINXBASE_EXPORT
cmn_t* cmn_init(int32 veclen);

/**
 * CMN for the whole sentence
*/
SPHINXBASE_EXPORT
void cmn (cmn_t *cmn,   /**< In/Out: cmn normalization, which contains the cmn_mean and cmn_var) */
          mfcc_t **mfc,	/**< In/Out: mfc[f] = mfc vector in frame f */
	  int32 varnorm,/**< In: if not FALSE, variance normalize the input vectors
			   to have unit variance (along each dimension independently);
			   Irrelevant if no cmn is performed */
	  int32 n_frame /**< In: Number of frames of mfc vectors */
	);

#define CMN_WIN_HWM     800     /* #frames after which window shifted */
#define CMN_WIN         500

/**
 * CMN for one block of data, using live mean
 */
SPHINXBASE_EXPORT
void cmn_live(cmn_t *cmn,        /**< In/Out: cmn normalization, which contains
                                    the cmn_mean and cmn_var) */
               mfcc_t **incep,  /**< In/Out: mfc[f] = mfc vector in frame f*/
	       int32 varnorm,    /**< This flag should always be 0 for live */
	       int32 nfr         /**< Number of incoming frames */
    );

/**
 * Update live mean based on observed data
 */
SPHINXBASE_EXPORT
void cmn_live_update(cmn_t *cmn);

/**
 * Set the live mean.
 */
SPHINXBASE_EXPORT
void cmn_live_set(cmn_t *cmn, mfcc_t const *vec);

/**
 * Get the live mean.
 */
SPHINXBASE_EXPORT
void cmn_live_get(cmn_t *cmn, mfcc_t *vec);

/* RAH, free previously allocated memory */
SPHINXBASE_EXPORT
void cmn_free (cmn_t *cmn);

```

---

#### [LICENSE](#CIGLET)<a name = "CMUSphinx_license"></a>  

https://cmusphinx.github.io/wiki/about/  
+ **BSD-like** license which allows commercial distribution  


+ fe_noise.h

<pre>
/* -*- c-basic-offset: 4; indent-tabs-mode: nil -*- */
/* ====================================================================
 * Copyright (c) 2013 Carnegie Mellon University.  All rights 
 * reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer. 
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in
 *    the documentation and/or other materials provided with the
 *    distribution.
 *
 * This work was supported in part by funding from the Defense Advanced 
 * Research Projects Agency and the National Science Foundation of the 
 * United States of America, and the CMU Sphinx Speech Consortium.
 *
 * THIS SOFTWARE IS PROVIDED BY CARNEGIE MELLON UNIVERSITY ``AS IS'' AND 
 * ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, 
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL CARNEGIE MELLON UNIVERSITY
 * NOR ITS EMPLOYEES BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT 
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, 
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY 
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * ====================================================================

</pre>

---
