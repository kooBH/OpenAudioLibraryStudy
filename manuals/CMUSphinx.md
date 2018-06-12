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
+ [cmn.h](#cmn.h)
+ [fe_internal.h](#fe_internal.h)

### sphinxbase/~/[cmn.h](#CMUSphinx_proto)<a name="cmn.h"></a>

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

+ ### sphinxbase/src/libsphinxbase/fe/[fe_inernal.h](#CMUSphinx_proto)<a name="fe_internal.h"></a>

```c++

/* Values for the 'logspec' field. */
enum {
	RAW_LOG_SPEC = 1,
	SMOOTH_LOG_SPEC = 2
};

/* Values for the 'transform' field. */
enum {
	LEGACY_DCT = 0,
	DCT_II = 1,
        DCT_HTK = 2
};

typedef struct melfb_s melfb_t;
/** Base Struct to hold all structure for MFCC computation. */
struct melfb_s {
    float32 sampling_rate;
    int32 num_cepstra;
    int32 num_filters;
    int32 fft_size;
    float32 lower_filt_freq;
    float32 upper_filt_freq;
    /* DCT coefficients. */
    mfcc_t **mel_cosine;
    /* Filter coefficients. */
    mfcc_t *filt_coeffs;
    int16 *spec_start;
    int16 *filt_start;
    int16 *filt_width;
    /* Luxury mobile home. */
    int32 doublewide;
    char const *warp_type;
    char const *warp_params;
    uint32 warp_id;
    /* Precomputed normalization constants for unitary DCT-II/DCT-III */
    mfcc_t sqrt_inv_n, sqrt_inv_2n;
    /* Value and coefficients for HTK-style liftering */
    int32 lifter_val;
    mfcc_t *lifter;
    /* Normalize filters to unit area */
    int32 unit_area;
    /* Round filter frequencies to DFT points (hurts accuracy, but is
       useful for legacy purposes) */
    int32 round_filters;
};

/* sqrt(1/2), also used for unitary DCT-II/DCT-III */
#define SQRT_HALF FLOAT2MFCC(0.707106781186548)

typedef struct vad_data_s {
    uint8 in_speech;
    int16 pre_speech_frames;
    int16 post_speech_frames;
    prespch_buf_t* prespch_buf;
} vad_data_t;

/** Structure for the front-end computation. */
struct fe_s {
    cmd_ln_t *config;
    int refcount;

    float32 sampling_rate;
    int16 frame_rate;
    int16 frame_shift;

    float32 window_length;
    int16 frame_size;
    int16 fft_size;

    uint8 fft_order;
    uint8 feature_dimension;
    uint8 num_cepstra;
    uint8 remove_dc;
    uint8 log_spec;
    uint8 swap;
    uint8 dither;
    uint8 transform;
    uint8 remove_noise;
    uint8 remove_silence;

    float32 pre_emphasis_alpha;
    int16 pre_emphasis_prior;
    int32 dither_seed;

    int16 num_overflow_samps;    
    size_t num_processed_samps;

    /* Twiddle factors for FFT. */
    frame_t *ccc, *sss;
    /* Mel filter parameters. */
    melfb_t *mel_fb;
    /* Half of a Hamming Window. */
    window_t *hamming_window;

    /* Noise removal  */
    noise_stats_t *noise_stats;

    /* VAD variables */
    int16 pre_speech;
    int16 post_speech;
    int16 start_speech;
    float32 vad_threshold;
    vad_data_t *vad_data;

    /* Temporary buffers for processing. */
    /* FIXME: too many of these. */
    int16 *spch;
    frame_t *frame;
    powspec_t *spec, *mfspec;
    int16 *overflow_samps;
};

void fe_init_dither(int32 seed);

/* Apply 1/2 bit noise to a buffer of audio. */
int32 fe_dither(int16 *buffer, int32 nsamps);

/* Load a frame of data into the fe. */
int fe_read_frame(fe_t *fe, int16 const *in, int32 len);

/* Shift the input buffer back and read more data. */
int fe_shift_frame(fe_t *fe, int16 const *in, int32 len);

/* Process a frame of data into features. */
void fe_write_frame(fe_t *fe, mfcc_t *feat, int32 store_pcm);

/* Initialization functions. */
int32 fe_build_melfilters(melfb_t *MEL_FB);
int32 fe_compute_melcosine(melfb_t *MEL_FB);
void fe_create_hamming(window_t *in, int32 in_len);
void fe_create_twiddle(fe_t *fe);

fixed32 fe_log_add(fixed32 x, fixed32 y);
fixed32 fe_log_sub(fixed32 x, fixed32 y);

/* Miscellaneous processing functions. */
void fe_spec2cep(fe_t * fe, const powspec_t * mflogspec, mfcc_t * mfcep);
void fe_dct2(fe_t *fe, const powspec_t *mflogspec, mfcc_t *mfcep, int htk);
void fe_dct3(fe_t *fe, const mfcc_t *mfcep, powspec_t *mflogspec);



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
