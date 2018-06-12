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


---
## [FUNCTION LIST](#CMUSphinx)<a name="CMUSphinx_list"></a>
* sphinxbase/include/sphinxbase
	+ [cmn.h](#cmn)
	+ [fe.h](#fe)
* sphinxbase/src/
	+ [fe_internal.h](#fe_internal)


+ sphinxbase/include/sphinxbase/clapack_lite.h
  sgemm_,sgemv_,ssymm_... parameter 도 gemm parm
  
+ sphinxbase/include/sphinxbase/cmn.h
//brief Apply Cepstral Mean Normaliztion (CMN) to the set of input mfc frames...

### sphinxbase/include/[fe.h](#CMUSphinx_list)<a name="fe"></a>
typedef float32/fixed32 mfcc_t
fe_init_auto
fe_get_args
fe_init_auto_r  
fe_get_config  
fe_start_stream  
fe_start_utt  
fe_get_output_size  
fe_get_input_size  
fe_get_vad_state  
fe_end_utt  
fe_retain  
fe_free  
fe_process_frames_ext  
fe_process_frames  
fe_process_utt  
fe_free_2d  
fe_mfcc_to_float  
fe_float_to_mfcc  
fe_logspec_to_mfcc  
fe_logspec_dct2  
fe_mfcc_dct3  

### sphinxbase/~/[cmn.h](#CMUSphinx)<a name="cmn"></a>
<pre>
cmn_init  
cmn   
cmn_live  
cmn_live_update  
cmn_live_set  
cmn_live_get  
cmn_free  
</pre>
 
### sphinxbse/src/libsphinxbase/fe/[fe_internal.h](#CMUSphinx)<a name="fe_internal"></a>  
  
fe_init_dither(  
fe_dither  
fe_read_frame  
fe_shift_frame  
 fe_write_frame  
**Initialization functions**  
fe_build_melfilters  
fe_compute_melcosine  
fe_create_hamming  
fe_create_twiddle  
    
fe_log_add  
fe_log_sub  
  
**Miscellaneous processing functions**  
fe_spec2cep   
fe_dct2   
fe_dct3   

---

## [FUNCTION PROTOTYPE](#CMUSphinx)<a name = "CMUSphinx_proto"></a>
* sphinxbase/include/sphinxbase
	+ [cmn.h](#cmn.h)
	+ [fe.h](#fe.h)
* sphinxbase/src
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
### sphinxbase/~/[fe.h](#CMUSphinx_proto)<a name="fe.h"></a>
```C++
/**
 * Structure for the front-end computation.
 */
typedef struct fe_s fe_t;

/**
 * Error codes returned by stuff.
 */
enum fe_error_e {
	FE_SUCCESS = 0,
	FE_OUTPUT_FILE_SUCCESS  = 0,
	FE_CONTROL_FILE_ERROR = -1,
	FE_START_ERROR = -2,
	FE_UNKNOWN_SINGLE_OR_BATCH = -3,
	FE_INPUT_FILE_OPEN_ERROR = -4,
	FE_INPUT_FILE_READ_ERROR = -5,
	FE_MEM_ALLOC_ERROR = -6,
	FE_OUTPUT_FILE_WRITE_ERROR = -7,
	FE_OUTPUT_FILE_OPEN_ERROR = -8,
	FE_ZERO_ENERGY_ERROR = -9,
	FE_INVALID_PARAM_ERROR =  -10
};

/**
 * Initialize a front-end object from global command-line.
 *
 * This is equivalent to calling fe_init_auto_r(cmd_ln_get()).
 *
 * @return Newly created front-end object.
 */
SPHINXBASE_EXPORT
fe_t* fe_init_auto(void);

/**
 * Get the default set of arguments for fe_init_auto_r().
 *
 * @return Pointer to an argument structure which can be passed to
 * cmd_ln_init() in friends to create argument structures for
 * fe_init_auto_r().
 */
SPHINXBASE_EXPORT
arg_t const *fe_get_args(void);

/**
 * Initialize a front-end object from a command-line parse.
 *
 * @param config Command-line object, as returned by cmd_ln_parse_r()
 *               or cmd_ln_parse_file().  Ownership of this object is
 *               claimed by the fe_t, so you must not attempt to free
 *               it manually.  Use cmd_ln_retain() if you wish to
 *               reuse it.
 * @return Newly created front-end object.
 */
SPHINXBASE_EXPORT
fe_t *fe_init_auto_r(cmd_ln_t *config);

/**
 * Retrieve the command-line object used to initialize this front-end.
 *
 * @return command-line object for this front-end.  This pointer is
 *         retained by the fe_t, so you should not attempt to free it
 *         manually.
 */
SPHINXBASE_EXPORT
const cmd_ln_t *fe_get_config(fe_t *fe);

/**
 * Start processing of the stream, resets processed frame counter
 */
SPHINXBASE_EXPORT
void fe_start_stream(fe_t *fe);

/**
 * Start processing an utterance.
 * @return 0 for success, <0 for error (see enum fe_error_e)
 */
SPHINXBASE_EXPORT
int fe_start_utt(fe_t *fe);

/**
 * Get the dimensionality of the output of this front-end object.
 *
 * This is guaranteed to be the number of values in one frame of
 * output from fe_end_utt() and fe_process_frames().  
 * It is usually the number of MFCC
 * coefficients, but it might be the number of log-spectrum bins, if
 * the <tt>-logspec</tt> or <tt>-smoothspec</tt> options to
 * fe_init_auto() were true.
 *
 * @param fe Front-end object
 * @return Dimensionality of front-end output.
 */
SPHINXBASE_EXPORT
int fe_get_output_size(fe_t *fe);

/**
 * Get the dimensionality of the input to this front-end object.
 *
 * This function retrieves the number of input samples consumed by one
 * frame of processing.  To obtain one frame of output, you must have
 * at least <code>*out_frame_size</code> samples.  To obtain <i>N</i>
 * frames of output, you must have at least <code>(N-1) *
 * *out_frame_shift + *out_frame_size</code> input samples.
 *
 * @param fe Front-end object
 * @param out_frame_shift Output: Number of samples between each frame start.
 * @param out_frame_size Output: Number of samples in each frame.
 */
SPHINXBASE_EXPORT
void fe_get_input_size(fe_t *fe, int *out_frame_shift,
                       int *out_frame_size);

/**
 * Get vad state for the last processed frame
 *
 * @return 1 if speech, 0 if silence
 */
SPHINXBASE_EXPORT
uint8 fe_get_vad_state(fe_t *fe);

/**
 * Finish processing an utterance.
 *
 * This function also collects any remaining samples and calculates a
 * final cepstral vector.  If there are overflow samples remaining, it
 * will pad with zeros to make a complete frame.
 *
 * @param fe Front-end object.
 * @param out_cepvector Buffer to hold a residual cepstral vector, or NULL
 *                      if you wish to ignore it.  Must be large enough
 * @param out_nframes Number of frames of residual cepstra created
 *                    (either 0 or 1).
 * @return 0 for success, <0 for error (see enum fe_error_e)
 */
SPHINXBASE_EXPORT
int fe_end_utt(fe_t *fe, mfcc_t *out_cepvector, int32 *out_nframes);

/**
 * Retain ownership of a front end object.
 *
 * @return pointer to the retained front end.
 */
SPHINXBASE_EXPORT
fe_t *fe_retain(fe_t *fe);

/**
 * Free the front end. 
 *
 * Releases resources associated with the front-end object.
 *
 * @return new reference count (0 if freed completely)
 */
SPHINXBASE_EXPORT
int fe_free(fe_t *fe);

/*
 * Do same as fe_process_frames, but also returns
 * voiced audio. Output audio is valid till next
 * fe_process_frames call.
 *
 * DO NOT MIX fe_process_frames calls
 *
 * @param voiced_spch Output: obtain voiced audio samples here
 *
 * @param voiced_spch_nsamps Output: shows voiced_spch length
 *
 * @param out_frameidx Output: index of the utterance start
 */
SPHINXBASE_EXPORT
int fe_process_frames_ext(fe_t *fe,
                      int16 const **inout_spch,
                      size_t *inout_nsamps,
                      mfcc_t **buf_cep,
                      int32 *inout_nframes,
                      int16 *voiced_spch,
                      int32 *voiced_spch_nsamps,
                      int32 *out_frameidx);

/** 
 * Process a block of samples.
 *
 * This function generates up to <code>*inout_nframes</code> of
 * features, or as many as can be generated from
 * <code>*inout_nsamps</code> samples.
 *
 * On exit, the <code>inout_spch</code>, <code>inout_nsamps</code>,
 * and <code>inout_nframes</code> parameters are updated to point to
 * the remaining sample data, the number of remaining samples, and the
 * number of frames processed, respectively.  This allows you to call
 * this repeatedly to process a large block of audio in small (say,
 * 5-frame) chunks:
 *
 *  int16 *bigbuf, *p;
 *  mfcc_t **cepstra;
 *  int32 nsamps;
 *  int32 nframes = 5;
 *
 *  cepstra = (mfcc_t **)
 *      ckd_calloc_2d(nframes, fe_get_output_size(fe), sizeof(**cepstra));
 *  p = bigbuf;
 *  while (nsamps) {
 *      nframes = 5;
 *      fe_process_frames(fe, &p, &nsamps, cepstra, &nframes);
 *      // Now do something with these frames...
 *      if (nframes)
 *          do_some_stuff(cepstra, nframes);
 *  }
 *
 * @param inout_spch Input: Pointer to pointer to speech samples
 *                   (signed 16-bit linear PCM).
 *                   Output: Pointer to remaining samples.
 * @param inout_nsamps Input: Pointer to maximum number of samples to
 *                     process.
 *                     Output: Number of samples remaining in input buffer.
 * @param buf_cep Two-dimensional buffer (allocated with
 *                ckd_calloc_2d()) which will receive frames of output
 *                data.  If NULL, no actual processing will be done,
 *                and the maximum number of output frames which would
 *                be generated is returned in
 *                <code>*inout_nframes</code>.
 * @param inout_nframes Input: Pointer to maximum number of frames to
 *                      generate.
 *                      Output: Number of frames actually generated.
 * @param out_frameidx Index of the first frame returned in a stream
 *
 * @return 0 for success, <0 for failure (see enum fe_error_e)
 */
SPHINXBASE_EXPORT
int fe_process_frames(fe_t *fe,
                      int16 const **inout_spch,
                      size_t *inout_nsamps,
                      mfcc_t **buf_cep,
                      int32 *inout_nframes,
                      int32 *out_frameidx);

/** 
 * Process a block of samples, returning as many frames as possible.
 *
 * This function processes all the samples in a block of data and
 * returns a newly allocated block of feature vectors.  This block
 * needs to be freed with fe_free_2d() after use.
 *
 * It is possible for there to be some left-over data which could not
 * fit in a complete frame.  This data can be processed with
 * fe_end_utt().
 *
 * This function is deprecated in favor of fe_process_frames().
 *
 * @return 0 for success, <0 for failure (see enum fe_error_e)
 */
SPHINXBASE_EXPORT
int fe_process_utt(fe_t *fe,  /**< A front end object */
                   int16 const *spch, /**< The speech samples */
                   size_t nsamps, /**< number of samples*/
                   mfcc_t ***cep_block, /**< Output pointer to cepstra */
                   int32 *nframes /**< Number of frames processed */
	);

/**
 * Free the output pointer returned by fe_process_utt().
 **/
SPHINXBASE_EXPORT
void fe_free_2d(void *arr);

/**
 * Convert a block of mfcc_t to float32 (can be done in-place)
 **/
SPHINXBASE_EXPORT
int fe_mfcc_to_float(fe_t *fe,
                     mfcc_t **input,
                     float32 **output,
                     int32 nframes);

/**
 * Convert a block of float32 to mfcc_t (can be done in-place)
 **/
SPHINXBASE_EXPORT
int fe_float_to_mfcc(fe_t *fe,
                     float32 **input,
                     mfcc_t **output,
                     int32 nframes);

/**
 * Process one frame of log spectra into MFCC using discrete cosine
 * transform.
 *
 * This uses a variant of the DCT-II where the first frequency bin is
 * scaled by 0.5.  Unless somebody misunderstood the DCT-III equations
 * and thought that's what they were implementing here, this is
 * ostensibly done to account for the symmetry properties of the
 * DCT-II versus the DFT - the first coefficient of the input is
 * assumed to be repeated in the negative frequencies, which is not
 * the case for the DFT.  (This begs the question, why not just use
 * the DCT-I, since it has the appropriate symmetry properties...)
 * Moreover, this is bogus since the mel-frequency bins on which we
 * are doing the DCT don't extend to the edge of the DFT anyway.
 *
 * This also means that the matrix used in computing this DCT can not
 * be made orthogonal, and thus inverting the transform is difficult.
 * Therefore if you want to do cepstral smoothing or have some other
 * reason to invert your MFCCs, use fe_logspec_dct2() and its inverse
 * fe_logspec_dct3() instead.
 *
 * Also, it normalizes by 1/nfilt rather than 2/nfilt, for some reason.
 **/
SPHINXBASE_EXPORT
int fe_logspec_to_mfcc(fe_t *fe,  /**< A fe structure */
                       const mfcc_t *fr_spec, /**< One frame of spectrum */
                       mfcc_t *fr_cep /**< One frame of cepstrum */
        );

/**
 * Convert log spectra to MFCC using DCT-II.
 *
 * This uses the "unitary" form of the DCT-II, i.e. with a scaling
 * factor of sqrt(2/N) and a "beta" factor of sqrt(1/2) applied to the
 * cos(0) basis vector (i.e. the one corresponding to the DC
 * coefficient in the output).
 **/
SPHINXBASE_EXPORT
int fe_logspec_dct2(fe_t *fe,  /**< A fe structure */
                    const mfcc_t *fr_spec, /**< One frame of spectrum */
                    mfcc_t *fr_cep /**< One frame of cepstrum */
        );

/**
 * Convert MFCC to log spectra using DCT-III.
 *
 * This uses the "unitary" form of the DCT-III, i.e. with a scaling
 * factor of sqrt(2/N) and a "beta" factor of sqrt(1/2) applied to the
 * cos(0) basis vector (i.e. the one corresponding to the DC
 * coefficient in the input).
 **/
SPHINXBASE_EXPORT
int fe_mfcc_dct3(fe_t *fe,  /**< A fe structure */
                 const mfcc_t *fr_cep, /**< One frame of cepstrum */
                 mfcc_t *fr_spec /**< One frame of spectrum */
        );


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
