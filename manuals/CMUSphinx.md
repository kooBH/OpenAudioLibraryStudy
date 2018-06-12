### [CMUSphinx](./openAudioLibs.md#TOP)<a name = "CMUSphinx"></a>
+ [funtion list](#CMUSphinx_list)
+ [function prototype](#CMUSphinx_proto)
+ [license](#CMUSphinx_license)
---

#### [SphinxBase Documnet]( https://cmusphinx.github.io/doc/sphinxbase/)  
#### [PoketSphinx Document]( https://cmusphinx.github.io/doc/pocketsphinx/index.html)

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

Header|Discription|&nbsp;
---|---|---
ad.h|brief generic live audio interface for recording and playback|O
agc.h|  Various forms of automatic gain control (AGC)|
bio.h| Sphinx-3 binary file I/O functions|
bitarr.h | brief An implementation bit array - memory efficient storage for digit int and float data|
bitvec.h | brief An implementation of bit vectors|
byteorder.h | Byte swapping ordering macros|
case.h |  Upper/lower case conversion routines|
ckd_alloc.h | Memory allocation package|
clapack_lite.h | (GUESS)minimum lapack function|O
cmd_ln.h| Command line argument parsing|
cmn.h| Various forms of cepstral mean normalization|O
err.h|brief Implementation of logging routines|
f2c.h| Standard Fortran to C header file|
feat.h|  Cepstral features computation|O
fe.h| (GUESS) about float-end|O
filename.h | File and path name operations|
fixpoint.h| Fixed-point arithmetic macros|
fsg_model.h| Word-level finite state graph|
genrand.h| a portable random generator|
glist.h| Module for maintaining a generic, linear linked-list structure|
hash_table.h| Hash table module|
heap.h| Generic heap structure for inserting in any and popping in sorted order|
jsgf.h| JSGF grammar compiler - Java Speech Grammar Format|
listelem_alloc.h  | brief Fast memory allocator for uniformly sized objects|
logmath.h| brief Fast integer logarithmic addition operations|
matrix.h|Matrix and linear algebra functions - 그냥 연산함 for for for 그리고 NxN만 됨|
mmio.h| brief Memory-mapped I/O wrappers for files|
ngram_model.h| brief N-Gram language models|
pio.h| Packaged I/O routines|
prim_type.h| Primitive types; more machine-independent|
priority_queue.h| Priority queue|
profile.h| For timing and event counting|
sbthread.h| brief Simple portable thread functions|
sphinxbase_export.h| (GUESS)About Shared Object|
strfuncs.h| brief Miscellaneous useful string functions|
yin.h| brief Implementation of pitch estimation|

+ ### sphinxbase/src/libsphinxbase/fe
HEADER|Discription|&nbsp;
---|---|---
fe_internal.h ||O
fe_noise.h | noise removal algorithm,noise removal algorithm,noise removal algorithm,noise removal algorithm|O
fe_prespch_buf.h |Buffer that maintains both features and raw audio for the VAD implementation|
fe_type.h | |
fe_warp_affine.h | Warp the frequency axis according to an affine function, i.e. : w' = a * w + b|O
fe_warp.h | Allows a caller to choose a warping function|O
fe_warp_inverse_linear.h| Warp the frequency axis according to an inverse_linear function, i.e : w' = w / a|O
fe_warp_piecewise_linear.h |  [1]|O

[1]
Warp the frequency axis according to an piecewise linear function. The function is linear up to a frequency F, where the slope changes so that the Nyquist frequency in the warped axis maps to the Nyquist frequency in the unwarped   
w' = a * w 
w < F  
w' = a' * w + b  
W > F  w'(0) = 0  
w'(F) = F  
w'(Nyq) = Nyq


+ ### sphinxbase/src/libsphinxbase/lm  
'lm' for Langauge Model  

HEADER|Discription|&nbsp;
---|---|---
jsgf_internal.h||
jsgf_parse.h||
jsgf_sacnner.h||
lm_trie.h||
lm_trie_quant.h||
ngram_model_internal.h||
ngram_model_trie.h||
ngram_raw.h||


---
## [FUNCTION LIST](#CMUSphinx)<a name="CMUSphinx_list"></a>
* ### sphinxbase/include/sphinxbase/
	+ [cmn.h](#cmn)
	+ [fe.h](#fe)
	+ [feat.h](#feat)
	+ [clapack_lite.h](#clapack_lite)
	+ [ad.h](#ad)
	
* ### sphinxbase/src/libsphinxbase/fe/
	+ [fe_internal.h](#fe_internal)
	+ [fe_noise.h](#fe_noise)
	+ [fe_warp_affine.h](#fe_warp_affine) 
	+ [fe_warp.h](#fe_warp)
	+ [fe_warp_inverse_linear.h](#fe_warp_inverse_linear)
	+ [fe_warp_piecewise_linear.h](#fe_warp_piecewise_linear)


### sphinxbase/include/[fe.h](#CMUSphinx_list)<a name="fe"></a>  
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
 
### sphinxbase/include/[feat.h](#CMUSphinx)<a name="feat"></a>
parse_subvecs  
subvecs_free  
feat_array_alloc  
feat_array_realloc  
feat_array_free  
feat_init  
int32 feat_read_lda  
feat_lda_transform  
feat_set_subvecs  
feat_print  
feat_s2mfc2feat  
feat_s2mfc2feat_live  
feat_update_stats  
feat_retain  
feat_free  
feat_report
 
### sphinxbase/include/[clapack_lite.h](#CMUSphinx_list)<a name="clapack_lite"></a>    
sgemm_  
sgemv_  
ssymm_  
sposv_   
spotrf_
 
 
### sphinxbase/include/[ad.h](#CMUSphinx_list)<a name="ad"></a>    
ad_open_dev  
ad_open_sps  
ad_open  
ad_start_rec  
ad_stop_rec  
ad_close  
ad_read

 
### sphinxbase/src/libsphinxbase/fe/[fe_internal.h](#CMUSphinx)<a name="fe_internal"></a>   
fe_init_dither    
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

### sphinxbase/src/libsphinxbase/fe/[fe_noise.h](#CMUSphinx)<a name="fe_noise"></a>  
fe_init_noisestats  
fe_reset_noisestats  
fe_free_noisestats  
fe_track_snr  
fe_vad_hangover

### sphinxbase/src/libsphinxbase/fe/[fe_warp_affine.h](#CMUSphinx)<a name="fe_warp_affine"></a>  
fe_warp_affine_doc  
fe_warp_affine_id  
fe_warp_affine_n_param  
fe_warp_affine_set_parameters  
fe_warp_affine_warped_to_unwarped  
fe_warp_affine_unwarped_to_warped  
fe_warp_affine_print  


### sphinxbase/src/libsphinxbase/fe/[fe_warp.h](#CMUSphinx)<a name="fe_warp"></a>  
fe_warp_set  
fe_warp_id  
fe_warp_doc  
fe_warp_set_parameters  
fe_warp_n_param  
fe_warp_warped_to_unwarped  
fe_warp_unwarped_to_warped  
fe_warp_print

### sphinxbase/src/libsphinxbase/fe/[fe_warp_inverse_linear.h](#CMUSphinx)<a name="fe_warp_inverse_linear"></a>  
fe_warp_inverse_linear_doc  
fe_warp_inverse_linear_id  
fe_warp_inverse_linear_n_param  
fe_warp_inverse_linear_set_parameters  
fe_warp_inverse_linear_warped_to_unwarped  
fe_warp_inverse_linear_unwarped_to_warped  
fe_warp_inverse_linear_print


### sphinxbase/src/libsphinxbase/fe/[fe_warp_piecewise_linear.h](#CMUSphinx)<a name="fe_warp_piecewise_linear"></a>  
fe_warp_piecewise_linear_doc  
fe_warp_piecewise_linear_id  
fe_warp_piecewise_linear_n_param  
fe_warp_piecewise_linear_set_parameters  
fe_warp_piecewise_linear_warped_to_unwarped  
fe_warp_piecewise_linear_unwarped_to_warped  
fe_warp_piecewise_linear_print 


---

## [FUNCTION PROTOTYPE](#CMUSphinx)<a name = "CMUSphinx_proto"></a>
* sphinxbase/include/sphinxbase
	+ [cmn.h](#cmn.h)
	+ [fe.h](#fe.h)
	+ [feat.h](#feat.h)
	+ [clapack_lite.h](#clapack_lite.h)
	+ [ad.h](#ad.h)
* sphinxbase/src/libsphinxbase/fe/
	+ [fe_internal.h](#fe_internal.h)
	+ [fe_noise.h](#fe_noise.h)
	+ [fe_warp_affine.h](#fe_warp_affine.h) 
	+ [fe_warp.h](#fe_warp.h)
	+ [fe_warp_inverse_linear.h](#fe_warp_inverse_linear.h)
	+ [fe_warp_piecewise_linear.h](#fe_warp_piecewise_linear.h)

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
+ ### sphinxbase/~/[feat.h](#CMUSphinx_proto)<a name="feat.h"></a>

```C++

/**
 * \struct feat_t
 * \brief Structure for describing a speech feature type
 * Structure for describing a speech feature type (no. of streams and stream widths),
 * as well as the computation for converting the input speech (e.g., Sphinx-II format
 * MFC cepstra) into this type of feature vectors.
 */
typedef struct feat_s {
    int refcount;       /**< Reference count. */
    char *name;		/**< Printable name for this feature type */
    int32 cepsize;	/**< Size of input speech vector (typically, a cepstrum vector) */
    int32 n_stream;	/**< Number of feature streams; e.g., 4 in Sphinx-II */
    uint32 *stream_len;	/**< Vector length of each feature stream */
    int32 window_size;	/**< Number of extra frames around given input frame needed to compute
                           corresponding output feature (so total = window_size*2 + 1) */
    int32 n_sv;         /**< Number of subvectors */
    uint32 *sv_len;      /**< Vector length of each subvector */
    int32 **subvecs;    /**< Subvector specification (or NULL for none) */
    mfcc_t *sv_buf;      /**< Temporary copy buffer for subvector projection */
    int32 sv_dim;       /**< Total dimensionality of subvector (length of sv_buf) */

    cmn_type_t cmn;	/**< Type of CMN to be performed on each utterance */
    int32 varnorm;	/**< Whether variance normalization is to be performed on each utt;
                           Irrelevant if no CMN is performed */
    agc_type_t agc;	/**< Type of AGC to be performed on each utterance */

    /**
     * Feature computation function. 
     * @param fcb the feat_t describing this feature type
     * @param input pointer into the input cepstra
     * @param feat a 2-d array of output features (n_stream x stream_len)
     * @return 0 if successful, -ve otherwise.
     *
     * Function for converting window of input speech vector
     * (input[-window_size..window_size]) to output feature vector
     * (feat[stream][]).  If NULL, no conversion available, the
     * speech input must be feature vector itself.
     **/
    void (*compute_feat)(struct feat_s *fcb, mfcc_t **input, mfcc_t **feat);
    cmn_t *cmn_struct;	/**< Structure that stores the temporary variables for cepstral 
                           means normalization*/
    agc_t *agc_struct;	/**< Structure that stores the temporary variables for acoustic
                           gain control*/

    mfcc_t **cepbuf;    /**< Circular buffer of MFCC frames for live feature computation. */
    mfcc_t **tmpcepbuf; /**< Array of pointers into cepbuf to handle border cases. */
    int32   bufpos;     /**< Write index in cepbuf. */
    int32   curpos;     /**< Read index in cepbuf. */

    mfcc_t ***lda; /**< Array of linear transformations (for LDA, MLLT, or whatever) */
    uint32 n_lda;   /**< Number of linear transformations in lda. */
    uint32 out_dim; /**< Output dimensionality */
} feat_t;

/**
 * Name of feature type.
 */
#define feat_name(f)		((f)->name)
/**
 * Input dimensionality of feature.
 */
#define feat_cepsize(f)		((f)->cepsize)
/**
 * Size of dynamic feature window.
 */
#define feat_window_size(f)	((f)->window_size)
/**
 * Number of feature streams.
 *
 * @deprecated Do not use this, use feat_dimension1() instead.
 */
#define feat_n_stream(f)	((f)->n_stream)
/**
 * Length of feature stream i.
 *
 * @deprecated Do not use this, use feat_dimension2() instead.
 */
#define feat_stream_len(f,i)	((f)->stream_len[i])
/**
 * Number of streams or subvectors in feature output.
 */
#define feat_dimension1(f)	((f)->n_sv ? (f)->n_sv : f->n_stream)
/**
 * Dimensionality of stream/subvector i in feature output.
 */
#define feat_dimension2(f,i)	((f)->lda ? (f)->out_dim : ((f)->sv_len ? (f)->sv_len[i] : f->stream_len[i]))
/**
 * Total dimensionality of feature output.
 */
#define feat_dimension(f)	((f)->out_dim)
/**
 * Array with stream/subvector lengths
 */
#define feat_stream_lengths(f)  ((f)->lda ? (&(f)->out_dim) : (f)->sv_len ? (f)->sv_len : f->stream_len)

/**
 * Parse subvector specification string.
 *
 * Format of specification:
 *   \li '/' separated list of subvectors
 *   \li each subvector is a ',' separated list of subranges
 *   \li each subrange is a single \verbatim <number> \endverbatim or
 *       \verbatim <number>-<number> \endverbatim (inclusive), where
 *       \verbatim <number> \endverbatim is a feature vector dimension
 *       specifier.
 *
 * E.g., "24,0-11/25,12-23/26,27-38" has:
 *   \li 3 subvectors
 *   \li the 1st subvector has feature dims: 24, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, and 11.
 *   \li etc.
 *
 * @param str subvector specification string.
 * @return allocated 2-D array of subvector specs (free with
 * subvecs_free()).  If there are N subvectors specified, subvec[N] =
 * NULL; and each subvec[0]..subvec[N-1] is -1 terminated vector of
 * feature dims.
 */
SPHINXBASE_EXPORT
int32 **parse_subvecs(char const *str);

/**
 * Free array of subvector specs.
 */
SPHINXBASE_EXPORT
void subvecs_free(int32 **subvecs);


/**
 * Allocate an array to hold several frames worth of feature vectors.  The returned value
 * is the mfcc_t ***data array, organized as follows:
 *
 * - data[0][0] = frame 0 stream 0 vector, data[0][1] = frame 0 stream 1 vector, ...
 * - data[1][0] = frame 1 stream 0 vector, data[0][1] = frame 1 stream 1 vector, ...
 * - data[2][0] = frame 2 stream 0 vector, data[0][1] = frame 2 stream 1 vector, ...
 * - ...
 *
 * NOTE: For I/O convenience, the entire data area is allocated as one contiguous block.
 * @return pointer to the allocated space if successful, NULL if any error.
 */
SPHINXBASE_EXPORT
mfcc_t ***feat_array_alloc(feat_t *fcb,	/**< In: Descriptor from feat_init(), used
					     to obtain number of streams and stream sizes */
                           int32 nfr	/**< In: Number of frames for which to allocate */
    );

/**
 * Realloate the array of features. Requires us to know the old size
 */
SPHINXBASE_EXPORT
mfcc_t ***feat_array_realloc(feat_t *fcb, /**< In: Descriptor from feat_init(), used
					      to obtain number of streams and stream sizes */
			     mfcc_t ***old_feat, /**< Feature array. Freed */
                             int32 ofr,	/**< In: Previous number of frames */
                             int32 nfr	/**< In: Number of frames for which to allocate */
    );

/**
 * Free a buffer allocated with feat_array_alloc()
 */
SPHINXBASE_EXPORT
void feat_array_free(mfcc_t ***feat);


/**
 * Initialize feature module to use the selected type of feature stream.  
 * One-time only initialization at the beginning of the program.  Input type 
 * is a string defining the  kind of input->feature conversion desired:
 *
 * - "s2_4x":     s2mfc->Sphinx-II 4-feature stream,
 * - "1s_c_d_dd": s2mfc->Sphinx 3.x single feature stream,
 * - "s3_1x39":   s2mfc->Sphinx 3.0 single feature stream,
 * - "n1,n2,n3,...": Explicit feature vector layout spec. with comma-separated 
 *   feature stream lengths.  In this case, the input data is already in the 
 *   feature format and there is no conversion necessary.
 *
 * @return (feat_t *) descriptor if successful, NULL if error.  Caller 
 * must not directly modify the contents of the returned value.
 */
SPHINXBASE_EXPORT
feat_t *feat_init(char const *type,/**< In: Type of feature stream */
                  cmn_type_t cmn, /**< In: Type of cepstram mean normalization to 
                                     be done before feature computation; can be 
                                     CMN_NONE (for none) */
                  int32 varnorm,  /**< In: (boolean) Whether variance 
                                     normalization done on each utt; only 
                                     applicable if CMN also done */
                  agc_type_t agc, /**< In: Type of automatic gain control to be 
                                     done before feature computation */
                  int32 breport, /**< In: Whether to show a report for feat_t */
                  int32 cepsize  /**< Number of components in the input vector
                                    (or 0 for the default for this feature type,
                                    which is usually 13) */
    );

/**
 * Add an LDA transformation to the feature module from a file.
 * @return 0 for success or -1 if reading the LDA file failed.
 **/
SPHINXBASE_EXPORT
int32 feat_read_lda(feat_t *feat,	 /**< In: Descriptor from feat_init() */
                    const char *ldafile, /**< In: File to read the LDA matrix from. */
                    int32 dim		 /**< In: Dimensionality of LDA output. */
    );

/**
 * Transform a block of features using the feature module's LDA transform.
 **/
SPHINXBASE_EXPORT
void feat_lda_transform(feat_t *fcb,		/**< In: Descriptor from feat_init() */
                        mfcc_t ***inout_feat,	/**< Feature block to transform. */
                        uint32 nfr		/**< In: Number of frames in inout_feat. */
    );

/**
 * Add a subvector specification to the feature module.
 *
 * The subvector splitting will be performed after dynamic feature
 * computation, CMN, AGC, and any LDA transformation.  The number of
 * streams in the dynamic feature type must be one, as with LDA.
 *
 * After adding a subvector specification, the output of feature
 * computation will be split into multiple subvectors, and
 * feat_array_alloc() will allocate pointers accordingly.  The number
 * of <em>streams</em> will remain the 
 *
 * @param fcb the feature descriptor.
 * @param subvecs subvector specification.  This pointer is retained
 * by the feat_t and should not be freed manually.
 * @return 0 for success or -1 if the subvector specification was
 * invalid.
 */
SPHINXBASE_EXPORT
int feat_set_subvecs(feat_t *fcb, int32 **subvecs);

/**
 * Print the given block of feature vectors to the given FILE.
 */
SPHINXBASE_EXPORT
void feat_print(feat_t *fcb,		/**< In: Descriptor from feat_init() */
		mfcc_t ***feat,		/**< In: Feature data to be printed */
		int32 nfr,		/**< In: Number of frames of feature data above */
		FILE *fp		/**< In: Output file pointer */
    );

  
/**
 * Read a specified MFC file (or given segment within it), perform
 * CMN/AGC as indicated by <code>fcb</code>, and compute feature
 * vectors.  Feature vectors are computed for the entire segment
 * specified, by including additional surrounding or padding frames to
 * accommodate the feature windows.
 *
 * @return Number of frames of feature vectors computed if successful;
 * -1 if any error.  <code>If</code> feat is NULL, then no actual
 * computation will be done, and the number of frames which must be
 * allocated will be returned.
 * 
 * A note on how the file path is constructed: If the control file
 * already specifies extension or absolute path, then these are not
 * applied. The default extension is defined by the application.
 */
SPHINXBASE_EXPORT
int32 feat_s2mfc2feat(feat_t *fcb,	/**< In: Descriptor from feat_init() */
		      const char *file,	/**< In: File to be read */
		      const char *dir,	/**< In: Directory prefix for file, 
					   if needed; can be NULL */
		      const char *cepext,/**< In: Extension of the
					   cepstrum file.It cannot be
					   NULL */
		      int32 sf, int32 ef,   /* Start/End frames
                                               within file to be read. Use
                                               0,-1 to process entire
                                               file */
		      mfcc_t ***feat,	/**< Out: Computed feature vectors; 
					   caller must allocate this space */
		      int32 maxfr	/**< In: Available space (number of frames) in 
					   above feat array; it must be 
					   sufficient to hold the result.
                                           Pass -1 for no limit. */
    );


/**
 * Feature computation routine for live mode decoder.
 *
 * This function computes features for blocks of incoming data. It
 * retains an internal buffer for computing deltas, which means that
 * the number of output frames will not necessarily equal the number
 * of input frames.
 *
 * <strong>It is very important</strong> to realize that the number of
 * output frames can be <strong>greater than</strong> the number of
 * input frames, specifically when <code>endutt</code> is true.  It is
 * guaranteed to never exceed <code>*inout_ncep +
 * feat_window_size(fcb)</code>.  You <strong>MUST</strong> have
 * allocated at least that many frames in <code>ofeat</code>, or you
 * will experience a buffer overflow.
 *
 * If beginutt and endutt are both true, CMN_CURRENT and AGC_MAX will
 * be done.  Otherwise only CMN_PRIOR and AGC_EMAX will be done.
 *
 * If beginutt is false, endutt is true, and the number of input
 * frames exceeds the input size, then end-of-utterance processing
 * won't actually be done.  This condition can easily be checked,
 * because <code>*inout_ncep</code> will equal the return value on
 * exit, and will also be smaller than the value of
 * <code>*inout_ncep</code> on entry.
 *
 * @return The number of output frames actually computed.
 **/
SPHINXBASE_EXPORT
int32 feat_s2mfc2feat_live(feat_t  *fcb,     /**< In: Descriptor from feat_init() */
                           mfcc_t **uttcep,  /**< In: Incoming cepstral buffer */
                           int32 *inout_ncep,/**< In: Size of incoming buffer.
                                                Out: Number of incoming frames consumed. */
                           int32 beginutt,   /**< In: Begining of utterance flag */
                           int32 endutt,     /**< In: End of utterance flag */
                           mfcc_t ***ofeat   /**< In: Output feature buffer.  See
                                                <strong>VERY IMPORTANT</strong> note
                                                about the size of this buffer above. */
    );


/**
 * Update the normalization stats, possibly in the end of utterance
 *
 */
SPHINXBASE_EXPORT
void feat_update_stats(feat_t *fcb);


/**
 * Retain ownership of feat_t.
 *
 * @return pointer to retained feat_t.
 */
SPHINXBASE_EXPORT
feat_t *feat_retain(feat_t *f);

/**
 * Release resource associated with feat_t
 *
 * @return new reference count (0 if freed)
 */
SPHINXBASE_EXPORT
int feat_free(feat_t *f /**< In: feat_t */
    );

/**
 * Report the feat_t data structure 
 */
SPHINXBASE_EXPORT
void feat_report(feat_t *f /**< In: feat_t */
    );


```


### sphinxbase/~/[clapack_lite.h](#CMUSphinx_proto)<a name="clapack_lite.h"></a>
```C++
/* Subroutine */ int sgemm_(char *transa, char *transb, integer *m, integer *
                            n, integer *k, real *alpha, real *a, integer *lda, real *b, integer *
                            ldb, real *beta, real *c__, integer *ldc);
/* Subroutine */ int sgemv_(char *trans, integer *m, integer *n, real *alpha,
                            real *a, integer *lda, real *x, integer *incx, real *beta, real *y,
                            integer *incy);
/* Subroutine */ int ssymm_(char *side, char *uplo, integer *m, integer *n,
                            real *alpha, real *a, integer *lda, real *b, integer *ldb, real *beta,
                            real *c__, integer *ldc);

/* Subroutine */ int sposv_(char *uplo, integer *n, integer *nrhs, real *a,
                            integer *lda, real *b, integer *ldb, integer *info);
/* Subroutine */ int spotrf_(char *uplo, integer *n, real *a, integer *lda,
                             integer *info);

#ifdef __cplusplus


```


### sphinxbase/~/[ad.h](#CMUSphinx_proto)<a name="ad.h"></a>
```C++  
/**
 * Open a specific audio device for recording.
 *
 * The device is opened in non-blocking mode and placed in idle state.
 *
 * @return pointer to read-only ad_rec_t structure if successful, NULL
 * otherwise.  The return value to be used as the first argument to
 * other recording functions.
 */
SPHINXBASE_EXPORT
ad_rec_t *ad_open_dev (
	const char *dev, /**< Device name (platform-specific) */
	int32 samples_per_sec /**< Samples per second */
	);

/**
 * Open the default audio device with a given sampling rate.
 */
SPHINXBASE_EXPORT
ad_rec_t *ad_open_sps (
		       int32 samples_per_sec /**< Samples per second */
		       );


/**
 * Open the default audio device.
 */
SPHINXBASE_EXPORT
ad_rec_t *ad_open ( void );


/* Start audio recording.  Return value: 0 if successful, <0 otherwise */
SPHINXBASE_EXPORT
int32 ad_start_rec (ad_rec_t *);


/* Stop audio recording.  Return value: 0 if successful, <0 otherwise */
SPHINXBASE_EXPORT
int32 ad_stop_rec (ad_rec_t *);


/* Close the recording device.  Return value: 0 if successful, <0 otherwise */
SPHINXBASE_EXPORT
int32 ad_close (ad_rec_t *);

/*
 * Read next block of audio samples while recording; read upto max samples into buf.
 * Return value: # samples actually read (could be 0 since non-blocking); -1 if not
 * recording and no more samples remaining to be read from most recent recording.
 */
SPHINXBASE_EXPORT
int32 ad_read (ad_rec_t *, int16 *buf, int32 max);

```


### sphinxbase/src/libsphinxbase/fe/[fe_inernal.h](#CMUSphinx_proto)<a name="fe_internal.h"></a>

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

+ ### sphinxbase/src/libsphinxbase/fe/[fe_noise.h](#CMUSphinx_proto)<a name="fe_noise.h"></a>
```C++
/* fe_noise.c
struct noise_stats_s {
    /* Smoothed power */
    powspec_t *power;
    /* Noise estimate */
    powspec_t *noise;
    /* Signal floor estimate */
    powspec_t *floor;
    /* Peak for temporal masking */
    powspec_t *peak;

    /* Initialize it next time */
    uint8 undefined;
    /* Number of items to process */
    uint32 num_filters;

    /* Sum of slow peaks for VAD */
    powspec_t slow_peak_sum;

    /* Precomputed constants */
    powspec_t lambda_power;
    powspec_t comp_lambda_power;
    powspec_t lambda_a;
    powspec_t comp_lambda_a;
    powspec_t lambda_b;
    powspec_t comp_lambda_b;
    powspec_t lambda_t;
    powspec_t mu_t;
    powspec_t max_gain;
    powspec_t inv_max_gain;

    powspec_t smooth_scaling[2 * SMOOTH_WINDOW + 3];
};
*/
typedef struct noise_stats_s noise_stats_t;

/* Creates noisestats object */
noise_stats_t *fe_init_noisestats(int num_filters);

/* Resets collected noise statistics */
void fe_reset_noisestats(noise_stats_t * noise_stats);

/* Frees allocated data */
void fe_free_noisestats(noise_stats_t * noise_stats);

/**
 * Process frame, update noise statistics, remove noise components if needed, 
 * and return local vad decision.
 */
void fe_track_snr(fe_t *fe, int32 *in_speech);

/**
 * Updates global state based on local VAD state smoothing the estimate.
 */
void fe_vad_hangover(fe_t *fe, mfcc_t *feat, int32 is_speech, int32 store_pcm);

```

+ ### sphinxbase/src/libsphinxbase/fe/[fe_warp_affine.h](#CMUSphinx_proto)<a name="fe_warp_affine.h"></a>
```C++
const char *
fe_warp_affine_doc(void);

uint32
fe_warp_affine_id(void);

uint32
fe_warp_affine_n_param(void);

void
fe_warp_affine_set_parameters(char const *param_str, float sampling_rate);

float
fe_warp_affine_warped_to_unwarped(float nonlinear);

float
fe_warp_affine_unwarped_to_warped(float linear);

void
fe_warp_affine_print(const char *label);

```

+ ### sphinxbase/src/libsphinxbase/fe/[fe_warp.h](#CMUSphinx_proto)<a name="fe_warp.h"></a>

```C++
typedef struct {
    void (*set_parameters)(char const *param_str, float sampling_rate);
    const char * (*doc)(void);
    uint32 (*id)(void);
    uint32 (*n_param)(void);
    float (*warped_to_unwarped)(float nonlinear);
    float (*unwarped_to_warped)(float linear);
    void (*print)(const char *label);
} fe_warp_conf_t;

int fe_warp_set(melfb_t *mel, const char *id_name);

uint32 fe_warp_id(melfb_t *mel);

const char * fe_warp_doc(melfb_t *mel);

void fe_warp_set_parameters(melfb_t *mel, char const *param_str, float sampling_rate);

uint32 fe_warp_n_param(melfb_t *mel);

float fe_warp_warped_to_unwarped(melfb_t *mel, float nonlinear);

float fe_warp_unwarped_to_warped(melfb_t *mel, float linear);

void fe_warp_print(melfb_t *mel, const char *label);

```

+ ### sphinxbase/src/libsphinxbase/fe/[fe_warp_inverse_linear.h](#CMUSphinx_proto)<a name="fe_warp_inverse_linear.h"></a>

```C++c
const char *
fe_warp_inverse_linear_doc(void);

uint32
fe_warp_inverse_linear_id(void);

uint32
fe_warp_inverse_linear_n_param(void);

void
fe_warp_inverse_linear_set_parameters(char const *param_str, float sampling_rate);

float
fe_warp_inverse_linear_warped_to_unwarped(float nonlinear);

float
fe_warp_inverse_linear_unwarped_to_warped(float linear);

void
fe_warp_inverse_linear_print(const char *label);

```

+ ### sphinxbase/src/libsphinxbase/fe/[fe_warp_piecewise_linear.h](#CMUSphinx_proto)<a name="fe_warp_piecewise_linear.h"></a>

```C++
const char *
fe_warp_piecewise_linear_doc(void);

uint32
fe_warp_piecewise_linear_id(void);

uint32
fe_warp_piecewise_linear_n_param(void);

void
fe_warp_piecewise_linear_set_parameters(char const *param_str, float sampling_rate);

float
fe_warp_piecewise_linear_warped_to_unwarped(float nonlinear);

float
fe_warp_piecewise_linear_unwarped_to_warped(float linear);

void
fe_warp_piecewise_linear_print(const char *label);


```



---

#### [LICENSE](#CIGLET)<a name = "CMUSphinx_license"></a>  

https://cmusphinx.github.io/wiki/about/  
+ **BSD-like** license which allows commercial distribution  

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
