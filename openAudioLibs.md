
# <a name ="TOP">Open Audio Libs</a>

1. ### [SUMMARY](#summary)  
2. ### [CIGLET](#CIGLET)
3. ### [HTK](#HTK)
---  

## SUMMARY<a name = "summary"></a>  
 
### [Ciglet](https://github.com/Sleepwalking/ciglet)
+ lightweight C library for digital signal processing
+ C-written sIGnal codeLETs
+ Matlab  to C conversion of frequently used ones 

### [Hidden Markov Model Toolkit (HTK)](http://htk.eng.cam.ac.uk/)
+ C source form  
+ a set of library modules and tools  
+ http://htk.eng.cam.ac.uk/prot-docs/htkbook.pdf  


### [CMUSphinx](https://cmusphinx.github.io/)
+ Pocketsphinx — lightweight recognizer library written in C.
+ Sphinxbase — support library required by Pocketsphinx

### [kaldi](https://github.com/kaldi-asr/kaldi)
+ C++
+ http://kaldi-asr.org/doc/
+ compile against the OpenFst toolkit (using it as a library)
+ include a matrix library that wraps standard BLAS and LAPACK routines
+ licensed under Apache 2.0, which is one of the least restrictive licenses available

### [openSMILE](https://audeering.com/technology/opensmile/)
+ c++ API
+ PortAudio 로 녹음,재생

### [BeamformIt](https://github.com/xanguera/BeamformIt)
+ acoustic beamforming tool

---  

### [CIGLET](#TOP)<a name = "CIGLET"></a>
+ [funtion list](#ciglet_list)
+ [function prototype](#ciglet_proto)
+ [license](#ciglet_license)
---
#### [FUNCTION LIST](#CIGLET)<a name="ciglet_list"></a>

+ Scalar operations

    random number generation: randu, randn  
    miscellaneous: max, min, linterp, fastatan2  
    complex arithmetics: c_cplx, c_conj, c_add, c_sub, c_mul, c_div, c_exp, c_abs, c_arg    
  
+ Vector operations and statistics
  
    vectorized arithmetics: sumfp, sumsqrfp, maxfp, minfp  
    descriptive statistics: meanfp, varfp, medianfp, xcorr, corr, cov  
    sorting: selectnth, sort  
    peak picking: find_peak, find_valley, find_maxima, find_minima  

+ Numerical routines

    fzero, polyval, roots  

+ Basic linear algebra

    products: matmul, mvecmul, dot  
    solving a linear system: lu, lusolve  
    pivoting: ppivot, permm, permv   

+ Memory (de)allocation

    enumeration: linspace, iota  
    2d array operations: malloc2d, free2d, copy2d, flatten, reshape, transpose  

+ Audio I/O

    wavread, wavwrite  

+ General DSP routines

    windows: boxcar, hanning, hamming, mltsine, blackman_harris, nuttall98, blackman  
    Fourier transform: fft, ifft, czt, iczt, idft, dct, fftshift    
    phase manipulation: wrap, unwrap, phase_diff  
    complex number conversion: abscplx, argcplx, polar2real, polar2imag, complete_symm, complete_asymm  
    cepstral analysis: rceps, irceps, minphase  
    filtering: fir1, conv, filter, filtfilt, moving_avg, moving_rms, medfilt1, kalmanf1d, kalmans1d  
    linear prediction: levinson, lpc, flpc, lpgain, lpspec, lpresf    
    interpolation: interp1, interp1u, sincinterp1u, interp_in_blank, rresample  
    operations on sinusoids: gensin, gensins, safe_aliased_sinc, safe_aliased_dsinc  
    miscellaneous: fetch_frame, diff, cumsum, flip ,white_noise, itakura_saito  

+ Audio/speech processing routines

    psychoacoustics: mel2freq, freq2mel, freq2bark, bark2freq, eqloud, melspace  
    frequency estimation: ifdetector_estimate, correlogram, invcrgm  
    spectrogram and STFT: stft, istft, qifft, spgm2cegm, cegm2spgm  
    filterbank analysis: filterbank_spgm, filterbank_spec, be2cc, be2ccgm  
    spectral envelope estimation: spec2env  
    glottal model: lfmodel_from_rd, lfmodel_spectrum, lfmodel_period  

+ Plotting utilities (Gnuplot interface, unavailable on Windows)

   plotopen, plot, imagesc, plotclose

---

#### [FUNCTION PROTOTYPE](#CIGLET)<a name = "ciglet_proto"></a>

-DFP_TPYE=float

```c++
void cig_fft(FP_TYPE* xr, FP_TYPE* xi, FP_TYPE* yr, FP_TYPE* yi,
  int n, FP_TYPE* buffer, FP_TYPE mode);

static inline void fft(FP_TYPE* xr, FP_TYPE* xi, FP_TYPE* yr, FP_TYPE* yi,
  int n, FP_TYPE* buffer) {
  cig_fft(xr, xi, yr, yi, n, buffer, -1.0);
}

static inline void ifft(FP_TYPE* xr, FP_TYPE* xi, FP_TYPE* yr, FP_TYPE* yi,
  int n, FP_TYPE* buffer) {
  cig_fft(xr, xi, yr, yi, n, buffer, 1.0);
}

// === Audio/speech processing routines ===

// instantaneous frequency detector
typedef struct {
  FP_TYPE fc;       // central frequency over sampling rate
  int nh;           // length of impulse responses
  FP_TYPE* hr;      // impulse response (real)
  FP_TYPE* hi;      // impulse response (imag)
  FP_TYPE* hdr;     // impulse response derivative (real)
  FP_TYPE* hdi;     // impulse response derivative (imag)
} ifdetector;

// fc: central frequency, fres: frequency resolution
// both are expressed as ratio of sampling rate
ifdetector* cig_create_ifdetector(FP_TYPE fc, FP_TYPE fres);
void cig_delete_ifdetector(ifdetector* dst);

static inline ifdetector* create_ifdetector(FP_TYPE fc, FP_TYPE fres) {
  return cig_create_ifdetector(fc, fres);
  }

static inline void delete_ifdetector(ifdetector* dst) {
  cig_delete_ifdetector(dst);
}

// estimate instantaneous frequency (as a ratio of sampling rate) at the center of x
// returns 0 if x is not long enough
FP_TYPE cig_ifdetector_estimate(ifdetector* ifd, FP_TYPE* x, int nx);

static inline FP_TYPE ifdetector_estimate(ifdetector* ifd, FP_TYPE* x, int nx) {
  return cig_ifdetector_estimate(ifd, x, nx);
}

typedef struct {
  int nchannel;     // number of channels/bands
  int nf;           // size of frequency response
  FP_TYPE fnyq;     // upperbound of frequency response
  FP_TYPE** fresp;  // array of frequency response of each channel

  int* lower_idx;   // index where the freq. resp. in each channel rises from 0
  int* upper_idx;   // index where the freq. resp. in each channel fades to 0
} filterbank;

static inline FP_TYPE mel2freq(FP_TYPE mel) {
  return 700.0 * (exp_2(mel / 1125.0) - 1.0);
}

static inline FP_TYPE freq2mel(FP_TYPE f) {
  return 1125.0 * log_2(1.0 + f / 700.0);
}

static inline FP_TYPE freq2bark(FP_TYPE f) {
  return 6.0 * asinh(f / 600.0);
}

static inline FP_TYPE bark2freq(FP_TYPE z) {
  return 600.0 * sinh(z / 6.0);
}

static inline FP_TYPE eqloud(FP_TYPE f) {
  FP_TYPE f2 = f * f;
  FP_TYPE f4 = f2 * f2;
  return f4 / (f2 + 1.6e5) / (f2 + 1.6e5) * (f2 + 1.44e6) / (f2 + 9.61e6);
}

static inline FP_TYPE* melspace(FP_TYPE fmin, FP_TYPE fmax, int n) {
  FP_TYPE* freq = malloc((n + 1) * sizeof(FP_TYPE));
  FP_TYPE mmin = freq2mel(fmin);
  FP_TYPE mmax = freq2mel(fmax);
  for(int i = 0; i <= n; i ++)
    freq[i] = mel2freq((FP_TYPE)i / n * (mmax - mmin) + mmin);
  return freq;
}

#define CIG_CORR_ACF     0
#define CIG_CORR_AMDF    1
#define CIG_CORR_SQRDIFF 2
#define CIG_CORR_YIN     3

// size of R: nfrm x max_period
void cig_correlogram(FP_TYPE* x, int nx, int* center, int* nwin, int nfrm,
  int max_period, int method, FP_TYPE** R);

// convert y axis of correlogram from period to frequency
FP_TYPE** cig_invcrgm(FP_TYPE** R, int nfrm, int max_period, int fs, FP_TYPE* faxis, int nf);

void cig_stft_forward(FP_TYPE* x, int nx, int* center, int* nwin, int nfrm,
  int nfft, char* window, int subt_mean, int optlv,
  FP_TYPE* norm_factor, FP_TYPE* weight_factor, FP_TYPE** Xmagn, FP_TYPE** Xphse);

FP_TYPE* cig_stft_backward(FP_TYPE** Xmagn, FP_TYPE** Xphse, int nhop, int nfrm,
  int offset, int hop_factor, int zp_factor, int nfade, FP_TYPE norm_factor, int* ny);

#define CIG_DEF_STFT(fname, optlv) \
static inline void fname(FP_TYPE* x, int nx, int nhop, int nfrm, int hopfc, int zpfc, \
  FP_TYPE* normfc, FP_TYPE* weightfc, FP_TYPE** Xmagn, FP_TYPE** Xphse) { \
  int* center = malloc(nfrm * sizeof(int)); \
  int* nwin = malloc(nfrm * sizeof(int)); \
  for(int i = 0; i < nfrm; i ++) { \
    center[i] = nhop * i; \
    nwin[i] = nhop * hopfc; \
  } \
  cig_stft_forward(x, nx, center, nwin, nfrm, nhop * hopfc * zpfc, "blackman", 0, \
     optlv, normfc, weightfc, Xmagn, Xphse); \
  free(center); \
  free(nwin); \
}

CIG_DEF_STFT(stft, 3);
CIG_DEF_STFT(stft_2, 2);

static inline FP_TYPE* istft(FP_TYPE** Xmagn, FP_TYPE** Xphse, int nhop, int nfrm,
  int hopfc, int zpfc, FP_TYPE normfc, int* ny) {
  return cig_stft_backward(Xmagn, Xphse, nhop, nfrm, 0, hopfc, zpfc, 32, normfc, ny);
}

FP_TYPE cig_qifft(FP_TYPE* magn, int k, FP_TYPE* dst_freq);

static inline FP_TYPE qifft(FP_TYPE* magn, int k, FP_TYPE* dst_freq) {
  return cig_qifft(magn, k, dst_freq);
}

static inline FP_TYPE** spgm2cegm(FP_TYPE** S, int nfrm, int nfft, int ncep) {
  FP_TYPE** C = (FP_TYPE**)malloc2d(nfrm, ncep, sizeof(FP_TYPE));
  FP_TYPE* xbuff = calloc(nfft, sizeof(FP_TYPE));
  FP_TYPE* cbuff = calloc(nfft, sizeof(FP_TYPE));
  FP_TYPE* fftbuff = calloc(nfft * 2, sizeof(FP_TYPE));
  
  for(int i = 0; i < nfrm; i ++) {
    for(int j = 0; j < nfft / 2 + 1; j ++)
      xbuff[j] = log_2(S[i][j] + M_EPS);
    complete_symm(xbuff, nfft);
    ifft(xbuff, NULL, cbuff, NULL, nfft, fftbuff);
    for(int j = 0; j < ncep; j ++) {
      C[i][j] = cbuff[j];
    }
  }
  
  free(fftbuff);
  free(cbuff);
  free(xbuff);
  return C;
}

static inline FP_TYPE** cegm2spgm(FP_TYPE** C, int nfrm, int nfft, int ncep) {
  FP_TYPE** S = (FP_TYPE**)malloc2d(nfrm, nfft / 2 + 1, sizeof(FP_TYPE));
  FP_TYPE* xbuff = calloc(nfft, sizeof(FP_TYPE));
  FP_TYPE* cbuff = calloc(nfft, sizeof(FP_TYPE));
  FP_TYPE* fftbuff = calloc(nfft * 2, sizeof(FP_TYPE));
  
  for(int i = 0; i < nfrm; i ++) {
    for(int j = 0; j < ncep; j ++)
      cbuff[j] = C[i][j];
    for(int j = ncep; j < nfft / 2 + 1; j ++)
      cbuff[j] = 0;
    complete_symm(cbuff, nfft);
    fft(cbuff, NULL, xbuff, NULL, nfft, fftbuff);
    for(int j = 0; j < nfft / 2 + 1; j ++)
      S[i][j] = exp_2(xbuff[j]);
  }
  
  free(fftbuff);
  free(cbuff);
  free(xbuff);
  return S;
}

filterbank* cig_create_empty_filterbank(int nf, FP_TYPE fnyq, int nchannel);
filterbank* cig_create_plp_filterbank(int nf, FP_TYPE fnyq, int nchannel);
filterbank* cig_create_melfreq_filterbank(int nf, FP_TYPE fnyq, int nchannel,
  FP_TYPE min_freq, FP_TYPE max_freq, FP_TYPE scale, FP_TYPE min_width);
void cig_delete_filterbank(filterbank* dst);

static inline filterbank* create_filterbank(int nf, FP_TYPE fnyq, int nchannel) {
  return cig_create_empty_filterbank(nf, fnyq, nchannel);
}

static inline filterbank* create_plpfilterbank(int nf, FP_TYPE fnyq, int nchannel) {
  return cig_create_plp_filterbank(nf, fnyq, nchannel);
}

static inline filterbank* create_melfilterbank(int nf, FP_TYPE fnyq, int nchannel,
  FP_TYPE min_freq, FP_TYPE max_freq) {
  return cig_create_melfreq_filterbank(
    nf, fnyq, nchannel, min_freq, max_freq, 1.0, 400.0);
}

static inline void delete_filterbank(filterbank* dst) {
  cig_delete_filterbank(dst);
}

FP_TYPE** cig_filterbank_spectrogram(filterbank* fbank, FP_TYPE** S, int nfrm,
  int nfft, int fs, int crtenergy);

static inline FP_TYPE** filterbank_spgm(filterbank* fbank, FP_TYPE** S, int nfrm,
  int nfft, int fs, int crtenergy) {
  return cig_filterbank_spectrogram(fbank, S, nfrm, nfft, fs, crtenergy);
}

FP_TYPE* cig_filterbank_spectrum(filterbank* fbank, FP_TYPE* S, int nfft, int fs,
  int crtenergy);

static inline FP_TYPE* filterbank_spec(filterbank* fbank, FP_TYPE* S, int nfft,
  int fs, int crtenergy) {
  return cig_filterbank_spectrum(fbank, S, nfft, fs, crtenergy);
}

static inline FP_TYPE* be2cc(FP_TYPE* band_energy, int nbe, int ncc, int with_energy) {
  FP_TYPE* dctcoef = dct(band_energy, nbe);
  FP_TYPE energy = dctcoef[0];
  ncc = min(nbe - 1, ncc);
  for(int i = 0; i < ncc; i ++)
    dctcoef[i] = dctcoef[i + 1];
  if(with_energy)
    dctcoef[ncc] = energy;
  return dctcoef;
}

static inline FP_TYPE** be2ccgm(FP_TYPE** E, int nfrm, int nbe, int ncc, int with_energy) {
  FP_TYPE** C = malloc(nfrm * sizeof(FP_TYPE*));
  for(int i = 0; i < nfrm; i ++)
    C[i] = be2cc(E[i], nbe, ncc, with_energy);
  return C;
}

// estimate speech spectral envelope on a spectrum S of size nfft / 2 + 1
// f0: ratio of fundamental frequency to sampling rate
FP_TYPE* cig_spec2env(FP_TYPE* S, int nfft, FP_TYPE f0, int nhar, FP_TYPE* Cout);

static inline FP_TYPE* spec2env(FP_TYPE* S, int nfft, FP_TYPE f0, FP_TYPE* Cout) {
  return cig_spec2env(S, nfft, f0, floor(nfft / 2 / f0), Cout);
}

typedef struct {
  FP_TYPE T0;
  FP_TYPE te;
  FP_TYPE tp;
  FP_TYPE ta;
  FP_TYPE Ee;
} lfmodel;

lfmodel cig_lfmodel_from_rd(FP_TYPE rd, FP_TYPE T0, FP_TYPE Ee);

static inline lfmodel lfmodel_from_rd(FP_TYPE rd, FP_TYPE T0, FP_TYPE Ee) {
  return cig_lfmodel_from_rd(rd, T0, Ee);
};

FP_TYPE* cig_lfmodel_spectrum(lfmodel model, FP_TYPE* freq, int nf, FP_TYPE* dst_phase);

static inline FP_TYPE* lfmodel_spectrum(lfmodel model, FP_TYPE* freq, int nf, FP_TYPE* dst_phase) {
  return cig_lfmodel_spectrum(model, freq, nf, dst_phase);
}

FP_TYPE* cig_lfmodel_period(lfmodel model, int fs, int n);

static inline FP_TYPE* lfmodel_period(lfmodel model, int fs, int n) {
  return cig_lfmodel_period(model, fs, n);
}

```


---

#### [LICENSE](#CIGLET)<a name = "ciglet_license"></a>

저작권 명시만 하면 무관  
<pre>
Copyright (c) 2016-2017, Kanru Hua
All rights reserved.
Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation and/or
other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software without
specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
</pre>

---

### [Hidden Markov Model Toolkit (HTK)](#TOP)<a name="HTK"></a>
+ [funtion list](#HTK_list)
+ [function prototype](#HTK_proto)
+ [license](#HTK_license)

|Header|Discription|
|---|---|
|HAdapt.h | Adaptation Library module   |
|HArc.h | Routines used in HFwdBkwdLat.c,  An alternative kind of lattice format used there   |
|HAudio.h | Audio Input/Output  |
|HDic.h| Dictionary Storage  |
|HEaxctMPE.h |MPE implementation (exact)  |
|HFB.h | forward Backward Routines module  
|HFBLat.h | Lattice Forward Backward routines  |
|HGraf.h | Minimal Graphics Interface   |
|HLabel.h | Speech Label File Input   |
|HLM.h | language model handling   |
|HMap.c | MAP Model Updates   |
|HMath.h | Math Support  
|HMem.h | Memory Management Module  |
|HModel.h | HMM Model Definition Data Type  |
|HNet.h | Network and Lattice Functions  |
|HParm.h | Speech Parameter Input/Output   |
|HRec.h | Viterbi Recognition Engine Library   |
|HShell.h | Interface to the Shell  |
|HSigP.h | Signal Processing Routines  |
|HTrain.h | HMM Training Support Routines  |
|HUtil.h | HMM utility routines  |
|HVQ.h | Vector Quantisation  |
|HWave.h | Speech Waveform File Input|



--- 
#### [FUNCTION LIST](#HTK)<a name="HTK_list"></a>
1. [HAudio](#HTK_list_HAudio)
2. [HMath](#HTK_list_HMath)
3. [HSigP](#HTK_list_HSigP)

---
+ <a name="HTK_list_HAudio">[HAudio](#HTK_list)</a> | [Proto](#HAudio)  
	OpenAudioInput    
	AttachReplayBuf  
	StartAudioInput  
	StopAudioInput  
	CloseAudioInput  
	FramesInAudio  
	SamplesInAudio  
	GetAIStatus  
	GetCurrentVol  
	SampsInAudioFrame  
	GetAudio  
	GetRawAudio  
	GetReplayBuf  
	OpenAudioOutput  
	StartAudioOutput  
	PlayReplayBuffer  
	CloseAudioOutput  
	etVolume  
	GetCurrentVol  
	AudioDevInput  
	AudioDevOutput  
	SamplesToPlay  
 
+ <a name= "HTK_list_HMath">[HMath](#HTK_list)</a> | [Proto](#HMath)
	1. **Vector Oriented Routines**  
	ZeroShortVec	
	ZeroIntVec	
	ZeroVector  
	ZeroDVector  
	CopyShortVec  
	CopyIntVec  
	CopyVector  
	CopyDVector  
	ReadShortVec  
	ReadIntVec  
	ReadVector  
	WriteShortVec 
	WriteIntVec  
	WriteVector  
	ShowShortVec  
	ShowIntVec  
	ShowVector  
	ShowDVector  
	2. **Matrix Oriented Routines**   
	ZeroMatrix  
	ZeroDMatrix 
	ZeroTriMat  
	CopyMatrix  
	CopyDMatrix  
	CopyTriMat    
	Mat2DMat  
	DMat2Mat  
	Mat2Tri     
	Tri2Mat 
	ReadTriMat      
	WriteMatrix  
	WriteTriMat  
	ShowMatrix  
	ShowDMatrix  
	ShowTriMat  
	3. **Linear Algebra Routines**   
	CovInvert  
	CovDet  
	MatDet  
	DMatDet  
	MatInvert  
	DMatInvert  
	DMatCofact  
	MatCofact  
	4. **Singular Value Decomposition Routines**   
	SVD  
	InvSVD  
	5. **Log Arithmetic Routines**   
	LAdd  
	LSub  
	L2F  
	6. **Random Number Routines**   
	RandInit  
	RandomValue  
	GaussDeviate  


+ <a name="HTK_list_HSigP">[HSigP](#HTK_list)</a> | [Proto](#HSigP)
  1. S**peech Signal Processing Operations**  
     ZeroMean  
     Ham  
     PreEmphasise  
  2. **Linear Prediction Coding Operations**  
     Wave2LPC  
     LPC2RefC  
     RefC2LPC     
     LPC2Cepstrum  
     Cepstrum2LPC  
  3. **FFT Based Operations**  
     FVec2Spectrum  
     FFT  
     Realft  
     SpecModulus  
     SpecLogModulus  
     SpecPhase  
  4. **FCC Related Operations**  
     Mel  
     InitFBank  
     Wave2FBank  
     FBank2MFCC  
     FBank2MelSpec  
     MelSpec2FBank  
     FBank2C0  
  5.  **PLP Related Operations**  
     InitPLP  
     FBank2ASpec  
     ASpec2LPCep  
  6.  **Feature Level Operations**   
     WeightCepstrum  
     UnWeightCepstrum  
     FZeroMean  
     AddRegression  
     AddHeadRegress  
     AddTailRegress  
     NormaliseLogEnergy  

+ <a name = "HTK_list_HWave">[HWave](#HTK_list)</a> | [Proto](#HWave)  
		InitWave  
		OpenWaveInput  
		CloseWaveInput  
		ZeroMeanWave  
		FramesInWave  
		SampsInWaveFrame  
		GetWave  
		GetWaveDirect  
		OpenWaveOutput  
		PutWaveSample  
		CloseWaveOutput  
		WaveFormat  
		Format2Str  
		FileFormat Str2Format  
**HTK Header Routines**   
		Boolean ReadHTKHeader  
		WriteHTKHeader  
		StoreESIGFieldList  
		RetrieveESIGFieldList  
		ReadEsignalHeader  



---
#### [FUNCTION PROTOTYPE](#HTK)<a name="HTK_proto"></a>
1. [HAudio.h](#HAudio)  
2. [HMath.h](#HMath)  
3. [HSigP.h](#HSigP)  
4. [HWave.h](#HWave)

---

+ ### [HAudio.h](#HTK_proto)<a name="HAudio"></a>

``` C++

typedef enum {
   AI_CLEARED,    /* Not sampling and buffer empty */
   AI_WAITSIG,    /* Wait for start signal */
   AI_SAMPLING,   /* Sampling speech and filling buffer */   
   AI_STOPPED,    /* Stopped but waiting for buffer to be emptied */
   AI_ERROR       /* Error state - eg buffer overflow */
}AudioInStatus;

typedef struct _AudioIn  *AudioIn;    /* Abstract audio input stream */
typedef struct _AudioOut *AudioOut;   /* Abstract audio output stream */

void InitAudio(void);
/*
   Initialise audio module
*/


AudioIn OpenAudioInput(MemHeap *x, HTime *sampPeriod, 
                       HTime winDur, HTime frPeriod);
/*
   Initialise and return an audio stream to sample with given period.
   Samples are returned in frames with duration winDur.  Period between
   successive frames is given by frPeriod.  If *sampPeriod is 0.0 then a 
   default period is used (eg it might be set by some type of audio 
   control panel) and this default period is assigned to *sampPeriod.  
   If audio is not supported then NULL is returned.
*/

void AttachReplayBuf(AudioIn a, int bufSize);
/*
   Attach a replay buffer of given size to a
*/

void StartAudioInput(AudioIn a, int sig);
/*
   if sig>NULLSIG then install sig and set AI_WAITSIG
   if sig<NULLSIG then wait for keypress
   Start audio device

   Signal handler: 
      if AI_WAITSIG then 
         Start audio device
      if AI_SAMPLING then
         Stop audio device and set AI_STOPPED
*/

void StopAudioInput(AudioIn a);
/*
   Stop audio device and set AI_STOPPED.
*/

void CloseAudioInput(AudioIn a);
/* 
   Terminate audio input stream and free buffer memory
*/

int FramesInAudio(AudioIn a);
/*
   CheckAudioInput and return number of whole frames which are 
   currently available from the given audio stream
*/

int SamplesInAudio(AudioIn a);
/*
   CheckAudioInput and return number of speech samples which are 
   currently available from the given audio stream
*/

AudioInStatus GetAIStatus(AudioIn a);
/*
   CheckAudioInput and return the current status of the given 
   audio stream
*/
float GetCurrentVol(AudioIn a);

int SampsInAudioFrame(AudioIn a);
/*
   Return number of samples in each frame of the given Audio
*/

void GetAudio(AudioIn a, int nFrames, float *buf);
/* 
   Get nFrames from given audio stream and store sequentially as
   floats in buf.  If a frame overlap has been set then samples will
   be duplicated in buf.  If a 'replay buffer' has been
   specified then samples are saved in this buffer (with wrap around).
   Every call to StartAudio resets this buffer.  If more frames
   are requested than are available, the call blocks.
*/

void GetRawAudio(AudioIn a, int nSamples, short *buf);
/* 
   Get nSamples from given audio stream and store sequentially in buf.
*/

int GetReplayBuf(AudioIn a, int nSamples, short *buf);
/* 
   Get upto nSamples from replay buffer and store sequentially in buf.
   Return num samples actually copied.
*/

AudioOut OpenAudioOutput(MemHeap *x, HTime *sampPeriod);
/*
   Initialise and return an audio stream for output at given sample
   rate.  If *sampPeriod is 0.0 then a default period is used (eg it might
   be set by some type of audio control panel) and this default
   period is assigned to *sampPeriod.  If audio is not supported then 
   NULL is returned.
*/

void StartAudioOutput(AudioOut a, long nSamples, short *buf);
/*
   Output nSamples to audio stream a using data stored in buf.
*/

void PlayReplayBuffer(AudioOut ao, AudioIn ai);
/*
   Output nSamples from ai's replay buffer to ao
*/

void CloseAudioOutput(AudioOut a);
/* 
   Terminate audio stream a
*/

void SetVolume(AudioOut a, int volume);
/*
   Set the volume of the given audio device. Volume range
   is 0 to 100.
*/

float GetCurrentVol(AudioIn a);
/* 
   Obtain current volume of audio input device
*/

int AudioDevInput(int *mask);
/* 
   Query/set audio device input - mic, linein, etc
*/

int AudioDevOutput(int *mask);
/* 
   Query/set audio device output - speaker, headphones, lineout, etc
*/

int SamplesToPlay(AudioOut a);
/*
   Return num samples left to play in output stream a
*/
```


+ ### [HMath.h](#HTK_proto)<a name ="HMath"></a>

```C++


#define PI   3.14159265358979
#define TPI  6.28318530717959     /* PI*2 */
#define LZERO  (-1.0E10)   /* ~log(0) */
#define LSMALL (-0.5E10)   /* log values < LSMALL are set to LZERO */
#define MINEARG (-708.3)   /* lowest exp() arg  = log(MINLARG) */
#define MINLARG 2.45E-308  /* lowest log() arg  = exp(MINEARG) */

/* NOTE: On some machines it may be necessary to reduce the
         values of MINEARG and MINLARG
*/

typedef float  LogFloat;   /* types just to signal log values */
typedef double LogDouble;

typedef enum {  /* Various forms of covariance matrix */
   DIAGC,         /* diagonal covariance */
   INVDIAGC,      /* inverse diagonal covariance */
   FULLC,         /* inverse full rank covariance */
   XFORMC,        /* arbitrary rectangular transform */
   LLTC,          /* L' part of Choleski decomposition */
   NULLC,         /* none - implies Euclidean in distance metrics */
   NUMCKIND       /* DON'T TOUCH -- always leave as final element */
} CovKind;

typedef union {
   SVector var;         /* if DIAGC or INVDIAGC */
   STriMat inv;         /* if FULLC or LLTC */
   SMatrix xform;       /* if XFORMC */
} Covariance;


/* ------------------------------------------------------------------- */

void InitMath(void);
/*
   Initialise the module
*/

/* ------------------ Vector Oriented Routines ----------------------- */

void ZeroShortVec(ShortVec v);
void ZeroIntVec(IntVec v);
void ZeroVector(Vector v);
void ZeroDVector(DVector v);
/*
   Zero the elements of v
*/

void CopyShortVec(ShortVec v1, ShortVec v2);
void CopyIntVec(IntVec v1, IntVec v2);
void CopyVector(Vector v1, Vector v2);
void CopyDVector(DVector v1, DVector v2);
/*
   Copy v1 into v2; sizes must be the same
*/

Boolean ReadShortVec(Source *src, ShortVec v, Boolean binary);
Boolean ReadIntVec(Source *src, IntVec v, Boolean binary);
Boolean ReadVector(Source *src, Vector v, Boolean binary);
/*
   Read vector v from source in ascii or binary
*/

void WriteShortVec(FILE *f, ShortVec v, Boolean binary);
void WriteIntVec(FILE *f, IntVec v, Boolean binary);
void WriteVector(FILE *f, Vector v, Boolean binary);
/*
   Write vector v to stream f in ascii or binary
*/

void ShowShortVec(char * title, ShortVec v,int maxTerms);
void ShowIntVec(char * title, IntVec v,int maxTerms);
void ShowVector(char * title,Vector v,int maxTerms);
void ShowDVector(char * title,DVector v,int maxTerms);
/*
   Print the title followed by upto maxTerms elements of v
*/

/* Quadratic prod of a full square matrix C and an arbitry full matrix transform A */
void LinTranQuaProd(Matrix Prod, Matrix A, Matrix C);

/* ------------------ Matrix Oriented Routines ----------------------- */

void ZeroMatrix(Matrix m);
void ZeroDMatrix(DMatrix m);
void ZeroTriMat(TriMat m);
/*
   Zero the elements of m
*/

void CopyMatrix (Matrix m1,  Matrix m2);
void CopyDMatrix(DMatrix m1, DMatrix m2);
void CopyTriMat (TriMat m1,  TriMat m2);
/*
   Copy matrix m1 to m2 which must have identical dimensions
*/

void Mat2DMat(Matrix m1,  DMatrix m2);
void DMat2Mat(DMatrix m1, Matrix m2);
void Mat2Tri (Matrix m1,  TriMat m2);
void Tri2Mat (TriMat m1,  Matrix m2);
/*
   Convert matrix format from m1 to m2 which must have identical 
   dimensions
*/

Boolean ReadMatrix(Source *src, Matrix m, Boolean binary);
Boolean ReadTriMat(Source *src, TriMat m, Boolean binary);
/*
   Read matrix from source into m using ascii or binary.
   TriMat version expects m to be in upper triangular form
   but converts to lower triangular form internally.
*/
   
void WriteMatrix(FILE *f, Matrix m, Boolean binary);
void WriteTriMat(FILE *f, TriMat m, Boolean binary);
/*
    Write matrix to stream in ascii or binary.  TriMat version 
    writes m in upper triangular form even though it is stored
    in lower triangular form!
*/

void ShowMatrix (char * title,Matrix m, int maxCols,int maxRows);
void ShowDMatrix(char * title,DMatrix m,int maxCols,int maxRows);
void ShowTriMat (char * title,TriMat m, int maxCols,int maxRows);
/*
   Print the title followed by upto maxCols elements of upto
   maxRows rows of m.
*/

/* ------------------- Linear Algebra Routines ----------------------- */

LogFloat CovInvert(TriMat c, Matrix invc);
/*
   Computes inverse of c in invc and returns the log of Det(c),
   c must be positive definite.
*/

LogFloat CovDet(TriMat c);
/*
   Returns log of Det(c), c must be positive definite.
*/

/* EXPORT->MatDet: determinant of a matrix */
float MatDet(Matrix c);

/* EXPORT->DMatDet: determinant of a double matrix */
double DMatDet(DMatrix c);

/* EXPORT-> MatInvert: puts inverse of c in invc, returns Det(c) */
  float MatInvert(Matrix c, Matrix invc);
  double DMatInvert(DMatrix c, DMatrix invc);
 
/* DMatCofact: generates the cofactors of row r of doublematrix c */
double DMatCofact(DMatrix c, int r, DVector cofact);

/* MatCofact: generates the cofactors of row r of doublematrix c */
double MatCofact(Matrix c, int r, Vector cofact);

/* ------------- Singular Value Decomposition Routines --------------- */

void SVD(DMatrix A, DMatrix U,  DMatrix V, DVector d);
/* 
   Singular Value Decomposition (based on MESCHACH)
   A is m x n ,  U is m x n,  W is diag N x 1, V is n x n
*/

void InvSVD(DMatrix A, DMatrix U, DVector W, DMatrix V, DMatrix Result);
/* 
   Inverted Singular Value Decomposition (calls SVD)
   A is m x n ,  U is m x n,  W is diag N x 1, V is n x n, Result is m x n 
*/

/* ------------------- Log Arithmetic Routines ----------------------- */

LogDouble LAdd(LogDouble x, LogDouble y);
/*
   Return x+y where x and y are stored as logs, 
   sum < LSMALL is floored to LZERO 
*/

LogDouble LSub(LogDouble x, LogDouble y);
/*
   Return x-y where x and y are stored as logs, 
   diff < LSMALL is floored to LZERO 
*/

double L2F(LogDouble x);
/*
   Convert log(x) to real, result is floored to 0.0 if x < LSMALL 
*/

/* ------------------- Random Number Routines ------------------------ */

void RandInit(int seed);
/* 
   Initialise random number generators, if seed is -ve, then system 
   clock is used.  RandInit(-1) is called by InitMath.
*/

float RandomValue(void);
/*
   Return a random number in range 0.0->1.0 with uniform distribution
*/

float GaussDeviate(float mu, float sigma);
/*
   Return a random number with a N(mu,sigma) distribution
*/


```


+ ### [HSigP.h](#HTK_proto)<a name="HSigP"></a> 

```C++
/* !HVER!HSigP:   3.4.1 [CUED 12/03/09] */

#ifndef _HSIGP_H_
#define _HSIGP_H_

#ifdef __cplusplus
extern "C" {
#endif

void InitSigP(void);
/*
   Initialise the signal processing module.  This must be called
   before any other operation
*/

/* --------------- Speech Signal Processing Operations ------------- */

void ZeroMean(short *data, long nSamples);
/* 
   zero mean a complete speech waveform nSamples long
*/

void Ham (Vector s);
/*
   Apply Hamming Window to Speech frame s
*/

void PreEmphasise (Vector s, float k);
/*
   Apply first order preemphasis filter y[n] = x[n] - K*x[n-1] to s
*/

/* --------------- Linear Prediction Coding Operations ------------- */

void Wave2LPC (Vector s, Vector a, Vector k, float *re, float *te);
/*
   Calculate LP Filter Coef in a and LP Refl Coef in k from speech s
   Either a and k can be NULL.  Residual Energy is returned in re and
   total energy in te.
*/

void LPC2RefC (Vector a, Vector k);
void RefC2LPC (Vector k, Vector a);
/*
   Convert between filter and reflection coefs 
*/

void LPC2Cepstrum (Vector a, Vector c);
void Cepstrum2LPC (Vector c, Vector a);
/*
   Convert between LP Cepstral Coef in c and LP Coef in a
*/

/* -------------------- FFT Based Operations ----------------------- */

void FVec2Spectrum (float fzero, Vector f, Vector s);
/*
   Pads f with zeroes and applies FFT to give spectrum s
   Only the useful half of the spectrum is returned eg 
   if VectorSize(s)=128 then a 128 point FFT will be used
   but only the first 64 complex points are returned in s.
   fzero is the value of the 0'th feature vector coefficient 
   which is typically omitted by HSigP routines eg a0 = 1.0 
   for LPC
*/

void FFT(Vector s, int invert);
/*
   When called s holds nn complex values stored in the
   sequence   [ r1 , i1 , r2 , i2 , .. .. , rn , in ] where
   n = VectorSize(s) DIV 2, n must be a power of 2. On exit s
   holds the fft (or the inverse fft if invert == 1) 
*/

void Realft (Vector s);
/*
   When called s holds 2*n real values, on exit s holds the
   first  n complex points of the spectrum stored in
   the same format as for fft
*/
   
void SpecModulus(Vector s, Vector m);
void SpecLogModulus(Vector s, Vector m, Boolean invert);
void SpecPhase(Vector s, Vector m);
/*
   On entry, s should hold n complex points; VectorSize(s)=n*2
   On return, m holds (log) modulus/phase of s in first n points
*/

/* -------------------- MFCC Related Operations -------------------- */

typedef struct{
   int frameSize;       /* speech frameSize */
   int numChans;        /* number of channels */
   long sampPeriod;     /* sample period */
   int fftN;            /* fft size */
   int klo,khi;         /* lopass to hipass cut-off fft indices */
   Boolean usePower;    /* use power rather than magnitude */
   Boolean takeLogs;    /* log filterbank channels */
   float fres;          /* scaled fft resolution */
   Vector cf;           /* array[1..pOrder+1] of centre freqs */
   ShortVec loChan;     /* array[1..fftN/2] of loChan index */
   Vector loWt;         /* array[1..fftN/2] of loChan weighting */
   Vector x;            /* array[1..fftN] of fftchans */
}FBankInfo;

float Mel(int k, float fres);
/* 
   return mel-frequency corresponding to given FFT index k.  
   Resolution is normally determined by fres field of FBankInfo
   record.
*/

FBankInfo InitFBank(MemHeap *x, int frameSize, long sampPeriod, int numChans,
                    float lopass, float hipass, Boolean usePower, Boolean takeLogs,
                    Boolean doubleFFT,
                    float alpha, float warpLowCut, float warpUpCut);
/*
   Initialise an FBankInfo record prior to calling Wave2FBank.
*/


void Wave2FBank(Vector s, Vector fbank, float *te, FBankInfo info);
/*
   Convert given speech frame in s into mel-frequency filterbank
   coefficients.  The total frame energy is stored in te.  The
   info record contains precomputed filter weights and should be set
   prior to using Wave2FBank by calling InitFBank.
*/

void FBank2MFCC(Vector fbank, Vector c, int n);
/*
   Apply the DCT to fbank and store first n cepstral coeff in c.
   Note that the resulting coef are normalised by sqrt(2/numChans)
*/ 

void FBank2MelSpec(Vector fbank);
/*
   Convert the given log filterbank coef, in place, to linear
*/ 

void MelSpec2FBank(Vector melspec);
/*
   Convert the given linear filterbank coef, in place, to log
*/ 

float FBank2C0(Vector fbank);
/*
   return zero'th cepstral coefficient for given filter bank, i.e.
   compute sum of fbank channels and do standard normalisation
*/


/* ------------------- PLP Related Operations ---------------------- */

void InitPLP(FBankInfo info, int lpcOrder, Vector eql, DMatrix cm);
/*
   Initialise equal-loudness curve and cosine matrix for IDFT
*/
void FBank2ASpec(Vector fbank, Vector as, Vector eql, float compressFact,
		 FBankInfo info);
/*
   Pre-emphasise with simulated equal-loudness curve and perform
   cubic root amplitude compression.
*/
void ASpec2LPCep(Vector as, Vector ac, Vector lp, Vector c, DMatrix cm);
/*
   Do IDFT giving autocorrelation values then do linear prediction
   and finally, transform into cepstral coefficients
*/

/* ------------------- Feature Level Operations -------------------- */

void WeightCepstrum (Vector c, int start, int count, int cepLiftering);
void UnWeightCepstrum(Vector c, int start, int count, int cepLiftering);
/*
   Apply weights w[1]..w[count] to c[start] to c[start+count-1] 
   where w[i] = 1.0 + (L/2.0)*sin(i*pi/L),  L=cepLiftering
*/

/* The following apply to a sequence of 'n' vectors 'step' floats apart  */

void FZeroMean(float *data, int vSize, int n, int step);
/* 
   Zero mean the given data sequence
*/

void AddRegression(float *data, int vSize, int n, int step, int offset, 
                   int delwin, int head, int tail, Boolean simpleDiffs);
/*
   Add regression vector at +offset from source vector.  

   Each regression component is given by Sum( t*(v[+t] - v[-t])) / 2*Sum(t*t) 
   where the sum ranges over 1 to delwin and v[+t/-t] is the corresponding 
   component t steps ahead/back assuming that this vector is in the valid
   range -head...n+tail.  If simple diffs is true, then slope is 
   calculated from (v[delwin] - v[-delwin])/(2*delwin).  
*/

void AddHeadRegress(float *data, int vSize, int n, int step, int offset, 
                    int delwin, Boolean simpleDiffs);
/* 
   As for AddRegression, but deals with start case where there are no
   previous frames to regress over (assumes that there are at least
   min(delwin,1) valid following frames).  If delwin==0, then a simple
   forward difference given by v[0] - v[-1] is used.  Otherwise, the first
   available frame in the window is replicated back in time to fill the
   window.
*/

void AddTailRegress(float *data, int vSize, int n, int step, int offset, 
                    int delwin, Boolean simpleDiffs);
/* 
   As for AddRegression, but deals with start case where there are no
   previous frames to regress over (assumes that there are at least
   min(delwin,1) valid preceding frames).  If delwin==0, then a simple
   forward difference given by v[0] - v[-1] is used.  Otherwise, the first
   available frame in the window is replicated back in time to fill the
   window.  
*/

void NormaliseLogEnergy(float *data, int n, int step, float silFloor, float escale);
/* 
   normalise log energy to range -X .. 1.0 by subtracting the max log
   energy and adding 1.0.  The lowest energy level is set by the value
   of silFloor which gives the ratio between the max and min energies
   in dB.  Escale is used to scale the normalised log energy.
*/

```
+ ### [HWave.h](#HTK_proto)<a name="HWave"></a>

```C++
typedef struct FieldSpec **HFieldList;

typedef enum {
        NOHEAD,            /* Headerless File */
        HAUDIO,            /* Direct Audio Input */
        HTK,               /* used for both wave and parm files */
        TIMIT,             /* Prototype TIMIT database */
        NIST,              /* NIST databases eg RM1,TIMIT */
        SCRIBE,            /* UK Scribe databases */
        AIFF,              /* Apple Audio Interchange format */
        SDES1,             /* Sound Designer I format */
        SUNAU8,            /* Sun 8 bit MuLaw .au format */
        OGI,               /* Oregon Institute format (similar to TIMIT) */
        ESPS,              /* used for both wave and parm files */
	ESIG,              /* used for both wave and parm files */
	WAV,               /* Microsoft WAVE format */
        UNUSED,
        ALIEN,             /* Unknown */
        UNDEFF
} FileFormat;

typedef struct _Wave *Wave;  /* Abstract type representing waveform file */

void InitWave(void);
/*
   Initialise module
*/

Wave OpenWaveInput(MemHeap *x, char *fname, FileFormat fmt, HTime winDur, 
                   HTime frPeriod, HTime *sampPeriod);
/*
   Open the named input file with the given format and return a
   Wave object. If fmt==UNDEFF then the value of the configuration
   parameter SOURCEFORMAT is used.  If this is not set, then the format
   HTK is assumed. Samples are returned in frames of duration winDur.  The 
   period between successive frames is given by frPeriod.  If the value of 
   sampPeriod is not 0.0, then it overrides the sample period specified in
   the file, otherwise the actual value is returned. Returns NULL on error.
*/

void CloseWaveInput(Wave w);
/* 
   Terminate Wave input and free any resources associated with w
*/

void ZeroMeanWave(Wave w);
/*
   Ensure that mean of wave w is zero
*/

int FramesInWave(Wave w);
/*
   Return number of whole frames which are currently
   available in the given Wave
*/

int SampsInWaveFrame(Wave w);
/*
   Return number of samples in each frame of the given Wave
*/

void GetWave(Wave w, int nFrames, float *buf);
/* 
   Get next nFrames from Wave input buffer and store sequentially in
   buf as floats.  If a frame overlap has been set then samples will be
   duplicated in buf.  It is a fatal error to request more frames
   than exist in the Wave (as determined by FramesInWave.
*/

short *GetWaveDirect(Wave w, long *nSamples);
/* 
   Returns a pointer to the waveform stored in w.
*/

Wave OpenWaveOutput(MemHeap *x, HTime *sampPeriod, long bufSize);
/*
   Initialise a Wave object to store waveform data at the given 
   sample period, using buffer of bufSize shorts.  
*/

void PutWaveSample(Wave w, long nSamples, short *buf);
/*
   Append given nSamples in buf to wave w.
*/

ReturnStatus CloseWaveOutput(Wave w, FileFormat fmt, char *fname);
/* 
   Output wave w to file fname in given fmt and free any 
   associated resources.  If fmt==UNDEFF then value of
   configuration variable TARGETFORMAT is used, if any,
   otherwise the HTK format is used. If an error then 
   returns FAIL and does not free any memory. 
*/

FileFormat WaveFormat(Wave w);
/* 
   Return format of given wave
*/

char *Format2Str(FileFormat format);
FileFormat Str2Format(char *fmt);
/*
   Convert between FileFormat enum type & string.
*/

/* --------------------- HTK Header Routines --------------------- */

Boolean ReadHTKHeader(FILE *f,long *nSamp,long *sampP,short *sampS,
                      short *kind, Boolean *bSwap);
/* 
   Get header info from HTK file f, return false if apparently not
   a HTK file.  If byte-swapped bswap returns true.  NB only
   the user can specify required byte order via NATREADORDER config var 
   since it is not defined for HTK files)
*/

void WriteHTKHeader(FILE *f, long nSamp, long sampP, short sampS, 
		    short kind, Boolean *bSwap);
/* 
   Write header info to HTK file f.  
   Sets bSwap to indicate whether header was byte swapped before writing.
*/

void StoreESIGFieldList(HFieldList fList);
/*
   Store the field list of an ESIG input file 
*/
void RetrieveESIGFieldList(HFieldList *fList);
/*
   Retrieve the field list of an ESIG input file 
*/

Boolean ReadEsignalHeader(FILE *f, long *nSamp, long *sampP, short *sampS,
 			  short *kind, Boolean *bSwap, long *hdrS,
 			  Boolean isPipe);
/*
    Get header from Esignal file f; return FALSE in case of failure.
*/

```

---
#### [LICENSE](#HTK)<a name="HTK_license"></a>

+ Licensee의 목적을 위해 사용,수정하는 것은 grant
+ HTK 전체든 부분이든 배포하거나 다른 라이센스 sub-licensed 하는 것은 금지
+ Copyright,trademakr, patent notices는 표시되어야함
  

<pre>
                        HTK END USER LICENSE AGREEMENT

1. Definitions

   Licensed Software:  All source code, object or executable code,
                       associated technical documentation and any 
                       data files in this HTK distribution.

   Licensor         :  University of Cambridge

   Licensee         :  The person/organisation who downloads the HTK
                       distribution or any part of it.
                 

2. License Rights and Obligations

   2.1 The Licensor hereby grants the Licensee a non-exclusive license to a) make 
   copies of the Licensed Software in source and object code form for use 
   within the Licensee's organisation; b) modify copies of the Licensed Software 
   to create derivative works thereof for use within the Licensee's organisation.

   2.2 The Licensed Software either in whole or in part can not be distributed
   or sub-licensed to any third party in any form.

   2.3 The Licensee agrees not to remove any copyright, trademark or patent notices 
   that appear in the Licensed Software.

   2.4 THE LICENSED SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY WARRANTY OF ANY KIND.
   TO THE MAXIMUM EXTENT PERMITTED BY LAW, ANY AND ALL EXPRESS AND IMPLIED 
   WARRANTIES OF ANY KIND WHATSOEVER, INCLUDING BUT NOT LIMITED TO THOSE OF 
   MERCHANTABILITY, OF FITNESS FOR A PARTICULAR PURPOSE, OF ACCURACY OR COMPLETENESS 
   OF RESPONSES, OF RESULTS, OF REASONABLE CARE OR WORKMANLIKE EFFORT, OF LACK OF 
   NEGLIGENCE, AND/OR OF A LACK OF VIRUSES, ALL WITH REGARD TO THE LICENSED SOFTWARE, 
   ARE EXPRESSLY EXCLUDED.  NEITHER LICENSOR, ENTROPIC OR MICROSOFT MAKE ANY WARRANTY 
   THAT THE LICENSED SOFTWARE WILL OPERATE PROPERLY AS INTEGRATED IN YOUR PRODUCT(S) 
   OR ON ANY CUSTOMER SYSTEM(S). 

   2.5 THE LICENSEE AGREES THAT NEITHER LICENSOR, ENTROPIC OR MICROSOFT SHALL BE 
   LIABLE FOR ANY CONSEQUENTIAL, INCIDENTAL, INDIRECT, ECONOMIC OR PUNITIVE DAMAGES 
   WHATSOEVER (INCLUDING BUT NOT LIMITED TO DAMAGES FOR LOSS OF BUSINESS OR PERSONAL 
   PROFITS, BUSINESS INTERRUPTION, LOSS OF BUSINESS OR PERSONAL OR CONFIDENTIAL 
   INFORMATION, OR ANY OTHER PECUNIARY LOSS, DAMAGES FOR LOSS OF PRIVACY, OR FOR FAILURE 
   TO MEET ANY DUTY, INCLUDING ANY DUTY OF GOOD FAITH, OR TO EXERCISE COMMERCIALLY 
   REASONABLE CARE OR FOR NEGLIGENCE) ARISING OUT OF OR IN ANY WAY RELATED TO THE 
   USE OF OR INABILITY TO USE THE LICENSED SOFTWARE, EVEN IF ENTROPIC HAS BEEN 
   ADVISED OF THE POSSIBILITY OF SUCH DAMAGES. 


3. Term of Agreement

   This agreement is effective upon the download of the Licensed Software and 
   shall remain in effect until terminated. Termination by the Licensee should be 
   accompanied by the complete destruction of the Licensed Software and any complete 
   or partial copies thereof. The Licensor may only terminate this  agreement if 
   the Licensee has failed to abide by the terms of this agreement in which case 
   termination may be without notice. All provisions of this agreement relating 
   to disclaimers of warranties, limitation of liability, remedies or damage 
   and the Licensor's proprietary rights shall survive termination.


4. Governing Law

   This agreement shall be construed and interpreted in accordance with the laws 
   of England.


5. Though not a license condition, Licensees are strongly encouraged to
   a) report all bugs, where possible with bug fixes, that are found.
   b) reference the use of HTK in any publications that use the Licensed Software.


6. Contributions to HTK 

   We strongly encourage contributions to the HTK source code base. These will 
   in general be additional tools or library modules which will not fall under this 
   HTK License Agreement.

</pre>
