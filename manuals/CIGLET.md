### [CIGLET](./openAudioLibs.md#TOP)<a name = "CIGLET"></a>
+ [fft](#ciglet_fft)
+ [Data I/O](#ciglet_dataIO)
+ [funtion list](#ciglet_list)
+ [function prototype](#ciglet_proto)
+ [license](#ciglet_license)

---

#### [fft](#TOP)<a name = "ciglet_fft"></a>

+ fft 
```C++
void cig_fft(FP_TYPE* xr, FP_TYPE* xi, FP_TYPE* yr, FP_TYPE* yi, int n, FP_TYPE* buffer, FP_TYPE mode)  
```
+ stft 
```C++
void cig_stft_forward(FP_TYPE* x, int nx, int* center, int* nwin, int nfrm, int nfft, char* window, int subt_mean, int optlv,FP_TYPE* norm_factor, FP_TYPE* weight_factor, FP_TYPE** Xmagn, FP_TYPE** Xphse)  
```


#### [Data I/O](#TOP)<a name = "ciglet_dataIO"></a>
external library using simple iostream  
only wav format supports  

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
으로 데이터 형식 지정  

<details><summary>ciglet.h</summary>

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

</details>

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
