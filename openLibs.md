

+ [summary](#summary)  
+ [CIGLET](#CIGLET)

---  

## SUMMARY<a name = "summary"></a>  
 
### [Ciglet](https://github.com/Sleepwalking/ciglet)
+ lightweight C library for digital signal processing
+ C-written sIGnal codeLETs
+ Matlab  to C conversion of frequently used ones 

### [Hidden Markov Model Toolkit (HTK)](http://htk.eng.cam.ac.uk/)
+ C source form


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

CIGLET<a name = "CIGLET"></a>
+ [funtion list](#ciglet_list)
+ [function prototype](#ciglet_proto)
+ [license](#ciglet_license)
---
#### FUNCTION LIST<a name="ciglet_list"></a>

Scalar operations

    random number generation: randu, randn
    miscellaneous: max, min, linterp, fastatan2
    complex arithmetics: c_cplx, c_conj, c_add, c_sub, c_mul, c_div, c_exp, c_abs, c_arg

Vector operations and statistics

    vectorized arithmetics: sumfp, sumsqrfp, maxfp, minfp
    descriptive statistics: meanfp, varfp, medianfp, xcorr, corr, cov
    sorting: selectnth, sort
    peak picking: find_peak, find_valley, find_maxima, find_minima

Numerical routines

fzero, polyval, roots
Basic linear algebra

    products: matmul, mvecmul, dot
    solving a linear system: lu, lusolve
    pivoting: ppivot, permm, permv

Memory (de)allocation

    enumeration: linspace, iota
    2d array operations: malloc2d, free2d, copy2d, flatten, reshape, transpose

Audio I/O

wavread, wavwrite
General DSP routines

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

Audio/speech processing routines

    psychoacoustics: mel2freq, freq2mel, freq2bark, bark2freq, eqloud, melspace
    frequency estimation: ifdetector_estimate, correlogram, invcrgm
    spectrogram and STFT: stft, istft, qifft, spgm2cegm, cegm2spgm
    filterbank analysis: filterbank_spgm, filterbank_spec, be2cc, be2ccgm
    spectral envelope estimation: spec2env
    glottal model: lfmodel_from_rd, lfmodel_spectrum, lfmodel_period

Plotting utilities (Gnuplot interface, unavailable on Windows)

plotopen, plot, imagesc, plotclose

---

#### [function prototype](#CIGLET)<a name = "ciglet_proto"></a>

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
```

---


```c++
FP_TYPE* cig_xcorr(FP_TYPE* x, FP_TYPE* y, int nx, int maxlag);

static inline FP_TYPE* xcorr(FP_TYPE* x, FP_TYPE* y, int nx) {
  return cig_xcorr(x, y, nx, nx);
}

```


---

#### [LICENSE](#CIGLET)<a name = "ciglet_license"></a>

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


