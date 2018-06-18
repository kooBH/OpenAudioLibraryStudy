### [Kaldi](./openAudioLibs.md#TOP)<a name = "Kaldi"></a>

+ [license](#ciglet_license)

---

+ OpenFst  
  we compile against this and use it heavily.
+ IRSTLM  
this a language modeling toolkit. Some of the example scripts require it but it is not tightly integrated with Kaldi; we can convert any Arpa format language model to an FST.
    *    The IRSTLM build process requires automake, aclocal, and libtoolize (the corresponding packages are automake and libtool).  
    *    Note: some of the example scripts now use SRILM; we make it easy to install that, although you still have to register online to download it.  
+ SRILM  
ome of the example scripts use this. It's generally a better and more complete language modeling toolkit than IRSTLM; the only drawback is the license, which is not free for commercial use. You have to enter your name on the download page to download it, so the installation script requires some human interaction.
+ sph2pipe  
this is for converting sph format files into other formats such as wav. It's needed for the example scripts that use LDC data.
+    sclite  
this is for scoring and is not necessary as we have our own, simple scoring program (compute-wer.cc).
 ATLAS, the linear algebra library. This is only needed for the headers; in typical setups we expect that ATLAS will be on your system. However, if it not already on your system you can compile ATLAS as long as your machine does not have CPU throttling enabled.
+    CLAPACK  
the linear algebra library (we download the headers). This is useful only on systems where you don't have ATLAS and are instead compiling with CLAPACK.
+    OpenBLAS  
this is an alernative to ATLAS or CLAPACK. The scripts don't use it by default but we provide installation scripts so you can install it if you want to compare it against ATLAS (it's more actively maintained than ATLAS).

comment
---

 + 사용 및 수정의 용이성과 명확성을 위해 bin 파일의 수가 많고 코드도 많이 분할되어있다  
 

## Kaldi Matrix

The Kaldi matrix library is mostly a C++ wrapper for standard BLAS and LAPACK linear algebra routines. With lots of #define s.  

### Kaldi supports
+    **ATLAS**, which is an implementation of BLAS plus a subset of LAPACK (with a different interface)
+    Intel's **MKL**, which provides both BLAS and LAPACK
+    **OpenBLAS**, which provides BLAS and LAPACK



## Kaldi FFT

+ SplitRadixComplexFft(SRFFT)  
* only works for power of 2
* faster

<details><summary>srfft.cc</summary>
   
```C++
#include "matrix/srfft.h"
#include "matrix/matrix-functions.h"

namespace kaldi {


template<typename Real>
SplitRadixComplexFft<Real>::SplitRadixComplexFft(MatrixIndexT N) {
  if ( (N & (N-1)) != 0 || N <= 1)
    KALDI_ERR << "SplitRadixComplexFft called with invalid number of points "
              << N;
  N_ = N;
  logn_ = 0;
  while (N > 1) {
    N >>= 1;
    logn_ ++;
  }
  ComputeTables();
}

template <typename Real>
SplitRadixComplexFft<Real>::SplitRadixComplexFft(
    const SplitRadixComplexFft<Real> &other):
    N_(other.N_), logn_(other.logn_) {
  // This code duplicates tables from a previously computed object.
  // Compare with the code in ComputeTables().
  MatrixIndexT lg2 = logn_ >> 1;
  if (logn_ & 1) lg2++;
  MatrixIndexT brseed_size = 1 << lg2;
  brseed_ = new MatrixIndexT[brseed_size];
  std::memcpy(brseed_, other.brseed_, sizeof(MatrixIndexT) * brseed_size);

  if (logn_ < 4) {
    tab_ = NULL;
  } else {
    tab_ = new Real*[logn_ - 3];
    for (MatrixIndexT i = logn_; i >= 4 ; i--) {
      MatrixIndexT m = 1 << i, m2 = m / 2, m4 = m2 / 2;
      MatrixIndexT this_array_size = 6 * (m4 - 2);
      tab_[i-4] = new Real[this_array_size];
      std::memcpy(tab_[i-4], other.tab_[i-4],
                  sizeof(Real) * this_array_size);
    }
  }
}

template<typename Real>
void SplitRadixComplexFft<Real>::ComputeTables() {
  MatrixIndexT    imax, lg2, i, j;
  MatrixIndexT     m, m2, m4, m8, nel, n;
  Real    *cn, *spcn, *smcn, *c3n, *spc3n, *smc3n;
  Real    ang, c, s;

  lg2 = logn_ >> 1;
  if (logn_ & 1) lg2++;
  brseed_ = new MatrixIndexT[1 << lg2];
  brseed_[0] = 0;
  brseed_[1] = 1;
  for (j = 2; j <= lg2; j++) {
    imax = 1 << (j - 1);
    for (i = 0; i < imax; i++) {
      brseed_[i] <<= 1;
      brseed_[i + imax] = brseed_[i] + 1;
    }
  }

  if (logn_ < 4) {
    tab_ = NULL;
  } else {
    tab_ = new Real* [logn_-3];
    for (i = logn_; i>=4 ; i--) {
      /* Compute a few constants */
      m = 1 << i; m2 = m / 2; m4 = m2 / 2; m8 = m4 /2;

      /* Allocate memory for tables */
      nel = m4 - 2;

      tab_[i-4] = new Real[6*nel];

      /* Initialize pointers */
      cn = tab_[i-4]; spcn  = cn + nel;  smcn  = spcn + nel;
      c3n = smcn + nel;  spc3n = c3n + nel; smc3n = spc3n + nel;

      /* Compute tables */
      for (n = 1; n < m4; n++) {
        if (n == m8) continue;
        ang = n * M_2PI / m;
        c = std::cos(ang); s = std::sin(ang);
        *cn++ = c; *spcn++ = - (s + c); *smcn++ = s - c;
        ang = 3 * n * M_2PI / m;
        c = std::cos(ang); s = std::sin(ang);
        *c3n++ = c; *spc3n++ = - (s + c); *smc3n++ = s - c;
      }
    }
  }
}

template<typename Real>
SplitRadixComplexFft<Real>::~SplitRadixComplexFft() {
  delete [] brseed_;
  if (tab_ != NULL) {
    for (MatrixIndexT i = 0; i < logn_-3; i++)
      delete [] tab_[i];
    delete [] tab_;
  }
}

template<typename Real>
void SplitRadixComplexFft<Real>::Compute(Real *xr, Real *xi, bool forward) const {
  if (!forward) {  // reverse real and imaginary parts for complex FFT.
    Real *tmp = xr;
    xr = xi;
    xi = tmp;
  }
  ComputeRecursive(xr, xi, logn_);
  if (logn_ > 1) {
    BitReversePermute(xr, logn_);
    BitReversePermute(xi, logn_);
  }
}

template<typename Real>
void SplitRadixComplexFft<Real>::Compute(Real *x, bool forward,
                                         std::vector<Real> *temp_buffer) const {
  KALDI_ASSERT(temp_buffer != NULL);
  if (temp_buffer->size() != N_)
    temp_buffer->resize(N_);
  Real *temp_ptr = &((*temp_buffer)[0]);
  for (MatrixIndexT i = 0; i < N_; i++) {
    x[i] = x[i * 2];  // put the real part in the first half of x.
    temp_ptr[i] = x[i * 2 + 1];  // put the imaginary part in temp_buffer.
  }
  // copy the imaginary part back to the second half of x.
  memcpy(static_cast<void*>(x + N_),
         static_cast<void*>(temp_ptr),
         sizeof(Real) * N_);

  Compute(x, x + N_, forward);
  // Now change the format back to interleaved.
  memcpy(static_cast<void*>(temp_ptr),
         static_cast<void*>(x + N_),
         sizeof(Real) * N_);
  for (MatrixIndexT i = N_-1; i > 0; i--) {  // don't include 0,
    // in case MatrixIndexT is unsigned, the loop would not terminate.
    // Treat it as a special case.
    x[i*2] = x[i];
    x[i*2 + 1] = temp_ptr[i];
  }
  x[1] = temp_ptr[0];  // special case of i = 0.
}

template<typename Real>
void SplitRadixComplexFft<Real>::Compute(Real *x, bool forward) {
  this->Compute(x, forward, &temp_buffer_);
}

template<typename Real>
void SplitRadixComplexFft<Real>::BitReversePermute(Real *x, MatrixIndexT logn) const {
  MatrixIndexT      i, j, lg2, n;
  MatrixIndexT      off, fj, gno, *brp;
  Real    tmp, *xp, *xq;

  lg2 = logn >> 1;
  n = 1 << lg2;
  if (logn & 1) lg2++;

  /* Unshuffling loop */
  for (off = 1; off < n; off++) {
    fj = n * brseed_[off]; i = off; j = fj;
    tmp = x[i]; x[i] = x[j]; x[j] = tmp;
    xp = &x[i];
    brp = &(brseed_[1]);
    for (gno = 1; gno < brseed_[off]; gno++) {
      xp += n;
      j = fj + *brp++;
      xq = x + j;
      tmp = *xp; *xp = *xq; *xq = tmp;
    }
  }
}


template<typename Real>
void SplitRadixComplexFft<Real>::ComputeRecursive(Real *xr, Real *xi, MatrixIndexT logn) const {

  MatrixIndexT    m, m2, m4, m8, nel, n;
  Real    *xr1, *xr2, *xi1, *xi2;
  Real    *cn, *spcn, *smcn, *c3n, *spc3n, *smc3n;
  Real    tmp1, tmp2;
  Real   sqhalf = M_SQRT1_2;

  /* Check range of logn */
  if (logn < 0)
    KALDI_ERR << "Error: logn is out of bounds in SRFFT";

  /* Compute trivial cases */
  if (logn < 3) {
    if (logn == 2) {  /* length m = 4 */
      xr2  = xr + 2;
      xi2  = xi + 2;
      tmp1 = *xr + *xr2;
      *xr2 = *xr - *xr2;
      *xr  = tmp1;
      tmp1 = *xi + *xi2;
      *xi2 = *xi - *xi2;
      *xi  = tmp1;
      xr1  = xr + 1;
      xi1  = xi + 1;
      xr2++;
      xi2++;
      tmp1 = *xr1 + *xr2;
      *xr2 = *xr1 - *xr2;
      *xr1 = tmp1;
      tmp1 = *xi1 + *xi2;
      *xi2 = *xi1 - *xi2;
      *xi1 = tmp1;
      xr2  = xr + 1;
      xi2  = xi + 1;
      tmp1 = *xr + *xr2;
      *xr2 = *xr - *xr2;
      *xr  = tmp1;
      tmp1 = *xi + *xi2;
      *xi2 = *xi - *xi2;
      *xi  = tmp1;
      xr1  = xr + 2;
      xi1  = xi + 2;
      xr2  = xr + 3;
      xi2  = xi + 3;
      tmp1 = *xr1 + *xi2;
      tmp2 = *xi1 + *xr2;
      *xi1 = *xi1 - *xr2;
      *xr2 = *xr1 - *xi2;
      *xr1 = tmp1;
      *xi2 = tmp2;
      return;
    }
    else if (logn == 1) {   /* length m = 2 */
      xr2  = xr + 1;
      xi2  = xi + 1;
      tmp1 = *xr + *xr2;
      *xr2 = *xr - *xr2;
      *xr  = tmp1;
      tmp1 = *xi + *xi2;
      *xi2 = *xi - *xi2;
      *xi  = tmp1;
      return;
    }
    else if (logn == 0) return;   /* length m = 1 */
  }

  /* Compute a few constants */
  m = 1 << logn; m2 = m / 2; m4 = m2 / 2; m8 = m4 /2;


  /* Step 1 */
  xr1 = xr; xr2 = xr1 + m2;
  xi1 = xi; xi2 = xi1 + m2;
  for (n = 0; n < m2; n++) {
    tmp1 = *xr1 + *xr2;
    *xr2 = *xr1 - *xr2;
    xr2++;
    *xr1++ = tmp1;
    tmp2 = *xi1 + *xi2;
    *xi2 = *xi1 - *xi2;
    xi2++;
    *xi1++ = tmp2;
  }

  /* Step 2 */
  xr1 = xr + m2; xr2 = xr1 + m4;
  xi1 = xi + m2; xi2 = xi1 + m4;
  for (n = 0; n < m4; n++) {
    tmp1 = *xr1 + *xi2;
    tmp2 = *xi1 + *xr2;
    *xi1 = *xi1 - *xr2;
    xi1++;
    *xr2++ = *xr1 - *xi2;
    *xr1++ = tmp1;
    *xi2++ = tmp2;
    // xr1++; xr2++; xi1++; xi2++;
  }

  /* Steps 3 & 4 */
  xr1 = xr + m2; xr2 = xr1 + m4;
  xi1 = xi + m2; xi2 = xi1 + m4;
  if (logn >= 4) {
    nel = m4 - 2;
    cn  = tab_[logn-4]; spcn  = cn + nel;  smcn  = spcn + nel;
    c3n = smcn + nel;  spc3n = c3n + nel; smc3n = spc3n + nel;
  }
  xr1++; xr2++; xi1++; xi2++;
  // xr1++; xi1++;
  for (n = 1; n < m4; n++) {
    if (n == m8) {
      tmp1 =  sqhalf * (*xr1 + *xi1);
      *xi1 =  sqhalf * (*xi1 - *xr1);
      *xr1 =  tmp1;
      tmp2 =  sqhalf * (*xi2 - *xr2);
      *xi2 = -sqhalf * (*xr2 + *xi2);
      *xr2 =  tmp2;
    } else {
      tmp2 = *cn++ * (*xr1 + *xi1);
      tmp1 = *spcn++ * *xr1 + tmp2;
      *xr1 = *smcn++ * *xi1 + tmp2;
      *xi1 = tmp1;
      tmp2 = *c3n++ * (*xr2 + *xi2);
      tmp1 = *spc3n++ * *xr2 + tmp2;
      *xr2 = *smc3n++ * *xi2 + tmp2;
      *xi2 = tmp1;
    }
    xr1++; xr2++; xi1++; xi2++;
  }

  /* Call ssrec again with half DFT length */
  ComputeRecursive(xr, xi, logn-1);

  /* Call ssrec again twice with one quarter DFT length.
     Constants have to be recomputed, because they are static! */
  // m = 1 << logn; m2 = m / 2;
  ComputeRecursive(xr + m2, xi + m2, logn - 2);
  // m = 1 << logn;
  m4 = 3 * (m / 4);
  ComputeRecursive(xr + m4, xi + m4, logn - 2);
}


template<typename Real>
void SplitRadixRealFft<Real>::Compute(Real *data, bool forward) {
  Compute(data, forward, &this->temp_buffer_);
}


// This code is mostly the same as the RealFft function.  It would be
// possible to replace it with more efficient code from Rico's book.
template<typename Real>
void SplitRadixRealFft<Real>::Compute(Real *data, bool forward,
                                      std::vector<Real> *temp_buffer) const {
  MatrixIndexT N = N_, N2 = N/2;
  KALDI_ASSERT(N%2 == 0);
  if (forward) // call to base class
    SplitRadixComplexFft<Real>::Compute(data, true, temp_buffer);

  Real rootN_re, rootN_im;  // exp(-2pi/N), forward; exp(2pi/N), backward
  int forward_sign = forward ? -1 : 1;
  ComplexImExp(static_cast<Real>(M_2PI/N *forward_sign), &rootN_re, &rootN_im);
  Real kN_re = -forward_sign, kN_im = 0.0;  // exp(-2pik/N), forward; exp(-2pik/N), backward
  // kN starts out as 1.0 for forward algorithm but -1.0 for backward.
  for (MatrixIndexT k = 1; 2*k <= N2; k++) {
    ComplexMul(rootN_re, rootN_im, &kN_re, &kN_im);

    Real Ck_re, Ck_im, Dk_re, Dk_im;
    // C_k = 1/2 (B_k + B_{N/2 - k}^*) :
    Ck_re = 0.5 * (data[2*k] + data[N - 2*k]);
    Ck_im = 0.5 * (data[2*k + 1] - data[N - 2*k + 1]);
    // re(D_k)= 1/2 (im(B_k) + im(B_{N/2-k})):
    Dk_re = 0.5 * (data[2*k + 1] + data[N - 2*k + 1]);
    // im(D_k) = -1/2 (re(B_k) - re(B_{N/2-k}))
    Dk_im =-0.5 * (data[2*k] - data[N - 2*k]);
    // A_k = C_k + 1^(k/N) D_k:
    data[2*k] = Ck_re;  // A_k <-- C_k
    data[2*k+1] = Ck_im;
    // now A_k += D_k 1^(k/N)
    ComplexAddProduct(Dk_re, Dk_im, kN_re, kN_im, &(data[2*k]), &(data[2*k+1]));

    MatrixIndexT kdash = N2 - k;
    if (kdash != k) {
      // Next we handle the index k' = N/2 - k.  This is necessary
      // to do now, to avoid invalidating data that we will later need.
      // The quantities C_{k'} and D_{k'} are just the conjugates of C_k
      // and D_k, so the equations are simple modifications of the above,
      // replacing Ck_im and Dk_im with their negatives.
      data[2*kdash] = Ck_re;  // A_k' <-- C_k'
      data[2*kdash+1] = -Ck_im;
      // now A_k' += D_k' 1^(k'/N)
      // We use 1^(k'/N) = 1^((N/2 - k) / N) = 1^(1/2) 1^(-k/N) = -1 * (1^(k/N))^*
      // so it's the same as 1^(k/N) but with the real part negated.
      ComplexAddProduct(Dk_re, -Dk_im, -kN_re, kN_im, &(data[2*kdash]), &(data[2*kdash+1]));
    }
  }

  {  // Now handle k = 0.
    // In simple terms: after the complex fft, data[0] becomes the sum of real
    // parts input[0], input[2]... and data[1] becomes the sum of imaginary
    // pats input[1], input[3]...
    // "zeroth" [A_0] is just the sum of input[0]+input[1]+input[2]..
    // and "n2th" [A_{N/2}] is input[0]-input[1]+input[2]... .
    Real zeroth = data[0] + data[1],
        n2th = data[0] - data[1];
    data[0] = zeroth;
    data[1] = n2th;
    if (!forward) {
      data[0] /= 2;
      data[1] /= 2;
    }
  }
  if (!forward) {  // call to base class
    SplitRadixComplexFft<Real>::Compute(data, false, temp_buffer);
    for (MatrixIndexT i = 0; i < N; i++)
      data[i] *= 2.0;
    // This is so we get a factor of N increase, rather than N/2 which we would
    // otherwise get from [ComplexFft, forward] + [ComplexFft, backward] in dimension N/2.
    // It's for consistency with our normal FFT convensions.
  }
}

template class SplitRadixComplexFft<float>;
template class SplitRadixComplexFft<double>;
template class SplitRadixRealFft<float>;
template class SplitRadixRealFft<double>;


} // end namespace kaldi
```
   
</details>

&nbsp;  
&nbsp;  
+ ComplexFft
<details><summary>matrix-funtions.cc</summary>

```C++
#include "matrix/matrix-functions.h"
#include "matrix/sp-matrix.h"

namespace kaldi {

template<typename Real> void ComplexFt (const VectorBase<Real> &in,
                                     VectorBase<Real> *out, bool forward) {
  int exp_sign = (forward ? -1 : 1);
  KALDI_ASSERT(out != NULL);
  KALDI_ASSERT(in.Dim() == out->Dim());
  KALDI_ASSERT(in.Dim() % 2 == 0);
  int twoN = in.Dim(), N = twoN / 2;
  const Real *data_in = in.Data();
  Real *data_out = out->Data();

  Real exp1N_re, exp1N_im;  //  forward -> exp(-2pi / N), backward -> exp(2pi / N).
  Real fraction = exp_sign * M_2PI / static_cast<Real>(N);  // forward -> -2pi/N, backward->-2pi/N
  ComplexImExp(fraction, &exp1N_re, &exp1N_im);

  Real expm_re = 1.0, expm_im = 0.0;  // forward -> exp(-2pi m / N).

  for (int two_m = 0; two_m < twoN; two_m+=2) {  // For each output component.
    Real expmn_re = 1.0, expmn_im = 0.0;  // forward -> exp(-2pi m n / N).
    Real sum_re = 0.0, sum_im = 0.0;  // complex output for index m (the sum expression)
    for (int two_n = 0; two_n < twoN; two_n+=2) {
      ComplexAddProduct(data_in[two_n], data_in[two_n+1],
                        expmn_re, expmn_im,
                        &sum_re, &sum_im);
      ComplexMul(expm_re, expm_im, &expmn_re, &expmn_im);
    }
    data_out[two_m] = sum_re;
    data_out[two_m + 1] = sum_im;


    if (two_m % 10 == 0) {  // occasionally renew "expm" from scratch to avoid
      // loss of precision.
      int nextm = 1 + two_m/2;
      Real fraction_mult = fraction * nextm;
      ComplexImExp(fraction_mult, &expm_re, &expm_im);
    } else {
      ComplexMul(exp1N_re, exp1N_im, &expm_re, &expm_im);
    }
  }
}

template
void ComplexFt (const VectorBase<float> &in,
                VectorBase<float> *out, bool forward);
template
void ComplexFt (const VectorBase<double> &in,
                VectorBase<double> *out, bool forward);


#define KALDI_COMPLEXFFT_BLOCKSIZE 8192
// This #define affects how we recurse in ComplexFftRecursive.
// We assume that memory-caching happens on a scale at
// least as small as this.


//! ComplexFftRecursive is a recursive function that computes the
//! complex FFT of size N.  The "nffts" arguments specifies how many
//! separate FFTs to compute in parallel (we assume the data for
//! each one is consecutive in memory).  The "forward argument"
//! specifies whether to do the FFT (true) or IFFT (false), although
//! note that we do not include the factor of 1/N (the user should
//! do this if required.  The iterators factor_begin and factor_end
//! point to the beginning and end (i.e. one past the last element)
//! of an array of small factors of N (typically prime factors).
//! See the comments below this code for the detailed equations
//! of the recursion.


template<typename Real>
void ComplexFftRecursive (Real *data, int nffts, int N,
                          const int *factor_begin,
                          const int *factor_end, bool forward,
                          Vector<Real> *tmp_vec) {
  if (factor_begin == factor_end) {
    KALDI_ASSERT(N == 1);
    return;
  }

  {  // an optimization: compute in smaller blocks.
    // this block of code could be removed and it would still work.
    MatrixIndexT size_perblock = N * 2 * sizeof(Real);
    if (nffts > 1 && size_perblock*nffts > KALDI_COMPLEXFFT_BLOCKSIZE) {  // can break it up...
      // Break up into multiple blocks.  This is an optimization.  We make
      // no progress on the FFT when we do this.
      int block_skip = KALDI_COMPLEXFFT_BLOCKSIZE / size_perblock;  // n blocks per call
      if (block_skip == 0) block_skip = 1;
      if (block_skip < nffts) {
        int blocks_left = nffts;
        while (blocks_left > 0) {
          int skip_now = std::min(blocks_left, block_skip);
          ComplexFftRecursive(data, skip_now, N, factor_begin, factor_end, forward, tmp_vec);
          blocks_left -= skip_now;
          data += skip_now * N*2;
        }
        return;
      } // else do the actual algorithm.
    } // else do the actual algorithm.
  }

  int P = *factor_begin;
  KALDI_ASSERT(P > 1);
  int Q = N / P;


  if (P > 1 && Q > 1) {  // Do the rearrangement.   C.f. eq. (8) below.  Transform
    // (a) to (b).
    Real *data_thisblock = data;
    if (tmp_vec->Dim() < (MatrixIndexT)N) tmp_vec->Resize(N);
    Real *data_tmp = tmp_vec->Data();
    for (int thisfft = 0; thisfft < nffts; thisfft++, data_thisblock+=N*2) {
      for (int offset = 0; offset < 2; offset++) {  // 0 == real, 1 == im.
        for (int p = 0; p < P; p++) {
          for (int q = 0; q < Q; q++) {
            int aidx = q*P + p, bidx = p*Q + q;
            data_tmp[bidx] = data_thisblock[2*aidx+offset];
          }
        }
        for (int n = 0;n < P*Q;n++) data_thisblock[2*n+offset] = data_tmp[n];
      }
    }
  }

  {  // Recurse.
    ComplexFftRecursive(data, nffts*P, Q, factor_begin+1, factor_end, forward, tmp_vec);
  }

  int exp_sign = (forward ? -1 : 1);
  Real rootN_re, rootN_im;  // Nth root of unity.
  ComplexImExp(static_cast<Real>(exp_sign * M_2PI / N), &rootN_re, &rootN_im);

  Real rootP_re, rootP_im;  // Pth root of unity.
  ComplexImExp(static_cast<Real>(exp_sign * M_2PI / P), &rootP_re, &rootP_im);

  {  // Do the multiplication
    // could avoid a bunch of complex multiplies by moving the loop over data_thisblock
    // inside.
    if (tmp_vec->Dim() < (MatrixIndexT)(P*2)) tmp_vec->Resize(P*2);
    Real *temp_a = tmp_vec->Data();

    Real *data_thisblock = data, *data_end = data+(N*2*nffts);
    for (; data_thisblock != data_end; data_thisblock += N*2) {  // for each separate fft.
      Real qd_re = 1.0, qd_im = 0.0;  // 1^(q'/N)
      for (int qd = 0; qd < Q; qd++) {
        Real pdQ_qd_re = qd_re, pdQ_qd_im = qd_im;  // 1^((p'Q+q') / N) == 1^((p'/P) + (q'/N))
                                              // Initialize to q'/N, corresponding to p' == 0.
        for (int pd = 0; pd < P; pd++) {  // pd == p'
          {  // This is the p = 0 case of the loop below [an optimization].
            temp_a[pd*2] = data_thisblock[qd*2];
            temp_a[pd*2 + 1] = data_thisblock[qd*2 + 1];
          }
          {  // This is the p = 1 case of the loop below [an optimization]
            // **** MOST OF THE TIME (>60% I think) gets spent here. ***
            ComplexAddProduct(pdQ_qd_re, pdQ_qd_im,
                              data_thisblock[(qd+Q)*2], data_thisblock[(qd+Q)*2 + 1],
                              &(temp_a[pd*2]), &(temp_a[pd*2 + 1]));
          }
          if (P > 2) {
            Real p_pdQ_qd_re = pdQ_qd_re, p_pdQ_qd_im = pdQ_qd_im;  // 1^(p(p'Q+q')/N)
            for (int p = 2; p < P; p++) {
              ComplexMul(pdQ_qd_re, pdQ_qd_im, &p_pdQ_qd_re, &p_pdQ_qd_im);  // p_pdQ_qd *= pdQ_qd.
              int data_idx = p*Q + qd;
              ComplexAddProduct(p_pdQ_qd_re, p_pdQ_qd_im,
                                data_thisblock[data_idx*2], data_thisblock[data_idx*2 + 1],
                                &(temp_a[pd*2]), &(temp_a[pd*2 + 1]));
            }
          }
          if (pd != P-1)
            ComplexMul(rootP_re, rootP_im, &pdQ_qd_re, &pdQ_qd_im);  // pdQ_qd *= (rootP == 1^{1/P})
          // (using 1/P == Q/N)
        }
        for (int pd = 0; pd < P; pd++) {
          data_thisblock[(pd*Q + qd)*2] = temp_a[pd*2];
          data_thisblock[(pd*Q + qd)*2 + 1] = temp_a[pd*2 + 1];
        }
        ComplexMul(rootN_re, rootN_im, &qd_re, &qd_im);  // qd *= rootN.
      }
    }
  }
}

/* Equations for ComplexFftRecursive.
   We consider here one of the "nffts" separate ffts; it's just a question of
   doing them all in parallel.  We also write all equations in terms of
   complex math (the conversion to real arithmetic is not hard, and anyway
   takes place inside function calls).


   Let the input (i.e. "data" at start) be a_n, n = 0..N-1, and
   the output (Fourier transform) be d_k, k = 0..N-1.  We use these letters because
   there will be two intermediate variables b and c.
   We want to compute:

     d_k = \sum_n a_n 1^(kn/N)                                             (1)

   where we use 1^x as shorthand for exp(-2pi x) for the forward algorithm
   and exp(2pi x) for the backward one.

   We factorize N = P Q (P small, Q usually large).
   With p = 0..P-1 and q = 0..Q-1, and also p'=0..P-1 and q'=0..P-1, we let:

    k == p'Q + q'                                                           (2)
    n == qP + p                                                             (3)

   That is, we let p, q, p', q' range over these indices and observe that this way we
   can cover all n, k.  Expanding (1) using (2) and (3), we can write:

      d_k = \sum_{p, q}  a_n 1^((p'Q+q')(qP+p)/N)
          = \sum_{p, q}  a_n 1^(p'pQ/N) 1^(q'qP/N) 1^(q'p/N)                 (4)

   using 1^(PQ/N) = 1 to get rid of the terms with PQ in them.  Rearranging (4),

     d_k =  \sum_p 1^(p'pQ/N) 1^(q'p/N)  \sum_q 1^(q'qP/N) a_n              (5)

   The point here is to separate the index q.  Now we can expand out the remaining
   instances of k and n using (2) and (3):

     d_(p'Q+q') =  \sum_p 1^(p'pQ/N) 1^(q'p/N)  \sum_q 1^(q'qP/N) a_(qP+p)   (6)

   The expression \sum_q varies with the indices p and q'.  Let us define

         C_{p, q'} =  \sum_q 1^(q'qP/N) a_(qP+p)                            (7)

   Here, C_{p, q'}, viewed as a sequence in q', is just the DFT of the points
   a_(qP+p) for q = 1..Q-1.  These points are not consecutive in memory though,
   they jump by P each time.  Let us define b as a rearranged version of a,
   so that

         b_(pQ+q) = a_(qP+p)                                                  (8)

   How to do this rearrangement in place?  In

   We can rearrange (7) to be written in terms of the b's, using (8), so that

         C_{p, q'} =  \sum_q 1^(q'q (P/N)) b_(pQ+q)                            (9)

   Here, the sequence of C_{p, q'} over q'=0..Q-1, is just the DFT of the sequence
   of b_(pQ) .. b_(p(Q+1)-1).  Let's arrange the C_{p, q'} in a single array in
   memory in the same way as the b's, i.e. we define
         c_(pQ+q') == C_{p, q'}.                                                (10)
   Note that we could have written (10) with q in place of q', as there is only
   one index of type q present, but q' is just a more natural variable name to use
   since we use q' elsewhere to subscript c and C.

   Rewriting (9), we have:
         c_(pQ+q')  = \sum_q 1^(q'q (P/N)) b_(pQ+q)                            (11)
    which is the DFT computed by the recursive call to this function [after computing
    the b's by rearranging the a's].  From the c's we want to compute the d's.
    Taking (6), substituting in the sum (7), and using (10) to write it as an array,
    we have:
         d_(p'Q+q') =  \sum_p 1^(p'pQ/N) 1^(q'p/N)  c_(pQ+q')                   (12)
    This sum is independent for different values of q'.  Note that d overwrites c
    in memory.  We compute this in  a direct way, using a little array of size P to
    store the computed d values for one value of q' (we reuse the array for each value
    of q').

    So the overall picture is this:
    We get a call to compute DFT on size N.

    - If N == 1 we return (nothing to do).
    - We factor N = P Q (typically, P is small).
    - Using (8), we rearrange the data in memory so that we have b not a in memory
       (this is the block "do the rearrangement").
       The pseudocode for this is as follows.  For simplicity we use a temporary array.

          for p = 0..P-1
             for q = 0..Q-1
                bidx = pQ + q
                aidx = qP + p
                tmp[bidx] = data[aidx].
             end
          end
          data <-- tmp
        else

        endif


        The reason this accomplishes (8) is that we want pQ+q and qP+p to be swapped
        over for each p, q, and the "if m > n" is a convenient way of ensuring that
        this swapping happens only once (otherwise it would happen twice, since pQ+q
        and qP+p both range over the entire set of numbers 0..N-1).

    - We do the DFT on the smaller block size to compute c from b (this eq eq. (11)).
      Note that this is actually multiple DFTs, one for each value of p, but this
      goes to the "nffts" argument of the function call, which we have ignored up to now.

    -We compute eq. (12) via a loop, as follows
         allocate temporary array e of size P.
         For q' = 0..Q-1:
            for p' = 0..P-1:
               set sum to zero [this will go in e[p']]
               for p = p..P-1:
                  sum += 1^(p'pQ/N) 1^(q'p/N)  c_(pQ+q')
               end
               e[p'] = sum
            end
            for p' = 0..P-1:
               d_(p'Q+q') = e[p']
            end
         end
         delete temporary array e

*/

// This is the outer-layer calling code for ComplexFftRecursive.
// It factorizes the dimension and then calls the FFT routine.
template<typename Real> void ComplexFft(VectorBase<Real> *v, bool forward, Vector<Real> *tmp_in) {
  KALDI_ASSERT(v != NULL);

  if (v->Dim()<=1) return;
  KALDI_ASSERT(v->Dim() % 2 == 0);  // complex input.
  int N = v->Dim() / 2;
  std::vector<int> factors;
  Factorize(N, &factors);
  int *factor_beg = NULL;
  if (factors.size() > 0)
    factor_beg = &(factors[0]);
  Vector<Real> tmp;  // allocated in ComplexFftRecursive.
  ComplexFftRecursive(v->Data(), 1, N, factor_beg, factor_beg+factors.size(), forward, (tmp_in?tmp_in:&tmp));
}

//! Inefficient version of Fourier transform, for testing purposes.
template<typename Real> void RealFftInefficient (VectorBase<Real> *v, bool forward) {
  KALDI_ASSERT(v != NULL);
  MatrixIndexT N = v->Dim();
  KALDI_ASSERT(N%2 == 0);
  if (N == 0) return;
  Vector<Real> vtmp(N*2);  // store as complex.
  if (forward) {
    for (MatrixIndexT i = 0; i < N; i++)  vtmp(i*2) = (*v)(i);
    ComplexFft(&vtmp, forward);  // this is already tested so we can use this.
    v->CopyFromVec( vtmp.Range(0, N) );
    (*v)(1) = vtmp(N);  // Copy the N/2'th fourier component, which is real,
    // to the imaginary part of the 1st complex output.
  } else {
    // reverse the transformation above to get the complex spectrum.
    vtmp(0) = (*v)(0);  // copy F_0 which is real
    vtmp(N) = (*v)(1);  // copy F_{N/2} which is real
    for (MatrixIndexT i = 1; i < N/2; i++) {
      // Copy i'th to i'th fourier component
      vtmp(2*i) = (*v)(2*i);
      vtmp(2*i+1) = (*v)(2*i+1);
      // Copy i'th to N-i'th, conjugated.
      vtmp(2*(N-i)) = (*v)(2*i);
      vtmp(2*(N-i)+1) = -(*v)(2*i+1);
    }
    ComplexFft(&vtmp, forward);  // actually backward since forward == false
    // Copy back real part.  Complex part should be zero.
    for (MatrixIndexT i = 0; i < N; i++)
      (*v)(i) = vtmp(i*2);
  }
}

template void RealFftInefficient (VectorBase<float> *v, bool forward);
template void RealFftInefficient (VectorBase<double> *v, bool forward);

template
void ComplexFft(VectorBase<float> *v, bool forward, Vector<float> *tmp_in);
template
void ComplexFft(VectorBase<double> *v, bool forward, Vector<double> *tmp_in);


// See the long comment below for the math behind this.
template<typename Real> void RealFft (VectorBase<Real> *v, bool forward) {
  KALDI_ASSERT(v != NULL);
  MatrixIndexT N = v->Dim(), N2 = N/2;
  KALDI_ASSERT(N%2 == 0);
  if (N == 0) return;

  if (forward) ComplexFft(v, true);

  Real *data = v->Data();
  Real rootN_re, rootN_im;  // exp(-2pi/N), forward; exp(2pi/N), backward
  int forward_sign = forward ? -1 : 1;
  ComplexImExp(static_cast<Real>(M_2PI/N *forward_sign), &rootN_re, &rootN_im);
  Real kN_re = -forward_sign, kN_im = 0.0;  // exp(-2pik/N), forward; exp(-2pik/N), backward
  // kN starts out as 1.0 for forward algorithm but -1.0 for backward.
  for (MatrixIndexT k = 1; 2*k <= N2; k++) {
    ComplexMul(rootN_re, rootN_im, &kN_re, &kN_im);

    Real Ck_re, Ck_im, Dk_re, Dk_im;
    // C_k = 1/2 (B_k + B_{N/2 - k}^*) :
    Ck_re = 0.5 * (data[2*k] + data[N - 2*k]);
    Ck_im = 0.5 * (data[2*k + 1] - data[N - 2*k + 1]);
    // re(D_k)= 1/2 (im(B_k) + im(B_{N/2-k})):
    Dk_re = 0.5 * (data[2*k + 1] + data[N - 2*k + 1]);
    // im(D_k) = -1/2 (re(B_k) - re(B_{N/2-k}))
    Dk_im =-0.5 * (data[2*k] - data[N - 2*k]);
    // A_k = C_k + 1^(k/N) D_k:
    data[2*k] = Ck_re;  // A_k <-- C_k
    data[2*k+1] = Ck_im;
    // now A_k += D_k 1^(k/N)
    ComplexAddProduct(Dk_re, Dk_im, kN_re, kN_im, &(data[2*k]), &(data[2*k+1]));

    MatrixIndexT kdash = N2 - k;
    if (kdash != k) {
      // Next we handle the index k' = N/2 - k.  This is necessary
      // to do now, to avoid invalidating data that we will later need.
      // The quantities C_{k'} and D_{k'} are just the conjugates of C_k
      // and D_k, so the equations are simple modifications of the above,
      // replacing Ck_im and Dk_im with their negatives.
      data[2*kdash] = Ck_re;  // A_k' <-- C_k'
      data[2*kdash+1] = -Ck_im;
      // now A_k' += D_k' 1^(k'/N)
      // We use 1^(k'/N) = 1^((N/2 - k) / N) = 1^(1/2) 1^(-k/N) = -1 * (1^(k/N))^*
      // so it's the same as 1^(k/N) but with the real part negated.
      ComplexAddProduct(Dk_re, -Dk_im, -kN_re, kN_im, &(data[2*kdash]), &(data[2*kdash+1]));
    }
  }

  {  // Now handle k = 0.
    // In simple terms: after the complex fft, data[0] becomes the sum of real
    // parts input[0], input[2]... and data[1] becomes the sum of imaginary
    // pats input[1], input[3]...
    // "zeroth" [A_0] is just the sum of input[0]+input[1]+input[2]..
    // and "n2th" [A_{N/2}] is input[0]-input[1]+input[2]... .
    Real zeroth = data[0] + data[1],
        n2th = data[0] - data[1];
    data[0] = zeroth;
    data[1] = n2th;
    if (!forward) {
      data[0] /= 2;
      data[1] /= 2;
    }
  }

  if (!forward) {
    ComplexFft(v, false);
    v->Scale(2.0);  // This is so we get a factor of N increase, rather than N/2 which we would
    // otherwise get from [ComplexFft, forward] + [ComplexFft, backward] in dimension N/2.
    // It's for consistency with our normal FFT convensions.
  }
}

template void RealFft (VectorBase<float> *v, bool forward);
template void RealFft (VectorBase<double> *v, bool forward);

/* Notes for real FFTs.
   We are using the same convention as above, 1^x to mean exp(-2\pi x) for the forward transform.
   Actually, in a slight abuse of notation, we use this meaning for 1^x in both the forward and
   backward cases because it's more convenient in this section.

   Suppose we have real data a[0...N-1], with N even, and want to compute its Fourier transform.
   We can make do with the first N/2 points of the transform, since the remaining ones are complex
   conjugates of the first.  We want to compute:
       for k = 0...N/2-1,
       A_k = \sum_{n = 0}^{N-1}  a_n 1^(kn/N)                 (1)

   We treat a[0..N-1] as a complex sequence of length N/2, i.e. a sequence b[0..N/2 - 1].
   Viewed as sequences of length N/2, we have:
       b = c + i d,
   where c = a_0, a_2 ... and d = a_1, a_3 ...

   We can recover the length-N/2 Fourier transforms of c and d by doing FT on b and
   then doing the equations below.  Derivation is marked by (*) in a comment below (search
   for it).  Let B, C, D be the FTs.
   We have
       C_k = 1/2 (B_k + B_{N/2 - k}^*)                                 (z0)
       D_k =-1/2i (B_k - B_{N/2 - k}^*)                                (z1)
so: re(D_k)= 1/2 (im(B_k) + im(B_{N/2-k}))                             (z2)
    im(D_k) = -1/2 (re(B_k) - re(B_{N/2-k}))                             (z3)

    To recover the FT A from C and D, we write, rearranging (1):

       A_k = \sum_{n = 0, 2, ..., N-2} a_n 1^(kn/N)
            +\sum_{n = 1, 3, ..., N-1} a_n 1^(kn/N)
           = \sum_{n = 0, 1, ..., N/2-1} a_n 1^(2kn/N)  + a_{n+1} 1^(2kn/N) 1^(k/N)
           = \sum_{n = 0, 1, ..., N/2-1} c_n 1^(2kn/N)  + d_n  1^(2kn/N) 1^(k/N)
       A_k =  C_k + 1^(k/N) D_k                                              (a0)

    This equation is valid for k = 0...N/2-1, which is the range of the sequences B_k and
    C_k.  We don't use is for k = 0, which is a special case considered below.  For
    1 < k < N/2, it's convenient to consider the pair k, k', where k' = N/2 - k.
    Remember that C_k' = C_k^ *and D_k' = D_k^* [where * is conjugation].  Also,
    1^(N/2 / N) = -1.  So we have:
       A_k' = C_k^* - 1^(k/N) D_k^*                                          (a0b)
    We do (a0) and (a0b) together.



    By symmetry this gives us the Fourier components for N/2+1, ... N, if we want
    them.  However, it doesn't give us the value for exactly k = N/2.  For k = 0 and k = N/2, it
    is easiest to argue directly about the meaning of the A_k, B_k and C_k in terms of
    sums of points.
       A_0 and A_{N/2} are both real, with A_0=\sum_n a_n, and A_1 an alternating sum
       A_1 = a_0 - a_1 + a_2 ...
     It's easy to show that
              A_0 = B_0 + C_0            (a1)
              A_{N/2} = B_0 - C_0.       (a2)
     Since B_0 and C_0 are both real, B_0 is the real coefficient of D_0 and C_0 is the
     imaginary coefficient.

     *REVERSING THE PROCESS*

     Next we want to reverse this process.  We just need to work out C_k and D_k from the
     sequence A_k.  Then we do the inverse complex fft and we get back where we started.
     For 0 and N/2, working from (a1) and (a2) above, we can see that:
          B_0 = 1/2 (A_0 + A_{N/2})                                       (y0)
          C_0 = 1/2 (A_0 + A_{N/2})                                       (y1)
     and we use
         D_0 = B_0 + i C_0
     to get the 1st complex coefficient of D.  This is exactly the same as the forward process
     except with an extra factor of 1/2.

     Consider equations (a0) and (a0b).  We want to work out C_k and D_k from A_k and A_k'.  Remember
     k' = N/2 - k.

     Write down
         A_k     =  C_k + 1^(k/N) D_k        (copying a0)
         A_k'^* =   C_k - 1^(k/N) D_k       (conjugate of a0b)
      So
             C_k =            0.5 (A_k + A_k'^*)                    (p0)
             D_k = 1^(-k/N) . 0.5 (A_k - A_k'^*)                    (p1)
      Next, we want to compute B_k and B_k' from C_k and D_k.  C.f. (z0)..(z3), and remember
      that k' = N/2-k.  We can see
      that
              B_k  = C_k + i D_k                                    (p2)
              B_k' = C_k - i D_k                                    (p3)

     We would like to make the equations (p0) ... (p3) look like the forward equations (z0), (z1),
     (a0) and (a0b) so we can reuse the code.  Define E_k = -i 1^(k/N) D_k.  Then write down (p0)..(p3).
     We have
             C_k  =            0.5 (A_k + A_k'^*)                    (p0')
             E_k  =       -0.5 i   (A_k - A_k'^*)                    (p1')
             B_k  =    C_k - 1^(-k/N) E_k                            (p2')
             B_k' =    C_k + 1^(-k/N) E_k                            (p3')
     So these are exactly the same as (z0), (z1), (a0), (a0b) except replacing 1^(k/N) with
     -1^(-k/N) .  Remember that we defined 1^x above to be exp(-2pi x/N), so the signs here
     might be opposite to what you see in the code.

     MODIFICATION: we need to take care of a factor of two.  The complex FFT we implemented
     does not divide by N in the reverse case.  So upon inversion we get larger by N/2.
     However, this is not consistent with normal FFT conventions where you get a factor of N.
     For this reason we multiply by two after the process described above.

*/


/*
   (*) [this token is referred to in a comment above].

   Notes for separating 2 real transforms from one complex one.  Note that the
   letters here (A, B, C and N) are all distinct from the same letters used in the
   place where this comment is used.
   Suppose we
   have two sequences a_n and b_n, n = 0..N-1.  We combine them into a complex
   number,
      c_n = a_n + i b_n.
   Then we take the fourier transform to get
      C_k = \sum_{n = 0}^{N-1} c_n 1^(n/N) .
   Then we use symmetry.  Define A_k and B_k as the DFTs of a and b.
   We use A_k = A_{N-k}^*, and B_k = B_{N-k}^*, since a and b are real.  Using
      C_k     = A_k    +  i B_k,
      C_{N-k} = A_k^*  +  i B_k^*
              = A_k^*  -  (i B_k)^*
   So:
      A_k     = 1/2  (C_k + C_{N-k}^*)
    i B_k     = 1/2  (C_k - C_{N-k}^*)
->    B_k     =-1/2i (C_k - C_{N-k}^*)
->  re(B_k)   = 1/2 (im(C_k) + im(C_{N-k}))
    im(B_k)   =-1/2 (re(C_k) - re(C_{N-k}))

 */

template<typename Real> void ComputeDctMatrix(Matrix<Real> *M) {
  //KALDI_ASSERT(M->NumRows() == M->NumCols());
  MatrixIndexT K = M->NumRows();
  MatrixIndexT N = M->NumCols();

  KALDI_ASSERT(K > 0);
  KALDI_ASSERT(N > 0);
  Real normalizer = std::sqrt(1.0 / static_cast<Real>(N));  // normalizer for
  // X_0.
  for (MatrixIndexT j = 0; j < N; j++) (*M)(0, j) = normalizer;
  normalizer = std::sqrt(2.0 / static_cast<Real>(N));  // normalizer for other
   // elements.
  for (MatrixIndexT k = 1; k < K; k++)
    for (MatrixIndexT n = 0; n < N; n++)
      (*M)(k, n) = normalizer
          * std::cos( static_cast<double>(M_PI)/N * (n + 0.5) * k );
}


template void ComputeDctMatrix(Matrix<float> *M);
template void ComputeDctMatrix(Matrix<double> *M);


template<typename Real>
void ComputePca(const MatrixBase<Real> &X,
                MatrixBase<Real> *U,
                MatrixBase<Real> *A,
                bool print_eigs,
                bool exact) {
  // Note that some of these matrices may be transposed w.r.t. the
  // way it's most natural to describe them in math... it's the rows
  // of X and U that correspond to the (data-points, basis elements).
  MatrixIndexT N = X.NumRows(), D = X.NumCols();
  // N = #points, D = feature dim.
  KALDI_ASSERT(U != NULL && U->NumCols() == D);
  MatrixIndexT G = U->NumRows();  // # of retained basis elements.
  KALDI_ASSERT(A == NULL || (A->NumRows() == N && A->NumCols() == G));
  KALDI_ASSERT(G <= N && G <= D);
  if (D < N) {  // Do conventional PCA.
    SpMatrix<Real> Msp(D);  // Matrix of outer products.
    Msp.AddMat2(1.0, X, kTrans, 0.0);  // M <-- X^T X
    Matrix<Real> Utmp;
    Vector<Real> l;
    if (exact) {
      Utmp.Resize(D, D);
      l.Resize(D);
      //Matrix<Real> M(Msp);
      //M.DestructiveSvd(&l, &Utmp, NULL);
      Msp.Eig(&l, &Utmp);
    } else {
      Utmp.Resize(D, G);
      l.Resize(G);
      Msp.TopEigs(&l, &Utmp);
    }
    SortSvd(&l, &Utmp);

    for (MatrixIndexT g = 0; g < G; g++)
      U->Row(g).CopyColFromMat(Utmp, g);
    if (print_eigs)
      KALDI_LOG << (exact ? "" : "Retained ")
                << "PCA eigenvalues are " << l;
    if (A != NULL)
      A->AddMatMat(1.0, X, kNoTrans, *U, kTrans, 0.0);
  } else {  // Do inner-product PCA.
    SpMatrix<Real> Nsp(N);  // Matrix of inner products.
    Nsp.AddMat2(1.0, X, kNoTrans, 0.0);  // M <-- X X^T

    Matrix<Real> Vtmp;
    Vector<Real> l;
    if (exact) {
      Vtmp.Resize(N, N);
      l.Resize(N);
      Matrix<Real> Nmat(Nsp);
      Nmat.DestructiveSvd(&l, &Vtmp, NULL);
    } else {
      Vtmp.Resize(N, G);
      l.Resize(G);
      Nsp.TopEigs(&l, &Vtmp);
    }

    MatrixIndexT num_zeroed = 0;
    for (MatrixIndexT g = 0; g < G; g++) {
      if (l(g) < 0.0) {
        KALDI_WARN << "In PCA, setting element " << l(g) << " to zero.";
        l(g) = 0.0;
        num_zeroed++;
      }
    }
    SortSvd(&l, &Vtmp); // Make sure zero elements are last, this
    // is necessary for Orthogonalize() to work properly later.

    Vtmp.Transpose();  // So eigenvalues are the rows.

    for (MatrixIndexT g = 0; g < G; g++) {
      Real sqrtlg = sqrt(l(g));
      if (l(g) != 0.0) {
        U->Row(g).AddMatVec(1.0 / sqrtlg, X, kTrans, Vtmp.Row(g), 0.0);
      } else {
        U->Row(g).SetZero();
        (*U)(g, g) = 1.0;  // arbitrary direction.  Will later orthogonalize.
      }
      if (A != NULL)
        for (MatrixIndexT n = 0; n < N; n++)
          (*A)(n, g) = sqrtlg * Vtmp(g, n);
    }
    // Now orthogonalize.  This is mainly useful in
    // case there were zero eigenvalues, but we do it
    // for all of them.
    U->OrthogonalizeRows();
    if (print_eigs)
      KALDI_LOG << "(inner-product) PCA eigenvalues are " << l;
  }
}


template
void ComputePca(const MatrixBase<float> &X,
                MatrixBase<float> *U,
                MatrixBase<float> *A,
                bool print_eigs,
                bool exact);

template
void ComputePca(const MatrixBase<double> &X,
                MatrixBase<double> *U,
                MatrixBase<double> *A,
                bool print_eigs,
                bool exact);


// Added by Dan, Feb. 13 2012.
// This function does: *plus += max(0, a b^T),
// *minus += max(0, -(a b^T)).
template<typename Real>
void AddOuterProductPlusMinus(Real alpha,
                              const VectorBase<Real> &a,
                              const VectorBase<Real> &b,
                              MatrixBase<Real> *plus,
                              MatrixBase<Real> *minus) {
  KALDI_ASSERT(a.Dim() == plus->NumRows() && b.Dim() == plus->NumCols()
               && a.Dim() == minus->NumRows() && b.Dim() == minus->NumCols());
  int32 nrows = a.Dim(), ncols = b.Dim(), pskip = plus->Stride() - ncols,
      mskip = minus->Stride() - ncols;
  const Real *adata = a.Data(), *bdata = b.Data();
  Real *plusdata = plus->Data(), *minusdata = minus->Data();

  for (int32 i = 0; i < nrows; i++) {
    const Real *btmp = bdata;
    Real multiple = alpha * *adata;
    if (multiple > 0.0) {
      for (int32 j = 0; j < ncols; j++, plusdata++, minusdata++, btmp++) {
        if (*btmp > 0.0) *plusdata += multiple * *btmp;
        else *minusdata -= multiple * *btmp;
      }
    } else {
      for (int32 j = 0; j < ncols; j++, plusdata++, minusdata++, btmp++) {
        if (*btmp < 0.0) *plusdata += multiple * *btmp;
        else *minusdata -= multiple * *btmp;
      }
    }
    plusdata += pskip;
    minusdata += mskip;
    adata++;
  }
}

// Instantiate template
template
void AddOuterProductPlusMinus<float>(float alpha,
                                     const VectorBase<float> &a,
                                     const VectorBase<float> &b,
                                     MatrixBase<float> *plus,
                                     MatrixBase<float> *minus);
template
void AddOuterProductPlusMinus<double>(double alpha,
                                      const VectorBase<double> &a,
                                      const VectorBase<double> &b,
                                      MatrixBase<double> *plus,
                                      MatrixBase<double> *minus);


} // end namespace kaldi

```

</details>




---

#### [LICENSE](#CIGLET)<a name = "ciglet_license"></a>

<details><summary>Apache License 2.0</summary>
<pre>

Update to legal notice, made Feb 2012, modified Sep 2013.  We would like to
 clarify that we are using a convention where multiple names in the Apache
 copyright headers, for example

  // Copyright 2009-2012  Yanmin Qian  Arnab Ghoshal
  //                2013  Vassil Panayotov

 does not signify joint ownership of copyright of that file, except in cases
 where all those names were present in the original release made in March 2011--
 you can use the version history to work this out, if this matters to you.
 Instead, we intend that those contributors who later modified the file, agree
 to release their changes under the Apache license.  The conventional way of
 signifying this is to duplicate the Apache headers at the top of each file each
 time a change is made by a different author, but this would quickly become
 impractical.

 Where the copyright header says something like:

 // Copyright    2013   Johns Hopkins University (author: Daniel Povey)

 it is because the individual who wrote the code was at that institution as an
 employee, so the copyright is owned by the university (and we will have checked
 that the contributions were in accordance with the open-source policies of the
 institutions concerned, including getting them vetted individually where
 necessary).  From a legal point of view the copyright ownership is that of the
 institution concerned, and the (author: xxx) in parentheses is just
 informational, to identify the actual person who wrote the code, and is not
 intended to have any legal implications.  In some cases, however, particularly
 early on, we just wrote the name of the university or company concerned,
 without the actual author's name in parentheses.  If you see something like

 //  Copyright  2009-2012   Arnab Ghoshal  Microsoft Corporation

 it does not imply that Arnab was working for Microsoft, it is because someone
 else contributed to the file while working at Microsoft (this would be Daniel
 Povey, in fact, who was working at Microsoft Research at the outset of the
 project).

 The list of authors of each file is in an essentially arbitrary order, but is
 often chronological if they contributed in different years.

 The original legal notice is below.  Note: we are continuing to modify it by
 adding the names of new contributors, but at any given time, the list may
 be out of date.

---
                          Legal Notices

Each of the files comprising Kaldi v1.0 have been separately licensed by
their respective author(s) under the terms of the Apache License v 2.0 (set
forth below).  The source code headers for each file specifies the individual
authors and source material for that file as well the corresponding copyright
notice.  For reference purposes only: A cumulative list of all individual
contributors and original source material as well as the full text of the Apache
License v 2.0 are set forth below.

Individual Contributors (in alphabetical order)

      Mohit Agarwal
      Tanel Alumae
      Gilles Boulianne
      Lukas Burget
      Dogan Can
      Guoguo Chen
      Gaofeng Cheng
      Cisco Corporation
      Pavel Denisov
      Ilya Edrenkin
      Ewald Enzinger
      Joachim Fainberg
      Daniel Galvez
      Pegah Ghahremani
      Arnab Ghoshal
      Ondrej Glembek
      Go Vivace Inc.
      Allen Guo
      Hossein Hadian
      Lv Hang
      Mirko Hannemann
      Hendy Irawan
      Navdeep Jaitly
      Johns Hopkins University
      Shiyin Kang
      Kirill Katsnelson
      Tom Ko
      Danijel Korzinek
      Gaurav Kumar
      Ke Li
      Matthew Maciejewski
      Vimal Manohar
      Yajie Miao
      Microsoft Corporation
      Petr Motlicek
      Xingyu Na
      Vincent Nguyen
      Lucas Ondel
      Vassil Panayotov
      Vijayaditya Peddinti
      Phonexia s.r.o.
      Ondrej Platek
      Daniel Povey
      Yanmin Qian
      Ariya Rastrow
      Saarland University
      Omid Sadjadi
      Petr Schwarz
      Yiwen Shao
      Nickolay V. Shmyrev
      Jan Silovsky
      Eduardo Silva
      Peter Smit
      David Snyder
      Alexander Solovets
      Georg Stemmer
      Pawel Swietojanski
      Jan "Yenda" Trmal
      Albert Vernon
      Karel Vesely
      Yiming Wang
      Shinji Watanabe
      Minhua Wu
      Haihua Xu
      Hainan Xu
      Xiaohui Zhang

Other Source Material

    This project includes a port and modification of materials from JAMA: A Java
  Matrix Package under the following notice: "This software is a cooperative
  product of The MathWorks and the National Institute of Standards and Technology
  (NIST) which has been released to the public domain." This notice and the
  original code is available at http://math.nist.gov/javanumerics/jama/

   This project includes a modified version of code published in Malvar, H.,
  "Signal processing with lapped transforms," Artech House, Inc., 1992.  The
  current copyright holder, Henrique S. Malvar, has given his permission for the
  release of this modified version under the Apache License 2.0.

  This project includes material from the OpenFST Library v1.2.7 available at
  http://www.openfst.org and released under the Apache License v. 2.0.

  [OpenFst COPYING file begins here]

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use these files except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    Copyright 2005-2010 Google, Inc.

  [OpenFst COPYING file ends here]


 -------------------------------------------------------------------------

                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
limitations under the License.

</pre>

</details>

---
