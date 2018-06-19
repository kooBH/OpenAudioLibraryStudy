# [BeamformIt](./openAudioLibs.md)<a name = "TOP"></a>
1. [Dependency](#BeamformIt_dependency)
1. [FFT](#BeamformIt_fft)
1. [NoiseFilter](#BeamformIt_noise)
1. [Citation](#BeamformIt_citation)



## [Dependency](#TOP)<a name = "BeamformIt_dependency"></a>
+ [Libsndfile](http://www.mega-nerd.com/libsndfile/) is a C library for reading and writing files containing sampled sound (such as MS Windows WAV and the Apple/SGI AIFF format) through one standard library interface  

```bash
$ sudo apt-get install libsndfile1-dev
```


## [FFT](#TOP)<a name = "BeamformIt_fft"></a>
Using external fft library "[FFTReal](http://ldesoras.free.fr/prod.html#src_fftreal)"    

FFTReal is a library to compute Discrete Fourier Transforms (DFT) with the  
FFT algorithm (Fast Fourier Transform) on arrays of real numbers. It can  
also compute the inverse transform.  

You should find in this package a lot of files ; some of them are of  
particular interest:  
- readme.txt          : you are reading it 
- ffft/FFTReal.h      : FFT, length fixed at run-time  
- ffft/FFTRealFixLen.h: FFT, length fixed at compile-time  
- delphi/FFTReal.pas : Pascal implementation (working but not up-to-date)  

## [NoiseFilter](#TOP)<a name = "BeamformIt_noise"></a>

<details><summary>noise_filter</summary>

```C++ 
/*!
  Applies a noise filter based on thresholding the xcorr values to avoid using non reliable TDOa data
  \param threshold Threshold to be applied, from 0 to 1
*/

void TDOA::noise_filter(float threshold)
{
  //apply a minimum value for the xcorrelation. Assign the previous delay in case it doesn't pass

  for(int channel_count=0; channel_count < m_numCh; channel_count++)
  {
    //go through all computed delays
    for(int count=0; count<(int)(m_chDelays[channel_count].size()); count++)
    {
	  //check wether the xcorr value for the best match is below the threshold
	  // when it is, I propagate the previous values forward, smoothing the delays
      if(m_chXcorrValues[channel_count][count][0] < threshold)
      {
		//printf("Channel %d Frame %d, Xcorr %f is below threshold %f\n", channel_count, count, m_chXcorrValues[channel_count][count][0], threshold);
        //enable the flag saying that a noise threshold hs been applied
        m_delayFilters[channel_count][count] += F_NOISE;
	      
        // I don't repeat the value as I let the viterbi handle the situation
        if(count>0)
        {
          //I just copy the delay value, but not the xcorr value
          m_chDelays[channel_count][count] = m_chDelays[channel_count][count-1];
        }
        else
        {
          //if the first delay is silence, we turn it to 0;
          m_chDelays[channel_count][count].assign(m_chDelays[channel_count][count].size(), (*m_config).marginFrames); //a big discouraging value (the max we allow as delay)
          m_chDelays[channel_count][count][0]=0;
          m_chXcorrValues[channel_count][count].assign(m_chXcorrValues[channel_count][count].size(),0);
          m_chXcorrValues[channel_count][count][0]=1;
        }     
      }
	//printf("Channel %d frame %d: %d %d %d %d\n", channel_count, count, m_chDelays[channel_count][count][0], m_chDelays[channel_count][count][1], m_chDelays[channel_count][count][2], m_chDelays[channel_count][count][3]);
    }
  }
}

```

</details>

## [Citation](#TOP)<a name = "BeamformIt_citation"></a>     
http://www.xavieranguera.com/beamformit/  
" If you use the software for research I would very much appreciate if you could cite my work, you can use any of the following citations "
+ "Acoustic beamforming for speaker diarization of meetings", Xavier Anguera, Chuck Wooters and Javier Hernando, IEEE Transactions on Audio, Speech and Language Processing, September 2007, volume 15, number 7, pp.2011-2023.  
+ "Robust Speaker Diarization for Meetings", Xavier Anguera, PhD Thesis, UPC Barcelona, 2006.   






