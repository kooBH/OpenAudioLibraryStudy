
### [Hidden Markov Model Toolkit (HTK)](./openAudioLibs.md#TOP)<a name="HTK"></a>
+ [function list](#HTK_list)
+ [license](#HTK_license)

|Header|Discription||
|---|---|---|
|HAdapt.h | Adaptation Library module   ||
|HArc.h | Routines used in HFwdBkwdLat.c,  An alternative kind of lattice format used there   ||
|HAudio.h | Audio Input/Output  |O|
|HDic.h| Dictionary Storage  ||
|HEaxctMPE.h |MPE implementation (exact)  ||
|HFB.h | forward Backward Routines module  |
|HFBLat.h | Lattice Forward Backward routines  ||
|HGraf.h | Minimal Graphics Interface   ||
|HLabel.h | Speech Label File Input   |O|
|HLM.h | language model handling   ||
|HMap.c | MAP Model Updates   ||
|HMath.h | Math Support  |O|
|HMem.h | Memory Management Module  ||
|HModel.h | HMM Model Definition Data Type  ||
|HNet.h | Network and Lattice Functions  ||
|HParm.h | Speech Parameter Input/Output   |O|
|HRec.h | Viterbi Recognition Engine Library   ||
|HShell.h | Interface to the Shell  ||
|HSigP.h | Signal Processing Routines  |O|
|HTrain.h | HMM Training Support Routines  ||
|HUtil.h | HMM utility routines  ||
|HVQ.h | Vector Quantisation  |O|
|HWave.h | Speech Waveform File Input|O|



---
#### [FUNCTION LIST](#HTK)<a name="HTK_list"></a>
1. [HAudio](#HTK_list_HAudio)
2. [HLabel](#HTK_list_HLabel)
2. [HMath](#HTK_list_HMath)
4. [HParm](#HTK_list_HParm)
3. [HSigP](#HTK_list_HSigP)  
4. [HVQ](#HTK_list_HVQ)  
5. [HWave](#HTK_list_HWave)  

---
+ <a name="HTK_list_HAudio">[HAudio](#HTK_list)</a> 
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
 
+ <a name = "HTK_list_HLabel">[HLabel](#HTK_list)</a> | 
    1. **Label/Name Handling**    
        InitLabel    
        LabId GetLabId  
        PrintNameTabStats  
        ReadLabel  
    2. **Transcription and Label List Handling**  
        CreateTranscription  
        CopyTranscription  
        void PrintTranscription  
        CreateLabelList  
        void AddLabelList  
        GetLabelList  
        CopyLabelList  
        CreateLabel  
        AddLabel  
        AddAuxLab  
        DeleteLabel  
        NumCases  
        NumAuxCases  
        GetCase  
        GetAuxCase  
        GetLabN  
        GetAuxLabN  
        CountLabs  
        CountAuxLabs  
        AuxLabEndTime
    3. **Label File Opening/Closing**  
        LOpen  
        SaveToMasterfile  
        CloseMLFSaveFile  
        LSave  
    4. **TriPhone Stripping**  
        TriStrip  
        LTriStrip  
    5. **Master Label File Handling**  
        LoadMasterFile  
        NumMLFFiles  
        NumMLFEntries    
        GetMLFFile  
        IsMLFFile  
        GetMLFTable  

 
 
+ <a name= "HTK_list_HMath">[HMath](#HTK_list)</a> 
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

+ <a name = "HTK_list_HParm">[HParm](#HTK_list)</a> 
    1. **Initialisation**  
        InitParm  
    2. **Channel functions**   
        SetChannel   
        ResetChannelSession  
        SetNewConfig    
        ResetCurCepMean    
    3. **Buffer Input Routines**   
        OpenBuffer   
        BufferStatus   
        ObsInBuffer   
        StartBuffer   
        StopBuffer   
        CloseBuffer   
        ReadAsBuffer   
        ReadAsTable   
        GetBufferInfo   
    4. **External Data Source Handling**  
        CreateSrcExt  
        OpenExtBuffer  
    5. **New Buffer Creation Routines**  
        EmptyBuffer    
        SaveBuffer  
        AddToBuffer  
    6. **Observation Handling Routines**   
        MakeObservation  
        ExplainObservation  
        PrintObservation  
        ZeroStreamWidths  
        SetStreamWidths  
        SetParmHMMSet    
    7. **Parameter Kind Conversions**  
        ParmKind2Str  
        Str2ParmKind  
        BaseParmKind  
        HasEnergy  
        HasDelta  
        HasNulle   
        HasAccs   
        HasThird   
        HasCompx   
        HasCrcc   
        HasZerom   
        HasZeroc   
        HasVQ   
        ValidConversion   

+ <a name="HTK_list_HSigP">[HSigP](#HTK_list)</a> 
  1. **Speech Signal Processing Operations**  
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

+ <a name = "HTK_list_HVQ">[HVQ](#HTK_list)</a> 
        InitVQ  
        CreateVQTab  
        CreateVQNode  
        LoadVQTab  
        StoreVQTab  
        PrintVQTab  
        VQNodeScore  
        GetVQ  

+ <a name = "HTK_list_HWave">[HWave](#HTK_list)</a> 
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


