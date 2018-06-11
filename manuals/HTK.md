
### [Hidden Markov Model Toolkit (HTK)](./openAudioLibs.md#TOP)<a name="HTK"></a>
+ [function list](#HTK_list)
+ [function prototype](#HTK_proto)
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
 
+ <a name = "HTK_list_HLabel">[HLabel](#HTK_list)</a> | [Proto](#HLabel)
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

+ <a name = "HTK_list_HParm">[HParm](#HTK_list)</a> | [Proto](#HParm)
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

+ <a name="HTK_list_HSigP">[HSigP](#HTK_list)</a> | [Proto](#HSigP)
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

+ <a name = "HTK_list_HVQ">[HVQ](#HTK_list)</a> | [Proto](#HVQ)  
		InitVQ  
		CreateVQTab  
		CreateVQNode  
		LoadVQTab  
		StoreVQTab  
		PrintVQTab  
		VQNodeScore  
		GetVQ  

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
2. [HLabel](#HLabel)
2. [HMath.h](#HMath)  
4. [HParm](#HParm)
3. [HSigP.h](#HSigP)
4. [HVQ.h](#HVQ)
5. [HWave.h](#HWave)

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

+ ### [HLabel.h](#HTK_proto)<a name = "HLabel"></a>

```C++
/* 
   This module allows an internal data structure representing a
   transcription to be constructed by reading labels from an external
   file.  Each transcription consists of a list of label lists where
   each label list represents one distinct interpretation of the
   input.  Each time segment in a label list has a primary label.  It
   can also have one or more auxiliary labels attached to it.  In 
   this case, each additional label corresponds to a different
   level of representation.
   
   A number of single alternative/single level input formats are
   supported but multiple alternatives and multiple levels are limited
   to the HTK format.
   
   The module also provides the internal data structures needed to
   support master label files and the hash table for all string to
   LabId mapping.
*/

/*  Configuration Parameters:
   SOURCELABEL    - format of input label files
   TARGETLABEL    - format of output label files
   STRIPTRIPHONES - strip contexts from triphones
   TRANSALT       - filter alt lab list on read in
   TRANSLEV       - filter level on read in
   TRACE          - trace level
*/

typedef struct _NameCell{  /* Hash Table Linked List Item */
   char * name;             /* Label Name */
   Ptr aux;                 /* User pointer */
   struct _NameCell *next;  /* Chain */
}NameCell;

typedef NameCell *LabId;   /* Internal representation of names */
  
typedef struct _Label *LLink;
typedef struct _Label{     /* Information for each label */
   LabId labid;             /* primary label id */
   float score;             /* primary score eg. logP */
   LabId *auxLab;           /* array [1..maxAuxLab] OF LabId */
   float *auxScore;         /* array [1..maxAuxLab] OF float */
   HTime start,end;         /* Start and end times in 100ns units */
   LLink succ;              /* Successor label */
   LLink pred;              /* Preceding label */
}Label; /* NB: head and tail of every list are dummy sentinels */

typedef struct _LabList{
   LLink head;              /* Pointer to head of List */
   LLink tail;              /* Pointer to tail of List */
   struct _LabList *next;   /* Next label list */
   int maxAuxLab;           /* max aux labels (default=0) */
}LabList;

typedef struct {
   LabList *head;          /* Pointer to head of Label List */
   LabList *tail;          /* Pointer to tail of Label List */
   int numLists;           /* num label lists (default=1) */
}Transcription;

enum _MLFDefType {
   MLF_IMMEDIATE, MLF_SIMPLE, MLF_FULL
};
typedef enum _MLFDefType MLFDefType;

enum _MLFPatType {
   PAT_FIXED,     /* whole "pattern" is hashed */
   PAT_ANYPATH,   /* pat is "* / name" and name is hashed */
   PAT_GENERAL    /* general pattern - no hashing */
};
typedef enum _MLFPatType MLFPatType;

typedef struct{
   int fidx;      /* MLF file index in mlfile */
   long offset;   /* offset into MLF file */
}ImmDef;

typedef union{
   char *subdir;     /* Sub-directory to search for MLF_SIMPLE & MLF_FULL */
   ImmDef immed;     /* Immediate Definition for MLF_IMMEDIATE */
}MLFDef;

typedef struct _MLFEntry{
   char *pattern;       /* pattern to match for this definition */
   MLFPatType patType;  /* type of pattern */
   unsigned patHash;    /* hash of pattern if not general */
   MLFDefType type;     /* type of this definition */
   MLFDef def;          /* the actual def */
   struct _MLFEntry *next;    /* next in chain */
}MLFEntry;

/* ------------------- Label/Name Handling ------------------- */

void InitLabel(void);
/*
   Initialises hash table - must be called before using any of
   the routines in this module.
*/

LabId GetLabId(char *name, Boolean insert);
/*
   Lookup given name in hash table and return its id.  If it
   is not there and insert is true then insert the new name
   otherwise return NULL.
*/

void PrintNameTabStats(void);
/* 
   Print out statistics on hash table usage: string heap,
   name cell heap and average search length
*/

Boolean ReadLabel(FILE *f, char *buf);
/*
   Read the next label from f into buf.  A label is any
   sequence of printing chars.  Returns false if no label
   found. Skips white space and puts terminator back into
   f.
*/

/* ---------- Transcription and Label List Handling --------------- */

/* 
   MemHeap must be a STAK or a CHEAP.  For stack case, deallocation
   can be by Disposing back to pointer to transcription (normally the 
   first object allocated) or by reseting the heap
*/

Transcription *CreateTranscription(MemHeap *x);
/*
   Create a transcription with no label lists.
*/

Transcription *CopyTranscription(MemHeap *x, Transcription *t);
/*
   Return a copy of transcription t, allocated in x.
*/

void PrintTranscription(Transcription *t, char *title);
/*
   Print transcription t (for debugging/diagnostics)
*/

LabList* CreateLabelList(MemHeap *x, int maxAuxLab);
/* 
   Create and return a new label list with upto maxAuxLab 
   alternative labels.  This will have sentinel labels
   with NULL labid fields at the head and tail.
*/

void AddLabelList(LabList *ll, Transcription *t);
/* 
   Add given label list to transcription t. 
*/

LabList* GetLabelList(Transcription *t, int n);
/*
   Return pointer to n'th label list from transcription t indexed
   1 .. numLists
*/

LabList* CopyLabelList(MemHeap *x, LabList* ll);
/*
   Return a copy of given label list, allocated in x.
*/

LLink CreateLabel(MemHeap *x, int maxAux);
/*
   create a label with maxAux auxiliary slots 
*/

LLink AddLabel(MemHeap *x, LabList *ll, LabId id,
               HTime st, HTime en, float score);
/*
   Append a new item to end of given list and store the given info.
   Return a pointer to the newly created label item
*/

void AddAuxLab(LLink lab, int n, LabId *auxLab, float *auxScore);
/*
   Store n auxiliary label/score in lab
*/

void DeleteLabel(LLink item);
/*
   Unlink given item from a label list 
*/

int NumCases(LabList *ll, LabId id);
int NumAuxCases(LabList *ll, LabId id, int i);
/* 
   find number of cases of primary label/ i'th auxiliary
   label in given label list 
*/

LLink GetCase(LabList *ll, LabId id, int n);
LLink GetAuxCase(LabList *ll, LabId id, int n, int i);
/*
   return the nth occurrence of given primary label/ i'th auxiliary
   label in given label list 
*/

LLink GetLabN(LabList *ll, int n);
LLink GetAuxLabN(LabList *ll, int n, int i);
/* 
   return n'th primary label / i'th auxiliary label in given label list
*/

int CountLabs(LabList *ll);
int CountAuxLabs(LabList *ll, int i);
/*
   return number of primary labels / i'th auxiliary labels in
   given label list
*/

HTime AuxLabEndTime(LLink p, int i);
/* 
   return the end time for the i'th aux lab in label p.  This will be 
   the end time of the label before the next one containing an aux lab i
   or the end of the last label, whichever comes first
*/

/* ------------------ Label File Opening/Closing -------------------- */

Transcription *LOpen(MemHeap *x, char *fname, FileFormat fmt);
/*
   Reads the labels stored in file fname and build the internal
   representation in a transcription allocated in x.  If a Master
   Label File has been loaded, then that file is searched before
   looking for fname directly.  If fmt is UNDEFF then source format
   will be SOURCEFORMAT if set else source format will be HTK.
   If TRANSALT is set to N then all but the N'th alternative is
   discarded on read in.  If TRANSLEV is set to L then all but the 
   L'th level is discarded on read in.
*/

ReturnStatus SaveToMasterfile(char *fname);
/*
   Once called, all subsequent LSave's go to the MLF file fname
   rather than to individual files.  Format must be HTK.
*/

void CloseMLFSaveFile(void);
/*
   Close the MLF output file.  Any subsequent LSave's return to
   normal behaviour.
*/

ReturnStatus LSave(char *fname, Transcription *t, FileFormat fmt);
/* 
   Save the given transcription in file fname.  If fmt is UNDEFF then
   target format will be TARGETFORMAT if set else target format will
   be HTK.
*/

/* -------------------- TriPhone Stripping -------------------- */

void TriStrip(char *s);
/* 
   Remove contexts of form A- and +B from s 
*/

void LTriStrip(Boolean enab);
/*
   When enab is set, all triphone labels with the form A-B+C
   are converted to B on read in by LOpen.
*/

/* ------------------ Master Label File Handling -------------------- */

void LoadMasterFile(char *fname);
/*
   Load the Master Label File stored in fname
*/

int NumMLFFiles(void);
int NumMLFEntries(void);
/*
   Return the number of loaded MLF files and the total number of
   entries in the MLF table
*/

FILE *GetMLFFile(int fidx);
/*
   Return the fidx'th loaded MLF file.  Index base is 0.
*/

Boolean IsMLFFile(char *fn);
/* 
   Return true if fn is an MLF file
*/

MLFEntry *GetMLFTable(void);
/*
   Return the first entry in the MLF table
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

+ ### [HParm.h](#HTK_proto)<a name = "HParm"></a>

```C++
enum _BaseParmKind{
      WAVEFORM,            /* Raw speech waveform (handled by HWave) */
      LPC,LPREFC,LPCEPSTRA,LPDELCEP,   /* LP-based Coefficients */
      IREFC,                           /* Ref Coef in 16 bit form */
      MFCC,                            /* Mel-Freq Cepstra */
      FBANK,                           /* Log Filter Bank */
      MELSPEC,                         /* Mel-Freq Spectrum (Linear) */
      USER,                            /* Arbitrary user specified data */
      DISCRETE,                        /* Discrete VQ symbols (shorts) */
      PLP,                             /* Standard PLP coefficients */
      ANON};
      
typedef short ParmKind;          /* BaseParmKind + Qualifiers */
                                 
#define HASENERGY  0100       /* _E log energy included */
#define HASNULLE   0200       /* _N absolute energy suppressed */
#define HASDELTA   0400       /* _D delta coef appended */
#define HASACCS   01000       /* _A acceleration coefs appended */
#define HASCOMPX  02000       /* _C is compressed */
#define HASZEROM  04000       /* _Z zero meaned */
#define HASCRCC  010000       /* _K has CRC check */
#define HASZEROC 020000       /* _0 0'th Cepstra included */
#define HASVQ    040000       /* _V has VQ index attached */
#define HASTHIRD 0100000       /* _T has Delta-Delta-Delta index attached */

#define BASEMASK  077         /* Mask to remove qualifiers */

/*
   An observation contains one or more stream values each of which 
   is either a vector of continuous values and/or a single
   discrete symbol.  The discrete vq symbol is included if the
   target kind is DISCRETE or the continuous parameter has the
   HASVQ qualifier. Observations are input via buffers or tables.  A
   buffer is a FIFO structure of potentially infinite length and it is
   always sourced via HAudio.  A table is a random access array of
   observations and it is sourced from a file possibly via HWave.
   Buffers are input only, a table can be input and output.
   Too allow discrete systems to be used directly from continuous
   data the observation also holds a separate parm kind for the
   parm buffer and routines which supply observations use this to
   determine stream widths when the observation kind is DISCRETE.
*/

typedef enum {
   FALSE_dup=FALSE, /*  0 */
   TRUE_dup=TRUE,   /*  1 */
   TRI_UNDEF=-1     /* -1 */
}
TriState;

typedef struct {
   Boolean eSep;         /* Energy is in separate stream */
   short swidth[SMAX];   /* [0]=num streams,[i]=width of stream i */
   ParmKind bk;          /* parm kind of the parm buffer */
   ParmKind pk;          /* parm kind of this obs (bk or DISCRETE) */
   short vq[SMAX];       /* array[1..swidth[0]] of VQ index */
   Vector fv[SMAX];      /* array[1..swidth[0]] of Vector */
} Observation;

/*
   A ParmBuf holds either a static table of parameter frames
   loaded from a file or a potentially infinite sequence
   of frames from an audio source. The key information relating 
   to the speech data in a buffer or table can be obtained via 
   a BufferInfo Record.  A static table behaves like a stopped
   buffer.
*/

typedef enum { 
   PB_INIT,     /* Buffer is initialised and empty */
   PB_WAITING,  /* Buffer is waiting for speech */
   PB_STOPPING, /* Buffer is waiting for silence */
   PB_FILLING,  /* Buffer is filling */
   PB_STOPPED,  /* Buffer has stopped but not yet empty */
   PB_CLEARED   /* Buffer has been emptied */
} PBStatus;

typedef struct _ParmBuf  *ParmBuf;

typedef struct {
   ParmKind srcPK;            /* Source ParmKind */ 
   FileFormat srcFF;          /* Source File format */ 
   HTime srcSampRate;         /* Source Sample Rate */ 
   int srcVecSize;            /* Size of source vector */
   long nSamples;             /* Number of source samples */
   int frSize;                /* Number of source samples in each frame */
   int frRate;                /* Number of source samples forward each frame */
   int nObs;                  /* Number of table observations */
   ParmKind tgtPK;            /* Target ParmKind */ 
   FileFormat tgtFF;          /* Target File format */ 
   HTime tgtSampRate;         /* Target Sample Rate */ 
   int tgtVecSize;            /* Size of target vector */
   AudioIn a;                 /* the audio source - if any */
   Wave w;                    /* the wave input - if any */
   Ptr i;                     /* the other input - if any */
   Boolean useSilDet;         /* Use Silence Detector */
   int audSignal;             /* Signal Number for Audio Control */
   char *vqTabFN;             /* Name of VQ Table Defn File */
   Boolean saveCompressed;    /* Save in compressed format */
   Boolean saveWithCRC;       /* Save with CRC check added */
   Boolean spDetParmsSet;     /* Parameters set for sp/sil detector */
   float spDetSil;            /* Silence level for channel */
   float chPeak;              /* Peak-to-peak input level for channel */
   float spDetSp;             /* Speech level for channel */
   float spDetSNR;            /* Speech/noise ratio for channel */
   float spDetThresh;         /* Silence/speech level threshold */
   float curVol;              /* Volume level of last frame (0.0-100.0dB) */
   int spDetSt;               /* Frame number of first frame of buffer */
   int spDetEn;               /* Frame number of last frame of buffer */
   char *matTranFN;           /* Matrix transformation name */
   Ptr xform;                 /* Used for input xform associated with this buffer */
}BufferInfo;

/*
   External source definition structure
*/

typedef struct hparmsrcdef *HParmSrcDef;

/* -------------------- Initialisation ------------------ */

ReturnStatus InitParm(void);
/*
   Initialise the module
*/

/* -------------------- Channel functions ------------------ */

ReturnStatus SetChannel(char *chanName);
/* 
   Set the current channel to use config parameters from chanName.
*/

void ResetChannelSession(char *chanName);
/* 
   Reset the session for the specified channel (NULL indicates default)
*/

/* 
   The next two functions have been kept to allow for backwards 
   compatibility.
*/
void SetNewConfig(char * libmod);
void ResetCurCepMean(void);

/* ---------------- Buffer Input Routines ------------------ */

ParmBuf OpenBuffer(MemHeap *x, char *fn, int maxObs, FileFormat ff, 
		   TriState enSpeechDet, TriState silMeasure);
/*
   Open and return a ParmBuf object connected to the current channel.
   If maxObs==0 blocks and reads whole of file/audio into memory and
   returns with status==PB_STOPPED ready for table access.  All 
   parameters, associated with the loading and conversion of the
   source are defined using configuration parameters.
   If maxObs!=0 buffer may be read as a stream.  In this case reading
   should be via ReadAsBuffer calls which should continue until either
   ReadAsBuffer returns FALSE or buffer status >= PB_STOPPED.  Note 
   that for some types of input (eg pipes) end of data can only be 
   determined by a failed attempt to read the final frame.
   If the speech detector is enabled (either by configuration or
   by explicit parameter in call) then silence measurement can be
   forced/prevented by setting silMeasure to TRUE/FALSE (if UNDEF
   will perform measurement if it is needed by config).
*/

PBStatus BufferStatus(ParmBuf pbuf);
/* 
   Return current status of buffer.
    PB_INIT - buffer ready for StartBuffer call. No observations available.
    PB_FILLING - buffer is currently reading from source.
    PB_STOPPED - source has closed and buffer can be used as a table.
    PB_CLEARED - same as PB_STOPPED but ReadAsBuffer has read final frame.
   Does not block.
*/

int ObsInBuffer(ParmBuf pbuf);
/* 
   Return number of observations available to ReadAsBuffer without blocking.
   This will be zero once the buffer is COMPLETE although ReadAsTable can
   still be used to access whole buffer (use GetBufferInfo to find
   range of allowable indexes).
   Note that final frame may not be read and the source closed until a
   ReadAsBuffer call has to read the final frame (this is only a problem
   for HParm/HWave file descriptors which normally get read immediately in
   full).  This is non-ideal but cannot otherwise guarantee ObsInBuffer 
   is non-blocking.
*/

void StartBuffer(ParmBuf pbuf);
/*
   Start and filling the buffer.  If signals have been enabled
   then effect is delayed until first signal is sent.  If
   silence/speech detection is enabled then frames will 
   accumulate when speech starts and buffer will stop filling
   when silence is detected.  If silence/speech detection is
   not enabled but signals are, then a second signal will stop
   the filling.  This operation will fail if pbuf status is not
   PB_INIT.
   This operation should now be non-blocking.
*/
   
void StopBuffer(ParmBuf pbuf);
/*   
   Filling the buffer is stopped regardless of whether signals
   and/or silence/speech detection is enabled.  After making 
   this call, the pbuf status will change to PB_STOPPED.  
   Only when the buffer has been emptied will the status change
   to PB_CLEARED.
*/

void CloseBuffer(ParmBuf pbuf);
/*
   Close the given buffer, close the associated audio stream if
   any and release any associated memory.
*/

Boolean ReadAsBuffer(ParmBuf pbuf, Observation *o);
/*
   Get next observation from buffer.  Buffer status must be PB_FILLING 
   or PB_STOPPED.  If no observation is available the function
   will block until one is available or the input is closed.
   Will returns FALSE if blocked but could not read new Observation.
*/

void ReadAsTable (ParmBuf pbuf, int index, Observation *o);
/* 
   Get the index'th observation from buffer.  Buffer status
   must be PB_STOPPED.  Index runs 0,1,2,....
   By definition this operation is non-blocking.
*/


void GetBufferInfo(ParmBuf pbuf, BufferInfo *info);
/*
   Get info associated with pbuf.
   Does not block.
*/

/* ---------------- External Data Source Handling---------------- */

HParmSrcDef CreateSrcExt(Ptr xInfo, ParmKind pk, int size, HTime sampPeriod,
                         Ptr (*fOpen)(Ptr xInfo,char *fn,BufferInfo *info),
                         void (*fClose)(Ptr xInfo,Ptr bInfo),
                         void (*fStart)(Ptr xInfo,Ptr bInfo),
                         void (*fStop)(Ptr xInfo,Ptr bInfo),
                         int (*fNumSamp)(Ptr xInfo,Ptr bInfo),
                         int (*fGetData)(Ptr xInfo,Ptr bInfo,int n,Ptr data));
/*
  Create and return a HParmSrcDef object handling an external data source.
  size: 1 for 8 bit u-law, 0x101 for 8 bit a-law or 2 for 16 bit linear
  Semantics of functions fClose() etc. described in HParm.c.
*/

ParmBuf OpenExtBuffer(MemHeap *x, char *fn, int maxObs,
                      FileFormat ff, HParmSrcDef ext,
                      TriState enSpeechDet, TriState silMeasure);

/*
  Open and return input buffer using an external source
*/

/* ----------------- New Buffer Creation Routines -------------- */

ParmBuf EmptyBuffer(MemHeap *x, int size, Observation o, BufferInfo info);
/*
   Create and return an empty ParmBuf object set-up as a table
   with initially size free observation slots.  Observation o is 
   used for sizing and info supplies associated configuration
   parameters.  The latter will typically be copied from a
   buffer created by an OpenBuffer call.
*/

ReturnStatus SaveBuffer(ParmBuf pbuf, char *fname, FileFormat ff);
/*
   Write contents of given buffer to fname.  If SAVEWITHCRC is set in
   config then a cyclic redundancy check code is added.  If
   SAVECOMPRESSED is set then the data in the table is compressed
   before writing out.  If ff is not UNDEFF then ff overrides
   target file format set in buffer.
*/

void AddToBuffer(ParmBuf pbuf, Observation o);
/*
   Append the given observation to the table.
*/

/* ----------------- Observation Handling Routines -------------- */

Observation MakeObservation(MemHeap *x, short *swidth, 
                            ParmKind pkind, Boolean forceDisc, Boolean eSep);
/*
   Create observation using info in swidth, eSep and pkind
   If forceDisc is true the observation will be DISCRETE but can
   read from a continuous parameter parmbuffer.
*/

void ExplainObservation(Observation *o, int itemsPerLine);
/* 
   Explain the structure of given observation by printing
   a template showing component structure
*/

void PrintObservation(int i, Observation *o, int itemsPerLine);
/*
   Print the given observation. If i>0 then print with an index.
*/

void ZeroStreamWidths(int numS, short *swidth);
/*
   Stores numS in swidth[0] and sets remaining components of
   swidth to zero
*/

void  SetStreamWidths(ParmKind pk, int size, short *swidth, Boolean *eSep);
/*
   If swidth has been 'zeroed' by ZeroStreamWidths, then this function
   sets up stream widths in swidth[1] to swidth[S] for number of streams
   S specified in swidth[0]. If eSep then energy is extracted as a 
   separate stream.  If swidth[n] is non-zero, then it only eSep is
   set.
*/

/* EXPORT->SyncBuffers: if matrix transformations are used this syncs the two buffers */
Boolean SyncBuffers(ParmBuf pbuf,ParmBuf pbuf2);

void SetParmHMMSet(Ptr hset);
/* 
   The prototype  for this should really be 
   void SetParmHMMSet(HMMSet *hset);
   However the .h files have an issue with this. A 
   cast is performed in the first line of the function.
*/


/* ------------------- Parameter Kind Conversions --------------- */

char *ParmKind2Str(ParmKind kind, char *buf);
ParmKind Str2ParmKind(char *str);
/*
   Convert between ParmKind type & string form.
*/

ParmKind BaseParmKind(ParmKind kind);
Boolean HasEnergy(ParmKind kind);
Boolean HasDelta(ParmKind kind);
Boolean HasNulle(ParmKind kind);
Boolean HasAccs(ParmKind kind);
Boolean HasThird(ParmKind kind);
Boolean HasCompx(ParmKind kind);
Boolean HasCrcc(ParmKind kind);
Boolean HasZerom(ParmKind kind);
Boolean HasZeroc(ParmKind kind);
Boolean HasVQ(ParmKind kind);
/* 
   Functions to separate base param kind from qualifiers 
*/

Boolean ValidConversion(ParmKind src, ParmKind tgt);
/* 
   Checks that src -> tgt conversion is possible 
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
+ ### [HVQ.h](#HTK_proto)<a name="HVQ"></a>

```C++
/*
   This module provides a datatype VQTable which is used to represent
   linear (flat) and binary tree VQ codebooks.  Externally, a VQ Table
   definition is stored in a file consisting of a header followed by a
   sequence of entries representing each tree node.  One tree is 
   built for each stream:
   
   header:
      magic type covkind numNodes numS sw1 sw2 ...
   node_entry:
      stream vqidx nodeId leftId rightId mean-vector [invcov matrix|vector]
      ...
   where 
      magic  = usually the original parmkind
      type   = 0 linTree, 1 binTree
      covkind   = 5 euclid, 1 inv diag covar, 2 inv full covar
      numNodes = total number of following node entries
      numS   = number of streams
      sw1,sw2 = width of each stream
      stream = stream number for this entry
      vqidx  = the vq code for this node
      nodeId,leftId,rightId = arbitrary but unique integers identifying
               each node in each tree.  Root id is always 1.
*/

#ifndef _HVQ_H_
#define _HVQ_H_

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
   linTree,    /* linear flat codebook (right branching tree) */
   binTree     /* binary tree - every node has 0 or 2 offspring */
} TreeType;

typedef struct _VQNodeRec *VQNode;
typedef struct _VQNodeRec{
   short vqidx;         /* vq index of this node */
   Vector mean;         /* centre of this node */
   Covariance cov;      /* null or inverse variance or covariance */
   float gconst;        /* const part of log prob for INVDIAGC & FULLC */
   void * aux;          /* available to 'user' */
   short nid,lid,rid;   /* used for mapping between mem and ext def */
   VQNode left,right;   /* offspring, only right is used in linTree */
}VQNodeRec;

typedef struct _VQTabRec *VQTable;
typedef struct _VQTabRec {
   char * tabFN;        /* name of this VQ table */
   short magic;         /* magic num, usually the ParmKind */
   TreeType type;       /* linear or binary */
   CovKind ckind;       /* kind of covariance used, if any*/
   short swidth[SMAX];  /* sw[0]=num streams, sw[i]=width of stream i */
   short numNodes;      /* total num nodes in all sub trees */
   VQNode tree[SMAX];   /* 1 tree per stream */
   VQTable next;        /* used internally for housekeeping */
}VQTabRec;

void InitVQ(void);
/*
   Initialise module
*/

VQTable CreateVQTab(char *tabFN, short magic, TreeType type,
                    CovKind ck, short *swidth);
/*
   Create an empty VQTable with given attributes.
*/

VQNode CreateVQNode(short vqidx, short nid, short lid, short rid, 
                    Vector mean, CovKind ck, Covariance cov);
/*
   Create a VQ node with given values
*/

VQTable LoadVQTab(char *tabFN, short magic);
/*
   Create a VQTable in memory and load the entries stored in the
   given file.  The value of magic must match the corresponding
   entry in the definition file unless it is zero in which case
   it is ignored.
*/

void StoreVQTab(VQTable vqTab, char *tabFN);
/*
   Store the given VQTable in the specified definition file tabFN.
   If tabFN is NULL then the existing tabFN stored in the table is
   used.
*/

void PrintVQTab(VQTable vqTab);
/*
   Print the given VQTable.
*/

float VQNodeScore(VQNode n, Vector v, int size, CovKind ck);
/* 
   compute score between vector v and VQ node n, smallest score is
   best.  If ck is NULLC then a euclidean distance is used otherwise
   -ve log prob is used.
*/

void GetVQ(VQTable vqTab, int numS, Vector *fv, short *vq);
/*
   fv is an array 1..numS of parameter vectors, each is 
   quantised using vqTab and the resulting indices are stored
   in vq[1..numS].
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