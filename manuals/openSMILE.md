# [openSMILE](./openAudioLibs.md#TOP)<a name="openSMILE"></a>
+ [CLASS LIST](#openSMILE_list)
+ [CLASS PROTOTYPE](#openSMILE_proto)
+ [LICENSE](#openSMILE_license)

https://audeering.com/technology/opensmile/  
#### [Document](https://www.audeering.com/research-and-open-source/files/openSMILE-book-latest.pdf)


+ **/src/include/core**

Header|Discription|&nbsp;
---|---|---
commandlineParser.hpp|   |
dataProcessor.hpp| This is an abstract base class for all components which read data from the data moemory and write new data to the data memory  |
dataWriter.hpp|   |
smileLogger.hpp|   |
vecToWinPorcessor.hpp|   |
componentList.hpp|   |
dataReader.hpp|   |
exceptions.hpp|   |
smileTypes.hpp|   |
versionInfo.hpp|   |
componentManager.hpp|   |
dataSelector.hpp|   |
nullSink
svn_version.hpp|   |
windowProcessor.hpp|   |
configManager.hpp|   |
dataSing.hpp|   |
smileCommon.hpp|   |
vectorProcessor.hpp|   |
winToVecProcessor.hpp|   |
dataMemory.hpp|   |
dataSource.hpp|   |
smileComponenet.hpp|   |
vectorTransform.hpp|   |


+ **/src/include/dsp**

Header|Discription|&nbsp;
---|---|---
dbA.hpp |applies dbX weighting to fft magnitudes|
smileResample.hpp|simple preemphasis : x(t) = x(t) -k*x(t-1)|
specScale.hpp||
signalGenerator.hpp| Signal source. Generates various noise types and pre-defined signals|
specResample.hpp|experimental resampling by ideal fourier interpolation and limiting this component taks a complex (!) dft spectrum (generated from real values) as input |
vadV1.hpp| voice activity detection based on LSF and Pitch features + smoothing|

+ **/src/include/dspcore**

Header|Discription|&nbsp;
---|---|---
acf.hpp| Autocorrelation Function (ACF) |
fftmagphase.hpp|  |
fullturnMean.hpp| computes mean of full input |
turnDetector.hpp| simple silence threshold based turn detector |
amdf.hpp| Average Magnitude Difference Function (AMDF) |
fftXg.h|  |
monoMixdown.hpp| simple mixer, which adds multiple channels (elements) to a single channel (element)  |
vextorMVN.hpp| This component extends the bases class cVectorTransform and implements mean/variance normalisation |
contourSmoother.hpp| smooth data contours by moving average filter |
framer.hpp| dataFramer |
preemphasis.hpp| simple preemphasis : x(t) = x(t) - k*x(t-1) |
vectorPreemphasis.hpp| pre-emphasis per frame (simplification, however, this is the way HTK does it... so for compatibility... here you go) (use before window function is applied) |
deltaRegression.hpp| comoute delta regression using regression formula |
fullinputMean.hpp| compute mean of full input |
transformFft.hpp| fast fourier transform using fft4g library, output : complex values of fft or real signal values (for iFFT) |
windower.hpp| data windower. takes frames from one level, applies windows function, and saves to other level |


+ **/src/include/functionals**

Header|Discription|&nbsp;
---|---|---
functionalCompoent.hpp|a single statistical functional or the like(template class)|
functionalModulation.hpp|number of segmenst based on delta thresholding|
functionalRegression.hpp|linear and quadratic regression coefficients|
functionalCrossings.hpp|zero-crossings, mean-crossings, arithmetic mean|
functionalMoments.hpp|statistical moments|
functionalSamples.hpp|rise/fall times, up/down-level times|
functionalDCT.hpp|number of segments based on delta thresholding |
functionalOnset.hpp|number of segments based on delta thresholing|
functionalSegments.hpp|number of segments based on delta thresholding |
functionalExtremes.hpp|extreme values and ranges|
functionalPeaks2.hpp|number of peaks and various measures associated with peaks|
functionals.hpp|passes unsorted row data array AND (if required) sorted row data array to functional processors|
functionalLpc.hpp|number of segments based on delta thresholding |
functionalPeaks.hpp|number of peaks and various measures associated with peaks|
functionalTimes.hpp|rise/fall times, up/down-level times|
functionalMeans.hpp|various means,arithmetic,geometric, quadratic, etc. also number of non-zero values, etc|
functionalPercentiles.hpp|percentiles and quadrtiles, and inter-percentile/quartile ranges|

+ /src/include/rnn

+ /src/include/lld

+ /src/include/lldcore

+ /src/include/portaudio

+ /src/include/smileutil

+ /src/include/classifiers

+ /src/include/io

+ /src/include/iocore

+ /src/include/rapidjson

+ /src/include/video  : openCV 




RNN :  Recurrent neural network  
OpenCV :  Open Source Computer Vision Library  
RapidJSON : RapidJSON is a JSON parser and generator for C++. It was inspired by RapidXml. Header-only  
PortAudio : cross-platform, open-source, audio I/O library


---




#### [CLASS LIST](#openSMILE)<a name="openSMILE_list"></a>

+ cVadV1 : cDataProcessor : cSmileComponent
  int computeFilters (long blocksize, double frameSizeSec, int idxc)
---

#### [CLASS PROTOTYPE](#openSMILE)<a name = "openSMILE_proto"></a>

+ ### cSmileComponent

```C++

class DLLEXPORT cSmileComponent {
  private:
    int id_;           // component ID in componentManager
    int EOI_;          // EOI counter, 0 only in first loop, then +1 for every nonEOI/EOI loop pair
    int EOIcondition_; // flag that indicates end of input
                      // i.e. if EOI is 1, myTick should show a different behaviour
                      //  esp. dataReaders should return right padded matrices, in getMatrix , getNextMatrix etc..
    int EOIlevel_;   // If set to >= 1, the isEOI will report EOI condition true only if EOI_  == EOIlevel_ ;
    int paused_; // flag that indicates whether processing (the tick loop) has been paused or is active
    
    smileMutex  messageMtx_;

    cComponentManager *compman_;  // pointer to component manager this component instance belongs to
    cSmileComponent *parent_;     // pointer to parent component (for dataReaders, etc.)
    char *iname_;   // name of component instance
    char *cfname_;  // name of config instance associated with this component instance

    // variables used for component profiling
    int doProfile_, printProfile_;
    double profileCur_, profileSum_; // exec. time of last tick, exec time total
    struct timeval startTime_;
    struct timeval endTime_;

    long lastNrun_;   // the number of Nrun in the last tickloop?

  protected:
    SMILECOMPONENT_STATIC_DECL_PR

    cConfigManager *confman_;  // pointer to configManager
    const char *cname_;        // name of the component (type)
    const char *description_;  // component description and usage information

    // component state variables
    int isRegistered_, isConfigured_, isFinalised_, isReady_;
    int runMe_;

    int manualConfig_;
    int override_;
    
    static sComponentInfo * makeInfo(cConfigManager *_confman,
                                     const char *_name, const char *_description,
                                     cSmileComponent * (*create) (const char *_instname),
                                     int regAgain=0, int _abstract=0, int _nodmem=0);

    // Gets a pointer to the component instance *name via the component manager.
    cSmileComponent * getComponentInstance(const char * name);

    // Gets the component instance type of the instance *name as string via the component manager.
    const char * getComponentInstanceType(const char * name);

    // Create a component instance of given type with instance name "name" in the component manager.
    cSmileComponent * createComponent(const char *name, const char *component_type);
    
    // Gets the number of components ran during the last tick.
    long getLastNrun()

    // Functions to get config values from the config manager from our config instance.
    // The _f functions internally free the string *name. Use these in conjunction with myvprint()...
    // NOTE: Yes, this is ineffective. TODO: smile memory manager, and fixed length text buffer which can be reused (can also grow if needed).
    void * getExternalPointer(const char *name) 
    
    void addExternalPointer(const char *name, void * ptr) 
    
    
    double getDouble(const char*name) 
    
    double getDouble_f(char*name) 
    
    int getInt(const char*name) 
    
    int getInt_f(char*name) 
    

    const char *getStr(const char*name)
    
    const char * getStr_f(char*name) 

    char getChar(const char*name) 
    
    const char getChar_f(char*name) 
    
    const ConfigValue *getValue(const char*name)
    
    const ConfigValue * getValue_f(char*name) 

    const ConfigValueArr *getArray(const char*name)
    
    const ConfigValueArr * getArray_f(char*name) 

    int getArraySize(const char*name)
    
    int getArraySize_f(char*name) 

    int isSet(const char*name) 
    int isSet_f(char*name)

    // Returns 1 if we are in an abort state (user requested abort).
    int isAbort();

    // Function that is called during initialisation. It should be used to fetch config variables
    // from the config manager and store them in local variables.
    // Each derived class is responsible for fetching configuration from confman or setting it manually BEFORE configureInstance is called!
    virtual void fetchConfig()=0;

    // Functions to be implemented by derived classes:

    // Sets the environment (pointers to parent, component manager, config manager).
    virtual void mySetEnvironment() 
    
    // Registers the component instance after creation (by the component Manager).
    virtual int myRegisterInstance(int *runMe=NULL) 

    // Configures the component instance. Data-memory levels and level parameters such as T are created and set here.
    virtual int myConfigureInstance() 

    // Finalises the component instance. Data-memory level names (fields and elements) are set here.
    virtual int myFinaliseInstance() 
    // Holds the actual implementation of the tick loop code for the derived component.
    // This function should return 1 if the component did process data, and 0 otherwise.
    // If all components return 0 (nothng to process), the component manager will switch
    // to the EOI (end-of-input) state (or advance to the next tick loop iteration).
    virtual int myTick(long long t) 

    // Called by the component manager when the tick loop is to be paused.
    // If the component needs to reject the pausing of the tick loop, it should return 0, otherwise always 1!
    virtual int pauseEvent() 

    // Called by the component manager when the tick loop is to be resumed.
    virtual void resumeEvent() { }

    // Signals EOI to componentManager (theoretically only useful for dataSource components, however we make it accessible to all smile components)
    // NOTE: you do not need to do this explicitely.. if all components fail, EOI is assumed, then a new tickLoop is started by the component manager
    // TODO: is this actually used and/or implemented properly?
    void signalEOI();

    // Checks if the given component message is of type "msgtype" (string name).
    int isMessageType(cComponentMessage *msg, const char * msgtype)

    // Functions for component message handling.
    // Note on data synchronisation for messages: The messages can arrive at any time, if
    // they are sent from another thread. Thus, a mutex is used to prevent collision when
    // accessing variables from processComponentMessage and the rest of the component code
    // (especially myTick). The mutex is automatically locked before processComponentMessage
    // is called, and unlocked afterwards. In the rest of the code, however, lockMessageMemory()
    // and unlockMessageMemory() must be used.

    // This function is called by the component manager, if there is a new message for this component.
    // A derived component must override this to receive and process messages.
    // Do NOT call lockMessageMemory() in this function! The *msg pointer is always valid.
    // Return value: 0, message was not processed; != 0, message was processed.
    // The return value will be passed on to the sender (?).
    // NOTE: the custData pointer in cComponentMessage (as well as the whole cComponentMessage object)
    //       is valid ONLY until processComponentMessage() returns!
    //       Thus, you are advised to copy data to local memory
    // NOTE2: This function is called directly from the context of another component/thread.
    //        Do NOT do anything here that is time consuming. It will block the other thread
    //        or the execution of the tick loop. Esp. don't wait here until your own myTick() is called.
    //        In signle threaded mode this will never happen!
    //        Always, only accept the message data and put it into a buffer. Process the buffer in myTick()
    //        or in a background thread.
    virtual int processComponentMessage(cComponentMessage *msg) { return 0; }
    
    // this function forwards the message to the componentMananger and sets the *sender pointer correctly
    // Return value: 0, message was not processed; != 0, message was processed.
    // TODO: return value passing for multiple recepients?
    int sendComponentMessage(const char *recepient, cComponentMessage *msg);

    // Locks the "message memory" (variables that are accessed by processComponentMessage).
    // You must call this function prior to accessing variables you are accessing in processComponentMessage().
    // Do not use this function in processComponentMessage() itself, though!
    void lockMessageMemory() 
    // Unlocks the "message memory" (variables that are accessed by processComponentMessage).
    void unlockMessageMemory() 
    
    // Gets the smile time from the component manager (time since start of the system).
    double getSmileTime();

    // Returns 1 if we are in an end-of-input condition.
    int isEOI() 

    // Get the EOI counter, i.e. the number of repeated tick loop sequences.
    int getEOIcounter() 

    // Request stopping of processing in tickLoop of component manager.
    // Calling this will make openSMILE stop the processing.
    virtual void abort_processing();

  public:
    // Statics:
    // these two must be overridden in a base class:
    //static sComponentInfo * registerComponent(cConfigManager *_confman);
    //static cSmileComponent * create(const char *_instname);
    SMILECOMPONENT_STATIC_DECL

    //method for setting config... a base class may implement it with arbirtrary parameters
    // values set will override those obtained by fetchConfig if override=1
    // if override=0, then only those which are not set are updated
    // void setConfig(..., int override=1);
    int isManualConfigSet()
    
    // Returns whether the tick loop is paused (1) or running (0).
    int isPaused() 

    // Gets pointer to the associated component manager object,
    // i.e. the object that created this smile component instance.
    cComponentManager *getCompMan() 

    // Constructor that creates an instance of this component with instance name "instname".
    cSmileComponent(const char *instname);
    
    // Sets component info after component creation, and calls fetchConfig.
    //  Pointer to config manager, component name and description
    virtual void setComponentInfo(cConfigManager *cm, const char *cname, const char *description) 
        
    // Sets component manager pointer and the component ID, as used by the component manager.
    // Called by the creator component (parent) or directly by the component manager.
    virtual void setComponentEnvironment(cComponentManager *compman, int id, cSmileComponent *parent=NULL);
    
    // Sets the name of the associated config instance this component instance is linked to.
    void setConfigInstanceName(const char *cfname) 
    
    // Gets the name of the associated config instance.
    const char * getConfigInstanceName() 
    
    // Performs component specific register operation, e.g. write/read requests with dataMemory..
    // *runMe return value for component manager : 0, don't call my tick of this component, 1, call myTick
    int registerInstance(int *runMe=NULL) 
    
    int isRegistered() 

    // Performs component configuration (data memory level configuration, creation of readers/writers etc.).
    int configureInstance() 
    
    int isConfigured() 

    // Performs component finalisation (loading of models, opening of files, data memory level names, ...).
    int finaliseInstance()
    int isFinalised() 

    // Returns 1, if component has been finalised and is ready for the tick loop. Returns 0 otherwise.
    int isReady()

    // The tick() function. This is called by the component manager, and internally executes myTick(),
    int tick(long long t, int EOIcond=0, long _lastNrun=-1) 
    
    // Configures profiling (measuring of the time spent in each tick).
    void setProfiling(int enable=1, int print=0) 

    // Starts time measurement (called at the beginning of the tick).
    void startProfile(long long t, int EOI);

    // Ends time measurement (called at the end of the tick).
    void endProfile(long long t, int EOI);

    // Gets the current profiling statistics.
    // If sum == 1, it returns the accumulated run-time in seconds
    // (i.e. the time spent in tick() by this component).
    // If sum == 0, the duration of the last tick() is returned.
    // NOTE: For multi-threaded code this method of profiling is not exact.
    //       The tick() function can be interrupted by other threads, but
    //       time measurement is done via the system timer from start to end of tick().
    double getProfile(int sum=1) 

    // Gets the component instance name.
    const char *getInstName() const

    // Gets the component type name.
    const char *getTypeName() const

    // Sets the EOI counter. Used by the component manager once a new tick loop with EOIcondition=1 starts.
    // NOTE: it would be possible for the components to manage their own EOI counters, however, like this we ensure sync and better performance.
    // If derived components create sub-components internally (without tick function - e.g. dataReaders/Writers),
    // they must override this, in order to properly set the EOI status in the sub-components.
    virtual int setEOIcounter(int cnt)

    // Sets and unsets the EOIcondition variable. This is called internally from tick() only.
    // If derived components create sub-components internally (without tick function - e.g. dataReaders/Writers),
    // they must override this, in order to properly set the EOI status in the sub-components.
    virtual void setEOI() 
    
    virtual void unsetEOI() 
    virtual void setEOIlevel(int level)
    virtual int EOIlevelIsMatch()
    
    // Called by the component manager. Notifies this component about a tick loop pause.
    virtual int pause() 

    // Called by the component manager. Notifies this component about a tick loop resume.
    virtual void resume()

    // This function is called externally by the component manager,
    //   if another component calls sendComponentMessage.
    // It receives the message, takes care of memory access synchronisation, and processes the message.
    int receiveComponentMessage(cComponentMessage *msg) 

```

---

#### [LICENSE](#openSMILE)<a name = "openSMILE_license"></a>

https://audeering.com/technology/opensmile/#licensing

+ 비영리목적으로 자유롭게 사용가능 (사용, 수정 ,배포, 출간, 시연, 연구)  
+ 영리목적으로 사용시 audEERING에게 license 받아야함  
+ 영리성이 있어도 주된 목적이 아니거나, 연구와 학술을 위한 목적이 더 클 경우, 상용제품의 성능테스트일 경우 사용가능  
+ 수정된 코드를 배포시 수정되었음을 명시  
+ binary 파일 수정, 역공학 금지  

**Licensing**

openSMILE is maintained by audEERING since 2013. Version 2.0 and above are distributed 
free of charge for research and personal use under the terms of the 
[openSMILE research only open-source license](https://www.audeering.com/research-and-open-source/files/openSMILE-open-source-license.txt).

For commercial use, we provide individualised, flexible commercial licensing 
options for any project size and budget. We also offer ready-to-use speech analysis 
services and software products based on our proprietary extensions to the openSMILE core. Expert technical 
support is also available to help you get started and integrate openSMILE in your developments quickly. 
[Contact us](www.audeering.com/contact) today to receive your customized offer and talk about your possibilities!

+ ### openSMILE research only open-source license
<pre>
  openSMILE Version 2.x
   - open Speech and Music Interpretation by Large-space Extraction -
  Authors: Florian Eyben, Felix Weninger, Martin Woellmer, Bjoern Schuller
  Copyright (C) 2008-2013, Institute for Human-Machine Communication, TUM
  Copyright (C) 2013-2014, audEERING UG (haftungsbeschrÃ¤nkt)
 
  audEERING UG (haftungsbeschrÃ¤nkt)
  Gilching, Germany
  http://www.audeeering.com/

 ********************************************************************** 
 If you use openSMILE or any code from openSMILE in your research work,
 you are kindly asked to acknowledge the use of openSMILE in your publications.
 See the file CITING.txt for details.
 **********************************************************************


This audEERING Research License Agreement (license, license agreement, 
or agreement in the ongoing), is a legal agreement between you 
and audEERING UG (haftungsbeschrÃ¤nkt), 
Gilching, Germany (audEERING or we) for the software 
or data mentioned above, which may include source code, 
and any associated materials, text or speech files, 
associated media and "on-line" or electronic documentation 
(all together called the "Software").  

By installing, copying, or otherwise using this Software, 
you agree to be bound by the terms in this license. 
If you do not agree, you must not install copy or use the Software. 
The Software is protected by copyright and other intellectual 
property laws and is licensed, not sold.

This license grants you the following rights:
A. You may use, copy, reproduce, and distribute this Software 
   for any non-commercial purpose, subject to the restrictions 
   set out below. Some purposes which can be non-commercial are teaching, 
   academic research, public demonstrations and personal experimentation 
   or personal home use. You may also distribute this Software with 
   books or other teaching materials, or publish the Software on websites, 
   that are intended to teach the use of the Software for academic or 
   other non-commercial purposes. 
   You may NOT use or distribute this Software or any derivative works 
   in any form for commercial purposes, except those outlined in (B). 
   Examples of commercial purposes are running business operations, 
   licensing, leasing, or selling the Software, distributing the 
   Software for use with commercial products (no matter whether free or paid), 
   using the Software in the creation or use of commercial products or any 
   other activity which purpose is to procure a commercial gain to you or 
   others (except for conditions set out in (B)).
B. Further, you may use the software for commercial research, which meets
   the following conditions: commercial research which is not directly
   associated with product development and has the primary purpose of 
   publishing and sharing results with the academic world; pre-product 
   evaluations of algorithms and methods, as long as these evaluations
   are more of an evaluatory, planning, and research nature than of a product development
   nature. 
   Any further commercial use requires you to obtain a commercial
   license or written approval from audEERING UG (limited) which grants
   you extended usage rights for this software. In particular any direct 
   (software) or indirect (models, features extracted with the software)
   use of the software, parts of the software, or derivatives in a 
   product (no matter whether free or paid), is not allowed without 
   an additional commercial license.
C. If the software includes source code or data, you may create 
   derivative works of such portions of the software and distribute the 
   modified software for non-commercial purposes, as provided herein.
   If you distribute the software or any derivative works of the Software, 
   you must distribute them under the same terms and conditions as in this
   license, and you must not grant other rights to the software or 
   derivative works that are different from those provided by this 
   license agreement. 
   If you have created derivative works of the software, and distribute 
   such derivative works, you will cause the modified files to carry 
   prominent notices so that recipients know that they are not receiving 
   the original software. Such notices must state: 
   (i) that you have altered the software; 
   and (ii) the date of any changes as well as your name.

In return for the above rights, you agree to:
1. That you will not remove any copyright or other notices (authors 
   and citing information, for example) from the software.
2. That if any of the software is in binary format, you will not attempt 
   to modify such portions of the software, or to reverse engineer or 
   decompile them, except and only to the extent authorized by applicable 
   law. 
3. That the copyright holders (audEERING) are granted back, 
   without any restrictions or limitations, a non-exclusive, perpetual, 
   irrevocable, royalty-free, assignable and sub-licensable license, 
   to reproduce, publicly perform or display, install, use, modify, post, 
   distribute, make and have made, sell and transfer your modifications 
   to and/or derivative works of the software source code or data, 
   for any purpose.  
4. That any feedback about the software provided by you to us is voluntarily 
   given, and audEERING shall be free to use the feedback 
   as they see fit without obligation or restriction of any kind, 
   even if the feedback is designated by you as confidential. 

5. THAT THE SOFTWARE COMES "AS IS", WITH NO WARRANTIES. 
   THIS MEANS NO EXPRESS, IMPLIED OR STATUTORY WARRANTY, INCLUDING 
   WITHOUT LIMITATION, WARRANTIES OF MERCHANTABILITY OR FITNESS FOR A 
   PARTICULAR PURPOSE, ANY WARRANTY AGAINST INTERFERENCE WITH YOUR 
   ENJOYMENT OF THE SOFTWARE OR ANY WARRANTY OF TITLE OR NON-INFRINGEMENT.
   THERE IS NO WARRANTY THAT THIS SOFTWARE WILL FULFILL ANY OF YOUR 
   PARTICULAR PURPOSES OR NEEDS. ALSO, YOU MUST PASS THIS DISCLAIMER ON 
   WHENEVER YOU DISTRIBUTE THE SOFTWARE OR DERIVATIVE WORKS.
   
6. THAT NEITHER TUM NOR AUDEERING NOR ANY AUTHOR OR CONTRIBUTOR TO THE 
   SOFTWARE WILL BE LIABLE FOR ANY DAMAGES RELATED TO THE SOFTWARE OR THIS 
   LICENSE, INCLUDING DIRECT, INDIRECT, SPECIAL, CONSEQUENTIAL OR INCIDENTAL 
   DAMAGES, TO THE MAXIMUM EXTENT THE LAW PERMITS, NO MATTER WHAT LEGAL 
   THEORY IT IS BASED ON. ALSO, YOU MUST PASS THIS LIMITATION OF LIABILITY 
   ON WHENEVER YOU DISTRIBUTE THE SOFTWARE OR DERIVATIVE WORKS.
   
7. That we have no duty of reasonable care or lack of negligence, 
   and we are not obligated to (and will not) provide technical support for 
   the Software.
8. That if you breach this license agreement or if you sue anyone over 
   patents that you think may apply to or read on the software or anyone's 
   use of the software, this license agreement (and your license and rights 
   obtained herein) terminate automatically. Upon any such termination, 
   you shall destroy all of your copies of the software immediately.  
   Sections 3, 4, 5, 6, 7, 10 and 11 of this license agreement shall survive 
   any termination of this license agreement.
9. That the patent rights, if any, granted to you in this license agreement 
   only apply to the software, not to any derivative works you make.
10.That the software may be subject to European export or import laws or such 
   laws in other places. You agree to comply with all such laws and regulations 
   that may apply to the software after the delivery of the software to you.
11.That all rights not expressly granted to you in this license agreement 
   are reserved by audEERING.
12.That this license agreement shall be construed and controlled by the laws 
   of the Federal Republic of Germany, without regard to conflicts of law. 
   If any provision of this license agreement shall be deemed unenforceable 
   or contrary to law, the rest of this license agreement shall remain in 
   full effect and interpreted in an enforceable manner that most closely 
   captures the intent of the original language. 

 ++ For other, such as commercial, licensing options, 
    please contact audEERING at info@audeering.com ++


openSMILE contains third-party contributions from the Speex codec package 
(in the cLsp component), which are distributed unter the following terms:
-----------------------------------------------------------------------
Copyright 2002-2008     Xiph.org Foundation
Copyright 2002-2008     Jean-Marc Valin
Copyright 2005-2007     Analog Devices Inc.
Copyright 2005-2008     Commonwealth Scientific and Industrial Research
                        Organisation (CSIRO)
Copyright 1993, 2002, 2006 David Rowe
Copyright 2003          EpicGames
Copyright 1992-1994     Jutta Degener, Carsten Bormann

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

- Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

- Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

- Neither the name of the Xiph.org Foundation nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE FOUNDATION OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
----------------------------------------------------------------------- 

About openSMILE:
================
openSMILE is a complete and open-source toolkit for audio analysis, 
processing and classification especially targeted at speech and music applications, 
e.g. ASR, emotion recognition, or beat tracking and chord detection.
The toolkit was developed at the Institute for Human-Machine Communication 
at the Technische Universitaet Muenchen in Munich, Germany.
It was started within the SEMAINE EU-FP7 research project.
The toolkit is now further maintained and developed
by audEERING UG (limited), which provides intelligent
audio analysis solutions and consulting services.
An open-source version for private use, education, and research
will always be available, next to commercial versions with
additional features (such as more interface components) or commercial
licenses for the basic version.

Third-party dependencies:
=========================

openSMILE uses LibSVM (by Chih-Chung Chang and Chih-Jen Lin) for 
classification tasks. It is distributed with openSMILE and is
 included in the src/classifiers/libsvm/ directory.

PortAudio is required for live recording from sound card and for 
the SEMAINE component.
You can get it from: http://www.portaudio.com
A working snapshot is included in thirdparty/portaudio.tgz

Optionally, openSMILE can be linked against the SEMAINE API 
and the Julius LVCSR engine, enabling an interface to the 
SEMAINE system and a keyword spotter component. 
See http://www.semaine-project.eu/ for details on running the 
SEMAINE system.
Only the older versions (1.0.1) are supported with the SEMAINE 
system.

Documentation/Installing/Using:
===============================

openSMILE is well documented in the openSMILE book, which can be 
found in doc/openSMILE_book.pdf.

Developers:
===========

Incomplete developer's documentation can be found in "doc/developer" 
and in the openSMILE book.

Information on how to write and compile run-time linkable plug-ins 
for openSMILE, see the openSMILE book or take a look at the files 
in the "plugindev" directory, especially the README file.

Getting more help:
==================

If you encounter problems with openSMILE, and solve them yourself, 
please do inform Florian Eyben via e-mail (fe@audeering.com), 
so the documentation can be updated!

If you cannot solve the problems yourself, please do also contact
Florian Eyben so we can solve the problem together and update 
the documentation.

</pre>

