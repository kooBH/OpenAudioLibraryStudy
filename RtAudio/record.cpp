/******************************************/
/*
  record.cpp
  by Gary P. Scavone, 2007

  This program records audio from a device and writes it to a
  header-less binary file.  Use the 'playraw', with the same
  parameters and format settings, to playback the audio.
*/
/******************************************/

#include "RtAudio.h"
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <stdio.h>

#include <string.h>
#include <time.h>

/*
typedef char MY_TYPE;
#define FORMAT RTAUDIO_SINT8
*/

typedef signed short MY_TYPE;
#define FORMAT RTAUDIO_SINT16

/*
typedef S24 MY_TYPE;
#define FORMAT RTAUDIO_SINT24

typedef signed long MY_TYPE;
#define FORMAT RTAUDIO_SINT32

typedef float MY_TYPE;
#define FORMAT RTAUDIO_FLOAT32

typedef double MY_TYPE;
#define FORMAT RTAUDIO_FLOAT64
*/

// Platform-dependent sleep routines.
#if defined( __WINDOWS_ASIO__ ) || defined( __WINDOWS_DS__ ) || defined( __WINDOWS_WASAPI__ )
  #include <windows.h>
  #define SLEEP( milliseconds ) Sleep( (DWORD) milliseconds ) 
#else // Unix variants
  #include <unistd.h>
  #define SLEEP( milliseconds ) usleep( (unsigned long) (milliseconds * 1000.0) )
#endif



void usage( void ) {
  // Error function in case of incorrect command-line
  // argument specifications
  std::cout << "\nusage: record <path> <device> <fs> <channels> <duration>\n";
  std::cout << "    path  = path output .wav file will be saved,\n";
  std::cout << "    device = optional device to use (default = 0),\n";
  std::cout << "    fs = the sample rate,\n";
  std::cout << "    channels = channels of input,\n";
  std::cout << "    duration = optional time in seconds to record (default = 2.0),\n\n";
  exit( 0 );
}



struct InputData {
  MY_TYPE* buffer;
  unsigned long bufferBytes;
  unsigned long totalFrames;
  unsigned long frameCounter;
  unsigned int channels;
};


// Interleaved buffers
int input( void * /*outputBuffer*/, void *inputBuffer, unsigned int nBufferFrames,
           double /*streamTime*/, RtAudioStreamStatus /*status*/, void *data )
{
  InputData *iData = (InputData *) data;

  // Simply copy the data to our allocated buffer.
  unsigned int frames = nBufferFrames;
  if ( iData->frameCounter + nBufferFrames > iData->totalFrames ) {
    frames = iData->totalFrames - iData->frameCounter;
    iData->bufferBytes = frames * iData->channels * sizeof( MY_TYPE );
  }

  unsigned long offset = iData->frameCounter * iData->channels;
  memcpy( iData->buffer+offset, inputBuffer, iData->bufferBytes );
  iData->frameCounter += frames;

  if ( iData->frameCounter >= iData->totalFrames ) return 2;
  return 0;
}
void run(char* a1,int a2,int a3,int a4,double a5)
{
  unsigned int channels, fs, bufferFrames, device = 0, offset = 0;
  double dtime = 60;
  FILE *fd;
  char time_name[100];
  
  //get current time
  time_t raw_time;
  struct tm * time_info;
  //time( &raw_time );
  raw_time = time(NULL);
  time_info = localtime(&raw_time);  
 
  sprintf(time_name, "%s/wavs/%d_%d_%d_%d_%d.wav",a1,time_info->tm_mon+1,time_info->tm_mday,time_info->tm_hour, time_info->tm_min, time_info->tm_sec);
   
	RtAudio adc;
	//Recording parameters
	fs = (unsigned int)a3;
	channels  = (unsigned int)a4;
	dtime = (double)a5;
	device = (unsigned int) a2;
   
  	std::cout<<"Recording .... "<<time_name<<"\n";
  
   	adc.showWarnings( true );
   bufferFrames = 512;
  RtAudio::StreamParameters iParams;
  if ( device == 0 )
    iParams.deviceId = adc.getDefaultInputDevice();
  else
    iParams.deviceId = device;
  iParams.nChannels = channels;
  iParams.firstChannel = offset;

  InputData data;
  data.buffer = 0;
  try {
    adc.openStream( NULL, &iParams, FORMAT, fs, &bufferFrames, &input, (void *)&data );
  }
  catch ( RtAudioError& e ) {
    std::cout << '\n' << e.getMessage() << '\n' << std::endl;
    goto cleanup;
  }

  data.bufferBytes = bufferFrames * channels * sizeof( MY_TYPE );
  data.totalFrames = (unsigned long) (fs * dtime);
  data.frameCounter = 0;
  data.channels = channels;
  unsigned long totalBytes;
  totalBytes = data.totalFrames * channels * sizeof( MY_TYPE );

  // Allocate the entire data buffer before starting stream.
  data.buffer = (MY_TYPE*) malloc( totalBytes );
  if ( data.buffer == 0 ) {
    std::cout << "Memory allocation error ... quitting!\n";
    goto cleanup;
  }

  try {
    adc.startStream();
  }
  catch ( RtAudioError& e ) {
    std::cout << '\n' << e.getMessage() << '\n' << std::endl;
    goto cleanup;
  }

 while ( adc.isStreamRunning() ) {
    SLEEP( 100 ); // wake every 100 ms to check if we're done
  }

  // Now write the entire data to the file.
  fd = fopen( time_name, "wb" );
  printf("open : %s\n",time_name);
  
  //If user doesn't have permission, returns NULL
  if(fd ==NULL)
	{
		printf("Failed to open file : %s\n",time_name);
		return;
	
	}
  char	riff_id[4];			//0
  int	riff_size;			//4
  char	wave_id[4];			//8
  char	format_id[4];		//12
  int	format_size;		//16
  short	format_type;		//20
  short	nChannels;			//22 (the number of channels)
  int	SamplePerSec;		//24 sampling frequency.
  int	AvgBytesPerSec;		//28
  short	nBlockAlign;		//32. block: samples set at the same time regardless channel. you guess total play time through block numbers.
  short	BitsPerSample;		//34
  short	cbsize;				//36 in case of non PCM format_type, there is a cbsize field.
  char	data_id[4];			//38(ex) 36(non ex)
  int	data_size;			//42(ex) 40(non ex)

  riff_id[0] = 'R';
  riff_id[1] = 'I';
  riff_id[2] = 'F';
  riff_id[3] = 'F';
  wave_id[0] = 'W';
  wave_id[1] = 'A';
  wave_id[2] = 'V';
  wave_id[3] = 'E';
  format_id[0] = 'f';
  format_id[1] = 'm';
  format_id[2] = 't';
  format_id[3] = ' ';
  cbsize = 0;
  data_id[0] = 'd';
  data_id[1] = 'a';
  data_id[2] = 't';
  data_id[3] = 'a';

  data_size = 0;
  riff_size = 0;

  nBlockAlign = channels * sizeof(MY_TYPE);
  BitsPerSample = sizeof(MY_TYPE) * 8;
  SamplePerSec = fs;
  AvgBytesPerSec = nBlockAlign * SamplePerSec;
  nChannels = channels;
  format_type = 0x0001;
  format_size = 16;
  data_size = data.totalFrames * nBlockAlign;
  riff_size = data_size - 36;

  fwrite(riff_id, sizeof(char), 4, fd);
  fwrite(&riff_size, sizeof(int), 1, fd);
  fwrite(wave_id, sizeof(char), 4, fd);
  fwrite(format_id, sizeof(char), 4, fd);
  fwrite(&format_size, sizeof(int), 1, fd);

  fwrite(&format_type, sizeof(short), 1, fd);
  fwrite(&nChannels, sizeof(short), 1, fd);
  fwrite(&SamplePerSec, sizeof(int), 1, fd);
  fwrite(&AvgBytesPerSec, sizeof(int), 1, fd);
  fwrite(&nBlockAlign, sizeof(short), 1, fd);
  fwrite(&BitsPerSample, sizeof(short), 1, fd);
  fwrite(data_id, sizeof(char), 4, fd);
  fwrite(&data_size, sizeof(int), 1, fd);

  fwrite( data.buffer, sizeof( MY_TYPE ), data.totalFrames * channels, fd );
  fclose( fd );
 // std::cout <<"record.raw is closed\n"; 
  std::cout<<time_name<<" is recorded\n"; 
cleanup:
  if ( adc.isStreamOpen() ) adc.closeStream();
  if ( data.buffer ) free( data.buffer );
 
  
  
}

int main( int argc, char *argv[] )
{
	int i;
	for(i=0;i<argc;i++)
		printf("arg[%d] : %s \n",i,argv[i]);
	
	if(argc < 5)
	{
		usage();
		return -1;
	}
	printf("path : %s\n", argv[1]);
	printf("device number : %d\n", atoi(argv[2]));
	printf("sample rate : %d\n", atoi(argv[3]));
	printf("channels : %d\n", atoi(argv[4]));
	printf("time unit : %f sec\n", atof(argv[5]));

  while(1)
   run(argv[1],atoi(argv[2]),atoi(argv[3]),atoi(argv[4]),atof(argv[5]));
	return 0;
}
