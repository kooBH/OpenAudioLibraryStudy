
int main()
{
	RtAudio dac;
	if ( dac.getDeviceCount() == 0 ) exit( 0 );

	RtAudio::StreamParameters parameters;
	parameters.deviceId = dac.getDefaultOutputDevice();
	parameters.nChannels = 2;

	unsigned int sampleRate = 44100;
	unsigned int bufferFrames = 256; // 256 sample frames

	RtAudio::StreamOptions options;
	options.flags = RTAUDIO_NONINTERLEAVED;


	try {
		dac.openStream( &parameters, NULL, RTAUDIO_FLOAT32,
		sampleRate, &bufferFrames, &myCallback, NULL, &options );
	}
	catch ( RtAudioError& e ) {
		std::cout << '\n' << e.getMessage() << '\n' << std::endl;
		exit( 0 );
	}
 
return 0;
}
