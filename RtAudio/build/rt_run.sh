#!/bin/sh
path="/home/ffe"
sleep 6
if [ -d "$path" ]
then
if [ ! -d "$path"/wavs ]
then
mkdir "$path"/wavs
fi
gnome-terminal -x 'bash' -c '/home/ffe/custom_rtaudio/rtaudio/build/monitor "/home/ffe" 5 48000 9 60 "/home/ffe/custom_rtaudio/rtaudio/build";bash'
else
echo 경로가 존재하지 않습니다. setting을 다시 해주세요.
fi 
