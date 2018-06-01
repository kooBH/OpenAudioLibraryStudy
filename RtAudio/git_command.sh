#!/bin/sh

#git clone https://rnqhsgur@bitbucket.org/rnqhsgur/custom_rtaudio.git
echo message? 
read msg

#넣을 파일들 위치에서
 
git add --all

git commit -m "$msg"

git push 
